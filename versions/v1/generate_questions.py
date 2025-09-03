#!/usr/bin/env python3
"""
한국 요양급여 심사 도메인의 질문 생성 및 증강 스크립트
엑셀 파일에서 데이터를 읽고 LLM을 사용해 질문을 생성합니다.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import aiohttp
import backoff
from rapidfuzz import fuzz
from dotenv import load_dotenv
import tiktoken
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('question_generation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

class QuestionGenerator:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", 
                 concurrency: int = 6, max_aug: int = 15, seed: int = 20250902):
        self.provider = provider
        self.model = model
        self.concurrency = concurrency
        self.max_aug = max_aug
        self.seed = seed
        random.seed(seed)
        
        # API 키 설정
        if provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")
        elif provider == "claude":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY가 환경변수에 설정되지 않았습니다.")
        
        # 토큰 계산기 초기화
        if provider == "openai":
            try:
                self.tokenizer = tiktoken.encoding_for_model(model)
            except:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # 세마포어로 동시성 제어
        self.semaphore = asyncio.Semaphore(concurrency)
        
        # 감사 로그
        self.audit_log = []
        
        # 금지어 목록
        self.forbidden_words = [
            "추정", "일반적으로", "대체로", "관행상", "아마도", 
            "추측", "예상", "통상", "보통", "대개"
        ]
    
    def load_excel_data(self, excel_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """엑셀 파일 로드 및 컬럼 매핑"""
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            logger.info(f"엑셀 파일 로드 완료: {df.shape[0]}행 {df.shape[1]}열")
            
            # 고정 매핑 (프롬프트에 명시된 대로)
            column_mapping = {
                '약제분류번호': 'code',
                '약제분류명': 'code_name', 
                '구분': 'title',
                '세부인정기준 및 방법': 'text'
            }
            
            # 컬럼명 매핑
            mapped_columns = {}
            for original_col in df.columns:
                for target_key, target_col in column_mapping.items():
                    if target_key in original_col:
                        mapped_columns[original_col] = target_col
                        break
            
            # 컬럼 이름 변경
            df = df.rename(columns=mapped_columns)
            
            # 필수 컬럼 확인
            required_cols = ['title', 'text']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"필수 컬럼 '{col}'이 없습니다.")
            
            logger.info(f"컬럼 매핑 완료: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"엑셀 파일 로드 실패: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> List[Dict]:
        """데이터 전처리 및 clause_id 생성"""
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                # 기본 필드 추출
                title = str(row['title']) if pd.notna(row['title']) else ""
                text = str(row['text']) if pd.notna(row['text']) else ""
                code = str(row['code']) if pd.notna(row.get('code')) else ""
                code_name = str(row['code_name']) if pd.notna(row.get('code_name')) else ""
                
                # title에서 카테고리 추출 ([일반원칙] 등)
                category = None
                title_clean = title
                category_match = re.search(r'\[(.*?)\]', title)
                if category_match:
                    category = category_match.group(1)
                    title_clean = re.sub(r'\[.*?\]\s*', '', title).strip()
                
                # clause_id 생성
                clause_id = self.generate_clause_id(code, title_clean, text)
                
                # group_id 생성
                group_id = code if code else self.slugify(title_clean)
                
                # 긴 텍스트 슬라이싱 (6000자 초과 시)
                if len(text) > 6000:
                    chunks = self.slice_text(text)
                    for i, chunk in enumerate(chunks, 1):
                        chunk_data = {
                            'clause_id': f"{clause_id}_p{i}",
                            'group_id': group_id,
                            'title': title,
                            'title_clean': title_clean,
                            'category': category,
                            'code': code if code else None,
                            'code_name': code_name if code_name else None,
                            'text': chunk,
                            'original_row': idx
                        }
                        processed_data.append(chunk_data)
                else:
                    chunk_data = {
                        'clause_id': clause_id,
                        'group_id': group_id,
                        'title': title,
                        'title_clean': title_clean,
                        'category': category,
                        'code': code if code else None,
                        'code_name': code_name if code_name else None,
                        'text': text,
                        'original_row': idx
                    }
                    processed_data.append(chunk_data)
                    
            except Exception as e:
                logger.warning(f"행 {idx} 전처리 실패: {e}")
                continue
        
        logger.info(f"데이터 전처리 완료: {len(processed_data)}개 항목")
        return processed_data
    
    def generate_clause_id(self, code: str, title_clean: str, text: str) -> str:
        """clause_id 생성 규칙"""
        # id_part 생성
        id_part = re.sub(r'\s+', '', code) if code else ""
        
        # title_slug 생성 (40자 이하)
        title_slug = self.slugify(title_clean)[:40]
        
        # hash8 생성
        hash8 = hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]
        
        if id_part:
            return f"{id_part}_{title_slug}"
        else:
            return f"{title_slug}_{hash8}"
    
    def slugify(self, text: str) -> str:
        """텍스트를 슬러그로 변환"""
        # 소문자 변환, 공백을 하이픈으로, 특수문자 제거
        text = text.lower()
        text = re.sub(r'[^\w\s가-힣-]', '', text)
        text = re.sub(r'\s+', '-', text)
        text = text.strip('-')
        return text
    
    def slice_text(self, text: str, chunk_size: int = 2500) -> List[str]:
        """긴 텍스트를 문단 경계로 슬라이싱"""
        if len(text) <= chunk_size * 2:
            return [text]
        
        # 문단 경계로 분할 시도
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size * 1.2:  # 20% 여유
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def create_prompt(self, data: Dict) -> str:
        """질문 생성 프롬프트 생성"""
        return f"""[역할] 너는 한국 요양급여 심사 도메인의 '질문 생성 에이전트'다.
[제약] '질문만 생성'. 답변/근거/해설 절대 금지. 외부 지식/추정 금지.
[입력]
- clause_id: {data['clause_id']}
- title: {data['title']}
- title_clean: {data['title_clean']}
- category: {data['category']}
- code: {data['code']}
- code_name: {data['code_name']}
- text: \"\"\"
{data['text']}
\"\"\"

[출력 스키마]
{{
  "questions": ["문장1","문장2","..."]
}}

[생성 규칙]
1) '기본 5문형'을 먼저 생성:
   - 정의/범위, 요건/기준, 제외/불인정, 증빙/서류, 엣지/경계(재시술/합병증/동반상병/진료장소 등 '원문에 언급된 경우에만')
2) 이어서 '증강 질문' 5~{self.max_aug}개:
   - WH 다양화(무엇/언제/어디서/누가/어떤 조건/어떻게/왜)
   - 문체 변형(~인가요/~해야 하나요/~가능한가요/~허용되나요/~인정되나요)
   - 길이 변형(짧은 요점형 ↔ 구체 조건형)
   - 주체/시점 명시(신청자/의료기관/담당부서, 최초/재시술/추적관찰)
   - 조건 조합(~인 경우에도 인정되나요? / ~이면 제외되나요?)
3) **괄호 안 용어 활용**: 원문에 괄호가 있는 경우(예: 품명 : 프로그랍캅셀·주사 등), 괄호 안의 구체적 약품명이나 제형명을 질문에 적극 활용하세요.
4) **약제명 활용**: 'code_name'에 약제명이 있다면 질문에서 구체적으로 언급하여 실무적인 질문을 만드세요.
5) 원문에 없는 개념·수치·기관명·연도 등 삽입 금지.
6) 각 질문은 1문장, 한국어, 길이 15~180자.
7) JSON 외 출력 금지."""
    
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=2)
    async def call_openai_api(self, session: aiohttp.ClientSession, prompt: str, data: Dict) -> List[str]:
        """OpenAI API 호출"""
        text_length = len(data['text'])
        max_tokens = min(1200, max(800, text_length // 10))
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 429:
                logger.warning(f"Rate limit hit for {data['clause_id']}, retrying...")
                await asyncio.sleep(2)
                raise aiohttp.ClientError("Rate limit")
            
            response.raise_for_status()
            result = await response.json()
            
            try:
                content = result['choices'][0]['message']['content']
                parsed = json.loads(content)
                
                return parsed.get('questions', [])
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"JSON 파싱 실패 for {data['clause_id']}: {e}")
                return []
    
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=2)
    async def call_claude_api(self, session: aiohttp.ClientSession, prompt: str, data: Dict) -> List[str]:
        """Claude API 호출"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": f"{prompt}\n\nJSON 형식으로만 응답하세요:"}
            ]
        }
        
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            try:
                content = result['content'][0]['text']
                # JSON 부분만 추출 (배열 또는 객체)
                json_match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    
                    return parsed.get('questions', [])
                return []
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"JSON 파싱 실패 for {data['clause_id']}: {e}")
                return []
    
    def post_process_questions(self, questions: List[str]) -> List[str]:
        """질문 후처리: 정규화, 중복 제거, 필터링"""
        if not questions:
            return []
        
        # 1. 정규화
        normalized = []
        for q in questions:
            q = re.sub(r'\s+', ' ', str(q)).strip()
            q = re.sub(r'[.]{2,}', '...', q)
            if not q.endswith('?'):
                if q.endswith('.'):
                    q = q[:-1] + '?'
                elif not q.endswith(('?', '.', '!', '다', '요')):
                    q += '?'
            normalized.append(q)
        
        # 2. 금지어 필터링
        filtered = []
        for q in normalized:
            if not any(word in q for word in self.forbidden_words):
                filtered.append(q)
        
        # 3. 길이 필터링
        length_filtered = [q for q in filtered if 15 <= len(q) <= 180]
        
        # 4. 중복 제거 (rapidfuzz 사용)
        deduplicated = []
        for q in length_filtered:
            is_duplicate = False
            for existing_q in deduplicated:
                if fuzz.token_set_ratio(q, existing_q) >= 90:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(q)
        
        return deduplicated
    
    async def generate_questions_for_item(self, session: aiohttp.ClientSession, data: Dict) -> Dict:
        """단일 항목에 대한 질문 생성"""
        async with self.semaphore:
            start_time = asyncio.get_event_loop().time()
            clause_id = data['clause_id']
            retries = 0
            
            try:
                # 첫 번째 시도
                prompt = self.create_prompt(data)
                
                if self.provider == "openai":
                    questions = await self.call_openai_api(session, prompt, data)
                else:
                    questions = await self.call_claude_api(session, prompt, data)
                
                # 후처리
                processed_questions = self.post_process_questions(questions)
                
                # 5개 미만이면 한 번 더 시도 (증강만)
                if len(processed_questions) < 5:
                    retries = 1
                    aug_prompt = self.create_prompt(data) + "\n추가로 증강 질문만 5~10개 더 생성해주세요."
                    
                    if self.provider == "openai":
                        additional_questions = await self.call_openai_api(session, aug_prompt, data)
                    else:
                        additional_questions = await self.call_claude_api(session, aug_prompt, data)
                    
                    all_questions = questions + additional_questions
                    processed_questions = self.post_process_questions(all_questions)
                
                # 결과 구성
                result = {
                    'clause_id': data['clause_id'],
                    'group_id': data['group_id'],
                    'title': data['title'],
                    'title_clean': data['title_clean'],
                    'category': data['category'],
                    'code': data['code'],
                    'code_name': data['code_name'],
                    'text': data['text'],
                    'questions': processed_questions,
                    'meta': {
                        'source_sheet': 'Sheet1',
                        'mapping_confidence': 0.95,
                        'dedup_rule': 'rapidfuzz>=90',
                        'neg_ratio': 0.0,
                        'version': 'qgen_v1.2',
                        'seed': self.seed
                    }
                }
                
                # 감사 로그
                elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
                self.audit_log.append({
                    'clause_id': clause_id,
                    'num_questions': len(processed_questions),
                    'retries': retries,
                    'provider': self.provider,
                    'model': self.model,
                    'tokens_req': len(prompt) // 4,  # 대략적 추정
                    'tokens_resp': len(str(questions)) // 4,
                    'elapsed_ms': elapsed_ms
                })
                
                logger.info(f"완료: {clause_id} - {len(processed_questions)}개 질문 생성")
                return result
                
            except Exception as e:
                logger.error(f"질문 생성 실패 {clause_id}: {e}")
                # 실패해도 기본 구조 반환
                return {
                    'clause_id': data['clause_id'],
                    'group_id': data['group_id'],
                    'title': data['title'],
                    'title_clean': data['title_clean'],
                    'category': data['category'],
                    'code': data['code'],
                    'code_name': data['code_name'],
                    'text': data['text'],
                    'questions': [],
                    'meta': {
                        'source_sheet': 'Sheet1',
                        'mapping_confidence': 0.95,
                        'dedup_rule': 'rapidfuzz>=90',
                        'neg_ratio': 0.0,
                        'version': 'qgen_v1.2',
                        'seed': self.seed,
                        'error': str(e)
                    }
                }
    
    async def generate_all_questions(self, processed_data: List[Dict]) -> List[Dict]:
        """모든 항목에 대한 질문 생성"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.generate_questions_for_item(session, item)
                for item in processed_data
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 예외 처리
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"작업 실패: {result}")
                else:
                    valid_results.append(result)
            
            return valid_results
    
    def save_results(self, results: List[Dict], output_path: str):
        """결과를 JSONL 파일로 저장"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
            
            logger.info(f"결과 저장 완료: {output_path} ({len(results)}개 항목)")
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            raise
    
    def save_questions_per_row(self, results: List[Dict], output_path: str):
        """질문 1개당 1행으로 엑셀/CSV 저장"""
        try:
            questions_data = []
            
            for result in results:
                # 기본 정보 추출
                code = result.get('code', '') or ''
                code_name = result.get('code_name', '') or ''
                title = result.get('title', '') or ''
                text = result.get('text', '') or ''
                
                # 각 질문마다 1행 생성
                for question in result.get('questions', []):
                    questions_data.append({
                        '약제분류번호': code,
                        '약제분류명': code_name,
                        '구분': title,
                        '세부인정기준 및 방법': text,
                        'question': question
                    })
            
            # DataFrame 생성 및 저장
            df = pd.DataFrame(questions_data)
            
            # 파일 확장자에 따라 저장 방식 결정
            if output_path.lower().endswith('.csv'):
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif output_path.lower().endswith(('.xlsx', '.xls')):
                df.to_excel(output_path, index=False)
            else:
                # 기본값으로 엑셀 저장
                excel_path = output_path.rsplit('.', 1)[0] + '_questions.xlsx'
                df.to_excel(excel_path, index=False)
                output_path = excel_path
            
            logger.info(f"질문별 결과 저장 완료: {output_path} ({len(questions_data)}개 질문)")
            return output_path
            
        except Exception as e:
            logger.error(f"질문별 결과 저장 실패: {e}")
            raise
    
    def save_audit_log(self, output_dir: str = "."):
        """감사 로그 저장"""
        try:
            audit_path = os.path.join(output_dir, "audit_log.csv")
            df = pd.DataFrame(self.audit_log)
            df.to_csv(audit_path, index=False, encoding='utf-8')
            logger.info(f"감사 로그 저장 완료: {audit_path}")
        except Exception as e:
            logger.error(f"감사 로그 저장 실패: {e}")
    
    def print_sample(self, results: List[Dict], n: int = 5):
        """샘플 질문 출력"""
        if not results:
            print("생성된 결과가 없습니다.")
            return
        
        sample_results = random.sample(results, min(n, len(results)))
        
        print(f"\n=== 샘플 질문 {n}개 ===")
        for i, result in enumerate(sample_results, 1):
            print(f"\n[{i}] {result['clause_id']}")
            print(f"제목: {result['title_clean']}")
            print(f"카테고리: {result.get('category', 'N/A')}")
            print(f"질문 수: {len(result['questions'])}")
            print("질문들:")
            for j, q in enumerate(result['questions'][:3], 1):  # 처음 3개만
                print(f"  {j}. {q}")
            if len(result['questions']) > 3:
                print(f"  ... (총 {len(result['questions'])}개)")


def main():
    parser = argparse.ArgumentParser(description="요양급여 심사 도메인 질문 생성기")
    
    # 필수 인자
    parser.add_argument("--excel", required=True, help="엑셀 파일 경로")
    
    # 선택적 인자
    parser.add_argument("--sheet", default="Sheet1", help="시트명 (기본: Sheet1)")
    parser.add_argument("--out", default="questions.jsonl", help="출력 파일명 (기본: questions.jsonl)")
    parser.add_argument("--provider", choices=["openai", "claude"], default="openai", help="LLM 제공자")
    parser.add_argument("--model", default="gpt-4o-mini", help="모델명")
    parser.add_argument("--concurrency", type=int, default=6, help="동시 실행 수")
    parser.add_argument("--max_aug", type=int, default=15, help="최대 증강 질문 수")
    parser.add_argument("--seed", type=int, default=20250902, help="랜덤 시드")
    parser.add_argument("--base5_only", action="store_true", help="기본 5문형만 생성")
    parser.add_argument("--neg_ratio", type=float, default=0.0, help="네거티브 라벨 비율")
    parser.add_argument("--print-sample", type=int, metavar="N", help="N개 샘플 질문 출력")
    
    args = parser.parse_args()
    
    try:
        # 질문 생성기 초기화
        generator = QuestionGenerator(
            provider=args.provider,
            model=args.model,
            concurrency=args.concurrency,
            max_aug=1 if args.base5_only else args.max_aug,
            seed=args.seed
        )
        
        # 엑셀 파일 로드
        df = generator.load_excel_data(args.excel, args.sheet)
        
        # 데이터 전처리
        processed_data = generator.preprocess_data(df)
        
        if not processed_data:
            logger.error("처리할 데이터가 없습니다.")
            sys.exit(1)
        
        # 질문 생성 실행
        logger.info(f"질문 생성 시작: {len(processed_data)}개 항목")
        results = asyncio.run(generator.generate_all_questions(processed_data))
        
        # 결과 저장
        generator.save_results(results, args.out)
        
        # 질문 1개당 1행 형식으로도 저장 (엑셀)
        excel_output = args.out.rsplit('.', 1)[0] + '_questions.xlsx'
        generator.save_questions_per_row(results, excel_output)
        
        generator.save_audit_log()
        
        # 통계 출력
        total_questions = sum(len(r['questions']) for r in results)
        logger.info(f"완료: {len(results)}개 항목, {total_questions}개 질문 생성")
        
        # 샘플 출력
        if args.print_sample:
            generator.print_sample(results, args.print_sample)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()