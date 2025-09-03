#!/usr/bin/env python3
"""
의료 보험 임베딩 모델 학습용 향상된 질문 생성기 V2
약제명/품명 균형, 질문 다양성, 실무 시나리오 기반 질문 생성
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
        logging.FileHandler('question_generation_v2.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

class EnhancedQuestionGenerator:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", 
                 concurrency: int = 6, max_aug: int = 20, seed: int = 20250902):
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
        
        # 질문 템플릿 정의
        self.question_templates = {
            'basic': [
                "{drug_name}의 요양급여 인정 기준은?",
                "{drug_name}은 어떤 질환에 사용 가능한가요?",
                "{drug_name}의 적응증은 무엇인가요?",
                "{drug_name}의 급여 적용 범위는?",
                "{drug_name}의 보험 코드는?"
            ],
            'conditional': [
                "{condition}에서 {drug_name}을 사용할 때 급여 기준은?",
                "{drug_name}을 {patient_group}에게 투여 시 제한사항은?",
                "만약 {condition}이라면 {drug_name} 사용이 가능한가요?",
                "{specific_situation}일 때 {drug_name} 투여 방법은?",
                "{drug_name}을 {duration} 사용 시 필요한 조건은?"
            ],
            'comparative': [
                "{drug_name}의 경구제와 주사제 급여 기준 차이는?",
                "{drug_name}과 다른 {drug_class}의 인정 기준 비교는?",
                "성인과 소아에서 {drug_name} 사용 기준의 차이점은?",
                "{drug_name}과 {alternative_drug}의 급여 기준 차이는?",
                "입원과 외래에서 {drug_name} 청구 방법 차이는?"
            ],
            'negative': [
                "{drug_name}이 급여 인정되지 않는 경우는?",
                "{drug_name}의 사용이 제한되는 환자군은?",
                "어떤 경우에 {drug_name} 삭감이 발생하나요?",
                "{drug_name} 처방이 불인정되는 상황은?",
                "{drug_name}의 금기사항은?"
            ],
            'procedural': [
                "{drug_name} 사용을 위한 사전승인 절차는?",
                "{drug_name} 처방 시 필요한 검사 항목은?",
                "어떻게 {drug_name}의 급여를 신청하나요?",
                "{drug_name} 투여 전 준비사항은?",
                "{drug_name} 청구 시 첨부 서류는?"
            ],
            'practical': [
                "{drug_name} 처방이 삭감된 경우 이의신청 근거는?",
                "{drug_name} 장기처방 시 삭감 예방 방법은?",
                "비급여 전환을 피하기 위한 {drug_name} 처방 전략은?",
                "{drug_name}와 관련 검사료 동시 청구 기준은?",
                "{drug_name} 투여 중 모니터링 항목은?"
            ]
        }
        
        # 난이도별 수식어
        self.difficulty_modifiers = {
            'easy': ['기본적인', '일반적인', '단순한'],
            'medium': ['특정 조건에서의', '복잡한 상황의', '예외적인'],
            'hard': ['다제내성', '복합 질환', '중증', '응급상황의']
        }
    
    def extract_drug_info(self, title: str, code_name: str = None) -> Dict:
        """제목에서 약제 정보 추출"""
        drug_info = {
            'main_name': '',
            'brand_names': [],
            'category': title,
            'generic_terms': []
        }
        
        # 주 약제명 추출 (괄호 앞 부분)
        main_match = re.match(r'^([^(]+)', title)
        if main_match:
            drug_info['main_name'] = main_match.group(1).strip()
        
        # 품명 추출 (품명 : 로 시작하는 부분)
        brand_match = re.search(r'품명\s*:\s*([^)]+)', title)
        if brand_match:
            brands = brand_match.group(1).split('·')
            drug_info['brand_names'] = [b.strip() for b in brands]
        
        # 괄호 안 모든 내용 추출
        all_brackets = re.findall(r'\(([^)]+)\)', title)
        for bracket_content in all_brackets:
            if '품명' not in bracket_content:
                drug_info['generic_terms'].append(bracket_content.strip())
        
        # code_name이 있으면 추가
        if code_name:
            drug_info['brand_names'].append(code_name)
        
        return drug_info
    
    def generate_balanced_drug_references(self, drug_info: Dict, count: int) -> List[str]:
        """약제명/품명을 균형있게 사용한 참조 생성"""
        references = []
        
        # 30-40% 주 약제명
        main_count = max(1, int(count * 0.35))
        references.extend([drug_info['main_name']] * main_count)
        
        # 30-40% 품명
        if drug_info['brand_names']:
            brand_count = max(1, int(count * 0.35))
            for i in range(brand_count):
                brand = random.choice(drug_info['brand_names'])
                references.append(brand)
        
        # 20-30% 둘 다 사용
        both_count = max(1, int(count * 0.25))
        if drug_info['brand_names']:
            for i in range(both_count):
                brand = random.choice(drug_info['brand_names'])
                combined = f"{drug_info['main_name']}({brand})"
                references.append(combined)
        
        # 나머지 10% 간접 지칭
        remaining = count - len(references)
        indirect_terms = ['이 약제', '해당 약물', '이 제제', '해당 치료제']
        for i in range(remaining):
            references.append(random.choice(indirect_terms))
        
        return references[:count]
    
    def create_enhanced_prompt(self, data: Dict) -> str:
        """향상된 질문 생성 프롬프트"""
        drug_info = self.extract_drug_info(data['title'], data.get('code_name'))
        
        return f"""당신은 의료보험 임베딩 모델 학습용 질문 생성 전문가입니다.
벡터 검색 성능 향상을 위해 다양하고 균형잡힌 질문을 생성하세요.

[약제 정보]
- 주 약제명: {drug_info['main_name']}
- 품명: {', '.join(drug_info['brand_names']) if drug_info['brand_names'] else '없음'}
- 기타 용어: {', '.join(drug_info['generic_terms']) if drug_info['generic_terms'] else '없음'}

[원문 내용]
{data['text']}

[생성 규칙]
1) **약제명 사용 균형** (매우 중요):
   - 30% 주 약제명만 사용: "{drug_info['main_name']}의 ..."
   - 30% 품명만 사용: "{', '.join(drug_info['brand_names'][:1])}의 ..." (있는 경우)
   - 25% 둘 다 사용: "{drug_info['main_name']}({', '.join(drug_info['brand_names'][:1])})의 ..." 
   - 15% 간접 지칭: "이 약제의 ...", "해당 치료제의 ..."

2) **질문 유형 다양화** (15-25개 생성):
   - 기본형(30%): "~의 급여 기준은?", "~은 어떤 질환에?"
   - 조건형(25%): "~한 경우 ~을 사용할 때", "만약 ~라면"
   - 비교형(15%): "~와 ~의 차이", "성인과 소아에서"
   - 부정형(15%): "~가 인정되지 않는 경우", "제한되는 상황"
   - 실무형(15%): "삭감 방어", "청구 방법", "이의신청"

3) **구체적 내용 활용**:
   - 원문의 수치, 용량, 기간을 질문에 포함
   - 특정 환자군, 질환명, 검사명 활용
   - 괄호 안 용어 적극 활용

4) **출력 형식**:
{{"questions": ["질문1", "질문2", "..."]}}

JSON만 출력하세요. 다른 설명은 불필요합니다."""

    def load_excel_data(self, excel_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """엑셀 파일 로드 및 컬럼 매핑"""
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            logger.info(f"엑셀 파일 로드 완료: {df.shape[0]}행 {df.shape[1]}열")
            
            # 고정 매핑
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
        """데이터 전처리"""
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                # 기본 필드 추출
                title = str(row['title']) if pd.notna(row['title']) else ""
                text = str(row['text']) if pd.notna(row['text']) else ""
                code = str(row['code']) if pd.notna(row.get('code')) else ""
                code_name = str(row['code_name']) if pd.notna(row.get('code_name')) else ""
                
                # title에서 카테고리 추출
                category = None
                title_clean = title
                category_match = re.search(r'\[(.*?)\]', title)
                if category_match:
                    category = category_match.group(1)
                    title_clean = re.sub(r'\[.*?\]\s*', '', title).strip()
                
                # clause_id 생성
                clause_id = self.generate_clause_id(code, title_clean, text)
                group_id = code if code else self.slugify(title_clean)
                
                # 슬라이싱 처리
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
        """clause_id 생성"""
        id_part = re.sub(r'\s+', '', code) if code else ""
        title_slug = self.slugify(title_clean)[:40]
        hash8 = hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]
        
        if id_part:
            return f"{id_part}_{title_slug}"
        else:
            return f"{title_slug}_{hash8}"
    
    def slugify(self, text: str) -> str:
        """텍스트를 슬러그로 변환"""
        text = text.lower()
        text = re.sub(r'[^\w\s가-힣-]', '', text)
        text = re.sub(r'\s+', '-', text)
        text = text.strip('-')
        return text
    
    def slice_text(self, text: str, chunk_size: int = 2500) -> List[str]:
        """긴 텍스트 슬라이싱"""
        if len(text) <= chunk_size * 2:
            return [text]
        
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size * 1.2:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=2)
    async def call_openai_api(self, session: aiohttp.ClientSession, prompt: str, data: Dict) -> List[str]:
        """OpenAI API 호출"""
        text_length = len(data['text'])
        max_tokens = min(1500, max(800, text_length // 8))  # 더 많은 토큰 할당
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,  # 다양성을 위해 조금 높임
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
    
    def post_process_questions(self, questions: List[str]) -> List[str]:
        """질문 후처리"""
        if not questions:
            return []
        
        # 정규화
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
        
        # 길이 필터링 (15~200자로 조금 더 여유롭게)
        length_filtered = [q for q in normalized if 15 <= len(q) <= 200]
        
        # 중복 제거
        deduplicated = []
        for q in length_filtered:
            is_duplicate = False
            for existing_q in deduplicated:
                if fuzz.token_set_ratio(q, existing_q) >= 85:  # 조금 더 엄격하게
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
                prompt = self.create_enhanced_prompt(data)
                
                if self.provider == "openai":
                    questions = await self.call_openai_api(session, prompt, data)
                else:
                    # Claude API는 기본 구현 생략 (필요시 추가)
                    questions = []
                
                # 후처리
                processed_questions = self.post_process_questions(questions)
                
                # 10개 미만이면 한 번 더 시도
                if len(processed_questions) < 10:
                    retries = 1
                    aug_prompt = prompt + "\n\n추가 요청: 더 다양한 관점에서 5-15개 추가 질문을 생성해주세요."
                    
                    if self.provider == "openai":
                        additional_questions = await self.call_openai_api(session, aug_prompt, data)
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
                        'version': 'embedding_v2.0',
                        'seed': self.seed,
                        'generation_strategy': 'balanced_drug_names',
                        'question_diversity': 'enhanced'
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
                    'tokens_req': len(prompt) // 4,
                    'tokens_resp': len(str(questions)) // 4,
                    'elapsed_ms': elapsed_ms
                })
                
                logger.info(f"완료: {clause_id} - {len(processed_questions)}개 질문 생성")
                return result
                
            except Exception as e:
                logger.error(f"질문 생성 실패 {clause_id}: {e}")
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
                        'version': 'embedding_v2.0',
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
                code = result.get('code', '') or ''
                code_name = result.get('code_name', '') or ''
                title = result.get('title', '') or ''
                text = result.get('text', '') or ''
                
                for question in result.get('questions', []):
                    questions_data.append({
                        '약제분류번호': code,
                        '약제분류명': code_name,
                        '구분': title,
                        '세부인정기준 및 방법': text,
                        'question': question
                    })
            
            df = pd.DataFrame(questions_data)
            
            if output_path.lower().endswith('.csv'):
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif output_path.lower().endswith(('.xlsx', '.xls')):
                df.to_excel(output_path, index=False)
            else:
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
            audit_path = os.path.join(output_dir, "audit_log_v2.csv")
            df = pd.DataFrame(self.audit_log)
            df.to_csv(audit_path, index=False, encoding='utf-8')
            logger.info(f"감사 로그 저장 완료: {audit_path}")
        except Exception as e:
            logger.error(f"감사 로그 저장 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="임베딩 모델 학습용 향상된 질문 생성기 V2")
    
    parser.add_argument("--excel", required=True, help="엑셀 파일 경로")
    parser.add_argument("--sheet", default="Sheet1", help="시트명")
    parser.add_argument("--out", default="questions_v2.jsonl", help="출력 파일명")
    parser.add_argument("--provider", choices=["openai", "claude"], default="openai", help="LLM 제공자")
    parser.add_argument("--model", default="gpt-4o-mini", help="모델명")
    parser.add_argument("--concurrency", type=int, default=6, help="동시 실행 수")
    parser.add_argument("--max_aug", type=int, default=20, help="최대 증강 질문 수")
    parser.add_argument("--seed", type=int, default=20250902, help="랜덤 시드")
    
    args = parser.parse_args()
    
    try:
        # 질문 생성기 초기화
        generator = EnhancedQuestionGenerator(
            provider=args.provider,
            model=args.model,
            concurrency=args.concurrency,
            max_aug=args.max_aug,
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
        logger.info(f"향상된 질문 생성 시작: {len(processed_data)}개 항목")
        results = asyncio.run(generator.generate_all_questions(processed_data))
        
        # 결과 저장
        generator.save_results(results, args.out)
        
        # 질문 1개당 1행 형식으로도 저장
        excel_output = args.out.rsplit('.', 1)[0] + '_questions.xlsx'
        generator.save_questions_per_row(results, excel_output)
        
        generator.save_audit_log()
        
        # 통계 출력
        total_questions = sum(len(r['questions']) for r in results)
        avg_questions = total_questions / len(results) if results else 0
        logger.info(f"완료: {len(results)}개 항목, {total_questions}개 질문 생성 (평균 {avg_questions:.1f}개/항목)")
        
        # 약제명/품명 사용 분석
        analyze_drug_name_balance(results)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        sys.exit(1)


def analyze_drug_name_balance(results: List[Dict]):
    """약제명/품명 사용 균형 분석"""
    all_questions = []
    for result in results:
        all_questions.extend(result['questions'])
    
    if not all_questions:
        return
    
    # 간단한 분석 (실제로는 더 정교한 분석 필요)
    main_name_count = sum(1 for q in all_questions if any(term in q for term in ['제제', 'Tacrolimus', '경구용']))
    brand_name_count = sum(1 for q in all_questions if any(term in q for term in ['프로그랍', '캅셀', '주사']))
    indirect_count = sum(1 for q in all_questions if any(term in q for term in ['이 약제', '해당 약물']))
    
    logger.info(f"약제명 사용 분석:")
    logger.info(f"- 주 약제명 추정: {main_name_count}개 ({main_name_count/len(all_questions)*100:.1f}%)")
    logger.info(f"- 품명 추정: {brand_name_count}개 ({brand_name_count/len(all_questions)*100:.1f}%)")
    logger.info(f"- 간접 지칭 추정: {indirect_count}개 ({indirect_count/len(all_questions)*100:.1f}%)")


if __name__ == "__main__":
    main()