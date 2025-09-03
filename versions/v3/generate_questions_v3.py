#!/usr/bin/env python3
"""
Pharma-Augment v3 - 임베딩 학습용 질문 생성기
Prompt3.md 스펙에 따른 엄격한 이름 사용 비율 및 대명사 차단 시스템
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import random
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import aiohttp
import backoff
from rapidfuzz import fuzz
from dotenv import load_dotenv
import tiktoken

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('question_generation_v3.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

class QuestionGeneratorV3:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", 
                 concurrency: int = 6, seed: int = 20250903):
        self.provider = provider
        self.model = model
        self.concurrency = concurrency
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
        
        # 대명사 차단 정규식 (Prompt3.md 6A)
        self.PRONOUN_RE = re.compile(r"(이|그|해당|본|동)\s?(약|약제|제제|제품)|이것|그것")
        
        # 카테고리 목록 (9개)
        self.CATEGORIES = [
            "범위", "요건", "오프라벨", "기간", "전환", 
            "증빙", "본인부담", "대상군", "절차"
        ]
        
        # 감사 로그
        self.audit_log = []
    
    def load_excel_data(self, excel_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """엑셀 파일 로드 및 컬럼 매핑"""
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            logger.info(f"엑셀 파일 로드 완료: {df.shape[0]}행 {df.shape[1]}열")
            
            # 고정 매핑 (요양심사약제-후처리 형식)
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
    
    def extract_drug_info(self, title: str, code_name: str = "", text: str = "") -> Tuple[str, List[str]]:
        """약제 정보 추출 (main_name, brand_names)"""
        # 기본 main_name은 code_name 또는 title에서 추출
        main_name = ""
        brand_names = []
        
        # code_name에서 main_name 추출
        if code_name:
            main_name = code_name.strip()
        elif title:
            # 제목에서 약제명 추출 시도
            title_clean = re.sub(r'\[.*?\]\s*', '', title).strip()
            main_name = title_clean
        
        # 텍스트에서 품명/브랜드명 추출
        # "품명 :" 패턴 찾기
        brand_pattern = r'품\s*명\s*[:：]\s*([^가-힣\n]*?)(?=\n|$|[가-힣])'
        brand_match = re.search(brand_pattern, text)
        
        if brand_match:
            brand_text = brand_match.group(1).strip()
            # 구분자로 분리 (·, /, ,, 등)
            brand_names = re.split(r'[·/,、]+', brand_text)
            brand_names = [name.strip() for name in brand_names if name.strip()]
        
        # 괄호 안의 품명도 추출
        paren_brands = re.findall(r'\(([^)]+(?:캅셀|주사|정|액|겔|크림)[^)]*)\)', text)
        for brand in paren_brands:
            if brand.strip() and brand.strip() not in brand_names:
                brand_names.append(brand.strip())
        
        return main_name, brand_names
    
    def calculate_ratios(self, brand_names: List[str]) -> Dict[str, float]:
        """브랜드명 개수에 따른 비율 계산 (Prompt3.md 2)"""
        num_brands = len(brand_names)
        
        if num_brands == 0:
            # 브랜드 0개: MAIN 70-80%, BOTH 20-30%
            main_ratio = round(random.uniform(0.70, 0.80), 2)
            both_ratio = round(1.0 - main_ratio, 2)
            return {"MAIN": main_ratio, "BRAND": 0.0, "BOTH": both_ratio}
        
        elif num_brands == 1:
            # 브랜드 1개: MAIN 35-45%, BRAND 30-40%, BOTH 20-30%
            main_ratio = round(random.uniform(0.35, 0.45), 2)
            brand_ratio = round(random.uniform(0.30, 0.40), 2)
            both_ratio = round(1.0 - main_ratio - brand_ratio, 2)
            return {"MAIN": main_ratio, "BRAND": brand_ratio, "BOTH": both_ratio}
        
        else:
            # 브랜드 2개 이상: MAIN 30-40%, BRAND 30-40%, BOTH 20-30%
            main_ratio = round(random.uniform(0.30, 0.40), 2)
            brand_ratio = round(random.uniform(0.30, 0.40), 2)
            both_ratio = round(1.0 - main_ratio - brand_ratio, 2)
            return {"MAIN": main_ratio, "BRAND": brand_ratio, "BOTH": both_ratio}
    
    def create_prompt_v3(self, drug_id: str, main_name: str, brand_names: List[str], content: str) -> str:
        """V3 프롬프트 생성 (Prompt3.md 5)"""
        brand_names_str = json.dumps(brand_names, ensure_ascii=False)
        
        return f"""[역할] 너는 의료 보험 임베딩 모델 학습용 질문 생성기다.

[입력]
- drug_id: {drug_id}
- main_name: {main_name}
- brand_names: {brand_names_str}
- content: \"\"\"{content}\"\"\"

[목표]
- content에 근거한 질문만 생성하라. 외부 지식/추정 금지.
- 아래 비율로 이름 사용을 강제:
  MAIN(주 약제명만) 30–40%, BRAND(품명만) 30–40%, BOTH(둘 다) 20–30%.
  *브랜드 0개/1개일 때의 재분배 규칙을 따를 것.*
- 대명사/간접 지칭은 사용 금지(예: "이/그/해당/본/동 제제·약제·제품", "이것/그것").

[형식/품질]
- 개방형 WH 문형 위주(무엇/어떤/언제/어떻게/왜)
- 1문항 1논점, 길이 15–70자
- 모호어·외부기관 금지
- 카테고리 9종 중 최소 4종 이상을 분산 태깅

[출력 스키마(필수)]
{{
  "drug_id": "...",
  "main_name": "...",
  "brand_names": ["..."],
  "questions": [
    {{"text":"...", "name_usage":"MAIN|BRAND|BOTH", "category":"범위|요건|오프라벨|기간|전환|증빙|본인부담|대상군|절차"}}
  ],
  "ratio": {{"MAIN":0.xx,"BRAND":0.xx,"BOTH":0.xx}}
}}

[검증]
- 각 질문의 text에 대명사/간접지칭이 포함되면 그 질문을 삭제하고 재생성하라.
- ratio 범위를 벗어나면 일부 질문의 name_usage와 문구를 조정해 재균형하라.
- JSON 외 텍스트 출력 금지."""
    
    def validate_pronoun(self, text: str) -> bool:
        """대명사 검증 (Prompt3.md 6A)"""
        return not self.PRONOUN_RE.search(text)
    
    def validate_name_usage(self, text: str, name_usage: str, main_name: str, brand_names: List[str]) -> bool:
        """이름 사용 검증 (Prompt3.md 6B)"""
        main_in_text = main_name in text if main_name else False
        brand_in_text = any(brand in text for brand in brand_names)
        
        if name_usage == "MAIN":
            return main_in_text and not brand_in_text
        elif name_usage == "BRAND":
            return brand_in_text and not main_in_text
        elif name_usage == "BOTH":
            return main_in_text and brand_in_text
        
        return False
    
    def validate_ratio(self, questions: List[Dict], expected_ratio: Dict[str, float]) -> bool:
        """비율 검증 (Prompt3.md 6C)"""
        if not questions:
            return False
        
        usage_counts = {"MAIN": 0, "BRAND": 0, "BOTH": 0}
        for q in questions:
            usage = q.get("name_usage", "")
            if usage in usage_counts:
                usage_counts[usage] += 1
        
        total = len(questions)
        actual_ratios = {k: v/total for k, v in usage_counts.items()}
        
        # ±0.02 허용
        for usage, expected in expected_ratio.items():
            actual = actual_ratios[usage]
            if abs(actual - expected) > 0.02:
                return False
        
        return True
    
    def post_process_v3(self, raw_response: Dict, main_name: str, brand_names: List[str]) -> Dict:
        """V3 후처리 및 검증"""
        questions = raw_response.get("questions", [])
        if not questions:
            return raw_response
        
        # 1. 대명사 필터링
        filtered_questions = []
        for q in questions:
            text = q.get("text", "")
            if self.validate_pronoun(text):
                filtered_questions.append(q)
            else:
                logger.warning(f"대명사 검출로 제거: {text}")
        
        # 2. 이름 사용 검증
        validated_questions = []
        for q in filtered_questions:
            text = q.get("text", "")
            name_usage = q.get("name_usage", "")
            if self.validate_name_usage(text, name_usage, main_name, brand_names):
                validated_questions.append(q)
            else:
                logger.warning(f"이름 사용 검증 실패: {text} ({name_usage})")
        
        # 3. 길이 필터링 (15-70자)
        length_filtered = []
        for q in validated_questions:
            text = q.get("text", "")
            if 15 <= len(text) <= 70:
                length_filtered.append(q)
            else:
                logger.warning(f"길이 제한 위반: {text} ({len(text)}자)")
        
        # 4. 카테고리 분산 확인
        categories = [q.get("category", "") for q in length_filtered]
        unique_categories = set(categories)
        if len(unique_categories) < 4:
            logger.warning(f"카테고리 부족: {unique_categories}")
        
        # 업데이트된 결과 반환
        raw_response["questions"] = length_filtered
        return raw_response
    
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3)
    async def call_api_v3(self, session: aiohttp.ClientSession, prompt: str, drug_id: str) -> Dict:
        """V3 API 호출"""
        if self.provider == "openai":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000,
                "response_format": {"type": "json_object"},
                "seed": self.seed
            }
            
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90)
            ) as response:
                if response.status == 429:
                    logger.warning(f"Rate limit for {drug_id}, retrying...")
                    await asyncio.sleep(3)
                    raise aiohttp.ClientError("Rate limit")
                
                response.raise_for_status()
                result = await response.json()
                
                try:
                    content = result['choices'][0]['message']['content']
                    return json.loads(content)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"JSON 파싱 실패 for {drug_id}: {e}")
                    return {}
        
        elif self.provider == "claude":
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.model,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                try:
                    content = result['content'][0]['text']
                    # JSON 부분만 추출
                    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    return {}
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"JSON 파싱 실패 for {drug_id}: {e}")
                    return {}
    
    async def generate_questions_for_drug(self, session: aiohttp.ClientSession, data: Dict) -> Dict:
        """단일 약제에 대한 질문 생성"""
        async with self.semaphore:
            start_time = asyncio.get_event_loop().time()
            
            # 기본 정보 추출
            clause_id = data['clause_id']
            title = data['title']
            text = data['text']
            code = data.get('code', '')
            code_name = data.get('code_name', '')
            
            try:
                # 약제 정보 추출
                main_name, brand_names = self.extract_drug_info(title, code_name, text)
                
                if not main_name:
                    main_name = f"Unknown_{clause_id}"
                
                # 기대 비율 계산
                expected_ratio = self.calculate_ratios(brand_names)
                
                # 프롬프트 생성
                prompt = self.create_prompt_v3(clause_id, main_name, brand_names, text)
                
                # API 호출
                raw_response = await self.call_api_v3(session, prompt, clause_id)
                
                # 후처리 및 검증
                validated_response = self.post_process_v3(raw_response, main_name, brand_names)
                
                # 최종 결과 구성
                result = {
                    "drug_id": clause_id,
                    "main_name": main_name,
                    "brand_names": brand_names,
                    "questions": validated_response.get("questions", []),
                    "ratio": expected_ratio,
                    "meta": {
                        "original_title": title,
                        "original_code": code,
                        "original_code_name": code_name,
                        "version": "v3",
                        "seed": self.seed,
                        "validation_passed": len(validated_response.get("questions", [])) > 0
                    }
                }
                
                # 감사 로그
                elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
                self.audit_log.append({
                    'drug_id': clause_id,
                    'main_name': main_name,
                    'brand_count': len(brand_names),
                    'questions_generated': len(result["questions"]),
                    'expected_ratio': expected_ratio,
                    'elapsed_ms': elapsed_ms,
                    'provider': self.provider,
                    'model': self.model
                })
                
                logger.info(f"완료: {clause_id} ({main_name}) - {len(result['questions'])}개 질문")
                return result
                
            except Exception as e:
                logger.error(f"질문 생성 실패 {clause_id}: {e}")
                # 실패해도 기본 구조 반환
                return {
                    "drug_id": clause_id,
                    "main_name": main_name if 'main_name' in locals() else "ERROR",
                    "brand_names": brand_names if 'brand_names' in locals() else [],
                    "questions": [],
                    "ratio": {"MAIN": 0.0, "BRAND": 0.0, "BOTH": 0.0},
                    "meta": {
                        "original_title": title,
                        "version": "v3",
                        "error": str(e)
                    }
                }
    
    def preprocess_data(self, df: pd.DataFrame) -> List[Dict]:
        """데이터 전처리"""
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                title = str(row['title']) if pd.notna(row['title']) else ""
                text = str(row['text']) if pd.notna(row['text']) else ""
                code = str(row['code']) if pd.notna(row.get('code')) else ""
                code_name = str(row['code_name']) if pd.notna(row.get('code_name')) else ""
                
                # clause_id 생성
                if code:
                    clause_id = re.sub(r'\s+', '', code)
                else:
                    title_slug = re.sub(r'[^\w\s가-힣]', '', title).replace(' ', '_')[:30]
                    hash8 = hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]
                    clause_id = f"{title_slug}_{hash8}"
                
                processed_data.append({
                    'clause_id': clause_id,
                    'title': title,
                    'text': text,
                    'code': code,
                    'code_name': code_name,
                    'original_row': idx
                })
                
            except Exception as e:
                logger.warning(f"행 {idx} 전처리 실패: {e}")
                continue
        
        logger.info(f"전처리 완료: {len(processed_data)}개 항목")
        return processed_data
    
    async def generate_all_questions(self, processed_data: List[Dict]) -> List[Dict]:
        """모든 약제에 대한 질문 생성"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.generate_questions_for_drug(session, item)
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
        """결과를 JSONL 형식으로 저장"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
            
            logger.info(f"결과 저장: {output_path} ({len(results)}개)")
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            raise
    
    def save_excel_results(self, results: List[Dict], output_path: str):
        """엑셀 형식으로 저장 (1질문당 1행)"""
        try:
            excel_data = []
            
            for result in results:
                drug_id = result.get('drug_id', '')
                main_name = result.get('main_name', '')
                brand_names = ', '.join(result.get('brand_names', []))
                original_title = result.get('meta', {}).get('original_title', '')
                
                for q in result.get('questions', []):
                    excel_data.append({
                        'drug_id': drug_id,
                        'main_name': main_name,
                        'brand_names': brand_names,
                        'original_title': original_title,
                        'question': q.get('text', ''),
                        'name_usage': q.get('name_usage', ''),
                        'category': q.get('category', ''),
                        'length': len(q.get('text', ''))
                    })
            
            df = pd.DataFrame(excel_data)
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            logger.info(f"엑셀 저장: {output_path} ({len(excel_data)}개 질문)")
            
        except Exception as e:
            logger.error(f"엑셀 저장 실패: {e}")
            raise
    
    def save_audit_log(self, output_dir: str = "."):
        """감사 로그 저장"""
        try:
            audit_path = os.path.join(output_dir, "audit_log_v3.csv")
            df = pd.DataFrame(self.audit_log)
            df.to_csv(audit_path, index=False, encoding='utf-8-sig')
            logger.info(f"감사 로그 저장: {audit_path}")
        except Exception as e:
            logger.error(f"감사 로그 저장 실패: {e}")
    
    def print_statistics(self, results: List[Dict]):
        """통계 출력"""
        if not results:
            print("결과가 없습니다.")
            return
        
        total_drugs = len(results)
        total_questions = sum(len(r.get('questions', [])) for r in results)
        
        # 이름 사용 통계
        usage_stats = {"MAIN": 0, "BRAND": 0, "BOTH": 0}
        category_stats = {cat: 0 for cat in self.CATEGORIES}
        
        for result in results:
            for q in result.get('questions', []):
                usage = q.get('name_usage', '')
                category = q.get('category', '')
                
                if usage in usage_stats:
                    usage_stats[usage] += 1
                if category in category_stats:
                    category_stats[category] += 1
        
        print(f"\n=== V3 질문 생성 통계 ===")
        print(f"총 약제 수: {total_drugs}")
        print(f"총 질문 수: {total_questions}")
        print(f"약제당 평균: {total_questions/total_drugs:.1f}개")
        
        print(f"\n=== 이름 사용 비율 ===")
        for usage, count in usage_stats.items():
            ratio = count / total_questions * 100 if total_questions > 0 else 0
            print(f"{usage}: {count}개 ({ratio:.1f}%)")
        
        print(f"\n=== 카테고리 분포 ===")
        for category, count in category_stats.items():
            ratio = count / total_questions * 100 if total_questions > 0 else 0
            print(f"{category}: {count}개 ({ratio:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Pharma-Augment V3 질문 생성기")
    
    # 필수 인자
    parser.add_argument("--excel", required=True, help="엑셀 파일 경로")
    
    # 선택적 인자
    parser.add_argument("--sheet", default="Sheet1", help="시트명")
    parser.add_argument("--out", default="questions_v3.jsonl", help="JSONL 출력 파일")
    parser.add_argument("--excel_out", default="questions_v3.xlsx", help="엑셀 출력 파일")
    parser.add_argument("--provider", choices=["openai", "claude"], default="openai", help="LLM 제공자")
    parser.add_argument("--model", default="gpt-4o-mini", help="모델명")
    parser.add_argument("--concurrency", type=int, default=6, help="동시 실행 수")
    parser.add_argument("--seed", type=int, default=20250903, help="랜덤 시드")
    
    args = parser.parse_args()
    
    try:
        # 질문 생성기 초기화
        generator = QuestionGeneratorV3(
            provider=args.provider,
            model=args.model,
            concurrency=args.concurrency,
            seed=args.seed
        )
        
        # 엑셀 데이터 로드
        df = generator.load_excel_data(args.excel, args.sheet)
        
        # 전처리
        processed_data = generator.preprocess_data(df)
        
        if not processed_data:
            logger.error("처리할 데이터가 없습니다.")
            sys.exit(1)
        
        # 질문 생성
        logger.info(f"V3 질문 생성 시작: {len(processed_data)}개 약제")
        results = asyncio.run(generator.generate_all_questions(processed_data))
        
        # 결과 저장
        generator.save_results(results, args.out)
        generator.save_excel_results(results, args.excel_out)
        generator.save_audit_log()
        
        # 통계 출력
        generator.print_statistics(results)
        
        logger.info("V3 질문 생성 완료!")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()