#!/usr/bin/env python3
"""
Pharma-Augment V5 Simple - 작동하는 간단한 버전
V2 스타일 복원 + 대명사 차단
"""

import argparse
import asyncio
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
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drug_generation_v5_simple.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

class DrugGeneratorV5Simple:
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
        
        # 세마포어로 동시성 제어
        self.semaphore = asyncio.Semaphore(concurrency)
        
        # 대명사 차단 정규식
        self.PRONOUN_RE = re.compile(r"(이|그|해당|본|동)\s?(약|약제|제제|제품)|이것|그것")
        
        # 감사 로그
        self.audit_log = []
        self.progress_bar = None
    
    def load_excel_data(self, excel_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """엑셀 파일 로드"""
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            logger.info(f"엑셀 파일 로드 완료: {df.shape[0]}행 {df.shape[1]}열")
            
            # 컬럼명 확인 및 매핑
            expected_cols = ['약제분류번호', '약제 분류명', '구분', '세부인정기준 및 방법']
            actual_cols = list(df.columns)
            
            logger.info(f"실제 컬럼: {actual_cols}")
            
            # 컬럼 순서대로 매핑
            if len(actual_cols) >= 4:
                column_mapping = dict(zip(actual_cols[:4], expected_cols))
                df = df.rename(columns=column_mapping)
                logger.info(f"컬럼 매핑 완료: {column_mapping}")
            
            return df
            
        except Exception as e:
            logger.error(f"엑셀 파일 로드 실패: {e}")
            raise
    
    def extract_drug_info(self, gubun: str) -> Tuple[str, List[str]]:
        """약제 정보 추출"""
        # 괄호 패턴 제거 후 추출
        clean_gubun = re.sub(r'\[.*?\]\s*', '', gubun).strip()
        
        # main_name = 괄호 앞
        main_name = ""
        brand_names = []
        
        paren_match = re.search(r'^([^(]+)', clean_gubun)
        if paren_match:
            main_name = paren_match.group(1).strip()
        
        # 품명 추출
        brand_match = re.search(r'품\s*명\s*[:：]\s*([^)]+)', clean_gubun)
        if brand_match:
            brand_text = brand_match.group(1).strip()
            brand_names = re.split(r'[·/]+', brand_text)
            brand_names = [name.strip() for name in brand_names if name.strip()]
        
        return main_name, brand_names
    
    def create_simple_prompt(self, row_data: Dict) -> str:
        """간단한 V5 프롬프트"""
        drug_code = row_data.get('약제분류번호', '')
        drug_name = row_data.get('약제 분류명', '')
        gubun = row_data.get('구분', '')
        content = row_data.get('세부인정기준 및 방법', '')
        
        main_name, brand_names = self.extract_drug_info(gubun)
        
        prompt = f"""너는 의료 보험 질문 생성 전문가다.

[입력 정보]
약제분류번호: {drug_code}
약제 분류명: {drug_name}
구분: {gubun}
주 약제명: {main_name}
품명들: {brand_names}
세부인정기준: {content}

[생성 규칙]
1. 8-15개의 질문을 생성하라
2. 질문 길이: 25-80자로 구체적이고 상세하게
3. 대명사 절대 금지: "이것", "그것", "해당 약제", "본 제제" 등 사용 금지
4. 구체적이고 실무적인 질문 생성:
   - 구체적 수치 포함: "AST 60U/L 이상일 때", "3개월 이상 처방 시", "200mg 투여 시"
   - 환자군 명확히 명시: "소아 환자", "고령자", "간기능 저하 환자", "신부전 환자"
   - 전문 의료 용어 사용: "급여 인정", "사전승인", "모니터링", "용량 조절", "투여 중단"
   - 임상적 상황 반영: "부작용 발생 시", "효과 부족 시", "병용 투여 시"
5. 다양한 질문 유형:
   - 기본 정보: "급여 인정 기준은 무엇인가요?"
   - 조건 상황: "어떤 조건에서 사용 가능한가요?"
   - 비교형: "경구제와 주사제 차이점은?"
   - 절차형: "처방 절차는 어떻게 되나요?"
   - 제한형: "사용이 제한되는 경우는?"

[라벨링]
- POSITIVE: 입력 내용에 직접 관련된 질문 (70%)
- NEGATIVE: 완전히 다른 약제나 상황 (15%)
- HARD_NEGATIVE: 비슷하지만 핵심이 다른 질문 (15%)

[출력 형식 - 유효한 JSON만 출력하라]
JSON 배열 형식으로 출력하되, 반드시 유효한 JSON 구문을 사용하라:
[
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"내용요약...","question":"구체적이고 상세한 질문 (25-80자)","라벨":"POSITIVE"}},
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"내용요약...","question":"또 다른 구체적인 질문","라벨":"POSITIVE"}}
]

반드시 25자 이상의 구체적이고 실무적인 질문을 생성하라. 출력은 JSON 배열만 해라."""

        return prompt
    
    def create_fallback_prompt(self, row_data: Dict) -> str:
        """간단한 Fallback 프롬프트 (더 관대한 기준)"""
        drug_code = row_data.get('약제분류번호', '')
        drug_name = row_data.get('약제 분류명', '')
        gubun = row_data.get('구분', '')
        content = row_data.get('세부인정기준 및 방법', '')
        
        main_name, brand_names = self.extract_drug_info(gubun)
        
        prompt = f"""너는 의료 보험 질문 생성 전문가다.

[입력 정보]  
약제분류번호: {drug_code}
약제 분류명: {drug_name}
구분: {gubun}
주 약제명: {main_name}
품명들: {brand_names}
세부인정기준: {content}

[생성 규칙 - 관대한 기준]
1. 10-15개의 질문을 생성하라
2. 질문 길이: 20자 이상 (더 짧아도 허용)  
3. 대명사 절대 금지: "이것", "그것", "해당 약제", "본 제제" 등 사용 금지
4. 기본적인 의료 질문 형태로 생성
5. 다양한 유형: 정보형, 조건형, 비교형, 절차형, 제한형

[출력 형식]
JSON 배열만 출력하라:
[
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"내용요약","question":"20자 이상의 질문","라벨":"POSITIVE"}}
]"""

        return prompt
    
    def post_process_simple(self, questions: List[Dict]) -> List[Dict]:
        """간단한 후처리"""
        if not questions:
            return []
        
        processed = []
        
        for q in questions:
            text = q.get("question", "")
            label = q.get("라벨", "")
            
            # 1. 대명사 검증 (엄격)
            if self.PRONOUN_RE.search(text):
                logger.warning(f"V5 대명사 검출: {text}")
                continue
            
            # 2. 길이 검증 (20-80자로 완화)
            if not (20 <= len(text) <= 80):
                if len(text) < 20:
                    logger.warning(f"V5 길이 부족: {text} ({len(text)}자)")
                    continue
                elif len(text) > 80:
                    logger.warning(f"V5 길이 초과: {text} ({len(text)}자)")
                    continue
            
            # 3. 라벨 검증 (완화)
            if label not in ["POSITIVE", "NEGATIVE", "HARD_NEGATIVE"]:
                q["라벨"] = "POSITIVE"
            
            processed.append(q)
        
        return processed
    
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3)
    async def call_api(self, session: aiohttp.ClientSession, prompt: str, row_id: str) -> List[Dict]:
        """API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 3000,
            "seed": self.seed
        }
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 429:
                logger.warning(f"Rate limit for {row_id}, retrying...")
                await asyncio.sleep(3)
                raise aiohttp.ClientError("Rate limit")
            
            response.raise_for_status()
            result = await response.json()
            
            try:
                content = result['choices'][0]['message']['content'].strip()
                
                # JSON 응답 정리
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].strip()
                
                # JSON 파싱 시도
                if content.startswith('['):
                    return json.loads(content)
                elif content.startswith('{'):
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        return [parsed]
                    return parsed
                else:
                    # JSON이 아닌 경우, 텍스트에서 JSON 추출 시도
                    json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    
                    # 완전 실패
                    logger.error(f"JSON 형식 불가 for {row_id}: {content[:100]}...")
                    return []
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"JSON 파싱 실패 for {row_id}: {e}")
                logger.error(f"Content: {content[:200]}...")
                return []
    
    async def generate_questions_for_drug(self, session: aiohttp.ClientSession, row_data: Dict, row_idx: int) -> List[Dict]:
        """단일 약제 질문 생성"""
        async with self.semaphore:
            start_time = time.time()
            row_id = f"v5_simple_row_{row_idx}"
            
            try:
                gubun = str(row_data.get('구분', ''))
                main_name, brand_names = self.extract_drug_info(gubun)
                content = str(row_data.get('세부인정기준 및 방법', ''))
                
                # 프롬프트 생성
                prompt = self.create_simple_prompt(row_data)
                
                # API 호출
                raw_questions = await self.call_api(session, prompt, row_id)
                
                # 후처리
                validated_questions = self.post_process_simple(raw_questions)
                
                # V5 Fallback: 질문이 부족하면 재시도
                if len(validated_questions) < 5:
                    logger.warning(f"V5 Fallback 시도 for {row_id}: {len(validated_questions)}개 -> 재시도")
                    fallback_prompt = self.create_fallback_prompt(row_data)
                    fallback_questions = await self.call_api(session, fallback_prompt, f"{row_id}_fallback")
                    fallback_validated = self.post_process_simple(fallback_questions)
                    validated_questions.extend(fallback_validated)
                
                # 감사 로그
                elapsed_ms = int((time.time() - start_time) * 1000)
                avg_length = sum(len(q.get('question', '')) for q in validated_questions) / max(1, len(validated_questions))
                
                self.audit_log.append({
                    'row_id': row_id,
                    'row_idx': row_idx,
                    'main_name': main_name,
                    'brand_count': len(brand_names),
                    'questions_generated': len(validated_questions),
                    'avg_length': avg_length,
                    'elapsed_ms': elapsed_ms,
                    'provider': self.provider,
                    'model': self.model,
                    'version': 'v5_simple'
                })
                
                # 진행상황 업데이트
                if self.progress_bar:
                    self.progress_bar.update(1)
                
                logger.info(f"V5 Simple 완료: {row_id} - {len(validated_questions)}개 질문 (평균 {int(avg_length)}자)")
                return validated_questions
                
            except Exception as e:
                logger.error(f"V5 Simple 질문 생성 실패 {row_id}: {e}")
                if self.progress_bar:
                    self.progress_bar.update(1)
                return []
    
    def preprocess_data(self, df: pd.DataFrame) -> List[Dict]:
        """데이터 전처리"""
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                row_dict = {}
                for col in ['약제분류번호', '약제 분류명', '구분', '세부인정기준 및 방법']:
                    if col in row:
                        val = row[col]
                        if pd.isna(val):
                            row_dict[col] = ""
                        else:
                            row_dict[col] = str(val).strip()
                    else:
                        row_dict[col] = ""
                
                # 필수 필드 검증
                if not row_dict.get('구분') or not row_dict.get('세부인정기준 및 방법'):
                    logger.warning(f"행 {idx}: 필수 필드 누락")
                    continue
                
                processed_data.append(row_dict)
                
            except Exception as e:
                logger.warning(f"행 {idx} 전처리 실패: {e}")
                continue
        
        logger.info(f"V5 Simple 전처리 완료: {len(processed_data)}개 항목")
        return processed_data
    
    async def generate_all_questions(self, processed_data: List[Dict]) -> List[Dict]:
        """모든 약제 질문 생성"""
        # 진행상황 바
        self.progress_bar = tqdm(total=len(processed_data), desc="V5 Simple 생성", unit="약제")
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.generate_questions_for_drug(session, item, idx)
                for idx, item in enumerate(processed_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            if self.progress_bar:
                self.progress_bar.close()
                self.progress_bar = None
            
            # 결과 정리
            all_questions = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"행 {idx} 실패: {result}")
                elif isinstance(result, list):
                    all_questions.extend(result)
            
            return all_questions
    
    def save_final_results(self, questions: List[Dict], output_path: str):
        """최종 결과 저장"""
        try:
            df = pd.DataFrame(questions)
            
            # 컬럼 순서 고정
            final_columns = ['약제분류번호', '약제 분류명', '구분', '세부인정기준 및 방법', 'question', '라벨']
            for col in final_columns:
                if col not in df.columns:
                    df[col] = ""
            
            df = df[final_columns]
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            logger.info(f"V5 Simple 결과 저장: {output_path} ({len(questions)}개 질문)")
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            raise
    
    def save_audit_log(self, output_dir: str = "."):
        """감사 로그 저장"""
        try:
            audit_path = os.path.join(output_dir, "audit_log_drug_v5_simple.csv")
            df = pd.DataFrame(self.audit_log)
            df.to_csv(audit_path, index=False, encoding='utf-8-sig')
            logger.info(f"V5 Simple 감사 로그 저장: {audit_path}")
        except Exception as e:
            logger.error(f"감사 로그 저장 실패: {e}")
    
    def print_statistics(self, questions: List[Dict]):
        """통계 출력"""
        if not questions:
            print("V5 Simple 결과가 없습니다.")
            return
        
        total_questions = len(questions)
        
        # 길이 통계
        lengths = [len(q.get('question', '')) for q in questions]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        min_length = min(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0
        
        # 라벨 분포
        label_counts = {}
        for q in questions:
            label = q.get('라벨', 'UNKNOWN')
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 통계 출력
        print(f"\n=== V5 Simple 약제 질문 생성 통계 ===")
        print(f"총 질문 수: {total_questions}")
        print(f"평균 길이: {avg_length:.1f}자")
        print(f"길이 범위: {min_length}-{max_length}자")
        
        print(f"\n=== 라벨 분포 ===")
        for label, count in label_counts.items():
            ratio = count / total_questions * 100 if total_questions > 0 else 0
            print(f"{label}: {count}개 ({ratio:.1f}%)")
        
        # V2와 비교
        print(f"\n=== V2 비교 ===")
        print(f"V2 평균 길이: 36.2자")
        print(f"V5 평균 길이: {avg_length:.1f}자")
        if avg_length >= 30:
            print("V5가 V2 수준 달성!")
        else:
            print("V5 길이 개선 필요")


def main():
    parser = argparse.ArgumentParser(description="Pharma-Augment V5 Simple 약제 질문 생성기")
    
    parser.add_argument("--excel", required=True, help="약제 엑셀 파일 경로")
    parser.add_argument("--sheet", default="Sheet1", help="시트명")
    parser.add_argument("--out", default="drug_questions_v5_simple.xlsx", help="출력 파일")
    parser.add_argument("--provider", choices=["openai"], default="openai", help="LLM 제공자")
    parser.add_argument("--model", default="gpt-4o-mini", help="모델명")
    parser.add_argument("--concurrency", type=int, default=4, help="동시 실행 수")
    parser.add_argument("--seed", type=int, default=20250903, help="랜덤 시드")
    
    args = parser.parse_args()
    
    try:
        generator = DrugGeneratorV5Simple(
            provider=args.provider,
            model=args.model,
            concurrency=args.concurrency,
            seed=args.seed
        )
        
        print("V5 Simple 데이터 로드 중...")
        df = generator.load_excel_data(args.excel, args.sheet)
        
        print("V5 Simple 전처리 중...")
        processed_data = generator.preprocess_data(df)
        
        if not processed_data:
            logger.error("처리할 데이터가 없습니다.")
            sys.exit(1)
        
        print(f"V5 Simple 질문 생성 시작: {len(processed_data)}개 행")
        results = asyncio.run(generator.generate_all_questions(processed_data))
        
        print("V5 Simple 결과 저장 중...")
        generator.save_final_results(results, args.out)
        generator.save_audit_log()
        
        generator.print_statistics(results)
        
        print("V5 Simple 질문 생성 완료!")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V5 Simple 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()