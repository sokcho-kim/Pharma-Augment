#!/usr/bin/env python3
"""
Pharma-Augment V5 - 향상된 질문 생성기
V2 스타일의 풍부하고 구체적인 질문 + V4의 대명사 차단 결합
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import random
import math
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import aiohttp
import backoff
from rapidfuzz import fuzz
from dotenv import load_dotenv
import tiktoken
from tqdm.asyncio import tqdm
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drug_generation_v5.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

class DrugGeneratorV5:
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
        
        # 세마포어로 동시성 제어
        self.semaphore = asyncio.Semaphore(concurrency)
        
        # 대명사 차단 정규식
        self.PRONOUN_RE = re.compile(r"(이|그|해당|본|동)\s?(약|약제|제제|제품)|이것|그것")
        
        # 라벨 타입
        self.LABELS = ["POSITIVE", "NEGATIVE", "HARD_NEGATIVE"]
        
        # 감사 로그
        self.audit_log = []
        self.progress_bar = None
    
    def load_excel_data(self, excel_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """엑셀 파일 로드"""
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            logger.info(f"엑셀 파일 로드 완료: {df.shape[0]}행 {df.shape[1]}열")
            
            # UTF-8 디코딩 문제 해결 시도
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
            
            # 컬럼명 확인 및 매핑
            expected_cols = ['약제분류번호', '약제 분류명', '구분', '세부인정기준 및 방법']
            actual_cols = list(df.columns)
            
            logger.info(f"실제 컬럼: {actual_cols}")
            
            # 컬럼 순서대로 매핑 (순서 기반)
            if len(actual_cols) >= 4:
                column_mapping = dict(zip(actual_cols[:4], expected_cols))
                df = df.rename(columns=column_mapping)
                logger.info(f"컬럼 매핑 완료: {column_mapping}")
            
            return df
            
        except Exception as e:
            logger.error(f"엑셀 파일 로드 실패: {e}")
            raise
    
    def extract_name_slots(self, gubun: str) -> Tuple[str, List[str]]:
        """구분 필드에서 main_name과 brand_names 추출"""
        # 괄호 패턴 제거 후 추출
        clean_gubun = re.sub(r'\[.*?\]\s*', '', gubun).strip()
        
        # main_name = 괄호 앞의 주 약제명
        main_name = ""
        brand_names = []
        
        # 괄호 앞 부분을 main_name으로
        paren_match = re.search(r'^([^(]+)', clean_gubun)
        if paren_match:
            main_name = paren_match.group(1).strip()
        
        # "품명:" 뒤의 품명들을 추출
        brand_match = re.search(r'품\s*명\s*[:：]\s*([^)]+)', clean_gubun)
        if brand_match:
            brand_text = brand_match.group(1).strip()
            # '·' 또는 '/' 기준으로 분리
            brand_names = re.split(r'[·/]+', brand_text)
            brand_names = [name.strip() for name in brand_names if name.strip()]
        
        return main_name, brand_names
    
    def create_drug_prompt_v5(self, row_data: Dict) -> str:
        """V5 프롬프트: V2 스타일의 풍부한 질문 + 대명사 차단"""
        drug_code = row_data.get('약제분류번호', '')
        drug_name = row_data.get('약제 분류명', '')
        gubun = row_data.get('구분', '')
        content = row_data.get('세부인정기준 및 방법', '')
        
        main_name, brand_names = self.extract_name_slots(gubun)
        brand_names_json = json.dumps(brand_names, ensure_ascii=False)
        
        return f"""[ROLE]
너는 의료 보험 심사 도메인의 전문 질문 생성 에이전트다. 
V2 스타일의 풍부하고 구체적인 질문들을 생성하되, 대명사는 절대 사용하지 않는다.

[INPUT]
- 약제분류번호: {drug_code}
- 약제 분류명: {drug_name}
- 구분: {gubun}
- main_name: {main_name}
- brand_names: {brand_names_json}
- 세부인정기준 및 방법: \"\"\"
{content}
\"\"\"

[V5 ENHANCED GENERATION RULES]
1) 질문 수: 8~15개 (양질 위주)
2) 질문 길이: 25~80자 (V2 수준으로 구체적이고 자세하게)
3) **대명사 절대 금지**: "이것", "그것", "해당 약제", "본 제제", "동 약물" 등 일체 사용 금지
4) 질문 유형을 다양하게 (V2 스타일):
   A) 기본 정보형: "{약제명}의 구체적인 급여 인정 기준은 무엇인가요?"
   B) 조건/상황형: "{특정수치/조건}일 때 {약제명} 사용이 가능한가요?"
   C) 비교형: "{약제명}의 경구제와 주사제 급여 기준 차이점은?"
   D) 절차형: "{약제명} 처방 시 필요한 사전승인 절차는 어떻게 되나요?"
   E) 제한형: "어떤 경우에 {약제명} 사용이 제한되거나 삭감되나요?"

5) 실무적 구체성 (V2 수준):
   - 구체적 수치 포함 (AST 60U/L, 3개월 이상 등)
   - 환자군 명시 (소아, 성인, 고령자 등)
   - 의료 용어 활용 (간기능, 신기능, 혈중농도 등)
   - 절차적 세부사항 (사전승인, 증빙서류, 모니터링 등)

6) 이름 사용 전략:
   - MAIN만: 30-40% (주 약제명만)
   - BRAND만: 30-40% (품명만)  
   - BOTH: 20-30% (둘 다)
   - 브랜드 없으면: MAIN 70%, BOTH 30%

[EXAMPLES - V2 스타일 참고]
✅ 좋은 예시:
- "AST 수치가 60U/L 이상일 때 {main_name} 급여요건은 어떻게 적용되나요?" (42자)
- "{brand_name}을 3개월 이상 장기 처방 시 필요한 모니터링 항목은?" (35자)
- "간기능 저하 환자에서 {main_name} 용량 조절 기준과 주의사항은?" (37자)
- "{brand_name}과 스테로이드 병용 투여 시 급여 심사에서 고려할 사항은?" (41자)

❌ 피해야 할 예시:
- "이 약제의 사용 기준은?" (대명사 사용)
- "{약제명} 기준은?" (너무 단순)
- "사용법은?" (비구체적)

[LABELING]
- POSITIVE (70%): 입력 내용에 직접 근거한 질문
- NEGATIVE (15%): 완전히 다른 약제나 상황 질문  
- HARD_NEGATIVE (15%): 비슷하지만 핵심이 다른 near-miss

[OUTPUT 스키마]
[
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"{content[:100]}...","question":"구체적이고 상세한 질문 (25-80자)","라벨":"POSITIVE|NEGATIVE|HARD_NEGATIVE"}},
  ...
]

반드시 V2 수준의 구체적이고 실무적인 질문을 생성하라. 단순한 질문은 금지한다."""

    def post_process_v5(self, questions: List[Dict], main_name: str, brand_names: List[str], content: str) -> List[Dict]:
        """V5 후처리: 품질 중심 필터링"""
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
            
            # 2. 길이 검증 (25-80자로 상향)
            if not (25 <= len(text) <= 80):
                if len(text) < 25:
                    logger.warning(f"V5 길이 부족: {text} ({len(text)}자)")
                    continue
                elif len(text) > 80:
                    logger.warning(f"V5 길이 초과: {text} ({len(text)}자)")
                    continue
            
            # 3. 라벨 검증
            if label not in self.LABELS:
                q["라벨"] = "POSITIVE"  # 기본값
            
            # 4. 구체성 검증 (간단한 패턴 매칭)
            concrete_patterns = [
                r'\d+', # 숫자 포함
                r'(개월|일|회|mg|mL|U/L)', # 단위 포함
                r'(환자|처방|투여|사용|적용)', # 의료 용어
                r'(기준|조건|절차|방법|사항)', # 구체적 명사
            ]
            
            has_concrete = any(re.search(pattern, text) for pattern in concrete_patterns)
            if not has_concrete:
                logger.warning(f"V5 구체성 부족: {text}")
                continue
            
            processed.append(q)
        
        return processed
    
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3)
    async def call_api_v5(self, session: aiohttp.ClientSession, prompt: str, row_id: str) -> List[Dict]:
        """V5 API 호출"""
        if self.provider == "openai":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,  # 창의성 증가
                "top_p": 0.95,
                "max_tokens": 3000,
                "response_format": {"type": "json_object"},
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
                    content = result['choices'][0]['message']['content']
                    # JSON 파싱 시도
                    if content.startswith('['):
                        return json.loads(content)
                    else:
                        # JSON 객체가 온 경우 배열로 감싸서 반환
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            return [parsed]
                        return parsed
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"V5 JSON 파싱 실패 for {row_id}: {e}")
                    logger.error(f"Response content: {content[:300]}...")
                    return []
        
        return []
    
    async def generate_questions_for_drug(self, session: aiohttp.ClientSession, row_data: Dict, row_idx: int) -> List[Dict]:
        """단일 약제에 대한 V5 질문 생성"""
        async with self.semaphore:
            start_time = time.time()
            row_id = f"v5_row_{row_idx}"
            
            try:
                gubun = str(row_data.get('구분', ''))
                main_name, brand_names = self.extract_name_slots(gubun)
                content = str(row_data.get('세부인정기준 및 방법', ''))
                
                # V5 프롬프트로 생성
                prompt = self.create_drug_prompt_v5(row_data)
                raw_questions = await self.call_api_v5(session, prompt, row_id)
                
                # V5 후처리
                validated_questions = self.post_process_v5(raw_questions, main_name, brand_names, content)
                
                # 품질 기준: 최소 5개
                if len(validated_questions) < 5:
                    logger.warning(f"{row_id}: V5 품질 기준 미달 ({len(validated_questions)}개)")
                
                # 감사 로그
                elapsed_ms = int((time.time() - start_time) * 1000)
                self.audit_log.append({
                    'row_id': row_id,
                    'row_idx': row_idx,
                    'main_name': main_name,
                    'brand_count': len(brand_names),
                    'questions_generated': len(validated_questions),
                    'avg_length': sum(len(q.get('question', '')) for q in validated_questions) / max(1, len(validated_questions)),
                    'elapsed_ms': elapsed_ms,
                    'provider': self.provider,
                    'model': self.model,
                    'version': 'v5'
                })
                
                # 진행상황 업데이트
                if self.progress_bar:
                    self.progress_bar.update(1)
                
                logger.info(f"V5 완료: {row_id} - {len(validated_questions)}개 질문 (평균 {int(sum(len(q.get('question', '')) for q in validated_questions) / max(1, len(validated_questions)))}자)")
                return validated_questions
                
            except Exception as e:
                logger.error(f"V5 질문 생성 실패 {row_id}: {e}")
                if self.progress_bar:
                    self.progress_bar.update(1)
                return []
    
    def preprocess_data(self, df: pd.DataFrame) -> List[Dict]:
        """데이터 전처리"""
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                # NaN 값 처리
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
        
        logger.info(f"V5 전처리 완료: {len(processed_data)}개 항목")
        return processed_data
    
    async def generate_all_questions(self, processed_data: List[Dict]) -> List[Dict]:
        """모든 약제에 대한 V5 질문 생성"""
        # 진행상황 바 초기화
        self.progress_bar = tqdm(total=len(processed_data), desc="V5 질문 생성", unit="약제")
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.generate_questions_for_drug(session, item, idx)
                for idx, item in enumerate(processed_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 진행상황 바 종료
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
        """최종 형식으로 결과 저장"""
        try:
            # 엑셀 형식으로 저장 (고정 컬럼)
            df = pd.DataFrame(questions)
            
            # 컬럼 순서 고정
            final_columns = ['약제분류번호', '약제 분류명', '구분', '세부인정기준 및 방법', 'question', '라벨']
            for col in final_columns:
                if col not in df.columns:
                    df[col] = ""
            
            df = df[final_columns]
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            logger.info(f"V5 결과 저장: {output_path} ({len(questions)}개 질문)")
            
        except Exception as e:
            logger.error(f"V5 결과 저장 실패: {e}")
            raise
    
    def save_audit_log(self, output_dir: str = "."):
        """감사 로그 저장"""
        try:
            audit_path = os.path.join(output_dir, "audit_log_drug_v5.csv")
            df = pd.DataFrame(self.audit_log)
            df.to_csv(audit_path, index=False, encoding='utf-8-sig')
            logger.info(f"V5 감사 로그 저장: {audit_path}")
        except Exception as e:
            logger.error(f"V5 감사 로그 저장 실패: {e}")
    
    def print_statistics(self, questions: List[Dict]):
        """V5 통계 출력"""
        if not questions:
            print("V5 결과가 없습니다.")
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
        
        # V5 통계 출력
        print(f"\n=== V5 약제 질문 생성 통계 ===")
        print(f"총 질문 수: {total_questions}")
        print(f"평균 길이: {avg_length:.1f}자")
        print(f"길이 범위: {min_length}-{max_length}자")
        
        print(f"\n=== 라벨 분포 ===")
        for label, count in label_counts.items():
            ratio = count / total_questions * 100 if total_questions > 0 else 0
            print(f"{label}: {count}개 ({ratio:.1f}%)")
        
        # V2와 비교 정보
        print(f"\n=== V2와 비교 ===")
        print(f"V2 평균 길이: 36.2자")
        print(f"V5 평균 길이: {avg_length:.1f}자")
        if avg_length >= 30:
            print("✅ V5가 V2 수준의 구체성 달성!")
        else:
            print("⚠️  V5 길이가 V2보다 짧음 - 구체성 개선 필요")


def main():
    parser = argparse.ArgumentParser(description="Pharma-Augment V5 약제 질문 생성기")
    
    # 필수 인자
    parser.add_argument("--excel", required=True, help="약제 엑셀 파일 경로")
    
    # 선택적 인자
    parser.add_argument("--sheet", default="Sheet1", help="시트명")
    parser.add_argument("--out", default="drug_questions_v5.xlsx", help="최종 출력 파일")
    parser.add_argument("--provider", choices=["openai", "claude"], default="openai", help="LLM 제공자")
    parser.add_argument("--model", default="gpt-4o-mini", help="모델명")
    parser.add_argument("--concurrency", type=int, default=6, help="동시 실행 수")
    parser.add_argument("--seed", type=int, default=20250903, help="랜덤 시드")
    
    args = parser.parse_args()
    
    try:
        # 질문 생성기 초기화
        generator = DrugGeneratorV5(
            provider=args.provider,
            model=args.model,
            concurrency=args.concurrency,
            seed=args.seed
        )
        
        print("📋 V5 약제 데이터 로드 중...")
        # 엑셀 데이터 로드
        df = generator.load_excel_data(args.excel, args.sheet)
        
        print("🔄 V5 데이터 전처리 중...")
        # 전처리
        processed_data = generator.preprocess_data(df)
        
        if not processed_data:
            logger.error("처리할 데이터가 없습니다.")
            sys.exit(1)
        
        print(f"🤖 V5 약제 질문 생성 시작: {len(processed_data)}개 행")
        print("🎯 V2 수준의 구체적이고 풍부한 질문을 생성합니다!")
        # 질문 생성
        results = asyncio.run(generator.generate_all_questions(processed_data))
        
        print("💾 V5 결과 저장 중...")
        # 결과 저장
        generator.save_final_results(results, args.out)
        generator.save_audit_log()
        
        # 통계 출력
        generator.print_statistics(results)
        
        print("✅ V5 약제 질문 생성 완료!")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V5 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()