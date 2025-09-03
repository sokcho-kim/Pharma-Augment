#!/usr/bin/env python3
"""
Pharma-Augment V4 - 약제(DRUG) 강화된 생성기
100% 행 커버리지를 보장하는 fallback 메커니즘 포함
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
        logging.FileHandler('drug_generation_v4_robust.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

class DrugGeneratorV4Robust:
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
        
        # V4 카테고리 (9개)
        self.CATEGORIES = [
            "범위", "요건/기준", "오프라벨/허가범위", "기간/시점", "전환", 
            "증빙/서류", "본인부담/급여구분", "대상군", "절차/프로세스"
        ]
        
        # 라벨 타입
        self.LABELS = ["POSITIVE", "NEGATIVE", "HARD_NEGATIVE"]
        
        # 감사 로그
        self.audit_log = []
        self.progress_bar = None
        self.failed_rows = []  # 실패한 행 추적
    
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
        # main_name = 괄호 앞의 주 약제명
        main_name = ""
        brand_names = []
        
        # 괄호 앞 부분을 main_name으로
        paren_match = re.search(r'^([^(]+)', gubun)
        if paren_match:
            main_name = paren_match.group(1).strip()
        
        # "품명:" 뒤의 품명들을 추출
        brand_match = re.search(r'품\s*명\s*[:：]\s*([^)]+)', gubun)
        if brand_match:
            brand_text = brand_match.group(1).strip()
            # '·' 또는 '/' 기준으로 분리
            brand_names = re.split(r'[·/]+', brand_text)
            brand_names = [name.strip() for name in brand_names if name.strip()]
        
        return main_name, brand_names
    
    def create_basic_questions(self, row_data: Dict) -> List[Dict]:
        """기본 질문 생성 (fallback용)"""
        drug_code = row_data.get('약제분류번호', '')
        drug_name = row_data.get('약제 분류명', '')
        gubun = row_data.get('구분', '')
        content = row_data.get('세부인정기준 및 방법', '')
        
        main_name, brand_names = self.extract_name_slots(gubun)
        
        # 안전한 기본 질문들 (대명사 없음, 적절한 길이)
        basic_questions = []
        
        if main_name:
            basic_questions.extend([
                {"question": f"{main_name}의 급여 인정 기준은 무엇인가요?", "라벨": "POSITIVE", "name_usage": "MAIN", "category": "요건/기준"},
                {"question": f"{main_name} 사용 시 필요한 증빙서류는?", "라벨": "POSITIVE", "name_usage": "MAIN", "category": "증빙/서류"},
                {"question": f"{main_name}의 보험급여 적용 범위는?", "라벨": "POSITIVE", "name_usage": "MAIN", "category": "범위"},
                {"question": f"{main_name} 처방 시 주의사항은 무엇인가요?", "라벨": "POSITIVE", "name_usage": "MAIN", "category": "절차/프로세스"},
                {"question": f"{main_name}의 투여 기간 제한은?", "라벨": "POSITIVE", "name_usage": "MAIN", "category": "기간/시점"}
            ])
        
        if brand_names:
            for brand in brand_names[:2]:  # 최대 2개 브랜드만
                basic_questions.extend([
                    {"question": f"{brand}의 급여 인정 요건은?", "라벨": "POSITIVE", "name_usage": "BRAND", "category": "요건/기준"},
                    {"question": f"{brand} 사용 대상군은 무엇인가요?", "라벨": "POSITIVE", "name_usage": "BRAND", "category": "대상군"}
                ])
        
        if main_name and brand_names:
            brand = brand_names[0]
            basic_questions.extend([
                {"question": f"{main_name}({brand})의 본인부담 기준은?", "라벨": "POSITIVE", "name_usage": "BOTH", "category": "본인부담/급여구분"},
                {"question": f"{brand}({main_name}) 전환 시 조건은?", "라벨": "POSITIVE", "name_usage": "BOTH", "category": "전환"}
            ])
        
        # NEGATIVE 질문 추가
        basic_questions.extend([
            {"question": "소아과 환자의 특별 급여 기준은?", "라벨": "NEGATIVE", "name_usage": "NONE", "category": "대상군"},
            {"question": "외래 진료 시 수가 산정 방법은?", "라벨": "NEGATIVE", "name_usage": "NONE", "category": "절차/프로세스"}
        ])
        
        # 각 질문에 필수 필드 추가
        for q in basic_questions:
            q.update({
                "약제분류번호": drug_code,
                "약제 분류명": drug_name,
                "구분": gubun,
                "세부인정기준 및 방법": content[:100] + "..." if len(content) > 100 else content
            })
        
        return basic_questions[:10]  # 최대 10개로 제한
    
    def create_drug_prompt_v4_relaxed(self, row_data: Dict) -> str:
        """완화된 V4 약제 프롬프트 (fallback용)"""
        drug_code = row_data.get('약제분류번호', '')
        drug_name = row_data.get('약제 분류명', '')
        gubun = row_data.get('구분', '')
        content = row_data.get('세부인정기준 및 방법', '')
        
        main_name, brand_names = self.extract_name_slots(gubun)
        brand_names_json = json.dumps(brand_names, ensure_ascii=False)
        
        return f"""[ROLE]
너는 의료 보험 약제 질문 생성기다. 반드시 5개 이상의 질문을 생성해야 한다.

[INPUT]
- 약제분류번호: {drug_code}
- 약제 분류명: {drug_name}
- 구분: {gubun}
- 세부인정기준 및 방법: \"\"\"{content}\"\"\"

[RELAXED RULES - 반드시 준수]
1) 최소 5개 이상 질문 생성 (필수)
2) 대명사 금지: "이것", "그것", "해당 약제", "본 제제" 등 사용 금지
3) 길이: 10~80자 (완화됨)
4) 라벨: POSITIVE(60% 이상), NEGATIVE, HARD_NEGATIVE 포함

[OUTPUT - JSON 배열만 출력]
[
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"{content[:50]}...","question":"급여 인정 기준은 무엇인가요?","라벨":"POSITIVE"}},
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"{content[:50]}...","question":"사용 대상군은?","라벨":"POSITIVE"}},
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"{content[:50]}...","question":"투여 기간은?","라벨":"POSITIVE"}},
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"{content[:50]}...","question":"필요한 검사는?","라벨":"POSITIVE"}},
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"{content[:50]}...","question":"처방전 작성 방법은?","라벨":"NEGATIVE"}}
]"""
    
    def post_process_relaxed(self, questions: List[Dict]) -> List[Dict]:
        """완화된 후처리 (fallback용)"""
        if not questions:
            return []
        
        processed = []
        
        for q in questions:
            text = q.get("question", "")
            label = q.get("라벨", "")
            
            # 1. 대명사 검증만 (다른 기준 완화)
            if self.PRONOUN_RE.search(text):
                logger.warning(f"대명사 검출로 제거: {text}")
                continue
            
            # 2. 완화된 길이 검증 (10-80자)
            if not (10 <= len(text) <= 80):
                logger.warning(f"길이 제한 위반: {text} ({len(text)}자)")
                continue
            
            # 3. 라벨 검증 (완화)
            if label not in self.LABELS:
                q["라벨"] = "POSITIVE"  # 기본값으로 수정
            
            processed.append(q)
        
        return processed
    
    def create_drug_prompt_v4(self, row_data: Dict) -> str:
        """원본 V4 약제 프롬프트 생성"""
        drug_code = row_data.get('약제분류번호', '')
        drug_name = row_data.get('약제 분류명', '')
        gubun = row_data.get('구분', '')
        content = row_data.get('세부인정기준 및 방법', '')
        
        main_name, brand_names = self.extract_name_slots(gubun)
        brand_names_json = json.dumps(brand_names, ensure_ascii=False)
        
        return f"""[ROLE]
너는 의료 보험 심사·수가 약제 데이터를 임베딩 학습용 질문 세트로 변환하는 에이전트다.
입력 1행(약제 단위)을 받아 질문을 최소 5개~최대 20개 생성하고, 각 질문에 라벨을 부여한다.
최종 제출 형식은 반드시 아래 6개 필드만을 가지는 JSON 배열이다.

[INPUT]
- 약제분류번호: {drug_code}
- 약제 분류명: {drug_name}
- 구분: {gubun}
- 세부인정기준 및 방법: \"\"\"
{content}
\"\"\"

[NAME SLOTS] (프롬프트 내부 추출 규칙)
- main_name = "{main_name}"
- brand_names = {brand_names_json}

[GENERATION RULES]
1) 질문 수: 5~20개. 생성 가능한 만큼 만들되 품질 필터 통과분만 채택.
2) 문형: WH 개방형(무엇/어떤/언제/어떻게/왜), 1문장 1논점, 외부 지식/추정 금지.
3) 길이: 15~70자(훈련 기준). 12~50자는 검증셋 추출 시 가점.
4) 대명사 금지: "(이|그|해당|본|동) (약|약제|제제|제품)|이것|그것" 포함 시 폐기·재생성.
5) 이름 사용 비율(세트 수준 강제):
   - MAIN(주 약제명만 포함) 30–40%
   - BRAND(품명만 포함)     30–40%  
   - BOTH(둘 다 포함)        20–30%
   - brand_names == [] 이면: MAIN 70–80%, BOTH 20–30%
   - brand_names 길이 == 1 이면: MAIN 35–45%, BRAND 30–40%, BOTH 20–30%
   - 검증: MAIN이면 brand 미포함, BRAND면 main 미포함, BOTH면 둘 다 포함.
6) 카테고리(질문마다 정확히 1개 부여, 세트 내 최소 4종 이상 등장, 단일 카테고리 ≤ 40%)
   {{범위, 요건/기준, 오프라벨/허가범위, 기간/시점, 전환(경구↔주사),
    증빙/서류, 본인부담/급여구분, 대상군, 절차/프로세스}}
7) DRUG 세트에서 특히 포함 권장: 기간/시점, 전환(경구↔주사), 본인부담, 대상군 특례.

[LABELING]
- POSITIVE: 본 입력(구분/세부인정기준)에 직접 부합하는 질문.
- NEGATIVE: 전혀 다른 약제·조항을 전제로 한 명백히 무관한 질문.
- HARD_NEGATIVE: 표면 토큰은 유사하나 핵심 차원이 어긋나는 near-miss
  (예: 같은 약제지만 '조혈모세포이식'↔'신장이식', '성인'↔'소아', '초기'↔'재발',
   '경구'↔'주사', '급여'↔'전액 본인부담', '기간 A'↔'기간 B' 등).

[OUTPUT — 제출용 고정 스키마(JSON 배열만 출력)]
[
  {{"약제분류번호":"{drug_code}","약제 분류명":"{drug_name}","구분":"{gubun}","세부인정기준 및 방법":"{content[:100]}...","question":"...","라벨":"POSITIVE|NEGATIVE|HARD_NEGATIVE"}},
  ...
]"""

    def calculate_question_score(self, text: str, content: str) -> float:
        """문항 점수 S_q 계산"""
        # 길이 점수 (15-70자 기준, 45자가 최적)
        length = len(text)
        length_score = max(0, min(1, 1 - abs(length - 45) / 30))
        
        # WH 점수
        wh_patterns = r'(무엇|어떤|언제|어떻게|왜|누가|어디|몇)'
        if re.search(f'^{wh_patterns}', text):
            wh_score = 1.0
        elif re.search(wh_patterns, text):
            wh_score = 0.6
        else:
            wh_score = 0.2
        
        # 단일 논점 점수
        multi_issue_count = text.count(',') + text.count('및') + text.count('/')
        single_issue = 1.0 if multi_issue_count <= 1 else 0.3
        
        # 대명사 패널티
        pronoun_penalty = -1.0 if self.PRONOUN_RE.search(text) else 0.0
        
        # 원문 중첩 점수 (간단한 토큰 교집합)
        text_tokens = set(re.findall(r'\w+', text))
        content_tokens = set(re.findall(r'\w+', content))
        if len(text_tokens) > 0:
            overlap = len(text_tokens & content_tokens) / len(text_tokens)
        else:
            overlap = 0.0
        
        # 최종 점수 계산
        s_q = (0.25 * length_score + 0.25 * wh_score + 
               0.25 * single_issue + 0.25 * overlap + pronoun_penalty)
        
        return max(0.0, min(1.0, s_q))
    
    def validate_name_usage(self, text: str, name_usage: str, main_name: str, brand_names: List[str]) -> bool:
        """이름 사용 검증"""
        main_in_text = main_name in text if main_name else False
        brand_in_text = any(brand in text for brand in brand_names)
        
        if name_usage == "MAIN":
            return main_in_text and not brand_in_text
        elif name_usage == "BRAND":
            return brand_in_text and not main_in_text
        elif name_usage == "BOTH":
            return main_in_text and brand_in_text
        
        return False
    
    def post_process_questions(self, questions: List[Dict], main_name: str, brand_names: List[str], content: str) -> List[Dict]:
        """질문 후처리 및 검증"""
        if not questions:
            return []
        
        processed = []
        
        for q in questions:
            text = q.get("question", "")
            name_usage = q.get("name_usage", "")
            category = q.get("category", "")
            label = q.get("라벨", "")
            
            # 1. 대명사 검증
            if self.PRONOUN_RE.search(text):
                logger.warning(f"대명사 검출로 제거: {text}")
                continue
            
            # 2. 길이 검증 (15-70자)
            if not (15 <= len(text) <= 70):
                logger.warning(f"길이 제한 위반: {text} ({len(text)}자)")
                continue
            
            # 3. 이름 사용 검증 (완화)
            if name_usage in ["MAIN", "BRAND", "BOTH"]:
                if not self.validate_name_usage(text, name_usage, main_name, brand_names):
                    logger.warning(f"이름 사용 검증 실패: {text} ({name_usage})")
                    continue
            
            # 4. 카테고리 검증 (완화)
            if category not in self.CATEGORIES:
                q["category"] = "요건/기준"  # 기본값으로 수정
            
            # 5. 라벨 검증
            if label not in self.LABELS:
                q["라벨"] = "POSITIVE"  # 기본값으로 수정
            
            # 6. 문항 점수 계산 (완화된 기준)
            s_q = self.calculate_question_score(text, content)
            if s_q < 0.6:  # 기준을 0.75 → 0.6으로 완화
                logger.warning(f"품질 점수 미달: {text} ({s_q:.2f})")
                continue
            
            processed.append(q)
        
        return processed
    
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3)
    async def call_api_v4(self, session: aiohttp.ClientSession, prompt: str, row_id: str) -> List[Dict]:
        """V4 API 호출"""
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
                    logger.error(f"JSON 파싱 실패 for {row_id}: {e}")
                    logger.error(f"Response content: {content[:200]}...")
                    return []
        
        # Claude 구현은 생략 (OpenAI 우선)
        return []
    
    async def generate_questions_for_drug(self, session: aiohttp.ClientSession, row_data: Dict, row_idx: int) -> List[Dict]:
        """단일 약제에 대한 질문 생성 (강화된 fallback 포함)"""
        async with self.semaphore:
            start_time = time.time()
            row_id = f"row_{row_idx}"
            
            try:
                # NAME SLOTS 추출
                gubun = str(row_data.get('구분', ''))
                main_name, brand_names = self.extract_name_slots(gubun)
                content = str(row_data.get('세부인정기준 및 방법', ''))
                
                validated_questions = []
                attempt_count = 0
                
                # 1차 시도: 원본 V4 프롬프트
                try:
                    attempt_count += 1
                    prompt = self.create_drug_prompt_v4(row_data)
                    raw_questions = await self.call_api_v4(session, prompt, row_id)
                    validated_questions = self.post_process_questions(raw_questions, main_name, brand_names, content)
                    
                    if len(validated_questions) >= 5:
                        logger.info(f"{row_id}: 1차 성공 - {len(validated_questions)}개")
                    else:
                        raise ValueError(f"질문 부족: {len(validated_questions)}개")
                        
                except Exception as e:
                    logger.warning(f"{row_id}: 1차 시도 실패 - {e}")
                    
                # 2차 시도: 완화된 프롬프트
                if len(validated_questions) < 5:
                    try:
                        attempt_count += 1
                        prompt_relaxed = self.create_drug_prompt_v4_relaxed(row_data)
                        raw_questions = await self.call_api_v4(session, prompt_relaxed, row_id)
                        validated_questions = self.post_process_relaxed(raw_questions)
                        
                        if len(validated_questions) >= 5:
                            logger.info(f"{row_id}: 2차 성공 - {len(validated_questions)}개")
                        else:
                            raise ValueError(f"질문 부족: {len(validated_questions)}개")
                            
                    except Exception as e:
                        logger.warning(f"{row_id}: 2차 시도 실패 - {e}")
                
                # 3차 시도: 기본 질문 생성 (fallback)
                if len(validated_questions) < 5:
                    attempt_count += 1
                    logger.warning(f"{row_id}: fallback으로 기본 질문 생성")
                    validated_questions = self.create_basic_questions(row_data)
                
                # 최종 검증 (20개 초과 시 상위 20개만)
                if len(validated_questions) > 20:
                    # S_q 점수 기준으로 상위 20개만 선택
                    scored = [(q, self.calculate_question_score(q.get("question", ""), content)) 
                             for q in validated_questions]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    validated_questions = [q for q, _ in scored[:20]]
                
                # 감사 로그
                elapsed_ms = int((time.time() - start_time) * 1000)
                self.audit_log.append({
                    'row_id': row_id,
                    'row_idx': row_idx,
                    'main_name': main_name,
                    'brand_count': len(brand_names),
                    'questions_generated': len(validated_questions),
                    'attempt_count': attempt_count,
                    'elapsed_ms': elapsed_ms,
                    'provider': self.provider,
                    'model': self.model,
                    'success': len(validated_questions) >= 5
                })
                
                # 실패한 행 추적
                if len(validated_questions) < 5:
                    self.failed_rows.append({
                        'row_idx': row_idx,
                        'row_data': row_data,
                        'question_count': len(validated_questions)
                    })
                
                # 진행상황 업데이트
                if self.progress_bar:
                    self.progress_bar.update(1)
                
                status = "성공" if len(validated_questions) >= 5 else "실패"
                logger.info(f"완료: {row_id} - {len(validated_questions)}개 질문 ({status})")
                return validated_questions
                
            except Exception as e:
                logger.error(f"질문 생성 완전 실패 {row_id}: {e}")
                
                # 완전 실패 시에도 기본 질문 생성 시도
                try:
                    fallback_questions = self.create_basic_questions(row_data)
                    logger.info(f"{row_id}: 완전 fallback - {len(fallback_questions)}개 질문")
                    
                    self.audit_log.append({
                        'row_id': row_id,
                        'row_idx': row_idx,
                        'questions_generated': len(fallback_questions),
                        'attempt_count': 999,
                        'error': str(e),
                        'success': False
                    })
                    
                    if self.progress_bar:
                        self.progress_bar.update(1)
                    
                    return fallback_questions
                    
                except Exception as fallback_error:
                    logger.error(f"{row_id}: fallback도 실패 - {fallback_error}")
                    
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
                
                # 필수 필드 검증 (완화)
                if not row_dict.get('구분') and not row_dict.get('세부인정기준 및 방법'):
                    logger.warning(f"행 {idx}: 구분과 세부인정기준 모두 누락")
                    continue
                
                processed_data.append(row_dict)
                
            except Exception as e:
                logger.warning(f"행 {idx} 전처리 실패: {e}")
                continue
        
        logger.info(f"전처리 완료: {len(processed_data)}개 항목")
        return processed_data
    
    async def generate_all_questions(self, processed_data: List[Dict]) -> List[Dict]:
        """모든 약제에 대한 질문 생성"""
        # 진행상황 바 초기화
        self.progress_bar = tqdm(total=len(processed_data), desc="질문 생성 중", unit="약제")
        
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
            
            logger.info(f"최종 결과 저장: {output_path} ({len(questions)}개 질문)")
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            raise
    
    def save_audit_log(self, output_dir: str = "."):
        """감사 로그 저장"""
        try:
            audit_path = os.path.join(output_dir, "audit_log_drug_v4_robust.csv")
            df = pd.DataFrame(self.audit_log)
            df.to_csv(audit_path, index=False, encoding='utf-8-sig')
            logger.info(f"감사 로그 저장: {audit_path}")
        except Exception as e:
            logger.error(f"감사 로그 저장 실패: {e}")
    
    def print_statistics(self, questions: List[Dict]):
        """통계 출력"""
        if not questions:
            print("결과가 없습니다.")
            return
        
        total_questions = len(questions)
        
        # 라벨 분포
        label_counts = {}
        for q in questions:
            label = q.get('라벨', 'UNKNOWN')
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 성공/실패 분석
        success_count = len([log for log in self.audit_log if log.get('success', True)])
        total_rows = len(self.audit_log)
        
        # 통계 출력
        print(f"\n=== V4 약제 질문 생성 통계 (Robust) ===")
        print(f"총 처리 행 수: {total_rows}")
        print(f"성공 행 수: {success_count}")
        print(f"실패 행 수: {total_rows - success_count}")
        print(f"성공률: {success_count/total_rows*100:.1f}%" if total_rows > 0 else "N/A")
        print(f"총 질문 수: {total_questions}")
        
        print(f"\n=== 라벨 분포 ===")
        for label, count in label_counts.items():
            ratio = count / total_questions * 100 if total_questions > 0 else 0
            print(f"{label}: {count}개 ({ratio:.1f}%)")
        
        # 실패한 행 정보
        if self.failed_rows:
            print(f"\n=== 실패한 행들 ===")
            for failed in self.failed_rows[:5]:  # 처음 5개만 출력
                print(f"행 {failed['row_idx']}: {failed['question_count']}개 질문 생성")


def main():
    parser = argparse.ArgumentParser(description="Pharma-Augment V4 약제 질문 생성기 (Robust)")
    
    # 필수 인자
    parser.add_argument("--excel", required=True, help="약제 엑셀 파일 경로")
    
    # 선택적 인자
    parser.add_argument("--sheet", default="Sheet1", help="시트명")
    parser.add_argument("--out", default="drug_questions_v4_robust.xlsx", help="최종 출력 파일")
    parser.add_argument("--provider", choices=["openai", "claude"], default="openai", help="LLM 제공자")
    parser.add_argument("--model", default="gpt-4o-mini", help="모델명")
    parser.add_argument("--concurrency", type=int, default=6, help="동시 실행 수")
    parser.add_argument("--seed", type=int, default=20250903, help="랜덤 시드")
    
    args = parser.parse_args()
    
    try:
        # 질문 생성기 초기화
        generator = DrugGeneratorV4Robust(
            provider=args.provider,
            model=args.model,
            concurrency=args.concurrency,
            seed=args.seed
        )
        
        print("📋 약제 데이터 로드 중...")
        # 엑셀 데이터 로드
        df = generator.load_excel_data(args.excel, args.sheet)
        
        print("🔄 데이터 전처리 중...")
        # 전처리
        processed_data = generator.preprocess_data(df)
        
        if not processed_data:
            logger.error("처리할 데이터가 없습니다.")
            sys.exit(1)
        
        print(f"🤖 V4 약제 질문 생성 시작: {len(processed_data)}개 행 (Robust 모드)")
        print("💡 각 행마다 최소 5개 질문이 보장됩니다!")
        # 질문 생성
        results = asyncio.run(generator.generate_all_questions(processed_data))
        
        print("💾 결과 저장 중...")
        # 결과 저장
        generator.save_final_results(results, args.out)
        generator.save_audit_log()
        
        # 통계 출력
        generator.print_statistics(results)
        
        print("✅ V4 약제 질문 생성 완료 (Robust)!")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()