"""
V6 의료 보험 질문 생성기 - 약제 전용
핵심 변화: 자유생성(텍스트만) → 강한 전·후처리, 라벨 비율 POS 중심(6:3:0), JSON 금지
고시 관련 기능 제외
"""

import pandas as pd
import re
import random
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from rapidfuzz import fuzz
import logging
import traceback
from datetime import datetime
import openai
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drug_generation_v6.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class GenerationConfig:
    """생성 설정"""
    pos_ratio: float = 6.0
    hn_ratio: float = 3.0
    en_ratio: float = 0.0  # V6에서 기본 0
    max_chars: int = 80
    min_chars: int = 25
    slice_length: int = 2500
    max_retries: int = 3
    temperature: float = 0.7
    model: str = "gpt-4o-mini"

class V6QuestionGenerator:
    """V6 질문 생성기 - 강한 전후처리 중심"""
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.pronoun_pattern = re.compile(
            r'(이것|그것|해당|본|동)\s*(약|제제|제품)|\b(이것|그것)\b'
        )
        self.specificity_pattern = re.compile(
            r'\d|mg|㎎|U/L|%|회|개월|일|주|급여|비급여|본인부담|사전승인|수가|코드|기간|횟수'
        )
        self.policy_keywords = [
            '급여', '비급여', '본인부담', '사전승인', '수가', '코드', 
            '기간', '횟수'
        ]
        
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        logger.info(f"데이터 로드 중: {file_path}")
        
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)
        logger.info(f"원본 데이터: {len(df)}행, 컬럼: {list(df.columns)}")
        
        # 컬럼 매핑 (한국어 헤더를 표준 키로 변환)
        column_mapping = {
            '약제분류번호': 'code',
            '약제 분류명': 'code_name', 
            '구분': 'title',
            '세부인정기준 및 방법': 'text'
        }
        
        # 부분 일치로 컬럼 매핑
        mapped_columns = {}
        for col in df.columns:
            for ko_name, en_name in column_mapping.items():
                if ko_name in str(col):
                    mapped_columns[col] = en_name
                    break
        
        df = df.rename(columns=mapped_columns)
        
        # 필수 컬럼 확인
        required_cols = ['code', 'code_name', 'title', 'text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")
        
        # 텍스트 슬라이싱 (2-3k자)
        df['text_slices'] = df['text'].apply(self._slice_text)
        
        # 메타 태깅
        df['has_numbers'] = df['text'].apply(lambda x: bool(re.search(r'\d', str(x))))
        df['has_units'] = df['text'].apply(lambda x: bool(re.search(r'mg|㎎|U/L|%|회|개월|일|주', str(x))))
        df['has_policy'] = df['text'].apply(lambda x: any(keyword in str(x) for keyword in self.policy_keywords))
        
        # 약제명 파싱
        df['main_name'] = df['title'].apply(self._extract_main_name)
        df['brand_names'] = df['title'].apply(self._extract_brand_names)
        
        logger.info(f"전처리 완료: {len(df)}행")
        return df
        
    def _slice_text(self, text: str) -> List[str]:
        """텍스트를 2-3k자 단위로 슬라이싱"""
        if not text or len(text) <= self.config.slice_length:
            return [text] if text else []
        
        slices = []
        start = 0
        
        while start < len(text):
            end = start + self.config.slice_length
            
            # 문단 경계에서 끊기
            if end < len(text):
                # 다음 문단이나 문장 끝을 찾기
                next_break = text.find('\n', end)
                next_period = text.find('.', end)
                
                if next_break != -1 and next_break < end + 200:
                    end = next_break
                elif next_period != -1 and next_period < end + 200:
                    end = next_period + 1
            
            slice_text = text[start:end].strip()
            if slice_text:
                slices.append(slice_text)
            
            start = end
        
        return slices
    
    def _extract_main_name(self, title: str) -> str:
        """괄호 앞 주 약제명 추출"""
        if not title:
            return ""
        
        # 첫 번째 괄호 앞까지
        match = re.match(r'([^(]+)', title)
        return match.group(1).strip() if match else title.strip()
    
    def _extract_brand_names(self, title: str) -> List[str]:
        """품명 목록 추출"""
        if not title or '품명:' not in title:
            return []
        
        # 품명: 뒤의 내용 추출
        brand_part = title.split('품명:')[1] if '품명:' in title else ""
        if not brand_part:
            return []
        
        # · 또는 /로 분리
        brands = re.split(r'[·/]', brand_part)
        return [brand.strip() for brand in brands if brand.strip()]
    
    def generate_positive_questions(self, text_slice: str, drug_info: Dict) -> List[str]:
        """POSITIVE 질문 생성"""
        prompt = f"""You are generating Korean training questions for an insurance review embedding model.
Return many lines, each a single Korean question based ONLY on the document below.
Constraints:
- 25–80 characters, end with '?'
- Include at least one of: a number, a unit (mg, U/L, %, 회, 개월, 일, 주), or a policy term (급여, 비급여, 본인부담, 사전승인, 수가, 코드, 기간, 횟수)
- Strictly forbid pronouns like '이것/그것/해당/본/동 + 약·제제·제품'
- One issue per sentence, vary openings so no single opening exceeds 30% in a set
- Use main drug name and brand names if present; do NOT output JSON.

Document (use only this content):
<<<DOC
{text_slice}
DOC
>>>"""

        try:
            response = openai.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            questions = [line.strip() for line in content.split('\n') if line.strip() and line.strip().endswith('?')]
            
            logger.info(f"POS 질문 생성됨: {len(questions)}개")
            return questions
            
        except Exception as e:
            logger.error(f"POS 질문 생성 실패: {e}")
            return []
    
    def generate_hard_negative_questions(self, anchor_questions: List[str], text_slice: str, drug_info: Dict) -> List[str]:
        """HARD_NEGATIVE 질문 생성"""
        hn_questions = []
        
        for anchor in anchor_questions[:5]:  # 상위 5개만 사용
            # 앵커 문장에서 facet 1개만 변형
            mutated = self._mutate_anchor(anchor, drug_info)
            if not mutated:
                continue
            
            # HN rewriter로 자연스러운 질문으로 변환
            rewritten = self._rewrite_as_question(mutated)
            if rewritten and self._validate_hard_negative(anchor, rewritten, drug_info):
                hn_questions.append(rewritten)
        
        logger.info(f"HN 질문 생성됨: {len(hn_questions)}개")
        return hn_questions
    
    def _mutate_anchor(self, anchor: str, drug_info: Dict) -> Optional[str]:
        """앵커 문장에서 facet 1개만 변경"""
        mutation_strategies = [
            self._mutate_dosage,
            self._mutate_frequency, 
            self._mutate_duration,
            self._mutate_route,
            self._mutate_indication,
            self._mutate_coverage,
            self._mutate_approval
        ]
        
        for strategy in random.sample(mutation_strategies, len(mutation_strategies)):
            try:
                mutated = strategy(anchor, drug_info)
                if mutated and mutated != anchor:
                    return mutated
            except Exception as e:
                logger.warning(f"변형 전략 실패 ({strategy.__name__}): {e}")
                continue
        
        return None
    
    def _mutate_dosage(self, anchor: str, drug_info: Dict) -> Optional[str]:
        """용량 변형"""
        # 숫자 패턴 찾기
        number_match = re.search(r'(\d+(?:\.\d+)?)', anchor)
        if number_match:
            original_num = float(number_match.group(1))
            # 다른 용량으로 변경 (0.5배 또는 2배)
            new_num = original_num * random.choice([0.5, 2.0])
            if new_num.is_integer():
                new_num = int(new_num)
            return anchor.replace(number_match.group(1), str(new_num))
        return None
    
    def _mutate_frequency(self, anchor: str, drug_info: Dict) -> Optional[str]:
        """투약 빈도 변형"""
        freq_mappings = {
            '1일': '2일', '2일': '1일',
            '1회': '2회', '2회': '3회', '3회': '1회',
            '매일': '격일', '격일': '매일'
        }
        
        for original, replacement in freq_mappings.items():
            if original in anchor:
                return anchor.replace(original, replacement)
        return None
    
    def _mutate_duration(self, anchor: str, drug_info: Dict) -> Optional[str]:
        """기간 변형"""
        duration_mappings = {
            '1개월': '3개월', '3개월': '6개월', '6개월': '1개월',
            '1주': '2주', '2주': '4주', '4주': '1주'
        }
        
        for original, replacement in duration_mappings.items():
            if original in anchor:
                return anchor.replace(original, replacement)
        return None
    
    def _mutate_route(self, anchor: str, drug_info: Dict) -> Optional[str]:
        """투여 경로 변형"""
        route_mappings = {
            '경구': '주사', '주사': '경구',
            '내복': '주입', '주입': '내복'
        }
        
        for original, replacement in route_mappings.items():
            if original in anchor:
                return anchor.replace(original, replacement)
        return None
    
    def _mutate_indication(self, anchor: str, drug_info: Dict) -> Optional[str]:
        """적응증 변형"""
        indication_mappings = {
            '고혈압': '당뇨', '당뇨': '고혈압',
            '감염': '염증', '염증': '감염',
            '급성': '만성', '만성': '급성'
        }
        
        for original, replacement in indication_mappings.items():
            if original in anchor:
                return anchor.replace(original, replacement)
        return None
    
    def _mutate_coverage(self, anchor: str, drug_info: Dict) -> Optional[str]:
        """급여 상태 변형"""
        coverage_mappings = {
            '급여': '비급여', '비급여': '급여',
            '본인부담': '급여', '급여': '본인부담'
        }
        
        for original, replacement in coverage_mappings.items():
            if original in anchor:
                return anchor.replace(original, replacement)
        return None
    
    def _mutate_approval(self, anchor: str, drug_info: Dict) -> Optional[str]:
        """승인 관련 변형"""
        if '사전승인' in anchor:
            return anchor.replace('사전승인', '사후승인')
        elif '승인' in anchor and '사전' not in anchor:
            return anchor.replace('승인', '사전승인')
        return None
    
    def _rewrite_as_question(self, mutated_text: str) -> Optional[str]:
        """변형된 텍스트를 자연스러운 질문으로 변환"""
        prompt = f"""Rewrite the given Korean sentence into a natural Korean question.
Constraints: 25–80 characters, end with '?', keep any numbers/units/policy terms, no pronouns.
Return exactly one line with the question. No extra words, no JSON.

Original:
{mutated_text}"""

        try:
            response = openai.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # 더 일관된 rewriting을 위해 낮은 temperature
                max_tokens=200
            )
            
            rewritten = response.choices[0].message.content.strip()
            if rewritten.endswith('?') and self.config.min_chars <= len(rewritten) <= self.config.max_chars:
                return rewritten
            return None
            
        except Exception as e:
            logger.error(f"질문 재작성 실패: {e}")
            return None
    
    def _validate_hard_negative(self, anchor: str, hn_question: str, drug_info: Dict) -> bool:
        """HN 질문 검증"""
        # 핵심 키워드 공유 확인
        main_name = drug_info.get('main_name', '')
        if main_name and main_name in anchor and main_name not in hn_question:
            return False
        
        # 너무 유사하지 않은지 확인
        similarity = fuzz.token_set_ratio(anchor, hn_question)
        if similarity > 90:  # 너무 유사
            return False
        
        # 완전히 다르지 않은지 확인  
        if similarity < 30:  # 너무 다름
            return False
        
        return True
    
    def apply_post_processing_gates(self, questions: List[str]) -> List[str]:
        """강한 후처리 게이트 적용"""
        filtered = []
        
        for question in questions:
            # 길이 체크
            if not (self.config.min_chars <= len(question) <= self.config.max_chars):
                continue
            
            # 물음표 필수
            if not question.endswith('?'):
                continue
            
            # 대명사 금지
            if self.pronoun_pattern.search(question):
                continue
            
            # 구체성 체크 (숫자/단위/정책어 중 1개 이상)
            if not self.specificity_pattern.search(question):
                continue
            
            # 단일 논점 체크
            if question.count(',') + question.count('및') + question.count('/') >= 2:
                continue
            
            filtered.append(question)
        
        # 중복 제거
        unique_questions = []
        for question in filtered:
            is_duplicate = False
            for existing in unique_questions:
                if fuzz.token_set_ratio(question, existing) >= 82:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_questions.append(question)
        
        return unique_questions
    
    def balance_labels(self, pos_questions: List[str], hn_questions: List[str]) -> Tuple[List[str], List[str]]:
        """라벨 비율 6:3:0으로 균형 맞추기"""
        total_needed = len(pos_questions) + len(hn_questions)
        
        # 목표 개수 계산
        pos_target = math.floor(total_needed * self.config.pos_ratio / 9)
        hn_target = math.floor(total_needed * self.config.hn_ratio / 9)
        
        # 남는 개수 배분 (POS 우선)
        remaining = total_needed - pos_target - hn_target
        pos_target += remaining
        
        # 샘플링
        balanced_pos = random.sample(pos_questions, min(len(pos_questions), pos_target))
        balanced_hn = random.sample(hn_questions, min(len(hn_questions), hn_target))
        
        logger.info(f"라벨 균형: POS {len(balanced_pos)}, HN {len(balanced_hn)}")
        return balanced_pos, balanced_hn
    
    def process_drug_data(self, df: pd.DataFrame) -> List[Dict]:
        """약제 데이터 전체 처리"""
        results = []
        
        for idx, row in df.iterrows():
            logger.info(f"처리 중: {idx+1}/{len(df)} - {row.get('main_name', 'Unknown')}")
            
            drug_info = {
                'code': row['code'],
                'code_name': row['code_name'], 
                'title': row['title'],
                'main_name': row['main_name'],
                'brand_names': row['brand_names']
            }
            
            # 각 텍스트 슬라이스별로 처리
            for slice_idx, text_slice in enumerate(row['text_slices']):
                if not text_slice.strip():
                    continue
                
                # POS 질문 생성
                pos_questions = self.generate_positive_questions(text_slice, drug_info)
                pos_questions = self.apply_post_processing_gates(pos_questions)
                
                if not pos_questions:
                    logger.warning(f"POS 질문 생성 실패: row {idx}, slice {slice_idx}")
                    continue
                
                # HN 질문 생성
                hn_questions = self.generate_hard_negative_questions(pos_questions, text_slice, drug_info)
                hn_questions = self.apply_post_processing_gates(hn_questions)
                
                # 라벨 균형 맞추기
                balanced_pos, balanced_hn = self.balance_labels(pos_questions, hn_questions)
                
                # 결과 저장
                for question in balanced_pos:
                    results.append({
                        '약제분류번호': row['code'],
                        '약제 분류명': row['code_name'],
                        '구분': row['title'],
                        '세부인정기준 및 방법': text_slice,
                        'question': question,
                        '라벨': 'POSITIVE'
                    })
                
                for question in balanced_hn:
                    results.append({
                        '약제분류번호': row['code'],
                        '약제 분류명': row['code_name'],
                        '구분': row['title'],
                        '세부인정기준 및 방법': text_slice,
                        'question': question,
                        '라벨': 'HARD_NEGATIVE'
                    })
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """결과를 엑셀 파일로 저장"""
        if not results:
            logger.warning("저장할 결과가 없습니다.")
            return
        
        df = pd.DataFrame(results)
        df.to_excel(output_path, index=False)
        logger.info(f"결과 저장 완료: {output_path} ({len(results)}행)")
        
        # 라벨별 통계
        label_counts = df['라벨'].value_counts()
        logger.info(f"라벨 분포: {dict(label_counts)}")

def main():
    """메인 실행 함수"""
    try:
        # 고정된 데이터 파일 경로
        data_path = r"C:\Jimin\Pharma-Augment\data\요양심사약제_후처리_v2.xlsx"
        output_path = r"C:\Jimin\Pharma-Augment\versions\v6\drug_questions_v6.xlsx"
        
        logger.info("V6 질문 생성기 시작")
        
        # 생성기 초기화
        generator = V6QuestionGenerator()
        
        # 데이터 로드 및 전처리
        df = generator.load_and_preprocess_data(data_path)
        
        # 질문 생성
        results = generator.process_drug_data(df)
        
        # 결과 저장
        generator.save_results(results, output_path)
        
        logger.info("V6 질문 생성 완료!")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()