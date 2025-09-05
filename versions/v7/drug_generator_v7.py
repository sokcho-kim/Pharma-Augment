"""
V7 의료 보험 질문 생성기 - ReasonIR 기반 멀티밴드 개선
핵심 변화: SR/MR/LR 길이 밴드 (60%/25%/15%) + 멀티턴 생성 + 앵커팩 수집
"""

import pandas as pd
import re
import random
import json
import time
import math
import uuid
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
from enum import Enum

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drug_generation_v7.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

class LengthBand(Enum):
    """길이 밴드 정의"""
    SR = "SR"  # Short Range: 25-80 chars
    MR = "MR"  # Medium Range: 80-160 chars  
    LR = "LR"  # Long Range: 200-600 chars (scenario-based)

@dataclass
class BandConfig:
    """밴드별 설정"""
    name: LengthBand
    min_chars: int
    max_chars: int
    ratio: float  # 전체 대비 비율
    allow_pronouns_in_scenario: bool = False  # 시나리오 부분에서 대명사 허용 여부

@dataclass 
class GenerationConfig:
    """생성 설정"""
    # 라벨 비율 (행 단위)
    pos_ratio: float = 6.0
    hn_ratio: float = 3.0
    en_ratio: float = 0.0  # 기본 0, 실험시에만 사용
    
    # 밴드 설정
    bands: Dict[LengthBand, BandConfig] = None
    
    # 기타 설정
    slice_length: int = 2500
    max_retries: int = 3
    temperature: float = 0.7
    model: str = "gpt-4o-mini"
    
    def __post_init__(self):
        if self.bands is None:
            self.bands = {
                LengthBand.SR: BandConfig(LengthBand.SR, 25, 80, 0.6),
                LengthBand.MR: BandConfig(LengthBand.MR, 80, 160, 0.25), 
                LengthBand.LR: BandConfig(LengthBand.LR, 200, 600, 0.15, allow_pronouns_in_scenario=True)
            }

@dataclass
class Question:
    """생성된 질문"""
    text: str
    label: str  # POSITIVE, HARD_NEGATIVE, EASY_NEGATIVE
    band: LengthBand
    anchor_id: str  # 앵커팩 ID
    doc_slice_id: str
    metadata: Dict[str, Any]

class V7QuestionGenerator:
    """V7 질문 생성기 - 멀티밴드 ReasonIR 방식"""
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        
        # 대명사 패턴 (밴드별 차등 적용)
        self.strict_pronoun_pattern = re.compile(
            r'(이것|그것|해당|본|동)\s*(약|제제|제품|고시|조항|내용|항)|\b(이것|그것)\b'
        )
        
        # 구체성 검증 패턴  
        self.specificity_pattern = re.compile(
            r'\d|mg|㎎|U/L|%|회|개월|일|주|급여|비급여|본인부담|사전승인|수가|코드|기간|횟수|시행일|개정'
        )
        
        # 문두 다양성 체크
        self.opening_words = ['무엇', '어떻게', '언제', '왜', '어떤', '어디서', '어느', '누가']
        
        # 정책 키워드
        self.policy_keywords = [
            '급여', '비급여', '본인부담', '사전승인', '수가', '코드', 
            '기간', '횟수', '시행일', '개정', '제출', '이의신청'
        ]
        
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        logger.info(f"데이터 로드 중: {file_path}")
        
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)
        logger.info(f"원본 데이터: {len(df)}행, 컬럼: {list(df.columns)}")
        
        # 컬럼 매핑 (실제 데이터에 맞게 수정)
        column_mapping = {
            '약제분류번호': 'code',
            '약제분류명': 'code_name',  # '약제 분류명' → '약제분류명' 
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
            
        # 핵심 컬럼만 결측치 확인 (약제분류번호, 분류명은 허용)
        essential_cols = ['title', 'text']  # 구분, 세부인정기준만 필수
        initial_len = len(df)
        df = df.dropna(subset=essential_cols)
        logger.info(f"필수 데이터 확인 후: {len(df)}행 (제거: {initial_len - len(df)}행)")
        
        # 약제분류번호, 분류명이 비어있는 경우 기본값 설정
        df['code'] = df['code'].fillna('Unknown')
        df['code_name'] = df['code_name'].fillna('미분류')
        
        return df
        
    def slice_text(self, text: str) -> List[str]:
        """텍스트를 적절한 크기로 슬라이싱 (문장 경계 보존)"""
        if len(text) <= self.config.slice_length:
            return [text]
        
        slices = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = min(current_pos + self.config.slice_length, len(text))
            
            # 문장 경계에서 자르기
            if end_pos < len(text):
                # 마지막 문장 끝을 찾기
                last_sentence_end = max(
                    text.rfind('.', current_pos, end_pos),
                    text.rfind('다.', current_pos, end_pos),
                    text.rfind('함.', current_pos, end_pos)
                )
                
                if last_sentence_end > current_pos:
                    end_pos = last_sentence_end + 1
            
            slice_text = text[current_pos:end_pos].strip()
            if slice_text:
                slices.append(slice_text)
                
            current_pos = end_pos
            
        return slices
        
    def parse_drug_info(self, title: str) -> Dict[str, Any]:
        """약제 정보 파싱"""
        result = {
            'main_name': '',
            'brand_names': [],
            'raw_title': title
        }
        
        # 괄호 앞 주 약제명 추출
        if '(' in title:
            result['main_name'] = title.split('(')[0].strip()
        else:
            result['main_name'] = title.strip()
            
        # 품명 추출 (품명: 뒤의 내용을 · 또는 // 로 분리)
        if '품명:' in title:
            brand_part = title.split('품명:')[1].strip()
            # · 또는 // 로 분리
            separators = ['·', '//', ',']
            brands = [brand_part]
            
            for sep in separators:
                new_brands = []
                for brand in brands:
                    new_brands.extend([b.strip() for b in brand.split(sep) if b.strip()])
                brands = new_brands
                
            result['brand_names'] = [b for b in brands if b and b != result['main_name']]
            
        return result
        
    def generate_questions_by_band(self, doc_slice: str, band: LengthBand, 
                                 drug_info: Dict[str, Any]) -> List[str]:
        """밴드별 질문 생성"""
        band_config = self.config.bands[band]
        
        # 밴드별 프롬프트
        if band == LengthBand.SR:
            prompt = self._get_sr_prompt(doc_slice)
        elif band == LengthBand.MR:
            prompt = self._get_mr_prompt(doc_slice)
        else:  # LR
            prompt = self._get_lr_prompt(doc_slice)
            
        logger.info(f"{band.value} 밴드 질문 생성 중...")
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            if band == LengthBand.LR:
                # LR의 경우 시나리오에서 질문 추출
                questions = self._extract_lr_questions(content)
            else:
                # SR/MR의 경우 각 줄이 질문
                questions = [line.strip() for line in content.split('\n') 
                           if line.strip() and line.strip().endswith('?')]
                           
            logger.info(f"{band.value} 밴드: {len(questions)}개 질문 생성")
            return questions
            
        except Exception as e:
            logger.error(f"{band.value} 밴드 생성 실패: {e}")
            return []
    
    def _get_sr_prompt(self, doc_slice: str) -> str:
        """SR 밴드 프롬프트"""
        return f"""You generate Korean questions for an insurance review embedding model.
Return many lines, each a single Korean question strictly based on the document below.
Constraints: 25–80 characters; end with '?'; include at least one number/unit/policy term;
no pronouns like '이것/그것/해당/본/동 + 약·제제·제품'; one issue per line; vary openings; no JSON.

Document:
<<<DOC
{doc_slice}
DOC
>>>"""

    def _get_mr_prompt(self, doc_slice: str) -> str:
        """MR 밴드 프롬프트"""  
        return f"""Generate many Korean questions as lines based only on the document.
Constraints: 80–160 characters; end with '?'; include at least one number/unit/policy term;
strict pronoun ban; one issue per line; vary openings; no JSON.

Document:
<<<DOC
{doc_slice}
DOC
>>>"""

    def _get_lr_prompt(self, doc_slice: str) -> str:
        """LR 밴드 프롬프트 (멀티턴 시나리오→질문)"""
        return f"""Create multiple Korean scenarios (2–4 sentences) followed by a final question line.
Separate each case with a blank line. Use only the document content.
Constraints: 200–600 characters per case; the last sentence must be a question ending with '?';
include at least one policy term or number/unit; avoid pronouns in the question; no JSON.

Document:
<<<DOC
{doc_slice}
DOC
>>>"""

    def _extract_lr_questions(self, content: str) -> List[str]:
        """LR 시나리오에서 질문 추출"""
        questions = []
        cases = content.strip().split('\n\n')
        
        for case in cases:
            if not case.strip():
                continue
                
            sentences = [s.strip() for s in case.strip().split('.') if s.strip()]
            
            # 마지막 문장이 질문인지 확인
            for sentence in reversed(sentences):
                if sentence.endswith('?'):
                    # 전체 시나리오 길이 체크
                    full_case = case.strip()
                    if 200 <= len(full_case) <= 600:
                        questions.append(sentence)
                    break
                    
        return questions
        
    def validate_question(self, question: str, band: LengthBand) -> bool:
        """질문 유효성 검증"""
        band_config = self.config.bands[band]
        
        # 길이 체크
        if not (band_config.min_chars <= len(question) <= band_config.max_chars):
            return False
            
        # 물음표 체크
        if not question.endswith('?'):
            return False
            
        # 대명사 체크 (밴드별 차등)
        if band in [LengthBand.SR, LengthBand.MR]:
            # SR/MR은 엄격한 대명사 금지
            if self.strict_pronoun_pattern.search(question):
                return False
        else:  # LR
            # LR은 질문 부분만 대명사 금지 (시나리오 부분은 허용)
            if self.strict_pronoun_pattern.search(question):
                return False
        
        # 구체성 체크 (숫자/단위/정책어 포함)
        if not self.specificity_pattern.search(question):
            return False
            
        # 단일 논점 체크 (복합 질문 방지)
        complex_markers = [',', ' 및 ', '/']
        complex_count = sum(question.count(marker) for marker in complex_markers)
        if complex_count >= 2:
            return False
            
        return True
        
    def generate_hard_negatives(self, positive_questions: List[Question]) -> List[Question]:
        """Hard Negative 생성 (앵커 기반 변형)"""
        hard_negatives = []
        
        # 상위 POS 질문들을 앵커로 선택
        anchors = positive_questions[:min(5, len(positive_questions))]
        
        for anchor in anchors:
            # facet 변형 (1개만)
            mutated_questions = self._mutate_facet(anchor)
            
            for mutated in mutated_questions:
                # LLM으로 리라이트
                rewritten = self._rewrite_question(mutated)
                if rewritten and self.validate_question(rewritten, anchor.band):
                    hn_question = Question(
                        text=rewritten,
                        label="HARD_NEGATIVE", 
                        band=anchor.band,
                        anchor_id=anchor.anchor_id,
                        doc_slice_id=anchor.doc_slice_id,
                        metadata={**anchor.metadata, 'mutation_type': mutated.get('mutation_type')}
                    )
                    hard_negatives.append(hn_question)
                    
        return hard_negatives
        
    def _mutate_facet(self, anchor: Question) -> List[Dict[str, Any]]:
        """단일 facet 변형"""
        mutations = []
        
        # 변형 타입들
        mutation_types = [
            'dosage_boundary',  # 용량 경계
            'duration_change',  # 기간 변경
            'route_change',     # 투여경로 변경
            'indication_shift', # 적응증 이동
            'age_boundary',     # 연령 경계
            'coverage_flip',    # 급여↔비급여
        ]
        
        for mutation_type in mutation_types[:2]:  # 최대 2개 변형
            mutated_text = self._apply_mutation(anchor.text, mutation_type)
            if mutated_text and mutated_text != anchor.text:
                mutations.append({
                    'text': mutated_text,
                    'mutation_type': mutation_type
                })
                
        return mutations
        
    def _apply_mutation(self, text: str, mutation_type: str) -> Optional[str]:
        """구체적인 변형 적용"""
        if mutation_type == 'dosage_boundary':
            # 숫자 경계 변형 (예: 10mg → 5mg)
            import re
            numbers = re.findall(r'\d+', text)
            if numbers:
                original = numbers[0]
                modified = str(int(int(original) * 0.5))  # 절반으로
                return text.replace(original, modified, 1)
                
        elif mutation_type == 'coverage_flip':
            # 급여↔비급여 뒤바꾸기
            if '급여' in text:
                return text.replace('급여', '비급여', 1)
            elif '비급여' in text:
                return text.replace('비급여', '급여', 1)
                
        # 다른 변형들도 구현...
        return None
        
    def _rewrite_question(self, mutated_data: Dict[str, Any]) -> Optional[str]:
        """변형된 텍스트를 자연스러운 질문으로 리라이트"""
        prompt = f"""Rewrite the given Korean sentence into a natural Korean question.
Constraints: 25–80 characters; end with '?'; keep numbers/units/policy terms; no pronouns.
Return exactly one line. No JSON.

Original:
{mutated_data['text']}"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # 낮은 temperature로 일관성 확보
                max_tokens=100
            )
            
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten.endswith('?') else None
            
        except Exception as e:
            logger.error(f"리라이트 실패: {e}")
            return None
    
    def balance_labels_per_band(self, questions: List[Question], 
                              target_total: int) -> List[Question]:
        """밴드별 라벨 비율 정규화"""
        # 밴드별 그룹화
        band_groups = {}
        for q in questions:
            if q.band not in band_groups:
                band_groups[q.band] = []
            band_groups[q.band].append(q)
        
        balanced_questions = []
        
        for band, band_questions in band_groups.items():
            band_config = self.config.bands[band]
            band_target = int(target_total * band_config.ratio)
            
            # 밴드 내에서 라벨 비율 적용
            pos_target = int(band_target * self.config.pos_ratio / 9)
            hn_target = int(band_target * self.config.hn_ratio / 9)
            en_target = int(band_target * self.config.en_ratio / 9)
            
            # 라벨별 분류
            pos_questions = [q for q in band_questions if q.label == "POSITIVE"]
            hn_questions = [q for q in band_questions if q.label == "HARD_NEGATIVE"]
            en_questions = [q for q in band_questions if q.label == "EASY_NEGATIVE"]
            
            # 샘플링
            selected_pos = random.sample(pos_questions, min(pos_target, len(pos_questions)))
            selected_hn = random.sample(hn_questions, min(hn_target, len(hn_questions)))
            selected_en = random.sample(en_questions, min(en_target, len(en_questions)))
            
            balanced_questions.extend(selected_pos + selected_hn + selected_en)
            
            logger.info(f"{band.value} 밴드: POS {len(selected_pos)}, HN {len(selected_hn)}, EN {len(selected_en)}")
        
        return balanced_questions
        
    def remove_duplicates(self, questions: List[Question]) -> List[Question]:
        """중복 제거 (RapidFuzz 기반)"""
        unique_questions = []
        
        for q in questions:
            is_duplicate = False
            for existing in unique_questions:
                # 토큰셋 유사도 체크
                similarity = fuzz.token_set_ratio(q.text, existing.text)
                if similarity >= 82:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_questions.append(q)
                
        logger.info(f"중복 제거: {len(questions)} → {len(unique_questions)}")
        return unique_questions
        
    def check_opening_diversity(self, questions: List[Question]) -> bool:
        """문두 다양성 체크"""
        opening_counts = {}
        total = len(questions)
        
        for q in questions:
            for opening in self.opening_words:
                if q.text.startswith(opening):
                    opening_counts[opening] = opening_counts.get(opening, 0) + 1
                    break
                    
        # 30% 초과하는 문두가 있는지 체크
        for opening, count in opening_counts.items():
            if count / total > 0.3:
                logger.warning(f"문두 '{opening}' 과다 사용: {count}/{total} ({count/total*100:.1f}%)")
                return False
                
        return True
        
    def generate_questions_for_row(self, row: pd.Series, target_count: int) -> List[Question]:
        """한 행에 대한 전체 질문 생성 파이프라인"""
        # 기본 정보 추출
        drug_info = self.parse_drug_info(row['title'])
        text_slices = self.slice_text(row['text'])
        
        all_questions = []
        
        for slice_idx, doc_slice in enumerate(text_slices):
            anchor_id = str(uuid.uuid4())
            slice_id = f"{row['code']}_{slice_idx}"
            
            # 각 밴드별 POS 질문 생성
            for band in [LengthBand.SR, LengthBand.MR, LengthBand.LR]:
                questions_text = self.generate_questions_by_band(doc_slice, band, drug_info)
                
                for q_text in questions_text:
                    if self.validate_question(q_text, band):
                        question = Question(
                            text=q_text,
                            label="POSITIVE",
                            band=band,
                            anchor_id=anchor_id,
                            doc_slice_id=slice_id,
                            metadata={
                                'code': row['code'],
                                'code_name': row['code_name'],
                                'drug_info': drug_info
                            }
                        )
                        all_questions.append(question)
        
        # Hard Negative 생성
        pos_questions = [q for q in all_questions if q.label == "POSITIVE"]
        if pos_questions:
            hn_questions = self.generate_hard_negatives(pos_questions)
            all_questions.extend(hn_questions)
        
        # 중복 제거
        all_questions = self.remove_duplicates(all_questions)
        
        # 라벨 비율 정규화
        balanced_questions = self.balance_labels_per_band(all_questions, target_count)
        
        logger.info(f"행 {row['code']}: 총 {len(balanced_questions)}개 질문 생성")
        
        return balanced_questions
        
    def save_results(self, all_questions: List[Question], original_df: pd.DataFrame, 
                    output_file: str):
        """결과 저장"""
        # 엑셀 형식으로 변환
        results = []
        
        for q in all_questions:
            row_data = {
                '약제분류번호': q.metadata['code'],
                '약제 분류명': q.metadata['code_name'], 
                '구분': q.metadata.get('title', ''),
                '세부인정기준 및 방법': q.metadata.get('text', ''),
                'question': q.text,
                '라벨': q.label
            }
            results.append(row_data)
            
        result_df = pd.DataFrame(results)
        result_df.to_excel(output_file, index=False, engine='openpyxl')
        
        logger.info(f"결과 저장 완료: {output_file}")
        logger.info(f"총 {len(results)}개 질문")
        
        # 통계 출력
        label_counts = result_df['라벨'].value_counts()
        band_counts = pd.Series([q.band.value for q in all_questions]).value_counts()
        
        logger.info("라벨 분포:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count}")
            
        logger.info("밴드 분포:")
        for band, count in band_counts.items():
            logger.info(f"  {band}: {count}")
            
        # 앵커팩 JSONL 저장 (선택사항)
        jsonl_file = output_file.replace('.xlsx', '_anchorpack.jsonl')
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for q in all_questions:
                anchor_data = {
                    'anchor_id': q.anchor_id,
                    'band': q.band.value,
                    'question': q.text,
                    'doc_slice_id': q.doc_slice_id,
                    'label': q.label
                }
                f.write(json.dumps(anchor_data, ensure_ascii=False) + '\n')
                
        logger.info(f"앵커팩 저장 완료: {jsonl_file}")

def main():
    """메인 실행 함수"""
    # 설정
    config = GenerationConfig()
    generator = V7QuestionGenerator(config)
    
    # 고정 데이터 경로
    data_file = "C:/Jimin/Pharma-Augment/data/요양심사약제_후처리_v2.xlsx"
    output_file = "C:/Jimin/Pharma-Augment/versions/v7/drug_questions_v7.xlsx"
    
    try:
        # 데이터 로드
        df = generator.load_and_preprocess_data(data_file)
        
        # 전체 질문 생성
        all_questions = []
        target_per_row = 15  # 행당 목표 질문 수
        
        for idx, row in df.iterrows():
            logger.info(f"처리 중: {idx+1}/{len(df)} - {row['code']}")
            questions = generator.generate_questions_for_row(row, target_per_row)
            all_questions.extend(questions)
            
            # 진행상황 체크
            if (idx + 1) % 10 == 0:
                logger.info(f"진행상황: {idx+1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%)")
        
        # 최종 정리 및 저장
        all_questions = generator.remove_duplicates(all_questions)
        generator.save_results(all_questions, df, output_file)
        
        logger.info("V7 질문 생성 완료!")
        
    except Exception as e:
        logger.error(f"실행 중 오류: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()