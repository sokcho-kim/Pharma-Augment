#!/usr/bin/env python3
"""
V8 Drug Question Generator - 임베딩 모델 파인튜닝 최적화
- 포괄질문 완전 제거
- 앵커 데이터 완벽 보존  
- 약물명/수치 권장 (필수 아님)
- 실행시간 단축
"""

import pandas as pd
import openai
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv("../../.env")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Question:
    text: str
    label: str
    anchor_id: str
    doc_slice_id: int
    metadata: Dict[str, Any]

@dataclass 
class V8Config:
    model: str = "gpt-4"
    temperature: float = 0.7
    questions_per_row: int = 8
    pos_ratio: float = 0.67  # POS 67%, HN 33%
    
class V8QuestionGenerator:
    def __init__(self, config: V8Config):
        self.config = config
        self.forbidden_patterns = [
            "무엇인가", "어떤", "어떻게", "적응증은", "몇", "어느", "언제", "왜",
            "방법은", "어떻게 하", "해야 하나", "해도 되나", "괜찮나",
            "적절한", "적합한", "필요한가", "가능한가", "어떤 경우"
        ]
        
    def generate_questions_for_row(self, row: pd.Series, idx: int) -> List[Question]:
        """행별 질문 생성 - 앵커 데이터 완벽 보존"""
        
        # 앵커 데이터 추출 (완벽 보존)
        anchor_data = {
            'code': str(row.get('약제분류번호', '')),
            'code_name': str(row.get('약제분류명', '')),
            'category': str(row.get('구분', '')),
            'criteria': str(row.get('세부인정기준 및 방법', '')),
            'original_index': idx
        }
        
        # 문서 컨텍스트 구성
        doc_context = f"""
약제분류: {anchor_data['code_name']} ({anchor_data['code']})
구분: {anchor_data['category']}
세부인정기준 및 방법: {anchor_data['criteria']}
"""
        
        # 질문 생성
        all_questions = []
        
        try:
            # POSITIVE 질문 생성
            pos_count = int(self.config.questions_per_row * self.config.pos_ratio)
            pos_questions = self._generate_questions(doc_context, pos_count, "POSITIVE", anchor_data)
            all_questions.extend(pos_questions)
            
            # HARD_NEGATIVE 질문 생성  
            hn_count = self.config.questions_per_row - pos_count
            hn_questions = self._generate_questions(doc_context, hn_count, "HARD_NEGATIVE", anchor_data)
            all_questions.extend(hn_questions)
            
            logger.info(f"Row {idx}: {len(pos_questions)} POS + {len(hn_questions)} HN = {len(all_questions)} 생성")
            
        except Exception as e:
            logger.error(f"Row {idx} 질문 생성 실패: {e}")
            
        return all_questions
        
    def _generate_questions(self, doc_context: str, count: int, label: str, anchor_data: Dict) -> List[Question]:
        """실제 질문 생성"""
        
        if label == "POSITIVE":
            prompt = self._get_positive_prompt(doc_context, count)
        else:
            prompt = self._get_hard_negative_prompt(doc_context, count)
            
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=1000
            )
            
            raw_text = response.choices[0].message.content.strip()
            question_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
            
            questions = []
            for i, line in enumerate(question_lines[:count]):
                # 품질 검증
                is_valid, reason = self._validate_question(line)
                if is_valid:
                    question = Question(
                        text=line,
                        label=label,
                        anchor_id=f"{anchor_data['code']}_{anchor_data['original_index']}",
                        doc_slice_id=i,
                        metadata=anchor_data.copy()
                    )
                    questions.append(question)
                else:
                    logger.warning(f"품질 불합격: {line} - {reason}")
                    
            return questions
            
        except Exception as e:
            logger.error(f"API 호출 실패: {e}")
            return []
            
    def _get_positive_prompt(self, doc_context: str, count: int) -> str:
        """POSITIVE 질문 프롬프트"""
        return f"""임베딩 모델 학습용 의료보험 급여기준 질문을 {count}개 생성하시오.

문서:
{doc_context}

요구사항:
- 위 급여기준에 부합하는 구체적인 임상 상황 질문
- 권장: 구체적 약물명, 수치, 단위 포함 (예: Metformin 500mg, ALT 40U/L)
- 필수: "무엇인가", "어떤", "어떻게", "적응증은", "몇" 등 포괄표현 금지
- 30-250자 길이, '?' 종료
- 각 줄에 하나씩, 번호 없이

예시 스타일:
- 당뇨병 환자에서 HbA1c 8% 이상일 때 Metformin 초기 용량은 얼마나 되는가?
- 간기능 저하 환자의 ALT 수치가 정상 상한의 3배 초과시 해당 약물 투여를 중단해야 하는가?"""

    def _get_hard_negative_prompt(self, doc_context: str, count: int) -> str:
        """HARD_NEGATIVE 질문 프롬프트"""
        return f"""임베딩 모델 학습용 Hard Negative 질문을 {count}개 생성하시오.

문서:
{doc_context}

Hard Negative 요구사항:
- 위 급여기준과 유사하지만 미묘하게 다른 조건/수치의 질문
- 같은 약물/질환이지만 기준에 맞지 않는 상황
- 권장: 약물명, 수치 포함하되 기준과 다르게
- 필수: "무엇인가", "어떤", "어떻게" 등 포괄표현 금지
- 30-250자 길이, '?' 종료

Hard Negative 예시:
- 원본이 "ALT 40U/L 이상"이면 → "ALT 30U/L 이하일 때"로 변형
- 원본이 "3개월 이상"이면 → "2개월 이하"로 변형
- 원본이 "성인"이면 → "소아"로 변형"""

    def _validate_question(self, question: str) -> Tuple[bool, str]:
        """질문 품질 검증 (권장사항 기반)"""
        
        # 기본 형식 검증
        if not question.endswith('?'):
            return False, "질문형식 오류"
            
        if len(question) < 30 or len(question) > 250:
            return False, f"길이 부적절({len(question)}자)"
            
        # 포괄표현 금지 (필수)
        for pattern in self.forbidden_patterns:
            if pattern in question:
                return False, f"포괄표현({pattern})"
                
        # 의미 있는 내용 검증
        if question.count(' ') < 3:
            return False, "내용 부족"
            
        # 권장사항 점수 (가산점)
        score = 100  # 기본 점수
        
        # 약물명 권장 (가산점)
        if re.search(r'[A-Z][a-z]{4,}', question):
            score += 20
            
        # 수치 권장 (가산점)  
        if re.search(r'\d+', question):
            score += 15
            
        # 단위 권장 (가산점)
        if re.search(r'(mg|ml|U/L|mmHg|일|개월|주|회|%)', question):
            score += 10
            
        return True, f"점수:{score}"
        
    def generate_dataset(self, file_path: str, output_file: str, max_rows: int = None):
        """전체 데이터셋 생성"""
        
        logger.info(f"🚀 V8 데이터셋 생성 시작: {file_path}")
        
        # 원본 데이터 로드
        df = pd.read_excel(file_path)
        if max_rows:
            df = df.head(max_rows)
            
        logger.info(f"대상 데이터: {len(df)}행")
        
        all_questions = []
        
        # 행별 질문 생성
        for idx, row in df.iterrows():
            logger.info(f"처리 중: {idx+1}/{len(df)} 행")
            
            row_questions = self.generate_questions_for_row(row, idx)
            all_questions.extend(row_questions)
            
            # 체크포인트 저장 (50행마다)
            if (idx + 1) % 50 == 0:
                self._save_checkpoint(all_questions, df, output_file, idx + 1)
                
        # 최종 저장
        self._save_final_results(all_questions, df, output_file)
        
        # 결과 검증
        self._verify_results(all_questions, output_file)
        
        logger.info(f"✅ V8 데이터셋 생성 완료: {len(all_questions)}개 질문")
        
    def _save_checkpoint(self, questions: List[Question], df: pd.DataFrame, output_file: str, processed_rows: int):
        """체크포인트 저장"""
        checkpoint_file = output_file.replace('.xlsx', f'_checkpoint_{processed_rows}.xlsx')
        self._save_questions_to_excel(questions, df, checkpoint_file)
        logger.info(f"체크포인트 저장: {checkpoint_file}")
        
    def _save_final_results(self, questions: List[Question], df: pd.DataFrame, output_file: str):
        """최종 결과 저장"""
        self._save_questions_to_excel(questions, df, output_file)
        logger.info(f"최종 결과 저장: {output_file}")
        
    def _save_questions_to_excel(self, questions: List[Question], original_df: pd.DataFrame, output_file: str):
        """질문을 Excel로 저장 - 앵커 데이터 완벽 보존"""
        
        result_rows = []
        for q in questions:
            row_data = {
                '약제분류번호': q.metadata['code'],
                '약제분류명': q.metadata['code_name'],
                '구분': q.metadata['category'],
                '세부인정기준 및 방법': q.metadata['criteria'],
                'question': q.text,
                '라벨': q.label
            }
            result_rows.append(row_data)
            
        result_df = pd.DataFrame(result_rows)
        result_df.to_excel(output_file, index=False)
        
    def _verify_results(self, questions: List[Question], output_file: str):
        """결과 검증"""
        
        pos_count = len([q for q in questions if q.label == "POSITIVE"])
        hn_count = len([q for q in questions if q.label == "HARD_NEGATIVE"])
        
        # 약물명 포함률 계산
        drug_count = len([q for q in questions if re.search(r'[A-Z][a-z]{4,}', q.text)])
        drug_ratio = drug_count / len(questions) * 100 if questions else 0
        
        # 수치 포함률 계산  
        number_count = len([q for q in questions if re.search(r'\d+', q.text)])
        number_ratio = number_count / len(questions) * 100 if questions else 0
        
        logger.info(f"📊 V8 최종 결과:")
        logger.info(f"  총 질문: {len(questions)}개")
        logger.info(f"  POSITIVE: {pos_count}개 ({pos_count/len(questions)*100:.1f}%)")
        logger.info(f"  HARD_NEGATIVE: {hn_count}개 ({hn_count/len(questions)*100:.1f}%)")
        logger.info(f"  약물명 포함: {drug_count}개 ({drug_ratio:.1f}%)")
        logger.info(f"  수치 포함: {number_count}개 ({number_ratio:.1f}%)")
        logger.info(f"  Triplet 구성 가능: {min(pos_count//2, hn_count)}개")
        
        # 품질 지표 검증
        if drug_ratio >= 70:
            logger.info("✅ 약물명 포함률 목표 달성 (70% 이상)")
        else:
            logger.warning(f"⚠️ 약물명 포함률 부족: {drug_ratio:.1f}% < 70%")
            
        if number_ratio >= 80:
            logger.info("✅ 수치 포함률 목표 달성 (80% 이상)")
        else:
            logger.warning(f"⚠️ 수치 포함률 부족: {number_ratio:.1f}% < 80%")

if __name__ == "__main__":
    # V8 설정
    config = V8Config(
        model="gpt-4",
        temperature=0.7,
        questions_per_row=8,
        pos_ratio=0.67
    )
    
    generator = V8QuestionGenerator(config)
    
    try:
        generator.generate_dataset(
            file_path='../../data/요양심사약제_후처리_v2.xlsx',
            output_file='../../results/V8_OPTIMIZED_DATASET.xlsx'
        )
        
    except Exception as e:
        logger.error(f"❌ V8 실행 실패: {e}")
        raise