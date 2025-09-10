#!/usr/bin/env python3
"""
V7 데이터 품질 최적화 - Triplet Mining 전용
POS/HN 품질 대폭 향상 + 완벽한 비율 균형
"""

import pandas as pd
import re
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V7DataCleaner:
    def __init__(self):
        self.forbidden_words = [
            'TODO', '미정', '검토', '예시', '샘플', 'placeholder',
            '[', ']', '{', '}', '()', '빈칸', '공란'
        ]
        
        self.vague_patterns = [
            '가능한가요', '어떻게 하죠', '방법은 무엇', '어떤 경우',  
            '언제인가요', '왜인가요', '무엇인가요', '어디서',
            '누구', '어느 것', '얼마나', '어떤 방법', '어떤', 
            '몇 개월', '몇 회', '몇 일', '몇 주', '특이사항',
            '필요한가', '해야 하나', '해도 되나', '괜찮나',
            '적절한', '적합한', '최대', '최소', '이상인가',
            '미만인가', '해당하나', '포함되나', '제외되나'
        ]
        
    def is_high_quality(self, question: str) -> Tuple[bool, List[str]]:
        """초엄격 품질 검사"""
        issues = []
        
        # 1. 길이 검사 (30-250자)
        if len(question) < 30:
            issues.append(f'너무짧음({len(question)}자)')
        elif len(question) > 250:
            issues.append(f'너무김({len(question)}자)')
            
        # 2. 금지어 검사
        for word in self.forbidden_words:
            if word in question:
                issues.append(f'금지어({word})')
                
        # 3. 포괄/추상 표현 검사
        for pattern in self.vague_patterns:
            if pattern in question:
                issues.append(f'포괄표현({pattern})')
                
        # 4. 구체성 검사 - 숫자+단위 모두 필수
        has_number = bool(re.search(r'\d+', question))
        has_unit = bool(re.search(r'(U/L|mg/kg|mg|ml|㎍|㎎|일|개월|주|회|세|점|년|mg/dL|mmol/L|mmHg)', question))
        has_medical_term = bool(re.search(r'(ALT|AST|크레아티닌|혈압|체중|BMI|HbA1c|HFMSE)', question))
        
        if not has_number:
            issues.append('숫자없음')
        if not (has_unit or has_medical_term):
            issues.append('단위없음')
            
        # 5. 약제/지표/대상자 구체성 검사
        has_drug_context = bool(re.search(r'(투여|처방|복용|주사|적용|사용)', question))
        has_target = bool(re.search(r'(환자|성인|소아|임산부|고령자)', question))
        
        if not (has_drug_context or has_target):
            issues.append('맥락부족(약제맥락필수)')
            
        # 6. 질문 형태 검사
        if not question.endswith('?'):
            issues.append('질문형식오류')
            
        # 7. 의미있는 내용 검사
        if len(question.replace('?', '').strip()) < 25:
            issues.append('내용부족')
            
        # 8. 숫자 패턴 검사 (단순 번호 매기기 제외)
        if re.search(r'^\d+\.', question.strip()):
            issues.append('번호매기기')
            
        # 9. 완전한 문장 검사
        if question.count(' ') < 3:  # 최소 4개 단어 이상
            issues.append('문장짧음')
            
        return len(issues) == 0, issues
        
    def extract_best_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """최고 품질 질문만 추출"""
        logger.info("🔍 초엄격 품질 검사 시작...")
        
        high_quality = []
        quality_stats = {'pos_good': 0, 'pos_bad': 0, 'hn_good': 0, 'hn_bad': 0}
        
        for idx, row in df.iterrows():
            question = row['question']
            label = row['라벨']
            
            is_good, issues = self.is_high_quality(question)
            
            if is_good:
                high_quality.append(row)
                if label == 'POSITIVE':
                    quality_stats['pos_good'] += 1
                else:
                    quality_stats['hn_good'] += 1
            else:
                if label == 'POSITIVE':
                    quality_stats['pos_bad'] += 1 
                else:
                    quality_stats['hn_bad'] += 1
                    
                logger.debug(f"품질불량: {question[:50]}... → {issues}")
                
        logger.info(f"품질 검사 완료:")
        logger.info(f"  POSITIVE: {quality_stats['pos_good']}개 합격, {quality_stats['pos_bad']}개 탈락")
        logger.info(f"  HN: {quality_stats['hn_good']}개 합격, {quality_stats['hn_bad']}개 탈락")
        
        return pd.DataFrame(high_quality)
        
    def optimize_for_triplet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Triplet Mining 최적화"""
        logger.info("🎯 Triplet Mining 최적화 시작...")
        
        pos_df = df[df['라벨'] == 'POSITIVE'].copy()
        hn_df = df[df['라벨'] == 'HARD_NEGATIVE'].copy()
        
        logger.info(f"고품질 데이터: POS {len(pos_df)}개, HN {len(hn_df)}개")
        
        # POS는 쌍으로 구성되므로 짝수로 맞추기
        pos_pairs = len(pos_df) // 2
        hn_count = len(hn_df)
        
        # Triplet 개수 결정 (더 작은 값)
        max_triplets = min(pos_pairs, hn_count)
        
        logger.info(f"최대 구성 가능 Triplet: {max_triplets}개")
        
        # 최적 데이터 선택
        needed_pos = max_triplets * 2  # 쌍이므로 2배
        needed_hn = max_triplets
        
        # POS 데이터 선택 (품질 점수로 정렬 후 상위 선택)
        selected_pos = pos_df.head(needed_pos)
        selected_hn = hn_df.head(needed_hn)
        
        # 결합
        result_df = pd.concat([selected_pos, selected_hn], ignore_index=True)
        
        logger.info(f"✅ Triplet 최적화 완료:")
        logger.info(f"  POS: {len(selected_pos)}개 ({len(selected_pos)/len(result_df)*100:.1f}%)")
        logger.info(f"  HN: {len(selected_hn)}개 ({len(selected_hn)/len(result_df)*100:.1f}%)")
        logger.info(f"  총 {max_triplets}개 완벽한 Triplet 세트")
        logger.info(f"  총 데이터: {len(result_df)}개")
        
        return result_df
        
    def clean_dataset(self, input_file: str, output_file: str):
        """전체 정제 프로세스"""
        logger.info(f"🚀 V7 데이터 정제 시작: {input_file}")
        
        # 데이터 로드
        df = pd.read_excel(input_file)
        logger.info(f"원본 데이터: {len(df)}개")
        
        # 1. 고품질 데이터 추출
        high_quality_df = self.extract_best_questions(df)
        logger.info(f"고품질 데이터: {len(high_quality_df)}개 ({len(high_quality_df)/len(df)*100:.1f}%)")
        
        # 2. Triplet 최적화
        optimized_df = self.optimize_for_triplet(high_quality_df)
        
        # 3. 저장
        optimized_df.to_excel(output_file, index=False)
        logger.info(f"✅ 정제 완료: {output_file}")
        
        # 4. 결과 검증
        self.verify_result(optimized_df)
        
        return optimized_df
        
    def verify_result(self, df: pd.DataFrame):
        """결과 검증"""
        logger.info("🔍 결과 검증 중...")
        
        pos_count = len(df[df['라벨'] == 'POSITIVE'])
        hn_count = len(df[df['라벨'] == 'HARD_NEGATIVE'])
        total = len(df)
        
        logger.info(f"최종 결과:")
        logger.info(f"  POSITIVE: {pos_count}개 ({pos_count/total*100:.1f}%)")
        logger.info(f"  HARD_NEGATIVE: {hn_count}개 ({hn_count/total*100:.1f}%)")
        logger.info(f"  총 데이터: {total}개")
        logger.info(f"  Triplet 개수: {min(pos_count//2, hn_count)}개")
        
        # 샘플 검사
        logger.info("샘플 질문 검사:")
        for label in ['POSITIVE', 'HARD_NEGATIVE']:
            sample = df[df['라벨'] == label].head(2)
            logger.info(f"[{label}]")
            for _, row in sample.iterrows():
                logger.info(f"  Q: {row['question']}")
                
if __name__ == "__main__":
    cleaner = V7DataCleaner()
    
    input_file = 'results/V7_FULL_DATASET_20250909.xlsx'
    output_file = 'results/V7_CLEANED_OPTIMIZED.xlsx'
    
    try:
        result_df = cleaner.clean_dataset(input_file, output_file)
        logger.info("🎉 모든 정제 작업 완료!")
        
    except Exception as e:
        logger.error(f"❌ 정제 작업 실패: {e}")