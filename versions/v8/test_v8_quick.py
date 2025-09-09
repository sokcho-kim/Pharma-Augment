#!/usr/bin/env python3
"""
V8 Quick Test - 3행 빠른 검증
"""

import sys
import os
sys.path.append('.')

from drug_generator_v8 import V8QuestionGenerator, V8Config
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_v8_quick():
    """V8 3행 빠른 테스트"""
    logger.info("🚀 V8 빠른 테스트 시작 (3행)")
    
    # V8 설정
    config = V8Config(
        questions_per_row=6,  # 테스트용으로 적게
        pos_ratio=0.67
    )
    
    generator = V8QuestionGenerator(config)
    
    # 테스트 실행
    output_file = '../../results/V8_QUICK_TEST.xlsx'
    try:
        generator.generate_dataset(
            file_path='../../data/요양심사약제_후처리_v2.xlsx',
            output_file=output_file,
            max_rows=3
        )
        
        # 결과 검증
        if os.path.exists(output_file):
            result_df = pd.read_excel(output_file)
            logger.info(f"✅ 테스트 성공!")
            logger.info(f"생성된 질문 수: {len(result_df)}개")
            logger.info(f"컬럼 구조: {list(result_df.columns)}")
            
            # 필수 컬럼 확인
            required_cols = ['약제분류번호', '약제분류명', '구분', '세부인정기준 및 방법', 'question', '라벨']
            missing_cols = [col for col in required_cols if col not in result_df.columns]
            
            if missing_cols:
                logger.error(f"❌ 누락된 컬럼: {missing_cols}")
                return False
            else:
                logger.info("✅ 모든 앵커 컬럼 완벽 보존")
                
            # 포괄질문 검증
            forbidden_patterns = ["무엇인가", "어떤", "어떻게", "적응증은", "몇"]
            bad_questions = []
            
            for _, row in result_df.iterrows():
                q = row['question']
                for pattern in forbidden_patterns:
                    if pattern in q:
                        bad_questions.append(q)
                        break
                        
            if bad_questions:
                logger.warning(f"⚠️ 포괄질문 {len(bad_questions)}개 발견")
                for bq in bad_questions:
                    logger.warning(f"  - {bq}")
            else:
                logger.info("✅ 포괄질문 완전 제거 성공")
                
            # 샘플 질문 출력
            logger.info("샘플 질문:")
            for i, row in result_df.head(5).iterrows():
                logger.info(f"  [{row['라벨']}] {row['question']}")
                
            return True
        else:
            logger.error("❌ 결과 파일이 생성되지 않음")
            return False
            
    except Exception as e:
        logger.error(f"❌ 테스트 실행 실패: {e}")
        return False

if __name__ == "__main__":
    success = test_v8_quick()
    if success:
        logger.info("🎉 V8 테스트 성공! 전체 실행 준비 완료")
    else:
        logger.error("💥 V8 테스트 실패! 코드 재검토 필요")
        sys.exit(1)