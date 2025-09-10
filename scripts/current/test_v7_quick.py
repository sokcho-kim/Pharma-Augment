#!/usr/bin/env python3
"""
V7 Quick Test Script - 첫 3행으로 빠른 검증
9시까지 완벽한 결과를 위한 테스트
"""

import os
import sys
sys.path.append('versions/v7')

from drug_generator_v7 import V7QuestionGenerator, GenerationConfig
import pandas as pd
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_v7_quick():
    """빠른 3행 테스트"""
    logger.info("🚀 V7 빠른 테스트 시작 (3행)")
    
    # 원본 데이터 로드
    df = pd.read_excel('data/요양심사약제_후처리_v2.xlsx')
    logger.info(f"원본 데이터: {len(df)}행")
    
    # 첫 3행만 테스트 
    test_df = df.head(3).copy()
    logger.info(f"테스트 데이터: {len(test_df)}행")
    
    # V7 생성기 초기화
    config = GenerationConfig()
    generator = V7QuestionGenerator(config)
    
    # 테스트 실행
    output_file = 'results/V7_QUICK_TEST.xlsx'
    try:
        generator.generate_dataset(
            file_path='data/요양심사약제_후처리_v2.xlsx',
            output_file=output_file,
            max_rows=3  # 3행으로 제한
        )
        
        # 결과 검증
        if os.path.exists(output_file):
            result_df = pd.read_excel(output_file)
            logger.info(f"✅ 테스트 성공!")
            logger.info(f"생성된 질문 수: {len(result_df)}")
            logger.info(f"컬럼 구조: {list(result_df.columns)}")
            
            # 필수 컬럼 검증
            required_cols = ['약제분류번호', '약제분류명', '구분', '세부인정기준 및 방법', 'question', '라벨']
            missing_cols = [col for col in required_cols if col not in result_df.columns]
            
            if missing_cols:
                logger.error(f"❌ 누락된 컬럼: {missing_cols}")
                return False
            else:
                logger.info("✅ 모든 필수 컬럼 존재")
                
            # 데이터 샘플 확인
            logger.info("첫 행 샘플:")
            first_row = result_df.iloc[0]
            for col in ['구분', '세부인정기준 및 방법']:
                logger.info(f"  {col}: {str(first_row[col])[:50]}...")
                
            return True
        else:
            logger.error("❌ 결과 파일이 생성되지 않음")
            return False
            
    except Exception as e:
        logger.error(f"❌ 테스트 실행 실패: {e}")
        return False

if __name__ == "__main__":
    success = test_v7_quick()
    if success:
        logger.info("🎉 테스트 성공! 이제 전체 실행 준비 완료")
    else:
        logger.error("💥 테스트 실패! 코드 재검토 필요")
        sys.exit(1)