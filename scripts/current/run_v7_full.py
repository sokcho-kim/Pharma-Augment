#!/usr/bin/env python3
"""
V7 Full Dataset Execution - 전체 688행 실행
9시까지 완벽한 결과를 위한 메인 실행 스크립트
"""

import os
import sys
sys.path.append('versions/v7')

from drug_generator_v7 import V7QuestionGenerator, GenerationConfig
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_v7_full():
    """전체 688행 V7 실행"""
    start_time = datetime.now()
    logger.info(f"🚀 V7 전체 데이터셋 실행 시작 - {start_time}")
    
    # V7 생성기 초기화
    config = GenerationConfig()
    generator = V7QuestionGenerator(config)
    
    # 전체 실행
    output_file = 'results/V7_FULL_DATASET_20250909.xlsx'
    try:
        generator.generate_dataset(
            file_path='data/요양심사약제_후처리_v2.xlsx',
            output_file=output_file
            # max_rows는 지정하지 않아서 전체 실행
        )
        
        # 완료 메시지
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"🎉 V7 전체 실행 완료!")
        logger.info(f"📅 시작 시간: {start_time}")
        logger.info(f"📅 완료 시간: {end_time}")
        logger.info(f"⏱️ 소요 시간: {duration}")
        logger.info(f"📁 결과 파일: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 전체 실행 실패: {e}")
        return False

if __name__ == "__main__":
    success = run_v7_full()
    if success:
        logger.info("🎉 모든 작업이 성공적으로 완료되었습니다!")
    else:
        logger.error("💥 실행 중 오류가 발생했습니다.")
        sys.exit(1)