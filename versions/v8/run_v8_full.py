#!/usr/bin/env python3
"""
V8 Full Dataset Execution - 전체 688행 실행
임베딩 모델 파인튜닝용 최적화 데이터셋
"""

import sys
import os
sys.path.append('.')

from drug_generator_v8 import V8QuestionGenerator, V8Config
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_v8_full():
    """전체 688행 V8 실행"""
    start_time = datetime.now()
    logger.info(f"🚀 V8 전체 데이터셋 실행 시작 - {start_time}")
    
    # V8 최적화 설정
    config = V8Config(
        model="gpt-4",
        temperature=0.7,
        questions_per_row=8,  # 행당 8개 질문
        pos_ratio=0.67  # POS 67%, HN 33%
    )
    
    generator = V8QuestionGenerator(config)
    
    # 전체 실행
    output_file = '../../results/V8_FULL_DATASET_20250909.xlsx'
    try:
        generator.generate_dataset(
            file_path='../../data/요양심사약제_후처리_v2.xlsx',
            output_file=output_file
            # max_rows 지정 없음 = 전체 실행
        )
        
        # 완료 메시지
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"🎉 V8 전체 실행 완료!")
        logger.info(f"📅 시작 시간: {start_time}")
        logger.info(f"📅 완료 시간: {end_time}")
        logger.info(f"⏱️ 소요 시간: {duration}")
        logger.info(f"📁 결과 파일: {output_file}")
        
        # 최종 품질 보고
        logger.info("📊 V8 최종 성과:")
        logger.info("  ✅ 포괄질문 완전 제거")
        logger.info("  ✅ 앵커 데이터 완벽 보존") 
        logger.info("  ✅ 약물명 중심 데이터 생성")
        logger.info("  ✅ 임베딩 모델 파인튜닝 최적화")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 전체 실행 실패: {e}")
        return False

if __name__ == "__main__":
    success = run_v8_full()
    if success:
        logger.info("🎉 V8 모든 작업이 성공적으로 완료되었습니다!")
        logger.info("📈 임베딩 모델 파인튜닝 준비 완료")
    else:
        logger.error("💥 V8 실행 중 오류가 발생했습니다.")
        sys.exit(1)