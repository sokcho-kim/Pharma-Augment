"""
V6 테스트 스크립트 - 소량 샘플로 파이프라인 검증
"""

import pandas as pd
from drug_generator_v6 import V6QuestionGenerator, GenerationConfig
import logging

# 테스트용 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_v6_pipeline():
    """V6 파이프라인 테스트"""
    try:
        logger.info("V6 테스트 시작")
        
        # 테스트용 설정 (소량)
        test_config = GenerationConfig()
        test_config.temperature = 0.8
        
        # 생성기 초기화
        generator = V6QuestionGenerator(test_config)
        
        # 데이터 로드 (첫 3행만)
        data_path = r"C:\Jimin\Pharma-Augment\data\요양심사약제_후처리_v2.xlsx"
        df = generator.load_and_preprocess_data(data_path)
        test_df = df.head(3)  # 첫 3행만 테스트
        
        logger.info(f"테스트 데이터: {len(test_df)}행")
        
        # 질문 생성 테스트
        results = generator.process_drug_data(test_df)
        
        # 결과 저장
        output_path = r"C:\Jimin\Pharma-Augment\versions\v6\test_v6_results.xlsx"
        generator.save_results(results, output_path)
        
        # 결과 분석
        if results:
            df_results = pd.DataFrame(results)
            print("\n=== 테스트 결과 ===")
            print(f"총 질문 수: {len(results)}")
            print(f"라벨 분포:\n{df_results['라벨'].value_counts()}")
            print(f"\n샘플 질문들:")
            for i, result in enumerate(results[:10]):
                print(f"{i+1}. [{result['라벨']}] {result['question']}")
        
        logger.info("V6 테스트 완료!")
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        raise

if __name__ == "__main__":
    test_v6_pipeline()