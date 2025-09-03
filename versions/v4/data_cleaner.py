#!/usr/bin/env python3
"""
데이터 전처리: [일반원칙], [심사지침] 등 괄호 패턴 제거
"""

import pandas as pd
import re
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_bracket_patterns(text):
    """괄호 패턴 제거"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # [일반원칙], [심사지침], [가], [나] 등 패턴 제거
    text = re.sub(r'\[.*?\]\s*', '', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

def clean_drug_data(input_path, output_path):
    """약제 데이터 정리"""
    try:
        # 엑셀 파일 로드
        df = pd.read_excel(input_path)
        logger.info(f"원본 데이터 로드: {df.shape[0]}행")
        
        # 컬럼명 확인
        logger.info(f"컬럼: {list(df.columns)}")
        
        # 구분 필드 정리
        if '구분' in df.columns:
            original_count = df['구분'].str.contains(r'\[.*?\]', na=False, regex=True).sum()
            logger.info(f"[괄호] 패턴 발견: {original_count}개")
            
            df['구분'] = df['구분'].apply(clean_bracket_patterns)
            
            # 정리 후 확인
            remaining_count = df['구분'].str.contains(r'\[.*?\]', na=False, regex=True).sum()
            logger.info(f"정리 후 남은 패턴: {remaining_count}개")
        
        # 세부인정기준 및 방법 필드도 정리
        if '세부인정기준 및 방법' in df.columns:
            # 문서 시작 부분의 괄호 패턴만 제거 (내용 중간은 보존)
            def clean_content_start(text):
                if pd.isna(text):
                    return ""
                text = str(text)
                # 문서 시작 부분의 패턴만 제거 (첫 100자 내)
                if len(text) > 10:
                    start_part = text[:100]
                    if re.match(r'^\[.*?\]\s*', start_part):
                        text = re.sub(r'^\[.*?\]\s*', '', text)
                return text
            
            df['세부인정기준 및 방법'] = df['세부인정기준 및 방법'].apply(clean_content_start)
        
        # 빈 구분 필드 제거
        original_len = len(df)
        df = df[df['구분'].str.strip() != '']
        logger.info(f"빈 구분 제거: {original_len} → {len(df)}행")
        
        # 결과 저장
        df.to_excel(output_path, index=False, engine='openpyxl')
        logger.info(f"정리된 데이터 저장: {output_path}")
        
        # 샘플 출력
        print("\n=== 정리 후 샘플 (처음 3개) ===")
        for i in range(min(3, len(df))):
            print(f"구분: {df.iloc[i]['구분'][:60]}")
            print(f"내용: {str(df.iloc[i]['세부인정기준 및 방법'])[:80]}...")
            print()
        
        return df
        
    except Exception as e:
        logger.error(f"데이터 정리 실패: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="약제 데이터 괄호 패턴 정리")
    parser.add_argument("--input", required=True, help="입력 엑셀 파일")
    parser.add_argument("--output", required=True, help="출력 엑셀 파일")
    
    args = parser.parse_args()
    
    clean_drug_data(args.input, args.output)
    print("✅ 데이터 정리 완료!")

if __name__ == "__main__":
    main()