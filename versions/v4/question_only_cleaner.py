#!/usr/bin/env python3
"""
생성된 질문 파일에서 question 필드만 [일반원칙], [심사지침] 등 괄호 패턴 제거
구분 필드는 그대로 유지
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

def clean_question_only(input_path, output_path):
    """question 필드만 정리"""
    try:
        # 엑셀 파일 로드
        df = pd.read_excel(input_path)
        logger.info(f"질문 파일 로드: {df.shape[0]}행")
        
        # question 필드만 괄호 패턴 확인 및 제거
        if 'question' in df.columns:
            # 패턴 확인
            original_count = df['question'].str.contains(r'\[.*?\]', na=False, regex=True).sum()
            logger.info(f"question 필드 [괄호] 패턴: {original_count}개")
            
            if original_count > 0:
                # question 필드만 패턴 제거
                df['question'] = df['question'].apply(clean_bracket_patterns)
                
                # 정리 후 확인
                remaining_count = df['question'].str.contains(r'\[.*?\]', na=False, regex=True).sum()
                logger.info(f"question 정리 후: {remaining_count}개")
        
        # 빈 질문 제거
        original_len = len(df)
        df = df[df['question'].str.strip() != '']
        if len(df) != original_len:
            logger.info(f"빈 질문 제거: {original_len} → {len(df)}행")
        
        # 결과 저장
        df.to_excel(output_path, index=False, engine='openpyxl')
        logger.info(f"정리된 질문 파일 저장: {output_path}")
        
        # 정리 전후 샘플 비교
        print("\n=== 정리 후 샘플 (처음 5개 질문) ===")
        for i in range(min(5, len(df))):
            print(f"{i+1}. 구분: {df.iloc[i]['구분']} (원본 유지)")
            print(f"   질문: {df.iloc[i]['question']} (괄호 제거됨)")
            print(f"   라벨: {df.iloc[i]['라벨']}")
            print()
        
        return df
        
    except Exception as e:
        logger.error(f"질문 파일 정리 실패: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="question 필드만 괄호 패턴 정리")
    parser.add_argument("--input", required=True, help="입력 질문 파일 (.xlsx)")
    parser.add_argument("--output", required=True, help="출력 질문 파일 (.xlsx)")
    
    args = parser.parse_args()
    
    clean_question_only(args.input, args.output)
    print("질문 필드만 정리 완료!")

if __name__ == "__main__":
    main()