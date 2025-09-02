#!/usr/bin/env python3
"""
JSONL 파일을 엑셀 파일로 변환하는 스크립트
생성된 질문들을 Excel에서 쉽게 볼 수 있도록 변환합니다.
"""

import argparse
import json
import pandas as pd
import sys
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def jsonl_to_excel(jsonl_path: str, excel_path: str, include_meta: bool = False):
    """JSONL 파일을 엑셀 파일로 변환"""
    try:
        data = []
        
        # JSONL 파일 읽기
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # 기본 정보
                    row = {
                        'clause_id': item.get('clause_id', ''),
                        'group_id': item.get('group_id', ''),
                        'title': item.get('title', ''),
                        'title_clean': item.get('title_clean', ''),
                        'category': item.get('category', ''),
                        'code': item.get('code', ''),
                        'code_name': item.get('code_name', ''),
                        'question_count': len(item.get('questions', [])),
                        'questions': '\n'.join(f"{i+1}. {q}" for i, q in enumerate(item.get('questions', [])))
                    }
                    
                    # 메타 정보 포함 옵션
                    if include_meta and 'meta' in item:
                        meta = item['meta']
                        row.update({
                            'source_sheet': meta.get('source_sheet', ''),
                            'mapping_confidence': meta.get('mapping_confidence', ''),
                            'dedup_rule': meta.get('dedup_rule', ''),
                            'version': meta.get('version', ''),
                            'seed': meta.get('seed', ''),
                            'error': meta.get('error', '')
                        })
                    
                    data.append(row)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"라인 {line_num}에서 JSON 파싱 실패: {e}")
                    continue
        
        if not data:
            logger.error("변환할 데이터가 없습니다.")
            return
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # 엑셀 파일로 저장 (여러 시트)
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 메인 시트: 전체 데이터
            df.to_excel(writer, sheet_name='전체_질문목록', index=False)
            
            # 통계 시트
            stats_data = []
            stats_data.append(['총 조항 수', len(df)])
            stats_data.append(['총 질문 수', df['question_count'].sum()])
            stats_data.append(['평균 질문 수', df['question_count'].mean()])
            stats_data.append(['최대 질문 수', df['question_count'].max()])
            stats_data.append(['최소 질문 수', df['question_count'].min()])
            
            # 카테고리별 통계
            if 'category' in df.columns:
                category_stats = df.groupby('category')['question_count'].agg(['count', 'sum', 'mean']).reset_index()
                category_stats.columns = ['카테고리', '조항수', '총질문수', '평균질문수']
                
                stats_data.append(['', ''])  # 빈 줄
                stats_data.append(['카테고리별 통계', ''])
                for _, row in category_stats.iterrows():
                    stats_data.append([f"  {row['카테고리']}", f"조항: {row['조항수']}, 질문: {row['총질문수']}, 평균: {row['평균질문수']:.1f}"])
            
            stats_df = pd.DataFrame(stats_data, columns=['항목', '값'])
            stats_df.to_excel(writer, sheet_name='통계', index=False)
            
            # 카테고리별 시트 (카테고리가 있는 경우)
            if 'category' in df.columns and df['category'].notna().any():
                for category in df['category'].dropna().unique():
                    category_df = df[df['category'] == category]
                    sheet_name = f"카테고리_{category}"[:31]  # 엑셀 시트명 길이 제한
                    category_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"엑셀 변환 완료: {excel_path}")
        logger.info(f"- 총 {len(df)}개 조항")
        logger.info(f"- 총 {df['question_count'].sum()}개 질문")
        logger.info(f"- 평균 {df['question_count'].mean():.1f}개 질문/조항")
        
    except Exception as e:
        logger.error(f"엑셀 변환 실패: {e}")
        raise

def create_question_list_excel(jsonl_path: str, excel_path: str):
    """질문만 별도 리스트로 추출하여 엑셀로 저장"""
    try:
        questions_data = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    clause_id = item.get('clause_id', '')
                    title = item.get('title_clean', item.get('title', ''))
                    category = item.get('category', '')
                    
                    for i, question in enumerate(item.get('questions', []), 1):
                        questions_data.append({
                            'clause_id': clause_id,
                            'title': title,
                            'category': category,
                            'question_no': i,
                            'question': question
                        })
                        
                except json.JSONDecodeError:
                    continue
        
        if questions_data:
            df = pd.DataFrame(questions_data)
            df.to_excel(excel_path, sheet_name='질문목록', index=False)
            logger.info(f"질문 리스트 엑셀 저장 완료: {excel_path} ({len(questions_data)}개 질문)")
        
    except Exception as e:
        logger.error(f"질문 리스트 엑셀 저장 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description="JSONL 파일을 엑셀 파일로 변환")
    
    parser.add_argument("jsonl_file", help="입력 JSONL 파일 경로")
    parser.add_argument("--excel", "-o", help="출력 엑셀 파일 경로 (기본: [jsonl파일명].xlsx)")
    parser.add_argument("--include-meta", action="store_true", help="메타데이터 포함")
    parser.add_argument("--questions-only", action="store_true", help="질문만 별도 엑셀로 저장")
    
    args = parser.parse_args()
    
    # 출력 파일명 결정
    if args.excel:
        excel_path = args.excel
    else:
        excel_path = args.jsonl_file.rsplit('.', 1)[0] + '.xlsx'
    
    try:
        # 메인 변환
        jsonl_to_excel(args.jsonl_file, excel_path, args.include_meta)
        
        # 질문만 별도 저장 옵션
        if args.questions_only:
            questions_excel_path = excel_path.rsplit('.', 1)[0] + '_questions_only.xlsx'
            create_question_list_excel(args.jsonl_file, questions_excel_path)
        
        print(f"✅ 변환 완료: {excel_path}")
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()