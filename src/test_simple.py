#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5 간단한 1개 질문 생성 테스트
"""

import asyncio
import aiohttp
import json
import os
import sys
import io
from dotenv import load_dotenv
from drug_generator_v5 import DrugGeneratorV5

# 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

load_dotenv()

async def test_simple_v5():
    """간단한 V5 1개 질문 생성 테스트"""
    
    # 생성기 초기화
    generator = DrugGeneratorV5()
    
    # 샘플 데이터
    sample_data = {
        '약제분류번호': 'A01AD02',
        '약제 분류명': 'Tacrolimus 제제',
        '구분': 'Tacrolimus 제제 (품명: 프로그랍캅셀, 프로그랩주사)',
        '세부인정기준 및 방법': '1. 조혈모세포이식 후 거부반응 억제에 사용하는 경우'
    }
    
    print("=== V5 간단 버전 테스트 ===")
    
    try:
        async with aiohttp.ClientSession() as session:
            # V5 API 호출
            result = await generator.call_api_v5(session, 
                generator.create_drug_prompt_v5(sample_data), 
                "test", 
                sample_data)
            
            if result:
                print(f"✅ 성공! 질문 수: {len(result)}")
                for i, q in enumerate(result):
                    print(f"{i+1}. {q['question']}")
                    print(f"   라벨: {q['라벨']}")
            else:
                print("❌ 실패 - 빈 결과")
                
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_v5())