@echo off
echo ========================================
echo  임베딩 최적화 질문 생성기 V2 테스트
echo ========================================
echo.

echo 샘플 데이터로 테스트를 시작합니다...
python generate_questions_v2.py ^
  --excel "..\data\sample_요양심사약제_후처리_v2.xlsx" ^
  --out "sample_embedding_v2.jsonl" ^
  --provider "openai" ^
  --model "gpt-4o-mini" ^
  --concurrency 2 ^
  --max_aug 20 ^
  --seed 20250902

echo.
echo ========================================
echo 테스트 완료! 결과 파일을 확인하세요:
echo - sample_embedding_v2.jsonl
echo - sample_embedding_v2_questions.xlsx
echo ========================================
pause