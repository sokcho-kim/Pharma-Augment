@echo off
echo ========================================
echo  임베딩 최적화 질문 생성기 V2 전체 실행
echo ========================================
echo.

echo 전체 688개 조항으로 질문 생성을 시작합니다...
echo 예상 소요시간: 15-20분
echo 예상 생성 질문 수: 15,000-20,000개
echo.

python generate_questions_v2.py ^
  --excel "..\data\요양심사약제_후처리_v2.xlsx" ^
  --out "embedding_questions_v2.jsonl" ^
  --provider "openai" ^
  --model "gpt-4o-mini" ^
  --concurrency 6 ^
  --max_aug 20 ^
  --seed 20250902

echo.
echo ========================================
echo 전체 생성 완료! 결과 파일:
echo - embedding_questions_v2.jsonl (JSONL 원본)
echo - embedding_questions_v2_questions.xlsx (엑셀 형식)
echo - audit_log_v2.csv (감사 로그)
echo ========================================
pause