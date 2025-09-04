@echo off
chcp 65001 >nul
echo V5 약제 질문 생성기 전체 실행

REM 환경 설정
set PYTHONIOENCODING=utf-8

REM 프로젝트 루트로 이동
cd /d C:\Sokcho\Pharma-Augment

REM 정리된 데이터로 전체 실행 (cleaned 파일 사용)
echo 전체 데이터로 V5 실행...
.venv\Scripts\python.exe versions\v5\drug_generator_v5.py ^
  --excel data\요양심사약제_후처리_v2_cleaned.xlsx ^
  --out versions\v5\drug_questions_v5_final.xlsx ^
  --concurrency 6

echo.
echo V5 전체 실행 완료!
pause