@echo off
echo ======================================
echo Pharma-Augment V4 자동 실행 스크립트
echo ======================================

cd /d "C:\Jimin\Pharma-Augment\versions\v4"

echo.
echo [1/4] 환경 확인 중...
if not exist "../../data/요양심사약제_후처리_v2.xlsx" (
    echo ERROR: 약제 파일을 찾을 수 없습니다.
    pause
    exit /b 1
)

if not exist "../../data/고시.xlsx" (
    echo ERROR: 고시 파일을 찾을 수 없습니다.
    pause
    exit /b 1
)

echo.
echo [2/4] 약제 질문 생성 시작...
python drug_generator_v4.py --excel "../../data/요양심사약제_후처리_v2.xlsx" --out "drug_questions_v4.xlsx"

if errorlevel 1 (
    echo ERROR: 약제 질문 생성 실패
    pause
    exit /b 1
)

echo.
echo [3/4] 고시 질문 생성 시작...
python notice_generator_v4.py --excel "../../data/고시.xlsx" --out "notice_questions_v4.xlsx"

if errorlevel 1 (
    echo ERROR: 고시 질문 생성 실패
    pause
    exit /b 1
)

echo.
echo [4/4] 완료!
echo ======================================
echo 결과 파일:
echo - drug_questions_v4.xlsx
echo - notice_questions_v4.xlsx
echo - audit_log_drug_v4.csv
echo - audit_log_notice_v4.csv
echo ======================================
pause