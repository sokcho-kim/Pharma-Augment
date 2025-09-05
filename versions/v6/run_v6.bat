@echo off
echo V6 질문 생성기 실행 중...
cd /d "C:\Jimin\Pharma-Augment\versions\v6"
python drug_generator_v6.py
if %ERRORLEVEL% NEQ 0 (
    echo 실행 중 오류가 발생했습니다.
    pause
) else (
    echo V6 질문 생성 완료!
    pause
)