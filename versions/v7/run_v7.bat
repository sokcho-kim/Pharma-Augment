@echo off
echo V7 ReasonIR 기반 질문 생성기 실행
echo =====================================

cd /d "C:\Jimin\Pharma-Augment\versions\v7"

echo 가상환경 활성화...
call "C:\Jimin\Pharma-Augment\.venv\Scripts\activate.bat"

echo V7 질문 생성 시작...
python drug_generator_v7.py

echo.
echo 실행 완료! 결과 파일 확인:
echo - drug_questions_v7.xlsx (기본 출력)  
echo - drug_questions_v7_anchorpack.jsonl (앵커팩)
echo - drug_generation_v7.log (로그)

pause