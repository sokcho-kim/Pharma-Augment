@echo off
chcp 65001 >nul
cd /d C:\Sokcho\Pharma-Augment
echo V5 간단 버전 테스트...
.venv\Scripts\python.exe versions\v5\test_simple_v5.py
pause