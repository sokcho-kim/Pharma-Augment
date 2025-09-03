@echo off
echo Installing required packages for Pharma-Augment...
pip install -r requirements.txt
echo.
echo Installation completed!
echo.
echo Usage example:
echo python generate_questions.py --excel "data/요양심사약제_후처리_v2.xlsx" --sheet "Sheet1" --out "questions.jsonl" --provider "openai" --model "gpt-4o-mini" --concurrency 6 --max_aug 15 --seed 20250902
pause