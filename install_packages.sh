#!/bin/bash
echo "Installing required packages for Pharma-Augment..."
pip install -r requirements.txt
echo
echo "Installation completed!"
echo
echo "Usage example:"
echo "python generate_questions.py \\"
echo "  --excel \"data/요양심사약제_후처리_v2.xlsx\" \\"
echo "  --sheet \"Sheet1\" \\"
echo "  --out \"questions.jsonl\" \\"
echo "  --provider \"openai\" \\"
echo "  --model \"gpt-4o-mini\" \\"
echo "  --concurrency 6 \\"
echo "  --max_aug 15 \\"
echo "  --seed 20250902"