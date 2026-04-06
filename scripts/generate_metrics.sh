#!/bin/bash
models=("gemini-2.5-flash-nothinking" "gemini-2.5-flash" "gpt-4o-mini-2024-07-18" "gpt-5-mini-2025-08-07" "gpt-5-nano-2025-08-07" "gpt-5-2025-08-07" "gpt-oss-20b-no-limit" "qwensft:latest")
# models=( "gpt-oss-20b-no-limit" )
for model in "${models[@]}"; do
    python -m evaluation.find_knowledge_used --model_name "$model"
done


for model in "${models[@]}"; do
    python -m evaluation.calculate_metrics --model_name "$model"
done
