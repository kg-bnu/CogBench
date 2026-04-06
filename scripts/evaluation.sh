#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL_NAME="$1"

python -m evaluation.response --model_name "$MODEL_NAME" && \
python -m evaluation.evaluate_response --model_name "$MODEL_NAME" && \
python -m evaluation.find_knowledge_used --model_name "$MODEL_NAME"  --use_multithreading False && \
python -m evaluation.calculate_metrics --model_name "$MODEL_NAME"