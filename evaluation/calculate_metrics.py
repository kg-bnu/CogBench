import argparse
import json
import os

from base.data import QuestionItem
from metrics.metrics import MetricsManager


def calculate_metrics(question_data: QuestionItem):
    return MetricsManager.get_all_metrics(question_data)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="gpt-5-nano-2025-08-07", help="LLM name to use")
    args = args.parse_args()
    model_name = args.model_name
    file_name = f"{model_name.replace('-', '_').replace('/', '_')}.json"
    file_path = os.path.join("./data/response/", file_name)
    question_data = QuestionItem.read_json_file(file_path)
    metrics = calculate_metrics(question_data)
    print(metrics)
    # 将metrics保存到json
    metrics_path = os.path.join("./data/metrics/", file_name)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
