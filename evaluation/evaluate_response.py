import argparse
import concurrent.futures
import os

from tqdm import tqdm

from base.data import QuestionItem
from evaluation.models import ChatManager
from evaluation.prompts import eval_answer_prompt

chat_manager = ChatManager()


def evaluate_answer_correctness(response_answer, full_answer, brief_answer, model_name):
    """
    使用大模型判断答案是否正确
    Args:
        response_answer (_type_): 待测评的答案
        full_answer (_type_): 原始答案
        brief_answer (_type_): 简要答案
        model_name (str, optional): 大模型名称
    """
    prompt = eval_answer_prompt.format(
        response_answer=response_answer, full_answer=full_answer, brief_answer=brief_answer
    )
    res = chat_manager.get_response(prompt, model_name)
    return "True" in res


def fill_response_is_correct(question: QuestionItem, model_name: str):
    for evaluation in question.evaluation:
        if evaluation.correct is None:
            evaluation.correct = evaluate_answer_correctness(
                response_answer=evaluation.response,
                full_answer=question.full_answer,
                brief_answer=question.brief_answer,
                model_name=model_name,
            )
    return question


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5-nano-2025-08-07", help="LLM name to use")
    parser.add_argument("--num_workers", type=int, default=144, help="Number of threads to use")
    parser.add_argument("--use_multithreading", type=bool, default=True, help="Whether to use multithreading")
    args = parser.parse_args()

    model_name = args.model_name
    file_name = f"{model_name.replace('-', '_').replace('/', '_')}.json"
    file_path = os.path.join("./data/response/", file_name)
    question_data = QuestionItem.read_json_file(file_path)

    def process_question(question):
        fill_response_is_correct(question, "gpt-5-nano-2025-08-07")

    if args.use_multithreading:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(process_question, question_data), total=len(question_data)))
    else:
        for question in tqdm(question_data):
            process_question(question)

    QuestionItem.save_json_file(question_data, file_path)
