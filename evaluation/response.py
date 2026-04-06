import argparse
import os

from loguru import logger
from tqdm import tqdm

from base.data import QuestionItem, ResponseSource, ResponseUnit
from evaluation.models import ChatManager
from evaluation.prompts import (
    prompt_generate_answerr_only_title,
    prompt_generate_answerr_title_grade,
    prompt_generate_answerr_title_knowledge_text,
)
from evaluation.utils import extract_grades_from_solutions

logger.remove(handler_id=None)
logger.add(sink="test.log", level="INFO", rotation="00:00", encoding="utf-8", enqueue=True)


def eval_question_is_leak(question):
    # 如果question.evaluation是空，则返回True
    # 如果evaluation数组中有对象的response是空字符串，则返回True
    if not question.evaluation or len(question.evaluation) == 0:
        logger.info(
            "Question {} has no evaluation.".format(question.url if hasattr(question, "url") else str(question))
        )
        return True
    for item in question.evaluation:
        if not item.response:
            logger.info(
                "Question {} has an evaluation item with empty response.".format(
                    question.url if hasattr(question, "url") else str(question)
                )
            )
            return True
    return False


def get_all_response(question: QuestionItem, model_name: str):
    chat_manager = ChatManager()
    # 从solution中提取不同的年级
    solution_grades = extract_grades_from_solutions(question.solution)
    # idx+1代表该solution对应的knowledge的下标,由于数据标注不一致的问题，在这里选择对应的knowledge
    different_grades = []
    seen = set()  # 需要对年级进行去重
    for idx, grade in enumerate(solution_grades):
        if grade and grade not in seen:  # grade不为空，且没有出现过
            knowledge_idx = idx + 1 if len(question.knowledge) > len(question.solution) else idx
            different_grades.append((grade, knowledge_idx))

    response_list = []
    title_options = question.title + str(question.options)
    # 生成title的response
    prompt1 = prompt_generate_answerr_only_title.format(title=title_options)
    logger.info(f"Generating response for TITLE_ONLY: {question.title}")
    response1 = chat_manager.get_response(prompt1, model_name)
    logger.info(f"Response (TITLE_ONLY): {response1}")
    response_list.append(ResponseUnit(response=response1, source=ResponseSource.TITLE_ONLY, grade=question.grade))

    # 原始grade的response
    prompt2 = prompt_generate_answerr_title_grade.format(title=title_options, target_grade=question.grade)
    logger.info(f"Generating response for TITLE_GRADE: {question.title} | Grade: {question.grade}")
    response2 = chat_manager.get_response(prompt2, model_name)
    logger.info(f"Response (TITLE_GRADE): {response2}")
    response_list.append(ResponseUnit(response=response2, source=ResponseSource.TITLE_GRADE, grade=question.grade))

    # 3. 为每个不同的年级生成响应，依次使用knowledge[1], knowledge[2], ...
    for grade, knowledge_idx in different_grades:
        # 第三个答案：根据题目和知识点生成
        knowledge_text = question.knowledge[knowledge_idx]
        prompt3 = prompt_generate_answerr_title_knowledge_text.format(
            title=title_options, knowledge_text=knowledge_text
        )
        logger.info(
            f"Generating response for TITLE_KNOWLEDGE: {question.title} | Grade: {grade} | KnowledgeIdx: {knowledge_idx}"
        )
        response3 = chat_manager.get_response(prompt3, model_name)
        logger.info(f"Response (TITLE_KNOWLEDGE, grade {grade}, idx {knowledge_idx}): {response3}")
        response_list.append(
            ResponseUnit(
                response=response3,
                source=ResponseSource.TITLE_KNOWLEDGE,
                grade=grade,
                knowledge_index=knowledge_idx,
                knowledge_used=knowledge_text,
            )
        )
    question.evaluation = response_list
    return question


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5-mini", help="LLM name to use for response generation")
    parser.add_argument("--input_file", type=str, default="data/CuratedQA.json", help="Path to the input JSON file")
    parser.add_argument("--num_workers", type=int, default=144, help="Number of threads to use for response generation")
    # 增加参数指定是否要使用多线程加速
    parser.add_argument("--use_multithreading", type=bool, default=True, help="Whether to use multithreading ")
    args = parser.parse_args()
    num_workers = args.num_workers
    model_name = args.model_name
    input_file = args.input_file

    logger.info(f"Loading questions from {input_file}")
    question_data = QuestionItem.read_json_file(input_file)
    logger.info(f"Loaded {len(question_data)} questions. Filtering questions that need evaluation responses.")

    # import random
    # random.seed(42)
    # question_data = random.sample(question_data, 100)

    filtered_questions = [q for q in question_data if eval_question_is_leak(q)]
    logger.info(f"Found {len(filtered_questions)} questions needing response generation.")

    import concurrent.futures

    def process_question(q):
        logger.info(f"Generating responses for question: {getattr(q, 'url', str(q.title))}")
        get_all_response(q, model_name)

    if args.use_multithreading:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(process_question, filtered_questions), total=len(filtered_questions)))
    else:
        for q in tqdm(filtered_questions):
            process_question(q)

    file_name = f"{model_name.replace('-', '_').replace('/', '_')}.json"
    save_path = os.path.join("./data/response/", file_name)
    logger.info(f"Saving output to {save_path}")
    QuestionItem.save_json_file(question_data, save_path)
