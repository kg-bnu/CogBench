import argparse
import concurrent
import os

from loguru import logger
from tqdm import tqdm

from base.data import KGRetrieverResult, QuestionItem
from evaluation.v1.cakg_retriever import CAKGTripleRetriever

logger.remove(handler_id=None)
logger.add(sink="test.log", level="INFO", rotation="00:00", encoding="utf-8", enqueue=True)

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
    retriever = CAKGTripleRetriever()

    def process_question(question):

        if question.standard_knowledge is None and question.full_answer:
            res = retriever.search_triples(question.full_answer, top_k=3)
            question.standard_knowledge = KGRetrieverResult.from_dict_list(res)

        for evaluation in question.evaluation:
            if evaluation.response and evaluation.kg_retriever_result is None:
                res = retriever.search_triples(evaluation.response, top_k=3)
                res_list = KGRetrieverResult.from_dict_list(res)
                evaluation.kg_retriever_result = res_list

    if args.use_multithreading:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(process_question, question_data), total=len(question_data)))
    else:
        for question in tqdm(question_data, total=len(question_data)):
            process_question(question)

    QuestionItem.save_json_file(question_data, file_path)
