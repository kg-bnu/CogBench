import itertools
from typing import Any, List

import numpy as np

from base.data import KGRetrieverResult, KnowledgeTriple, QuestionItem, ResponseSource
from evaluation.utils import get_grade_num


def calculate_knowledge_grade_above(kg_retriever_result: KGRetrieverResult, grade_str: str):
    """
    对一个response的kg搜索结果，判断其所用知识的年级是否小于等于其认知等级
    """
    grade = get_grade_num(grade_str)
    all_knowledge_grade_less_than_grade = True
    for kg_result in kg_retriever_result:
        knowledge_grade = get_grade_num(kg_result.triple.relation)
        if knowledge_grade is not None and knowledge_grade > grade:
            all_knowledge_grade_less_than_grade = False
            break
    return all_knowledge_grade_less_than_grade


def calculate_knowledge_intersection(list_1: List[KnowledgeTriple], list_2: List[KnowledgeTriple]):
    """
    计算两个三元组列表之间的交集
    """
    heads_1 = set([k.head for k in list_1])
    heads_2 = set([k.head for k in list_2])
    intersection = heads_1 & heads_2
    return len(intersection) / len(heads_1)


def calculate_knowledge_pad(list_1: List[KnowledgeTriple], list_2: List[KnowledgeTriple]):
    """
    计算pad
    """
    heads_1 = set([k.head for k in list_1])
    heads_2 = set([k.head for k in list_2])
    intersection = heads_1 & heads_2
    return len(intersection) / (len(heads_1) + len(heads_2) - len(intersection))


class MetricsManager:
    @classmethod
    def get_averge(cls, list_in: List[Any]):
        return np.array(list_in).mean().item()

    @classmethod
    def get_all_metrics(cls, question_data: QuestionItem):
        # 计算ACC
        title_only_acc = []
        title_grade_acc = []
        title_knowledge_acc = []
        # 计算CAR
        title_only_car = []
        title_grade_car = []
        title_knowledge_car = []
        # 计算KAS
        # kas_list = []
        title_only_kas = []
        title_grade_kas = []
        title_knowledge_kas = []
        # 计算PAD
        pad_list = []

        for question in question_data:
            for evaluation in question.evaluation:
                # used_knowledge_list = evaluation.knowledge_used
                # 判断是否使用小于当前年级的知识（CAR）
                if not evaluation.response:
                    continue
                if evaluation.kg_retriever_result is None:
                    continue
                if question.standard_knowledge is None:
                    continue

                kas = calculate_knowledge_intersection(
                    [r.triple for r in question.standard_knowledge], [r.triple for r in evaluation.kg_retriever_result]
                )
                car = calculate_knowledge_grade_above(evaluation.kg_retriever_result, evaluation.grade)
                if evaluation.source is ResponseSource.TITLE_ONLY:
                    title_only_acc.append(evaluation.correct)
                    title_only_car.append(car)
                    title_only_kas.append(kas)
                elif evaluation.source is ResponseSource.TITLE_GRADE:
                    title_grade_acc.append(evaluation.correct)
                    title_grade_car.append(car)
                    title_grade_kas.append(kas)
                elif evaluation.source is ResponseSource.TITLE_KNOWLEDGE:
                    title_knowledge_acc.append(evaluation.correct)
                    title_knowledge_car.append(car)
                    title_knowledge_kas.append(kas)
            # 大于3个代表有不同年级
            if len(question.evaluation) > 3:

                # 计算PAD
                # 对于所有evaluation.source为ResponseSource.TITLE_KNOWLEDGE的对象
                # 两两配对通过calculate_knowledge_intersection计算分值，并取平均
                title_knowledge_evals = [
                    ev for ev in question.evaluation if ev.source == ResponseSource.TITLE_KNOWLEDGE
                ]
                pad_scores = []
                # 两两配对
                for eval1, eval2 in itertools.combinations(title_knowledge_evals, 2):
                    if not eval1.response or not eval2.response:
                        continue
                    pad_score = 1 - calculate_knowledge_pad(
                        [r.triple for r in eval1.kg_retriever_result], [r.triple for r in eval2.kg_retriever_result]
                    )
                    pad_scores.append(pad_score)
                if len(pad_scores) > 0:
                    avg_pad_score = cls.get_averge(pad_scores)
                    pad_list.append(avg_pad_score)
        title_only_acc_avg = cls.get_averge(title_only_acc)
        title_grade_acc_avg = cls.get_averge(title_grade_acc)
        title_knowledge_acc_avg = cls.get_averge(title_knowledge_acc)

        title_only_car_avg = cls.get_averge(title_only_car)
        title_grade_car_avg = cls.get_averge(title_grade_car)
        title_knowledge_car_avg = cls.get_averge(title_knowledge_car)

        title_only_kas_avg = cls.get_averge(title_only_kas)
        title_grade_kas_avg = cls.get_averge(title_grade_kas)
        title_knowledge_kas_avg = cls.get_averge(title_knowledge_kas)

        acc_avg = cls.get_averge(title_only_acc + title_grade_acc + title_knowledge_acc)
        car_avg = cls.get_averge(title_only_car + title_grade_car + title_knowledge_car)
        kas_avg = cls.get_averge(title_only_kas + title_grade_kas + title_knowledge_kas)
        pad_avg = cls.get_averge(pad_list)

        as_avg = car_avg + kas_avg + pad_avg

        return {
            "title_only_acc": title_only_acc_avg,
            "title_grade_acc": title_grade_acc_avg,
            "title_knowledge_acc": title_knowledge_acc_avg,
            "title_only_car": title_only_car_avg,
            "title_grade_car": title_grade_car_avg,
            "title_knowledge_car": title_knowledge_car_avg,
            "title_only_kas": title_only_kas_avg,
            "title_grade_kas": title_grade_kas_avg,
            "title_knowledge_kas": title_knowledge_kas_avg,
            "acc": acc_avg,
            "car": car_avg,
            "kas": kas_avg,
            "pad": pad_avg,
            "as": as_avg,
        }
