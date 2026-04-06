import json
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, TypeAdapter, field_validator

from evaluation.utils import parse_solution


class JsonReaderBase(BaseModel):
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]):
        return [cls.model_validate(item) for item in dict_list]

    @classmethod
    def read_json_file(cls, filepath: str) -> List:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return [cls.model_validate(item) for item in data]

    @classmethod
    def save_json_file(cls, arr: List["JsonReaderBase"], filepath: str):
        adapter = TypeAdapter(List[type(arr[0])] if arr else List[cls])
        json_str = adapter.dump_json(arr, indent=2).decode("utf-8")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)


# 用于读取data/cakg.json
class KnowledgeTriple(JsonReaderBase):
    head: Optional[str] = None
    relation: Optional[str] = None
    tail: Optional[str] = None


class ResponseSource(str, Enum):
    TITLE_ONLY = "title_only"
    TITLE_GRADE = "title_grade"
    TITLE_KNOWLEDGE = "title_knowledge"


class KGRetrieverResult(JsonReaderBase):
    triple: KnowledgeTriple
    formatted_text: str
    similarity: float
    grade_range: str
    triple_id: int


class ResponseUnit(BaseModel):
    response: str
    source: ResponseSource
    grade: str
    knowledge_index: Optional[int] = None

    correct: Optional[bool] = None

    knowledge_used: Optional[List[KnowledgeTriple]] = None

    kg_retriever_result: Optional[List[KGRetrieverResult]] = None


# 用于读取data/CuratedQA.json
class QuestionItem(JsonReaderBase):
    url: str
    title: str
    brief_answer: str
    full_answer: str
    grade: str
    options: List[str] = Field(default_factory=list)
    solution: List[str]
    knowledge: List[List[KnowledgeTriple]]
    evaluation: Optional[List[ResponseUnit]] = None
    standard_knowledge: Optional[List[KGRetrieverResult]] = None

    @field_validator("options", mode="before")
    @classmethod
    def transform_options(cls, v) -> List[str]:
        """
        options字段可能为str List None
        将其统一处理为list
        目前不清楚options在未来是否会变成多个，保留处理逻辑
        """
        if isinstance(v, str):
            return [v]
        if v is None:
            return []

        if isinstance(v, list):
            return v
        return v


class SolutionGradeHelper(BaseModel):
    solution_text: Optional[str] = None
    solution_grade: Optional[int] = None
    solution_content: Optional[str] = None

    @classmethod
    def from_text(cls, solution_text):
        grade_num, content = parse_solution(solution_text)
        return cls(solution_text=solution_text, solution_grade=grade_num, solution_content=content)


class DPOPairItem(BaseModel):
    q: str
    s_plus: str
    s_minus: str


class DPOUnit(BaseModel):
    from_: str = Field(alias="from")
    value: str


class DPOItem(BaseModel):
    conversations: List[DPOUnit]
    chosen: DPOUnit
    rejected: DPOUnit
