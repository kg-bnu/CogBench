import json
import os
from typing import List

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


class CAKGTripleRetriever:
    """
    基于三元组嵌入的检索器
    """

    def __init__(
        self,
        embeddings_dir: str = "data/graph_embeddings/",
        embedding_api=None,
    ):
        self.embeddings_dir = embeddings_dir
        self.embedding_api = embedding_api
        self.grade_range_data = {}

        self.load_all_embeddings()

    def load_all_embeddings(self):
        """
        加载所有年级范围的嵌入数据
        """
        # 加载汇总信息
        summary_path = os.path.join(self.embeddings_dir, "embedding_summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        grade_ranges = summary["grade_ranges"]

        # 加载每个年级范围的数据
        for grade_range in grade_ranges:
            logger.info(f"加载 {grade_range} 的嵌入数据...")
            # 加载JSON数据
            json_path = os.path.join(self.embeddings_dir, f"{grade_range}_triple_embeddings.json")
            with open(json_path, "r", encoding="utf-8") as f:
                triple_data = json.load(f)
            # 加载向量数据
            vectors_path = os.path.join(self.embeddings_dir, f"{grade_range}_embedding_vectors.npy")
            vectors = np.load(vectors_path)

            self.grade_range_data[grade_range] = {
                "triple_data": triple_data,
                "vectors": vectors,
                "triple_count": len(triple_data),
            }

        logger.success(f"成功加载 {len(self.grade_range_data)} 个年级范围的嵌入数据")

    def search_triples(self, query: str, grade_range: str = "Grade1_to_Grade10", top_k: int = 3):
        """
        根据查询检索最相似的k个三元组

        Args:
            query: 查询文本
            grade_range: 指定年级范围，如果为None则在所有范围中搜索
            top_k: 返回的三元组数量

        Returns:
            最相似的k个三元组及其相似度
        """
        if not self.embedding_api:
            raise RuntimeError("LLM embedding API configuration has been removed; no embedding client is available.")

        # 生成查询向量
        query_vectors = self.embedding_api.get_embeddings([query])
        query_vector = query_vectors[0]

        all_candidates = []

        # 确定搜索范围
        search_ranges = grade_range if grade_range else "Grade1_to_Grade10"

        data = self.grade_range_data[search_ranges]
        vectors = data["vectors"]
        triple_data = data["triple_data"]

        # 计算相似度
        similarities = cosine_similarity([query_vector], vectors)[0]

        # 收集候选结果
        for i, (similarity, triple_info) in enumerate(zip(similarities, triple_data)):
            all_candidates.append(
                {
                    "triple": triple_info["triple"],
                    "formatted_text": triple_info["formatted_text"],
                    "similarity": float(similarity),
                    "grade_range": search_ranges,
                    "triple_id": triple_info["id"],
                }
            )

        # 按相似度排序并返回top-k
        all_candidates.sort(key=lambda x: x["similarity"], reverse=True)

        return all_candidates[:top_k]

    def get_available_grade_ranges(self) -> List[str]:
        """
        获取可用的年级范围列表
        """
        return list(self.grade_range_data.keys())
