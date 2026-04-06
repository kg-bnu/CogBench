import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


class CAKGTripleEmbedding:
    """
    CAKG三元组级别的嵌入生成器
    """

    def __init__(
        self,
        cakg_path: str = "data/CAKG/cakg.json",
        embedding_api=None,
        embedding_model: str = None,
    ):
        self.cakg_path = cakg_path
        self.cakg_data = None
        self.grade_triples = {}
        self.cumulative_triple_sets = {}
        self.embedding_api = embedding_api
        self.embedding_model = embedding_model

    def _require_embedding_api(self):
        if not self.embedding_api:
            raise RuntimeError("LLM embedding API configuration has been removed; no embedding client is available.")
        return self.embedding_api

    def load_cakg_data(self):
        """加载CAKG数据"""
        print("正在加载CAKG数据...")
        try:
            with open(self.cakg_path, "r", encoding="utf-8") as f:
                self.cakg_data = json.load(f)
            print(f"成功加载 {len(self.cakg_data)} 个三元组")

            # 按年级组织三元组
            self.organize_triples_by_grade()
            # 生成累积三元组集合
            self.generate_cumulative_triple_sets()
        except Exception as e:
            print(f"加载CAKG数据失败: {e}")
            raise

    def organize_triples_by_grade(self):
        """按年级组织三元组"""
        grade_triples = defaultdict(list)

        for triple in self.cakg_data:
            relation = triple.get("relation", "")
            if relation.startswith("Grade"):
                grade_triples[relation].append(triple)

        self.grade_triples = dict(grade_triples)
        print(f"按年级组织完成，共 {len(self.grade_triples)} 个年级")

        for grade, triples in self.grade_triples.items():
            print(f"  {grade}: {len(triples)} 个三元组")

    def generate_cumulative_triple_sets(self):
        """生成累积三元组集合"""
        cumulative_triples = []
        grade_order = sorted(self.grade_triples.keys(), key=lambda x: int(x.replace("Grade", "")))

        for grade in grade_order:
            cumulative_triples.extend(self.grade_triples[grade])
            grade_range = f"Grade1_to_{grade}"
            # 创建副本避免引用问题
            self.cumulative_triple_sets[grade_range] = [t.copy() for t in cumulative_triples]

        print("累积三元组集合生成完成:")
        for grade_range, triples in self.cumulative_triple_sets.items():
            print(f"  {grade_range}: {len(triples)} 个三元组")

    def format_triple_for_embedding(self, triple: Dict[str, str]) -> str:
        """
        将单个三元组格式化为适合嵌入的文本
        """
        head = triple["head"]
        tail = triple["tail"]
        relation = triple["relation"]

        # 简化tail内容，避免过长的描述
        # if len(tail) > 150:
        #     tail = tail[:150] + "..."

        # 格式化为简洁的自然语言描述
        formatted_text = f"数学知识：{head}。{tail}（{relation}）"

        return formatted_text

    def test_api_connection(self):
        """测试API连接"""
        print("测试API连接...")
        embedding_api = self._require_embedding_api()
        try:
            test_text = ["这是一个测试文本"]
            result = embedding_api.get_embeddings(test_text)
            if result and len(result) > 0:
                print(f"API连接测试成功！向量维度: {len(result[0])}")
                return True
            else:
                print("API连接测试失败：返回结果为空")
                return False
        except Exception as e:
            print(f"API连接测试失败: {e}")
            return False

    def process_grade_range_embeddings(
        self, grade_range: str, output_dir: str = "data/Curated QA/graph_embeddings/", max_triples: int = None
    ) -> Dict[str, Any]:
        """
        为指定年级范围的所有三元组生成嵌入向量
        """
        embedding_api = self._require_embedding_api()
        if grade_range not in self.cumulative_triple_sets:
            raise ValueError(f"未找到年级范围: {grade_range}")

        triples = self.cumulative_triple_sets[grade_range]

        # 限制处理数量（用于测试）
        if max_triples:
            triples = triples[:max_triples]
            print(f"限制处理数量为: {max_triples}")

        print(f"\n处理年级范围: {grade_range}，共 {len(triples)} 个三元组")

        # 格式化所有三元组为文本
        print("格式化三元组文本...")
        triple_texts = []
        for i, triple in enumerate(triples):
            formatted_text = self.format_triple_for_embedding(triple)
            triple_texts.append(formatted_text)

            # 显示进度
            if (i + 1) % 100 == 0:
                print(f"  已格式化 {i + 1}/{len(triples)} 个三元组")

        # 生成嵌入向量
        print("生成嵌入向量...")
        try:
            embedding_vectors = embedding_api.get_embeddings(triple_texts)
        except Exception as e:
            print(f"生成嵌入向量失败: {e}")
            raise

        # 组装结果
        triple_embeddings = []
        for i, (triple, text, vector) in enumerate(zip(triples, triple_texts, embedding_vectors)):
            triple_embeddings.append(
                {
                    "id": i,
                    "triple": triple,
                    "formatted_text": text,
                    "embedding_vector": vector,
                    "grade_range": grade_range,
                }
            )

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        self._save_grade_range_embeddings(triple_embeddings, grade_range, output_dir)

        print(f"完成 {grade_range}，生成 {len(triple_embeddings)} 个三元组嵌入")

        return {"grade_range": grade_range, "triple_count": len(triple_embeddings), "embeddings": triple_embeddings}

    def process_all_cumulative_embeddings(
        self, output_dir: str = "data/Curated QA/graph_embeddings/", test_mode: bool = False
    ) -> Dict[str, Any]:
        """
        处理所有累积年级范围的三元组嵌入
        """
        self._require_embedding_api()
        if not self.cakg_data:
            self.load_cakg_data()

        # 测试API连接
        if not self.test_api_connection():
            raise Exception("API连接测试失败，请检查配置")

        all_results = {}

        # 测试模式只处理部分数据
        max_triples = 10 if test_mode else None

        for grade_range in self.cumulative_triple_sets.keys():
            try:
                result = self.process_grade_range_embeddings(grade_range, output_dir, max_triples)
                all_results[grade_range] = result

                if test_mode:
                    print(f"测试模式：只处理了 {grade_range}")
                    break  # 测试模式只处理第一个年级范围

            except Exception as e:
                print(f"处理 {grade_range} 时出错: {e}")
                continue

        # 保存汇总信息
        summary = {
            "total_grade_ranges": len(all_results),
            "grade_ranges": list(all_results.keys()),
            "generation_timestamp": time.time(),
            "test_mode": test_mode,
        }

        summary_path = os.path.join(output_dir, "embedding_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n所有嵌入处理完成，结果保存在: {output_dir}")
        return all_results

    def _save_grade_range_embeddings(self, triple_embeddings: List[Dict[str, Any]], grade_range: str, output_dir: str):
        """
        保存指定年级范围的三元组嵌入
        """
        # 分离向量和其他数据
        embeddings_without_vectors = []
        vectors = []

        for item in triple_embeddings:
            embeddings_without_vectors.append({k: v for k, v in item.items() if k != "embedding_vector"})
            vectors.append(item["embedding_vector"])

        # 保存JSON数据（不含向量）
        json_path = os.path.join(output_dir, f"{grade_range}_triple_embeddings.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_without_vectors, f, ensure_ascii=False, indent=2)

        # 保存向量数据
        vectors_path = os.path.join(output_dir, f"{grade_range}_embedding_vectors.npy")
        np.save(vectors_path, np.array(vectors))

        print(f"  保存到: {json_path} 和 {vectors_path}")
