from dataclasses import dataclass


@dataclass
class Settings:
    data_path: str = "./data/"
    graph_embeddings_dir: str = "./data/graph_embeddings"
