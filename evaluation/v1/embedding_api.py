from typing import List


class OpenAIEmbeddingAPI:
    """
    OpenAI嵌入API封装类（适配openai库，可自定义API端点）
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        max_retries: int = 10,
        api_url: str = None,
    ):
        self.api_key = None
        self.model = model
        self.max_retries = max_retries
        self.api_url = None
        self.client = None

    def get_embeddings(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        raise RuntimeError("LLM embedding API configuration has been removed; no client is available for embeddings.")
