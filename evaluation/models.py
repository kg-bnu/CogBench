import asyncio
from typing import Dict, Iterable, List, Optional


class ChatManager:
    def __init__(self, model2client: Optional[Dict[str, object]] = None, default_client: object = None):
        # LLM API客户端初始化已移除；如需调用需外部注入 client。
        self.default_client = default_client
        self.model2client = model2client or {}

    def _get_client(self, model: str):
        client = self.model2client.get(model, self.default_client)
        if client is None:
            raise RuntimeError("LLM API configuration has been removed; no client is available.")
        return client

    def get_embedding(self, text, model="Qwen/Qwen3-Embedding-8B") -> List[float]:
        client = self._get_client(model)
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    def get_embeddings(self, texts: List[str], model: str = "Qwen/Qwen3-Embedding-8B") -> List[List[float]]:
        client = self._get_client(model)
        response = client.embeddings.create(model=model, input=texts)
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def _get_chat_args(self, prompt, model) -> dict:
        chat_args = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_completion_tokens": 1024,
            "stream": False,
            # "max_tokens": 1024,
        }
        model2param = {
            "gpt-5-mini-2025-08-07": {
                "temperature": 1,
                "reasoning_effort": "minimal",
                "verbosity": "low",
            },
            "gpt-5-nano-2025-08-07": {
                "temperature": 1,
                "reasoning_effort": "minimal",
                "verbosity": "low",
            },
            "gpt-5-2025-08-07": {
                "temperature": 1,
                "reasoning_effort": "minimal",
                "verbosity": "low",
            },
            "gpt-4o-mini-2024-07-18": {},
            "qwensft:latest": {
                "max_tokens": 1024,
            },
            "qwen3:8b": {
                "max_tokens": 1024,
            },
            "gpt-oss-20b": {
                "max_tokens": 1024,
            },
            "gemini-2.5-flash": {
                "max_tokens": 1024,
            },
            "gemini-2.5-flash-nothinking": {
                "max_tokens": 1024,
                "reasoning_effort": "low",
            },
            "gemini-2.5-pro-thinking-0": {
                "max_tokens": 1024,
                "reasoning_effort": "low",
            },
            "qwen3-30b-a3b": {
                "max_tokens": 1024,
            },
            "qwen3-235b-a22b": {
                "max_tokens": 1024,
            },
            "llama-3.1-70b": {
                "max_tokens": 1024,
            },
            "llama-3.1-405b": {
                "max_tokens": 1024,
            },
            "claude-sonnet-4-20250514": {
                "max_tokens": 1024,
            },
            "claude-opus-4-20250514": {
                "max_tokens": 1024,
            },
        }
        chat_args.update(model2param.get(model, {}))
        return chat_args

    def get_response(self, prompt, model) -> str:
        client = self._get_client(model)
        chat_args = self._get_chat_args(prompt, model)
        completion = client.chat.completions.create(**chat_args)
        answer = completion.choices[0].message.content
        # 对于token不足的情况特殊处理
        if len(answer) == 0:
            chat_args.update({"max_completion_tokens": 4096})
            completion = client.chat.completions.create(**chat_args)
            answer = completion.choices[0].message.content
        return answer

    async def aget_embeddings(
        self, texts: List[str], model: str = "Qwen/Qwen3-Embedding-8B", batch_size: int = 64
    ) -> List[List[float]]:
        client = self._get_client(model)

        async def _emb_batch(batch: List[str]) -> List[List[float]]:
            resp = await client.embeddings.create(model=model, input=batch)
            return [item.embedding for item in resp.data]

        tasks = []
        for i in range(0, len(texts), batch_size):
            tasks.append(_emb_batch(texts[i : i + batch_size]))
        if not tasks:
            return []
        batches = await asyncio.gather(*tasks)
        return [vec for batch in batches for vec in batch]

    async def aget_response(self, prompt: str, model: str, temperature: float = 1.0, top_k: float = 1) -> str:
        client = self._get_client(model)
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_k,
            max_tokens=1024,
        )
        return completion.choices[0].message.content

    async def aget_responses(
        self,
        prompts: Iterable[str],
        model: str,
        temperature: float = 1.0,
        top_k: float = 1,
    ) -> List[str]:
        tasks = [self.aget_response(prompt, model=model, temperature=temperature, top_k=top_k) for prompt in prompts]
        if not tasks:
            return []
        return await asyncio.gather(*tasks)
