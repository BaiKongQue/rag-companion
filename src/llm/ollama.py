# from ollama import AsyncClient
import logging
from src.config.settings import settings
from src.llm.base import BaseLLMClient, BaseEmbedderClient

logger = logging.getLogger(__name__)

class OllamaLLMClient(BaseLLMClient):
    def __init__(self, host: str, model: str):
        self.client = AsyncClient(host=host)
        self.model = model

    async def chat(self, messages: list[dict]):
        response = await self.client.chat(model=self.model, messages=messages)
        return response['message']['content']

    async def close(self):
        await self.client.aclose()

class OllamaEmbedder(BaseEmbedderClient):
    def __init__(self, host: str, model: str):
        self.client = AsyncClient(host=host)
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings(model=self.model, prompt=texts)
        return response['embeddings']

    async def close(self):
        await self.client.aclose()