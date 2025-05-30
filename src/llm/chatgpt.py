from openai import AsyncOpenAI
from src.config.settings import settings
from src.llm.base import BaseLLMClient, BaseEmbedderClient
import logging

logger = logging.getLogger(__name__)

class ChatGPTClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.llm_api_key)

    async def chat(self, messages: list[dict]) -> str:
        response = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
        )
        return response.choices[0].message.content

    async def close(self):
        pass
        # await self.client.aiterator.aclose()  # In case streaming was used

class ChatGPTEmbedder(BaseEmbedderClient):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.embedding_api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=settings.embedding_model,
            input=texts
        )
        return [r.embedding for r in response.data]

    async def close(self):
        pass
        # await self.client.aiterator.aclose()