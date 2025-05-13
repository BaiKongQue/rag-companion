# src/pdf_rag_engine/llm/chatgpt.py

from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from src.config.settings import settings
from src.llm.base import BaseLLMClient

class ChatGPTClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.api_key)

    async def chat(self, messages: list[dict]) -> str:
        response = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
        )
        return response.choices[0].message.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=settings.embedding_model,
            input=texts
        )
        return [r.embedding for r in response.data]

    async def close(self):
        await self.client.aiterator.aclose()  # In case streaming was used
        # If not using streaming, this is optional

@asynccontextmanager
async def get_chatgpt_client():
    client = ChatGPTClient()
    try:
        yield client
    finally:
        await client.close()
