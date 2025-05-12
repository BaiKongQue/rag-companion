from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from src.config.settings import settings

class ChatGPTClient:
    def __init__(self, api_key: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def chat(self, messages: list[dict]):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

@asynccontextmanager
async def get_chatgpt_llm():
    client = ChatGPTClient(api_key=settings.llm_api_key, model=settings.llm_model)
    yield client


class ChatGPTEmbedder:
    def __init__(self, api_key: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [d.embedding for d in response.data]

@asynccontextmanager
async def get_chatgpt_embedder():
    embedder = ChatGPTEmbedder(api_key=settings.embedding_api_key, model=settings.embedding_model)
    yield embedder
