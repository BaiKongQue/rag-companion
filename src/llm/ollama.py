# src/pdf_rag_engine/llm/ollama.py

from contextlib import asynccontextmanager
from ollama import AsyncClient
from pdf_rag_engine.config.settings import Settings

class OllamaLLMClient:
    def __init__(self, host: str, model: str):
        self.client = AsyncClient(host=host)
        self.model = model

    async def chat(self, messages: list[dict]):
        response = await self.client.chat(model=self.model, messages=messages)
        return response['message']['content']

    async def close(self):
        await self.client.aclose()


class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.client = AsyncClient(host=host)
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings(model=self.model, prompt=texts)
        return response['embeddings']

    async def close(self):
        await self.client.aclose()


@asynccontextmanager
async def get_ollama_llm(settings: Settings):
    client = OllamaLLMClient(settings.llm_host, settings.llm_model)
    try:
        yield client
    finally:
        await client.close()


@asynccontextmanager
async def get_ollama_embedder(settings: Settings):
    embedder = OllamaEmbedder(settings.embedding_host, settings.embedding_model)
    try:
        yield embedder
    finally:
        await embedder.close()
