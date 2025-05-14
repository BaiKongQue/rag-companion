from contextlib import asynccontextmanager
from ollama import AsyncClient
from src.config.settings import settings

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
async def get_ollama_llm():
    client = OllamaLLMClient(settings.llm_host, settings.llm_model)
    try:
        yield client
    finally:
        await client.close()


@asynccontextmanager
async def get_ollama_embedder():
    embedder = OllamaEmbedder(settings.embedding_host, settings.embedding_model)
    try:
        yield embedder
    finally:
        await embedder.close()



# from contextlib import asynccontextmanager
# from ollama import AsyncClient
# from src.config.settings import settings

# @asynccontextmanager
# async def get_ollama_llm(host: str = settings.llm_host, model: str = settings.llm_model):
#     client = AsyncClient(host=host)
#     async def chat(messages: list[dict]):
#         response = await client.chat(model=model, messages=messages)
#         return response['message']['content']
#     try:
#         yield chat
#     finally:
#         await client.aclose()



# @asynccontextmanager
# async def get_ollama_embedder(host: str = settings.llm_host, model: str = settings.llm_model):
#     client = AsyncClient(host=host)
#     async def embed(texts: list[str]) -> list[list[float]]:
#         response = await client.embeddings(model=model, prompt=texts)
#         return response['embeddings']
#     try:
#         yield embed
#     finally:
#         await client.aclose()
