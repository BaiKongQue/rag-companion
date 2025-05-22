from contextlib import asynccontextmanager
from src.config.settings import settings
import chromadb


class ChromaDBClient:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_save_path)
        self.collection = self.client.create_collection(name=settings.chroma_collection)

    async def close(self):
        await self.client.close()

@asynccontextmanager
async def init_chromadb():
    client = ChromaDBClient()
    try:
        yield client
    finally:
        await client.close()