from contextlib import asynccontextmanager
from src.config.settings import settings
import chromadb
import logging

logger = logging.getLogger(__name__)

class ChromaDBClient:
    def __init__(self):
        if not settings.chroma_collection:
            raise ValueError("ChromaDB collection name must be set in settings.")
        if not settings.chroma_save_path:
            raise ValueError("ChromaDB save path must be set in settings.")
        
        self.client = chromadb.PersistentClient(path=settings.chroma_save_path)
        self.collection = self.client.get_or_create_collection(name=settings.chroma_collection)

        logger.info(f"ChromaDB client initialized with collection '{settings.chroma_collection}' at '{settings.chroma_save_path}'")

    async def close(self):
        await self.client.close()

@asynccontextmanager
async def init_chromadb():
    """Initialize and yield a ChromaDB client."""
    client = ChromaDBClient()

    if not client.collection:
        raise ValueError("Failed to initialize ChromaDB collection.")
    try:
        yield client
    finally:
        await client.close()