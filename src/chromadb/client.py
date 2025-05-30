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

    def test():
        logger.info("ChromaDB client Test method called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    async def close(self):
        logger.info("Closing ChromaDB client...")