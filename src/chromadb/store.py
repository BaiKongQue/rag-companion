from typing import List
from uuid import uuid4
import logging
from src.chromadb.client import ChromaDBClient
from src.llm.base import BaseEmbedderClient

logger = logging.getLogger(__name__)

# Optional: Use one collection per file
def get_collection_name(filename: str) -> str:
    return f"doc_{filename.replace('.', '_')}"

async def store_embeddings(filename: str, chunks: List[str], chromadb_client: ChromaDBClient, embedding_model: BaseEmbedderClient):
    logger.debug(f"Embedding class {embedding_model.__class__.__name__}")
    embeddings = await embedding_model.embed(chunks)
    ids = [str(uuid4()) for _ in chunks]

    chromadb_client.collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": filename}] * len(chunks)
    )
