from typing import List
from src.lifespan import embedding_model, chromadb_client
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

# Optional: Use one collection per file
def get_collection_name(filename: str) -> str:
    return f"doc_{filename.replace('.', '_')}"

async def store_embeddings(filename: str, chunks: List[str]):
    embeddings = await embedding_model.embed(chunks)

    collection = chromadb_client.get_or_create_collection(name=get_collection_name(filename))
    ids = [str(uuid4()) for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": filename}] * len(chunks)
    )
