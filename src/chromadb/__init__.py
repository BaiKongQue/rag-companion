from contextlib import asynccontextmanager
from src.chromadb.client import ChromaDBClient
import logging

logger = logging.getLogger(__name__)
chromadb_client = None

@asynccontextmanager
async def init_chromadb():
    """Initialize and yield a ChromaDB client."""
    global chromadb_client
    chromadb_client = ChromaDBClient()
    
    if not chromadb_client.collection:
        raise ValueError("Failed to initialize ChromaDB collection.")
    
    try:
        yield
    finally:
        await chromadb_client.close()