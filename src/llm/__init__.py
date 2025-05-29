from contextlib import asynccontextmanager
import logging
from src.config.settings import settings
from src.llm.base import BaseLLMClient, BaseEmbedderClient, __llm_client_map, __embedder_client_map

logger = logging.getLogger(__name__)

@asynccontextmanager
async def init_llm_client():
    if settings.llm_model.lower() not in __llm_client_map:
        raise ValueError(f"Unsupported LLM model: {settings.llm_model}")
    
    try:
        client: BaseLLMClient = __llm_client_map[settings.llm_model.lower()]()
        yield client
    finally:
        await client.close()

@asynccontextmanager
async def init_embedder_client():
    if settings.embedding_model.lower() not in __embedder_client_map:
        raise ValueError(f"Unsupported embedding model: {settings.embedding_model}")
    
    try:
        client: BaseEmbedderClient = __embedder_client_map[settings.embedding_model.lower()]()
        yield client
    finally:
        await client.close()