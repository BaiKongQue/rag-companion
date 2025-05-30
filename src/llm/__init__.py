from contextlib import asynccontextmanager
import logging
from src.config.settings import settings
from src.llm.base import BaseLLMClient, BaseEmbedderClient
from src.llm.chatgpt import ChatGPTClient, ChatGPTEmbedder
from src.llm.ollama import OllamaLLMClient, OllamaEmbedder

logger = logging.getLogger(__name__)
llm_model: BaseLLMClient = None
embedding_model: BaseEmbedderClient = None

@asynccontextmanager
async def init_llm_client():
    llm_client_map: dict[str, BaseLLMClient] = {
        "chatgpt": ChatGPTClient,
        "ollama": OllamaLLMClient,
    }

    if settings.llm.lower() not in llm_client_map:
        raise ValueError(f"Unsupported LLM model: {settings.llm}\nAvailable options: {', '.join(llm_client_map.keys())}")

    try:
        llm_model: BaseLLMClient = llm_client_map[settings.llm.lower()]()
        logger.info(f"Successfully initialized LLM client for {settings.llm}.")
        yield
    finally:
        await llm_model.close()

@asynccontextmanager
async def init_embedder_client():
    embedder_client_map: dict[str, BaseEmbedderClient] = {
        "chatgpt": ChatGPTEmbedder,
        "ollama": OllamaEmbedder,
    }
    
    if settings.embedding.lower() not in embedder_client_map:
        raise ValueError(f"Unsupported embedding model: {settings.embedding}\nAvailable options: {', '.join(embedder_client_map.keys())}")
    
    try:
        embedding_model: BaseEmbedderClient = embedder_client_map[settings.embedding.lower()]()
        logger.info(f"Successfully initialized embedder client for {settings.embedding}.")
        yield
    finally:
        await embedding_model.close()