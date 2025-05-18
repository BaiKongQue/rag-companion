# src/pdf_rag_engine/llm/__init__.py

from src.config.settings import Settings
from src.llm.chatgpt import ChatGPTClient
from src.llm.ollama import OllamaClient
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """
    Base class for all LLM clients.
    """
    @abstractmethod
    async def chat(self, messages: list[dict]) -> str:
        """
        Send a chat message to the LLM and get a response.
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Close the LLM client connection.
        """
        pass

class BaseEmbedderClient(ABC):
    """
    Base class for all embedding clients.
    """
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for the given texts.
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Close the embedder connection.
        """
        pass

llm_client_map: dict[str, BaseLLMClient] = {}
embedder_client_map: dict[str, BaseEmbedderClient] = {}

def LLMClient():
    """
    Decorator to register a class as a LLM client.
    """
    def decorator(cls, name):
        llm_client_map[name.lower()] = cls
        return cls
    return decorator

def EmbedderClient():
    """
    Decorator to register a class as an embedder client.
    """
    def decorator(cls, name):
        embedder_client_map[name.lower()] = cls
        return cls
    return decorator

@asynccontextmanager
async def init_llm_client(settings: Settings):
    if settings.llm_model.lower() not in llm_client_map:
        raise ValueError(f"Unsupported LLM model: {settings.llm_model}")
    
    try:
        client: BaseLLMClient = llm_client_map[settings.llm_model.lower()]()
        yield client
    finally:
        await client.close()

@asynccontextmanager
async def init_embedder_client(settings: Settings):
    if settings.embedding_model.lower() not in embedder_client_map:
        raise ValueError(f"Unsupported embedding model: {settings.embedding_model}")
    
    try:
        client: BaseEmbedderClient = embedder_client_map[settings.embedding_model.lower()]()
        yield client
    finally:
        await client.close()