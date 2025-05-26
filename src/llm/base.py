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

__llm_client_map: dict[str, BaseLLMClient] = {}
__embedder_client_map: dict[str, BaseEmbedderClient] = {}

def LLMClient():
    """
    Decorator to register a class as a LLM client.
    """
    def decorator(cls, name):
        __llm_client_map[name.lower()] = cls
        return cls
    return decorator

def EmbedderClient():
    """
    Decorator to register a class as an embedder client.
    """
    def decorator(cls, name):
        __embedder_client_map[name.lower()] = cls
        return cls
    return decorator
