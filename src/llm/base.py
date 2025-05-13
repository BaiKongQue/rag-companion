from abc import ABC, abstractmethod

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
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for the given texts.
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Close the LLM client connection.
        """
        pass