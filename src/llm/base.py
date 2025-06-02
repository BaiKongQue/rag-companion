from abc import ABC
import logging

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """
    Base class for all LLM clients.
    """
    async def chat(self, messages: list[str], relevant_docs: list[str]) -> str:
        """
        Send a chat message to the LLM and get a response.
        """
        logger.warning("Chat method not implemented in BaseLLMClient. Returning dummy response.")
        return "[LLM not implemented]"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for the given texts.
        """
        logger.warning("Embed method not implemented in BaseEmbedderClient. Returning dummy embeddings.")
        return [[0.0]]

    async def close(self):
        """
        Close the embedder connection.
        """
        pass

