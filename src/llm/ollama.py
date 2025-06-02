from ollama import AsyncClient
import logging
from src.config.settings import settings
from src.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

class OllamaLLMClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncClient(host=settings.model_host)

    async def chat(self, messages: list[str], relevant_docs: list[str]) -> str:
        msgs = [{"role": msg['role'], "content": msg['content']} for msg in messages]
        docs = '\n'.join(relevant_docs) if relevant_docs else ""
        
        logger.debug(f"Model host set to: {settings.model_host}. model: {settings.model_llm}")

        try:
            response = await self.client.chat(
                model=settings.model_llm,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *msgs,
                    {"role": "assistant", "content": docs}
                ],
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return "[No response]"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        logger.debug(f"Embedding texts with model: {settings.model_embedding}")

        response = await self.client.embed(
            input=texts,
            model=settings.model_embedding
        )
        if 'embeddings' not in response:
            logger.error("Ollama embed response missing 'embeddings'")
            return [[0.0] * 1536] * len(texts)
        return response['embeddings']

    async def close(self):
        pass
        # await self.client