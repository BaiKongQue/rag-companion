from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from src.config.settings import settings
from src.llm.base import BaseLLMClient
import logging

logger = logging.getLogger(__name__)

class ChatGPTClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.model_api_key)

    async def chat(self, messages: list[str], relevant_docs: list[str]) -> str:
        msgs = [ChatCompletionUserMessageParam(role="user", content=msg) for msg in messages]
        docs = '\n'.join(relevant_docs) if relevant_docs else ""
        response = await self.client.chat.completions.create(
            model=settings.model_llm,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                *msgs,
                {"role": "assistant", "content": docs}
            ],
        )

        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logger.error("No response from ChatGPT or response format is incorrect.")
            return "[No response]"

        return response.choices[0].message.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=settings.model_embedding,
            input=texts
        )
        return [r.embedding for r in response.data]

    async def close(self):
        pass
        # await self.client.aiterator.aclose()