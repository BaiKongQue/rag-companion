import logging
from src.config.settings import settings
from src.llm.base import BaseLLMClient
import httpx
import requests

logger = logging.getLogger(__name__)

class DockerLLMClient(BaseLLMClient):
    def __init__(self):
        pass

    async def chat(self, messages: list[str], relevant_docs: list[str]) -> str:
        msgs = [{
            "role": "user",
            "content": msg
        } for msg in messages]

        docs = '\n'.join(relevant_docs) if relevant_docs else ""
        response = await self.__chat_completion(
            model=settings.model_llm,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                *msgs,
                {"role": "assistant", "content": docs}
            ],
        )

        if not response['choices'] or not response['choices'][0]['message'] or not response['choices'][0]['message']['content']:
            logger.error("No response from ChatGPT or response format is incorrect.")
            return "[No response]"

        return response['choices'][0]['message']['content']

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.__embeddings(
            model=settings.model_embedding,
            input=texts
        )
        if not response or 'data' not in response:
            logger.error("No embeddings returned or response format is incorrect.")
            return [[0.0] * 1536] * len(texts)
        
        return [r['embedding'] for r in response['data']]

    async def close(self):
        # Placeholder for Docker client cleanup
        pass

    async def __chat_completion(self, messages: list[dict], model: str):
        payload = {
            "model": model,
            "messages": messages
        }

        # response = requests.post(
        #     f"{settings.model_host}/chat/completions",
        #     headers={
        #         "Content-Type": "application/json",
        #         'X-Content-Type-Options': 'nosniff',
        #         'X-Frame-Options': 'SAMEORIGIN',
        #         'X-XSS-Protection': '1; mode=block',
        #         'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; font-src 'self' https://cdnjs.cloudflare.com"
        #     },
        #     json=payload,
        #     timeout=130
        # )

        # if response.status_code != 200:
        #     raise Exception(f"API returned status code {response.status_code}: {response.text}")
            
        # # Parse the response
        # chat_response = response.json()

        # return chat_response

        timeout = httpx.Timeout(300.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Model host set to: {settings.model_host}. model: {model}")
            response = await client.post(
                settings.model_host + "/chat/completions",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': 'SAMEORIGIN',
                    'X-XSS-Protection': '1; mode=block',
                    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; font-src 'self' https://cdnjs.cloudflare.com"
                }
            )
            response.raise_for_status()
            data = response.json()
            if "choices" in data and data["choices"]:
                return data
            else:
                return {"choices": [{"message": {"content": "[No response]"}}]}

    async def __embeddings(self, input: list[str], model: str):
        payload = {
            "model": model,
            "input": input,
        }

        timeout = httpx.Timeout(120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Model host set to: {settings.model_host}. model: {model}")
            response = await client.post(
                settings.model_host + "/embeddings",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': 'SAMEORIGIN',
                    'X-XSS-Protection': '1; mode=block',
                    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; font-src 'self' https://cdnjs.cloudflare.com"
                }
            )
            response.raise_for_status()
            data = response.json()

            # Assumes standard OpenAI-like format
            return data
        # [item["embedding"] for item in data.get("data", [])]