# src/pdf_rag_engine/llm/__init__.py

from src.config.settings import Settings
from src.llm.chatgpt import ChatGPTClient
from src.llm.ollama import OllamaClient
from contextlib import asynccontextmanager
from .base import BaseLLMClient

@asynccontextmanager
async def get_llm_client(settings: Settings) -> BaseLLMClient:
    clientMap = {
        "chatgpt": ChatGPTClient,
        "ollama": OllamaClient
    }

    # if settings.llm_model.lower() == "chatgpt":
    #     async with get_chatgpt_client(settings) as client:
    #         yield client
    # elif settings.llm_model.lower() == "ollama":
    #     async with get_ollama_client(settings) as client:
    #         yield client
    # else:
    if settings.llm_model.lower() not in clientMap:
        raise ValueError(f"Unsupported LLM model: {settings.llm_model}")
    
    async with clientMap[settings.llm_model.lower()]() as client:
        yield client
