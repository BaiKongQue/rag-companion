from typing import Type
from contextlib import asynccontextmanager
import logging
from src.config.settings import settings
from src.llm.base import BaseLLMClient
from src.llm.chatgpt import ChatGPTClient
from src.llm.ollama import OllamaLLMClient
from src.llm.docker import DockerLLMClient

logger = logging.getLogger(__name__)

def new_llm_client() -> BaseLLMClient:
    llm_client_map: dict[str, Type[BaseLLMClient]] = {
        "chatgpt": ChatGPTClient,
        "ollama": OllamaLLMClient,
        "docker": DockerLLMClient
    }

    if settings.model.lower() not in llm_client_map:
        raise ValueError(f"Unsupported LLM model: {settings.model}\nAvailable options: {', '.join(llm_client_map.keys())}")

    return llm_client_map[settings.model.lower()]()