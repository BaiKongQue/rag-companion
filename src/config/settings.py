from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
import os
from typing_extensions import Self
from typing import Literal
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix='RAG_COMPANION_',
        # cli_parse_args=True
    )

    production: bool = True
    logging_level: str = "INFO"

    address: str = "0.0.0.0"
    port: int = 9000
    
    redis_url: str = "redis://localhost:6379/0"

    model: str = "chatgpt"
    model_host: str = "localhost"
    model_llm: str = "o4-mini"
    model_embedding: str = "text-embedding-3-small"
    model_api_key: str = ""
    
    chroma_save_path: str = "src/chromadb/data"
    chroma_collection: str = "rag-companion"

    # @model_validator(mode='after')
    # def validate_llm(self) -> Self:
    #     if self.model == "chatgpt":
    #         if not self.model_api_key:
    #             raise ValueError("RAG_COMPANION_LLM_API_KEY must be set when RAG_COMPANION_LLM_MODEL is chatgpt")
    #     return self

    @property
    def redis_host(self):
        return self.redis_url.split("//")[1].split(":")[0]
    
    @property
    def redis_port(self):
        return int(self.redis_url.split("//")[1].split(":")[1])

settings: Settings = Settings()