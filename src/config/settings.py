from pydantic_settings import BaseSettings, SettingsConfigDict, model_validator
import os
from typing_extensions import Self
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix='RAG_COMPANION_',
        cli_parse_args=True
    )

    production: bool = False
    logging_level: str = "INFO"

    address: str = "0.0.0.0"
    port: int = 9000
    
    redis_url: str = "redis://localhost:6379"

    llm: Literal['chatgpt', 'ollama'] = "chatgpt"
    llm_host: str = "localhost"
    llm_model: str = "o4-mini"
    llm_api_key: str = ""

    embedding: str = "chatgpt"
    embedding_host: str = "localhost"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    
    chroma_collection: str = "rag-companion"

    @model_validator(mode='after')
    def validate_llm(self) -> Self:
        if self.llm_model == "chatgpt":
            if not self.llm_api_key:
                raise ValueError("RAG_COMPANION_LLM_API_KEY must be set when RAG_COMPANION_LLM_MODEL is chatgpt")
        return self

    @property
    def redis_host(self):
        return self.redis_url.split("//")[1].split(":")[0]
    
    @property
    def redis_port(self):
        return int(self.redis_url.split("//")[1].split(":")[1])

def resolve_env_file() -> str:
    prod = os.environ.get("PRODUCTION", "").lower()
    return ".env" if prod == "true" else "dev.env"

settings = Settings(_env_file=resolve_env_file())