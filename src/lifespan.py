from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI
from src.config.settings import Settings
from src.retriever.indexer import init_chroma
from src.llm.chatgpt import get_chatgpt_llm, get_chatgpt_embedder
from src.llm.ollama import get_ollama_llm, get_ollama_embedder
from src.cache.redis_cache import init_redis_cache


def build_lifespan(settings: Settings):
    embeddings = {
        "chatgpt": get_chatgpt_embedder,
        "ollama": get_ollama_embedder
    }

    LLMs = {
        "chatgpt": get_chatgpt_llm,
        "ollama": get_ollama_llm
    }

    @asynccontextmanager
    async def app_lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            # Init ChromaDB
            chroma = await stack.enter_async_context(init_chroma(settings))
            app.state.chroma = chroma
            
            if settings.embedding_model not in embeddings:
                raise ValueError(f"Unsupported embedder: {settings.embedding_model}")
            app.state.embedder = await stack.enter_async_context(embeddings[settings.embedding_model]())

            # Init LLM model
            if settings.llm_model not in LLMs:
                raise ValueError(f"Unsupported LLM: {settings.llm_model}")
            app.state.llm = await stack.enter_async_context(LLMs[settings.llm_model]())

            # Optional Redis
            if settings.use_redis_cache:
                cache = await stack.enter_async_context(init_redis_cache(settings.redis_url))
                app.state.cache = cache

            yield

    return app_lifespan