from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI
from src.config.settings import Settings
from src.retriever.indexer import init_chroma
from src.llm.chatgpt import get_chatgpt_llm, get_chatgpt_embedder
from src.llm.ollama import get_ollama_llm, get_ollama_embedder
from src.cache.redis_cache import init_redis_cache

llm_model = None
embedding_model = None
redis_client = None
chroma_client = None

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
        global llm_model, embedding_model, redis_client, chroma_client
        async with AsyncExitStack() as stack:
            # Add any sub application lifespan here
            await stack.enter_async_context(
            )

            # Init ChromaDB
            chroma_client = chroma
            
            # Init Embedding model
            if settings.embedding_model not in embeddings:
                raise ValueError(f"Unsupported embedder: {settings.embedding_model}")
            embedding_model = await stack.enter_async_context(embeddings[settings.embedding_model]())

            # Init LLM model
            if settings.llm_model not in LLMs:
                raise ValueError(f"Unsupported LLM: {settings.llm_model}")
            llm_model = await stack.enter_async_context(LLMs[settings.llm_model]())

            # Optional Redis
            if settings.use_redis_cache:
                cache = await stack.enter_async_context(init_redis_cache(settings.redis_url))
                app.state.cache = cache

            yield
            
            # Cleanup
            await stack.aclose()
            if redis_client:
                await redis_client.close()
            if chroma_client:
                await chroma_client.close()
            if embedding_model:
                await embedding_model.close()
            if llm_model:
                await llm_model.close()

    return app_lifespan