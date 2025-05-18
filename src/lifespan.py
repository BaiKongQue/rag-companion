from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI
from src.config.settings import settings
from src.llm import BaseLLMClient, BaseEmbedderClient, init_embedder_client, init_llm_client
from src.chromadb import init_chromadb

chromadb_client = None
llm_model: BaseLLMClient = None
embedding_model: BaseEmbedderClient = None
redis_client = None

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    global llm_model, embedding_model, redis_client, chromadb_client
    async with AsyncExitStack() as stack:
        # Add any sub application lifespan here
        await stack.enter_async_context(
        )

        # Init ChromaDB
        chromadb_client = await stack.enter_async_context(init_chromadb())
        
        # Init Embedding model
        embedding_model = await stack.enter_async_context(init_embedder_client())

        # Init LLM model
        llm_model = await stack.enter_async_context(init_llm_client())

        # Optional Redis
        if settings.use_redis_cache:
            cache = await stack.enter_async_context(init_redis_cache(settings.redis_url))
            app.state.cache = cache

        yield
        
        # Cleanup
        await stack.aclose()