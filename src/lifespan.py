from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI
from src.config.settings import settings
from src.llm import BaseLLMClient, BaseEmbedderClient, init_embedder_client, init_llm_client
from src.chromadb.client import init_chromadb
import logging

logger = logging.getLogger(__name__)

chromadb_client = None
llm_model: BaseLLMClient = None
embedding_model: BaseEmbedderClient = None
redis_client = None

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    global llm_model, embedding_model, redis_client, chromadb_client
    async with AsyncExitStack() as stack:
        # Add any sub application lifespan here
        # await stack.enter_async_context(
        # )

        # Init ChromaDB
        logger.info("Initializing ChromaDB client...")
        chromadb_client = await stack.enter_async_context(init_chromadb())
        
        # Init Embedding model
        logger.info("Initializing embedding model client...")
        embedding_model = await stack.enter_async_context(init_embedder_client())

        # Init LLM model
        logger.info("Initializing LLM model client...")
        llm_model = await stack.enter_async_context(init_llm_client())

        # Optional Redis
        # if settings.use_redis_cache:
        #     cache = await stack.enter_async_context(init_redis_cache(settings.redis_url))
        #     app.state.cache = cache

        yield
        
        # Cleanup
        await stack.aclose()