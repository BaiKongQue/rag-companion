import sys
import os
import logging
from fastapi import FastAPI
from fastapi import FastAPI
from datetime import datetime
from src.api.routes import sub_app_api, app_lifespan
from src.config.settings import settings
from src.config.formatter import CustomFormatter
import uvicorn
from contextlib import AsyncExitStack, asynccontextmanager

# Set up logging configuration
__log_handler = logging.StreamHandler(sys.stdout)
__log_handler.setFormatter(CustomFormatter('%(levelname)s [%(asctime)s] {%(filename)s:%(lineno)d} %(message)s'))
logging.basicConfig(
    level=logging.INFO if settings.production is not None and settings.production == True else logging.DEBUG,
    handlers=[__log_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"Production mode: {settings.production}")


@asynccontextmanager
async def main_lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(
            app_lifespan(sub_app_api)
        )

        # Optional Redis
        # if settings.use_redis_cache:
        #     cache = await stack.enter_async_context(init_redis_cache(settings.redis_url))
        #     app.state.cache = cache

        yield
        
        # Cleanup
        await stack.aclose()

app = FastAPI(
    title="RAG Companion",
    version="0.1.0",
    lifespan=main_lifespan
)

@app.get("/health")
async def health():
    """Health endpoint for Kubernetes probes"""
    # if bot.is_healthy:
    #     return {"status": "healthy"}
    # else:
    #     raise HTTPException(status_code=500, detail="Status is unhealthy")
    return {"status": "healthy", "time": datetime.now().isoformat()}


# app.include_router(api_router, prefix="/api")

app.mount("/api", sub_app_api)

async def run():
    try:
        config = uvicorn.Config(app, host="0.0.0.0", port=8010, log_level="info")
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.error(f"FastApi encountered an error: {e}")
        exit(-1)

if __name__ == "__main__":
    logger.info("Starting RAG Companion server...")
    try:
        import asyncio
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"An error occurred while running the server: {e}")
        exit(-1)