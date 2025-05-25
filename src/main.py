import sys
import os
import logging
from fastapi import FastAPI
from fastapi import FastAPI
from datetime import datetime
from api.api import sub_app_api
from lifespan import app_lifespan
import uvicorn

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO if os.environ.get("PRODUCTION") is not None and os.environ.get("PRODUCTION").lower() == "true" else logging.DEBUG,
    format='- [%(levelname)s] [%(asctime)s] {%(filename)s:%(lineno)d} %(message)s',
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Companion",
    version="0.1.0",
    lifespan=app_lifespan
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

app.mount("/lab", sub_app_api)

async def run():
    try:
        config = uvicorn.Config(app, host="0.0.0.0", port=8010, log_level="info")
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.error(f"FastApi encountered an error: {e}")
        exit(-1)