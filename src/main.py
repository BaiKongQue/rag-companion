import sys
import os
import logging
from fastapi import FastAPI
from fastapi import FastAPI
from src.config.settings import settings
from src.lifespan import build_lifespan
from src.api.routes import router as api_router

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
    lifespan=build_lifespan(settings)
)

app.include_router(api_router, prefix="/api")