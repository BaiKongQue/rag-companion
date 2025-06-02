from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI
from src.config.settings import settings
from src.llm import init_embedder_client, init_llm_client
from src.chromadb import init_chromadb, chromadb_client
import logging

logger = logging.getLogger(__name__)

