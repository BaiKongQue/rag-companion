import logging
from contextlib import asynccontextmanager, AsyncExitStack
import chromadb.api
import chromadb.types
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from src.services.document_processor import process_document
from typing import Annotated, Any
import chromadb
from src.llm import new_llm_client
from src.llm.base import BaseLLMClient
from src.models.chat import ChatResponse, ChatRequest
from uuid import uuid4
import numpy as np
from chromadb.types import Metadata
from src.config.settings import settings

logger = logging.getLogger(__name__)
sub_app_api = FastAPI()

chromadb_client: chromadb.api.ClientAPI
llm_client: BaseLLMClient

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Application lifespan context manager for initialization and cleanup."""
    logger.info("Starting application lifespan context...")
    global chromadb_client, llm_client

    # Init ChromaDB
    logger.info("Initializing ChromaDB client...")
    chromadb_client = chromadb.PersistentClient(path=settings.chroma_save_path)
    
    # Init LLM model
    logger.info("Initializing LLM model client...")
    llm_client = new_llm_client()

    yield

@sub_app_api.post("/upload")
async def upload(file: UploadFile, collection: str = Form(...)):
    """Upload endpoint"""
    if not file.filename or not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files are supported.")

    try:
        # Step 1: Extract and split text
        chunks = await process_document(file)

        # Step 2: Embed and store in ChromaDB
        embeddings = await llm_client.embed(chunks)
        ids = [str(uuid4()) for _ in chunks]
        metadata: list[Metadata] = [{
            "filename": file.filename, 
            "chunk_index": i,
            # "embedding_function": settings.model_embedding
        } for i in range(len(chunks))]
        
        # Ensure the collection exists or create it
        chromadb_collection = chromadb_client.get_or_create_collection(name=collection.replace(" ", "_").lower())

        chromadb_collection.add(
            documents=chunks,
            embeddings=np.array(embeddings, dtype=np.float32),
            ids=ids,
            metadatas=metadata
        )

        return {"status": "success", "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@sub_app_api.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint to interact with the LLM asking questions about uploaded pdfs."""
    try:
        

        # load relevant documents from ChromaDB
        logger.debug("Loading relevant documents from ChromaDB...")
        message_embeddings = await llm_client.embed([request.message])
        chromadb_collection = chromadb_client.get_or_create_collection(
            name=request.collection.replace(" ", "_").lower()
            # metadata={"embedding_function": settings.model_embedding}
        )
        
        # Check if chromadb has data
        if chromadb_collection.count() <= 0:
            logger.warning("ChromaDB collection is empty.")
            return {"status": "failed", "response": "[No documents found in the collection. Please upload documents first.]"}
        
        relevant_docs = chromadb_collection.query(
            query_embeddings=message_embeddings[0],
            n_results=10,
            include=["documents", "metadatas"]
        )

        if not relevant_docs or not relevant_docs['documents']:
            logger.warning("No relevant documents found.")
            return {"status": "success", "response": "[No relevant documents found.]"}
        docs = [d for dd in relevant_docs['documents'] for d in dd]

        logger.debug(f"Relevant documents: {docs}")
        
        # Send the chat message to the LLM
        logger.debug("Sending chat message to LLM...")

        response = await llm_client.chat([request.message], relevant_docs=docs)
        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@sub_app_api.post("/test")
async def test_endpoint():
    """Test endpoint to check if the API is working."""
    try:
        logger.info("Testing LLM client with a sample message...")
        response = await llm_client.chat(["Hello, how are you?"], relevant_docs=["This is a test document."])
        logger.info("Test endpoint successful.")
        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Test endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Test failed. Check logs for details.")