import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from src.services.document_processor import process_document
from src.chromadb.store import store_embeddings

logger = logging.getLogger(__name__)
sub_app_api = FastAPI()

@sub_app_api.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload endpoint"""
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files are supported.")

    try:
        # Step 1: Extract and split text
        chunks = await process_document(file)

        # Step 2: Embed and store in ChromaDB
        await store_embeddings(file.filename, chunks)

        return {"status": "success", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@sub_app_api.get("/chat")
async def chat():
    """Chat endpoint"""
    logger.info("Chat endpoint called")