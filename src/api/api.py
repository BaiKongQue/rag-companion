import logging
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)
sub_app_api = FastAPI()

@sub_app_api.get("/upload")
async def upload():
    """Upload endpoint"""
    logger.info("Upload endpoint called")
    
@sub_app_api.get("/chat")
async def chat():
    """Chat endpoint"""
    logger.info("Chat endpoint called")