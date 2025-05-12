import logging
from fastapi import FastAPI, HTTPException
import uvicorn

import bot

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/health")
async def health():
    """Health endpoint for Kubernetes probes"""
    if bot.is_healthy:
        return {"status": "healthy"}
    else:
        raise HTTPException(status_code=500, detail="Status is unhealthy")

def start_fastapi():
    """Start the FastAPI app"""
    try:
        config = uvicorn.Config(app, host="0.0.0.0", port=8010, log_level="info")
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.error(f"FastApi encountered an error: {e}")
        exit(-1)