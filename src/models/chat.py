from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    collection: str

class ChatResponse(BaseModel):
    status: str
    response: str