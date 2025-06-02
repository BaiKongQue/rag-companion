import io
from typing import List
from fastapi import UploadFile
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)

def split_text(text: str, max_chunk_size: int = 500) -> List[str]:
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

async def process_document(file: UploadFile) -> List[str]:
    content = await file.read()

    if file.filename and file.filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        text = "".join(page.extract_text() or "" for page in reader.pages)
    else:  # .txt
        text = content.decode("utf-8")

    chunks = split_text(text)
    return chunks
