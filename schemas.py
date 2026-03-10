from pydantic import BaseModel
from typing import Optional


class RAGResponse(BaseModel):

    answer: str
    source_pages: Optional[list[int]] = None
    image_base64: Optional[str] = None


class RAGQuery(BaseModel):

    question: str
    mode: str = "fast"

class LLMAnswer(BaseModel):

    answer: str
    confidence: float

class AnswerSchema(BaseModel):
    answer: str