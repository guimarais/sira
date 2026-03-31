"""
Schemas for request and response models used in the API endpoints.
"""
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    intent: str

class IngestResponse(BaseModel):
    status: str
    detail: dict
