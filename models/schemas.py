"""
Schemas for request and response models used in the API endpoints.
"""
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    intent: str

class IngestResponse(BaseModel):
    status: str
    detail: dict
