# app/schemas/chat.py
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    k: int = 3  # number of chunks to use as context after retrieval/re-ranking
    session_id: str | None = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    session_id: str  # pass this back on the next request to continue the conversation
