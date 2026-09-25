from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    context: str | None = None


class ChatResponse(BaseModel):
    answer: str
