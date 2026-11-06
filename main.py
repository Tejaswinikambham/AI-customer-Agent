from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import run_agent

app = FastAPI(
    title="Agentic RAG Customer Support",
    description="AI-powered customer support backend",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def home():
    return {
        "message": "Agentic RAG Customer Support API is running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    answer = run_agent(request.question)

    return ChatResponse(
        question=request.question,
        answer=answer
    )