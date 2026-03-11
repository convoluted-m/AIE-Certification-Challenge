from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Agent logic
from agent import (
    init_pipeline_with_bm25,
    get_llm,
    answer_dream_query_hybrid,
)

app = FastAPI()

# CORS middleware to allow requests from all origins
# allows all methods and headers
# NOT FOR PRODUCTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# on startup  -runs once when server starts
# initializes vector store and llm
@app.on_event("startup")
async def startup_event() -> None:
    """
    Build the vector store, BM25 retriever, and LLM once when the server starts.
    Store them on app.state for reuse across requests.
    """
    vector_store, bm25_retriever = init_pipeline_with_bm25()
    app.state.vector_store = vector_store
    app.state.bm25_retriever = bm25_retriever
    app.state.llm = get_llm()

# health check endpoint
@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# resquest model for chat endpoint
class ChatRequest(BaseModel):
    question: str

# response model for chat endpoint
class ChatResponse(BaseModel):
    answer: str

# chat endpoint - takes a question and returns an answer
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint. Takes a dream-related question and returns a retrieval-grounded answer.
    """
    vector_store = app.state.vector_store
    bm25_retriever = app.state.bm25_retriever
    llm = app.state.llm

    answer = answer_dream_query_hybrid(
        query=request.question,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        llm=llm,
    )
    return ChatResponse(answer=answer)
