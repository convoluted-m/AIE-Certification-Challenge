"""
FastAPI backend for the DreamNest agentic RAG pipeline (agent orchestration logic in agent.py)
On startup, initialises the RAG pipeline.

Endpoints for:
- Health check
- Chat endpoint - calls the RAG pipeline to answer user questions
"""

## Imports
# FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Pydantic for data validation
from pydantic import BaseModel

# Import agent logic from agent.py
from agent import (
    init_pipeline_semantic,
    init_pipeline_hybrid,
    get_llm,
    answer_dream_query_semantic,
    answer_dream_query_hybrid,
)

# create FastAPI app
app = FastAPI()

# CORS middleware to allow requests from all origins - allows all methods and headers
# for dev, NOT FOR PRODUCTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# On startup - runs once when server starts (uvicorn)
# Initialises the RAG pipeline (vector store, retriever, LLM)
# currently set up to use hybrid retrieval
@app.on_event("startup")
async def startup_event() -> None:
    """
    Builds the vector store, retriever, and LLM once when the server starts.
    Stores them on app.state for reuse across requests.
    """
    vector_store, bm25_retriever = init_pipeline_hybrid()
    app.state.vector_store = vector_store
    app.state.bm25_retriever = bm25_retriever
    app.state.llm = get_llm()


# health check endpoint
@app.get("/api/health")
def health_check():
    return {"status": "ok"}


## Data validation with Pydantic
# question from user must be a json object with a string field 'question'
class ChatRequest(BaseModel):
    question: str
# answer from agent must be a json object with a string field 'answer'
class ChatResponse(BaseModel):
    answer: str


# chat endpoint (calls the RAG pipeline, returns the answer)
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint. 
    Takes the user's question and returns a retrieval-grounded answer.
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
