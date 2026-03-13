"""
FastAPI backend for the DreamNest agentic RAG pipeline.
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
    init_pipeline_hybrid,
    get_llm,
    build_retrieval_agent,
    dream_archive_search,
)
from langchain_core.messages import HumanMessage

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


# On startup - runs once when server starts
# Initialises the RAG pipeline (vector store, retriever, LLM)
@app.on_event("startup") # chekc this later, seems deprecated
async def startup_event() -> None:
    """
    Builds the vector store, retriever, and LLM once when the server starts.
    Stores them on app.state for reuse across requests.
    """
    vector_store, bm25_retriever = init_pipeline_hybrid()

    llm = get_llm()

    # for debugging
    app.state.vector_store = vector_store
    app.state.bm25_retriever = bm25_retriever
    app.state.llm = llm

    # Build the agent with tools 
    app.state.agent = build_retrieval_agent(llm=llm)


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
    Chat endpoint. Takes the user's question and returns an answer from the agent.
    The agent can call two tools:
    """
    agent = app.state.agent

    try:
        # create_agent LangChain >= 1.0 uses a messages-based interface
        # Input must be {"messages": [HumanMessage(...)]}
        # Output is {"messages": [...]} where last message is final answer
        result = agent.invoke({"messages": [HumanMessage(content=request.question)]})
        if isinstance(result, dict) and "messages" in result:
            answer = result["messages"][-1].content
        elif isinstance(result, dict) and "output" in result:
            answer = result["output"]
        else:
            answer = str(result)
    except Exception as e:
        # Fallback if agent raises error call RAG tool directly so user gets answer
        print(f"[FALLBACK] Agent raised {type(e).__name__}: {e} — routing to dream_archive_search directly")
        answer = dream_archive_search.invoke(request.question)

    return ChatResponse(answer=answer)

# debugging endpoint
# test if llm is working itself without langchain create_agent() and agent.invoke()
@app.get("/api/test-llm")
async def test_llm():
    llm = app.state.llm
    resp = llm.invoke("say 'hello' without tools.")
    return {"ok": True, "response": resp.content}