from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# import agent functions
from agent import init_pipeline, get_llm, answer_dream_query

app = FastAPI()

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
    Build the vector store and LLM once when the server starts.
    Store them on app.state for reuse across requests.
    """
    app.state.vector_store = init_pipeline()
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
    llm = app.state.llm

    answer = answer_dream_query(
        query=request.question,
        vector_store=vector_store,
        llm=llm,
    )
    return ChatResponse(answer=answer)
