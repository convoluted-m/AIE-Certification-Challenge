# Certification Challenge- DreamNest (Prototype)

Privacy-first agentic RAG app for personal dream exploration. Built with a FastAPI backend, open-source models, and run locally with Ollama. Runs locally on the user's machine to preserve user privacy. This is a prototype. Built for the AI Makerspace AIE9 Certification Challenge.

## 1) Problem and Audience

User: someone keeping a private dream journal. 
Success criteria: System retrieves relevant past dreams and surfaces recurring symbols.

## 2) Proposed Solution & Stack

Agentic RAG over personal dream journal. Tool - similarity retrieval, guardrail prompt to prevent psychoanalytical interpretation.

- Local inference: Ollama
- LLM: llama3.2:3b  https://ollama.com/library/llama3.2:3b
- Embedding model: embeddinggemma https://ollama.com/library/embeddinggemma
- Vector store: Qdrant (in-memory)
- Backend: Agentic RAG orchestration: LangChain
- Frontend with Node.js running locally served to the user's browser


## 3) Data & Chunking
Synthetic dream journal entries saved in PDF
Dream data is stored in-memory only.


## 4) End-to-End Prototype

User asks: When did I dream about swimming? -> agent retrieves relevant entries -> LLM wraps into a response 



## 5) Golden Test Set & RAGAS
Syntheti manually defined golden test set targeting key behaviours


## 6) Advanced Retrieval
Hybrid retriever using lexical seach with BM25

## 7) Performance & Next Steps
Check if dream reports from dream database can be used.



## How to run
### Repo layout
- `api/` — FastAPI service
- `data/` — dream PDFs
- `frontend/` — Next.js web UI
- Root configs/deps — `pyproject.toml`, `README`, etc.

### Prerequisites
- Python 3.11+ (3.11–3.12 recommended for LangChain compatibility)
- uv for dependency management
- Node.js (for frontend)
- Ollama for local model inference

### Setup

**1. Install dependencies**
```bash
uv sync
```

**2. Pull required models**
```bash
ollama pull llama3.2:3b      # chat model (~2GB)
ollama pull embeddinggemma  # embedding model (~622MB) 
```

### Run on local machine

**Terminal 1 — start Ollama server:**
```bash
ollama serve
```
Ollama will be available at `http://localhost:11434`.

**Terminal 2 — run the FastAPI backend:**
```bash
uv run uvicorn api.index:app --reload
```

**Terminal 3 - run frontend:**
```bash
   cd frontend
   npm install
   npm run dev
```
Frontend available at `http://localhost:3000`

**Test endpoints:**
```bash
curl http://localhost:8000/api/health
```

**Test chat endpoint:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Have I had a dream about a house before?"}'
```
