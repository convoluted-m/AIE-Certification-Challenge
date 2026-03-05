# DreamNest

Privacy-first agentic RAG app for personal dream exploration. Built with a FastAPI backend and open-source models and run locally. This is a prototype version. Runs locally only for privacy reasons.  Built for the AI Makerspace AIE9 Certification Challenge.

### Tech Stack
- Local inference: Ollama
- LLM: llama3.2:3b  https://ollama.com/library/llama3.2:3b
- Embedding model: embeddinggemma https://ollama.com/library/embeddinggemma
- Vector store: Qdrant (in-memory)
- Backend: Agentic RAG orchestration: LangChain (LangGraph planned)
- Frontend with Node.js running locally served to the user's browser

### Data
Synthetic dream journal entries saved in PDF

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

### Privacy
Inference and embedding runs locally via Ollama. No data is sent to any third-party API. Dream journal data is stored in-memory only.
