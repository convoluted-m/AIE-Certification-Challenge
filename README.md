# DreamNest — Prototype

Privacy-first agentic RAG system for reflective dream exploration. Built with open-source models, run locally on the user's machine to preserve privacy. Built for the AI Makerspace AIE9 Certification Challenge.

## 1. Problem + Audience

This app is built for people like me who keep dream journals and want to search through them in an easy way to reflect on their dream life. It should be easier than flipping through the pages of physical notebooks forever! Over months and years, one can end up with many notebooks full of recurring themes and be unable to connect the dots, to point to the exact dreams featuring specific patterns. It's hard to do it by hand, take my word. If you ever wondered, *"When have I dreamt about fire?"* or *"Do I dream of flying yettis often?", you're not alone. This app is a solution to this existential pain by providing grounded, descriptive answers dream queries drawn only from your own writing. And because I wanted something that feels like a safe cosy space where my intimate data is not sent to third-parties, this app runs fully locally. Finally, I wanted this to be a tool that preserves your own agency in meaning-making of your inner life, that's why it's a descriptive retrieval system, not a psychonalaytical engine - save that for your personal analysis sessions and long dark nights. Welcome to your DreamNest.

Sample list of queries:
 
| Query | Expected behaviour |
|---|---|
| "Have I dreamt about fire??" | Searches for dreams containing fire. If found, returns them with dates and excerpts. I|
| "Have I dreamt about whales?" | If the user hasn't recorded dreams featuring whales, system replies that no relevant dreams were found in the user's journal. |
| "When have I dreamt about water?" | Aggregates results and retrieves dreams with lake, river, sea, pool etc. |
| "What recurring locations appear in my dreams?" | Aggregates recurring locations across archive |
| "What does dreaming about fire mean?" | Guardrail: declines to interpret, offers to search dreamer's archive instead |
| "What should I cook for dinner?" | Guardrail: out of scope, redirects to dream queries |
| "Is it helpful to write down your dreams?" | Routes to Tavily web search for general info about dreaming |

## 2. Solution

DreamNest is an agentic RAG system built over the user's private dream journal. It features a minimal chat interface served locally in a browser. The flow is the folowing: the user asks a natural language question about their dream. The agent decides whether the query is about a specific dreams (routes to the private retrieval tool) or a general question about dreaming (routes to a public web search via Tavily). Agent responses are grounded only in the retrieved content and descriptive in nature — the system surfaces patterns and excerpts, it is not intended to provide psychoanalytical commentary. The stack is fully local using an open source LLM and embedding models and is run via Ollama on the user's machine.

```
User (browser)
     │
     ▼
Next.js frontend (localhost:3000)
     │  HTTP POST /api/chat
     ▼
FastAPI backend (localhost:8000)
     │
     ▼
LangChain agent orchestration   ◄────  create_agent(), SYSTEM_PROMPT (guardrails, routing rules)
     │
     ├──► Qdrant vector store (in-memory)
     │    ◄── OllamaEmbeddings (embeddinggemma, local)
     │         │
     │         ├── Semantic retriever → dream_archive_search (Tool 1, baseline)
     │         │   cosine similarity search, score-based filtering
     │         │
     │         └── Hybrid retriever → dream_archive_search (Tool 1, upgraded)
     │             semantic search + lexical keyword search = fused
     │
     └──► tavily_dream_info (Tool 2 — public web search via Tavily API)
     │
     ▼
ChatOllama (gpt-oss:20b, local via Ollama)
     │  generates final response from tool output
     ▼
FastAPI → Next.js → User
```


Tooling choices:

- LLM — `gpt-oss:20b` via Ollama. Open-source, runs locally — no user data sent externally.
- Embedding model — `embeddinggemma` via Ollama. Open-source, lightweight model, runs locally - good semantic similarity for dream narration.
- Agent orchestration — LangChain `create_agent()`. Handles agentic RAG and tool routing.
  - Tool 1 — `dream_archive_search`. Retrieves from private archive using lexical + semantic hybrid retrieval.
  - Tool 2 — `tavily_dream_info`  via Tavily API. Handles general questions about dreaming and dream journaling.
- Vector database — Qdrant (in-memory) - fully local; sufficient for prototype.
- Backend — FastAPI . Lightweight Python API.
- Frontend — Next.js. Served locally on user's machine; minimal chat UI for prototype.
- Evaluation — RAGAS. Standard RAG evaluation framework with LLM as judge.
- Deployment — Local only as a privacy requirement — user data does not leave the device.

RAG components:
- Retriever: hybrid retriever combining Qdrant semantic search and in-memory lexical keyword scan, fused by result de-duplication preferring semantic hits
- Generator: `ChatOllama (gpt-oss:20b)` summarises raw retrieved dream content into a natural language response

Agent components:
- `create_agent()` with two tools
- `SYSTEM_PROMPT` with routing rules, guardrails against interpretation, and response format guidance
- The agent decides per query whether to call `dream_archive_search`, `tavily_dream_info`, or respond directly (e.g. for guardrail cases)


## 3. Data + Chunking

Data: For the prototype, 32 synthetic dream journal entries, written as a single PDF (`data/dream_entries.pdf`). Each entry includes a date and a first-person dream narrative (100–300 words). The entries feature recurring motifs (water, houses, bridges, clocks, trains) to support retrieval and pattern evaluation.

Chunking strategy: The document is split one dream entry per chunk, using `"Dream "` as the separator in `RecursiveCharacterTextSplitter`. Chunk overlap is set to 0. This decision was made because the unit of retrieval should be a complete dream entry. Splitting a dream mid-narrative would break the semantic coherence of the entry (the beginning and end of a single dream would appear in different chunks and could be retrieved independently, degrading retrieval precision and response quality. Using entry boundaries as natural split points keeps each retrieved chunk self-contained and meaningful.

External API: Tavily Search API (`tavily_dream_info` tool). Used for general questions about dreaming and dream journalling (e.g. *"Is it helpful to keep a dream journal?"*). Tavily is not called for queries about the user's specific dreams. The tool was included in the prototype to meet the certification challenge requirements.

## 4. End-to-End Prototype

Full agent loop (LLM → tool call → retrieval → LLM summarises → response) is implemnted in the prototype:

- FastAPI backend (`api/index.py`) initialises the hybrid RAG pipeline and agent on startup
- LangChain `create_agent()` with two `@tool` functions handles user queries
- Next.js frontend (`frontend/`) provides a chat UI served at `localhost:3000`

NOTE: Public deployment was not implemented by design as this would violate the privacy-first assumption of this prototype.

## 5. Evaluation

Evaluation was run using the RAGAS framework with LLM as a judge.

The golden test set (10 manually defined cases) covers: positive retrieval (objects, motifs, animals), negative retrieval (no hallucination on absent terms), pattern aggregation, guardrail enforcement (interpretation refusal), and out-of-scope query handling.

Metrics evaluated:

- Faithfulness — checks whether the answer sticks to retrieved context
  Answer Relevancy — checks if the answer is relevant to the question asked
- Context Precision— checks how how much of the retrieved context is actually useful
- Context Recall — checks if  the relevant chunks were retrieved
- Noise Sensitivity — checks how much irrelevant retrieved context affects the answer

Baseline results (semantic retriever):

| Metric | Score |
|---|---|
| Faithfulness | 0.50 |
| Answer Relevancy | 0.38 |
| Context Precision | 0.60 |
| Context Recall | 0.45 |
| Noise Sensitivity | 0.44 |


## 6. Retriever Upgrade

Advanced retrieval technique: Hybrid retrieval (semantic + lexical keyword scan)

Semantic similarity alone struggles with exact keyword queries. Embedding models represent meaning in a continuous space, which is good for thematic queries ("recurring locations") but can miss exact term matches for rare or specific words ("fire", "train", "clock") if their cosine similarity scores fall below the retrieval threshold. BM25 lexical search is strong at exact matches but misses semantic variation. Combining both approaches improves recall for specific keyword queries while retaining semantic breadth for thematic ones.

Implementation:
- Semantic component: Qdrant similarity search with score-based filtering (minimum threshold 0.2; relative filter at 75% of top score)
- Lexical component: in-memory token scan across all loaded dream chunks for exact query term matches
- Fusion: de-duplicated merge preferring semantic hits, up to `k=3` final chunks

Hybrid results (agentic pipeline):

| Metric | Baseline (semantic) | Hybrid (semantic + lexical) |
|---|---|---|
| Faithfulness | 0.50 | **0.55** |
| Answer Relevancy | 0.38 | 0.37 |
| Context Precision | **0.60** | 0.50 |
| Context Recall | **0.45** | 0.35 |
| Noise Sensitivity | **0.44** | 0.43 |

## 7. Next Steps

- Symbol extraction: detect recurring symbols explicitly using a classifier rather than relying on keyword matching
- Emotional texture retrieval: embed and retrieve by emotional tone in addition to content keywords
- Track symbol or motif evolution over time: *"Do I always dream of water when I'm anxious?"*
- Dashboard view with statistics: most frequent locations, symbols, time-of-year patterns
- Visualisations of recurring patterns across the archive
- Timeline view of retrieved entries
- Allow user to load their own dream journal PDFs directly from the UI
- Replace Tavily with a fully local alternative (e.g. a local web scraper) to eliminate the only remaining external API call
- Explore real dream journal corpora from open archives (e.g. DreamBank) as an optional data source for users without their own archive

- Share with open-source self-analysis communities for feedback and co-development

## How to Run

### Repo layout
- `agent.py` — RAG pipeline, retrieval logic, tools, agent
- `api/index.py` — FastAPI backend
- `frontend/` — Next.js web UI
- `data/` — dream PDFs
- `evals/` — RAGAS evaluation notebook and results

### Prerequisites
- Python 3.11–3.12
- `uv` for dependency management
- Node.js (for frontend)
- Ollama for local model inference

### Setup

**1. Install Python dependencies**
```bash
uv sync
```

**2. Install frontend dependencies**
```bash
cd frontend && npm install
```

**3. Pull required Ollama models**
```bash
ollama pull gpt-oss:20b      # chat model (~13GB); substitute llama3.2:3b (~2GB)
ollama pull embeddinggemma   # embedding model (~622MB)
```

**4. Set environment variables**

Create a `.env` file at project root:
```
OPENAI_API_KEY=...   # required for RAGAS evaluation only
TAVILY_API_KEY=...   # required for Tavily web search tool
```

### Run locally

**Terminal 1 — start Ollama:**
```bash
ollama serve
```

**Terminal 2 — start FastAPI backend:**
```bash
uv run uvicorn api.index:app --reload
```

**Terminal 3 — start frontend:**
```bash
cd frontend && npm run dev
```

Open `http://localhost:3000` in your browser.

**Test endpoints:**
```bash
curl http://localhost:8000/api/health

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Have I dreamt about fire?"}'
```
