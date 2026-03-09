"""
Runs the DreamNest agent.
"""
## Imports

# Langchain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent

# Qdrant
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# For files 
from pathlib import Path
from typing import List

# other imports later
# tavily api? 
# cohere if reranker

## Constants
# Path to dream data
DATA_DIR = Path("data")
# for Qdrant vector store collection name
COLLECTION_NAME = "dreams"

# system prompt for retrieval - with guardrails
SYSTEM_PROMPT = """
You are a retrieval assistant operating overal a private dream journal archive.
Your role is strictly limited to:
- Identifying relevant past dream excerpts.
- Describing patterns that are explicitly visible in the retrieved text.
- Reporting occurence of locations, objects, themes or situations. 

You MUST NOT: 
- Interpret symbolic meaning.
- Provide psychological or psychoanalytical explanations.
- Offer therapeutic advice or diagnoses.
- Introduce information not present in the retrieved excerpts.

If the user asks for interpretation, meaning, psychological insight, or explanation, respond with:
"This system is designed for retrieval and descriptive reflection only. It does not provide interpretation or psychological analysis."

If no relevant excerpts are found, respond with:
"No strong matches were found in your dream archive for this query."

User question:
{query}

Retrieved dream excerpts:
{context}

Hard rules:
- Only use information contained in the retrieved excerpts of dream entries.
- Quote or reference specific excerpts where appropriate.
- Keep the tone neutral and observational.
- Do not speculate.
- Do not add new symbolic explanations.
- Keep the response concise.
"""

## Helper functions
# Initialize LLM
def get_llm() -> ChatOllama:
    """Initialize local llama chat model via Ollama server."""
    return ChatOllama(
        model="llama3.2:3b",
        base_url="http://10.56.69.207:11434", # default is http://localhost:11434
    )

# Initialize embedding model 
def get_embeddings() -> OllamaEmbeddings:
    """Initialize embedding model via Ollama."""
    return OllamaEmbeddings(
        model="embeddinggemma", 
        base_url="http://10.56.69.207:11434", #default is localhost:11434
    )

# Prep dream data: Load and chunk
def load_and_chunk_dreams() -> List[Document]:
    """
    Load dream PDF(s) from DATA_DIR and split into chunks for storage and retrieval.
    RecursiveCharacterTextSplitter splits docs into chunks (it's a character based splitter).
    """
    # error handling for missing data directory
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dream data directory '{DATA_DIR}' not found. ")

    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=0,separators=["Dream "])

    return splitter.split_documents(raw_docs)

# Build vector store
def build_vector_store(chunks: List[Document]) -> QdrantVectorStore:
    """
    Create in-memory Qdrant vector store and populate with dream chunks.
    Takes chunks, creates embeddings, and stores in Qdrant.
    Returns vector_stire object
    """
    embeddings = get_embeddings()
    sample = embeddings.embed_query("test")
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=len(sample), distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
    return vector_store

# call in fast api to initialize the pipeline
def init_pipeline() -> QdrantVectorStore:
    """Initialise the full RAG pipeline to build vector store: load, chunk, embed as vectors, store."""
    chunks = load_and_chunk_dreams()
    return build_vector_store(chunks)

# function to answer a dream query - is called by fastapi
def answer_dream_query(
    query: str,
    vector_store: QdrantVectorStore,
    llm: ChatOllama,
) -> str:
    """
    Core RAG helper for FastAPI.

    - Uses the vector store to retrieve similar dream chunks.
    - Applies a simple similarity threshold to avoid weak matches.
    - Formats the context and passes it to the LLM under SYSTEM_PROMPT.
    """
    results_with_scores = vector_store.similarity_search_with_score(query, k=5)

    if not results_with_scores:
        return "No strong matches were found in your dream archive for this query."

    top_doc, top_score = results_with_scores[0]
    if top_score < 0.4:
        return "No strong matches were found in your dream archive for this query."

    formatted_chunks: list[str] = []
    for idx, (doc, score) in enumerate(results_with_scores, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", "unknown page")
        formatted_chunks.append(
            f"Dream {idx} (Source: {source}, Page: {page}, Similarity: {score:.2f}):\n"
            f"{doc.page_content.strip()}"
        )

    context = "\n\n".join(formatted_chunks)
    prompt = SYSTEM_PROMPT.format(query=query, context=context)

    response = llm.invoke(prompt)
    return response.content