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

# system prompt for retrieval
SYSTEM_PROMPT = """
You are a retrieval assistant operating over a private dream journal archive. Your role is to answer the user query {query} about their dreams based only on the retrieved context {context}.

Important: The sections marked <Steps>, <Examples>, <Rules and Guidelines>, and <Hard Rules> are instructions for you only. Do NOT repeat, quote, or paraphrase any of that instructional text in your answer. Your reply should only talk about the user's question and the retrieved dreams.

<Steps>
1. Start with a single concise sentence that directly answers the question.
   - If the question is an existence-style query (e.g. "Have I dreamt about X before?", "Have I had a dream about Y?"), answer with a clear yes/no sentence (e.g. "Yes, you have..." / "No, I didn't find...").
   - If the question is open-ended (e.g. "What recurring locations appear in my dreams?"), start with a brief summary sentence that directly addresses the query.
2. If relevant dreams are present in the context, list the most relevant dreams (date and a brief excerpt, and, if provided, similarity score).
3. Do not say that no relevant dreams were found if you have any retrieved dreams in the context.
</Steps>

<Examples>
1. User query: Have I dreamt about a house before?
   Expected response:
   "Searching your private dream database. Yes, you have dreamt about a house before. You dreamt about a childhood house (Dream 3) and a small wooden house (Dream 31). Here are the retrieved dreams:

   Retrieved dreams:
   - Dream 3 — Date: 2023-02-10
     Excerpt: I was back in my childhood house...
     Similarity score: 0.78

   - Dream 31 — Date: 2024-02-03
     Excerpt: I arrived at a small wooden house...
     Similarity score: 0.85"

2. User query: Show me dreams with whales.
   (Assume the context is empty or indicates no strong matches.)
   Expected response:
   "Searching your private dream database. I didn't find any dreams featuring whales in your archive. Would you like to try a different search?"

3. User query: What recurring locations appear in my dreams?
   Expected response:
   "Searching your private dream archive. You most commonly dream about houses, bodies of water, and bridges. Here are some retrieved dreams that illustrate these locations:
   - Dream X — Date: ...
     Excerpt: ...
   - Dream Y — Date: ...
     Excerpt: ..."
</Examples>

<Rules and Guidelines>
You are strictly limited to identifying relevant past dream entries and describing patterns that are explicitly visible in the retrieved text. You may report occurrences of themes, locations, objects, or situations, but you must not go beyond what the retrieved text supports.

- Only use information contained in the retrieved excerpts of dream entries.
- Quote or reference specific entries, dates, or similarity scores where appropriate.
- When you mention similarity scores, state them only as "Similarity score: X.XX" for each individual dream. Do NOT invent scores or placeholders like "N/A"; if no score is present in the context, simply omit it.
- Keep the tone neutral and observational.
- Keep the response concise.

If the user asks for dream interpretation or psychological meaning, respond that this system is designed for retrieval and descriptive reflection only and does not provide interpretation or psychological analysis.
</Rules and Guidelines>

<Hard Rules>
You MUST NOT:
- Interpret symbolic meaning.
- Provide psychological or psychoanalytical explanations.
- Offer therapeutic advice or diagnoses.
- Introduce information not present in the retrieved context.
- Invent or describe pairwise similarity relationships between different dreams; you may only repeat the numeric similarity score attached to each dream in the context.
- Repeat, quote, or paraphrase any text from the <Steps>, <Examples>, <Rules and Guidelines>, or <Hard Rules> sections in your answer.
</Hard Rules>
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
    - Applies a similarity threshold to avoid weak matches.
    - Formats the context and passes it to the LLM under SYSTEM_PROMPT.
    """
    # Retrieve a small set of top semantic matches.
    # We keep k low here to reduce the chance of including loosely related dreams.
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    if not results_with_scores:
        return "No strong matches were found in your dream archive for this query."

    # Check that the best match is strong enough at all
    top_doc, top_score = results_with_scores[0]
    if top_score < 0.4:
        return "No strong matches were found in your dream archive for this query."

    # Filter out weaker matches and keep only clearly relevant dreams.
    # We keep documents that are very close to the best score (>= 80% of top_score)
    # while also enforcing an absolute floor of 0.45, so weak matches are always dropped.
    min_score = max(0.45, top_score * 0.80)
    filtered_results = [
        (doc, score) for doc, score in results_with_scores if score >= min_score
    ]

    if not filtered_results:
        return "No strong matches were found in your dream archive for this query."

    formatted_chunks: list[str] = []
    for idx, (doc, score) in enumerate(filtered_results, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", "unknown page")
        formatted_chunks.append(
            f"Retrieved dream {idx} (Source: {source}, Page: {page}, Similarity score: {score:.2f}):\n"
            f"{doc.page_content.strip()}"
        )

    context = "\n\n".join(formatted_chunks)
    prompt = SYSTEM_PROMPT.format(query=query, context=context)

    response = llm.invoke(prompt)
    return response.content