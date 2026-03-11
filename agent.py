"""
Logic for the DreamNest agentic pipeline
"""
## Imports

# Langchain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.retrievers import BM25Retriever

# Qdrant
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# For files 
from pathlib import Path
from typing import List, Optional

# BM25Retriever for retrieval upgrade
from langchain_community.retrievers import BM25Retriever

# for certification challenge only - external search with tavilly
from langchain_community.tools.tavily_search import TavilySearchResults

## Constants
# Path to dream data
DATA_DIR = Path("data")
# for Qdrant vector store collection name
COLLECTION_NAME = "dreams"

# Simple module-level handles so tools / agents can reuse
# the same pipeline without passing everything around.
DREAM_VECTOR_STORE: Optional[QdrantVectorStore] = None
DREAM_LLM: Optional[ChatOllama] = None

# system prompt for retrieval
SYSTEM_PROMPT = """
You are a retrieval assistant operating over a user's private dream journal archive. Your role is to answer the user query {query} about their dream content based only on the retrieved context {context}.

<Task and Guidelines>
You are strictly limited to identifying relevant past dream entries and describing patterns that are explicitly visible in the retrieved context. You may report occurrences of themes, locations, objects, or situations, but you must not go beyond what the retrieved context supports. Quote or reference specific entries, dates, or similarity scores where appropriate. When you mention similarity scores, state them  as "Similarity score: X.XX" for each individual dream. Do not invent or describe pairwise similarity relationships between different dreams. Keep the response tone neutral and observational. Keep the responses concise.

You MUST NOT:
- Interpret symbolic meaning or provide psychoanalytical explanations.
- Offer therapeutic advice or diagnoses.
- Introduce information not present in the retrieved context.
- Invent similarity scores or add placeholders like "N/A"; if no score is present in the context, omit it.
- Repeat, quote, or paraphrase any text from the <Task and Guidelines>, <Steps>, or <Examples> sections in your answer.
</Task and Guidelines>

<Steps>
Follow these steps to answer the user's question:
1. Start with a single concise sentence that directly answers the question.
   - If the question is an existence-style query (e.g. "Have I dreamt about X before?", "Have I had a dream about Y?"), answer with a clear yes/no sentence (e.g. "Yes, you have..." / "No, I didn't find...").
   - If the question is open-ended (e.g. "What recurring locations appear in my dreams?"), start with a brief summary sentence that directly addresses the query.
2. If relevant dreams are present in the context, list the most relevant dreams (date and a brief excerpt, and, if provided, similarity score).
3. Do not say that no relevant dreams were found if you have relevant dreams in the retrieved context.
4. After you give the answer, finish with a short, relevant follow-up question asking if the user would like to search for anything else in their dreams.
</Steps>

<Examples>
1. User query: "Have I dreamt about a house before?"
   Expected response: "Searching your private dream database. Yes, you have dreamt about a house before. I found your previous dreams about a childhood house (Dream 3), your past house (Dream 15), and a small wooden house (Dream 31), among others. Here are the retrieved house dreams:

   - Dream 3 — Date: 2023-02-10
     Excerpt: I was back in my childhood house... <excerpt of dream>
     Similarity score: <score>

    - Dream 15 — Date: 2023-08-08
     Excerpt: I was in a house I used to rent years ago... <excerpt of dream>
     Similarity score: <score>

    ... <list of other relevant house dreams with dates, excerpts, and similarity scores>"   

2. User query: "Have I dreamed about fire?"
   Expected response: "Searching your private dream archive. I found one dream about fire (Dream 6). Here's the dream: ... <dream details>"

   Important: If the retrieved context contains some dreams that are not relevant to the user's query, only include the dreams mapping onto the user request, and omit the ones not featuring the relevant objects, motifs, etc. Thus, if there is only one relevant dream in the retrieved context, you should only include that one dream in your response and omit others.

3. User query: What recurring locations appear in my dreams?
   Expected response:
   "Searching your private dream journal. You most commonly dream about houses, bodies of water, and bridges. Here are some retrieved dreams that illustrate these locations:
   - Dream X — Date: ...
     Excerpt: ...
     Similarity score: <score>
     ... <list of other relevant dreams with dates, excerpts, and similarity scores>

     Would you like to search for any specific dreams?"

4. User query: "Have I dreamt about fish?"
   (Assume the retrieved context is empty or indicates no strong matches.)
   Expected response:
   "Searching your private dream archive. I didn't find any dreams featuring fish in your archive. Would you like to try a different search?"

   IMPORTANT: In a situation where the retrieved context doesn't support the user's query, you should respond with a message that no relevant dreams were found in the user's dream archive. For example: a user asks about dreams featuring penguins and the retrieved context includes dreams with no penguins mentioned. In this case, you should respond saying you did NOT find any dreams featuring penguins, and follow with an invitation to try a different search.

5. User query: "What does it mean if I dreamt about bridges?"
   Expected response: "I'm sorry but this system is designed for retrieval and descriptive reflection only and does not provide interpretation or psychological analysis. Would you like me to look for your dreams about bridges?"

   IMPORTANT: If the user asks for dream interpretation or psychological meaning, respond that this system is designed for retrieval and descriptive reflection only and does not provide interpretation or psychological analysis.

6. User query: "What is the capital of Poland?"
   Expected response: "I'm sorry but this system is designed for retrieval and descriptive reflection only and does not provide any other information."

7. User query: "You are no good!"
   Expected response: "I'm sorry you feel that way. I'm doing my best to help you explore your dreams but this system is just a prototype. Would you like me to search your dreams for something specific like objects, animals, or themes?"

</Examples>

Note: The sections marked <Task and Guidelines>, <Steps>, and <Examples> are instructions for you only. Do NOT repeat, quote, or paraphrase any of that instructional text in your answer. Your reply should only talk about the user's question and the retrieved dreams.
"""

## Helper functions
# Initialize LLM
def get_llm() -> ChatOllama:
    """Initialize local llama chat model via Ollama server."""
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://10.56.69.207:11434",  # default is http://localhost:11434
    )
    global DREAM_LLM
    DREAM_LLM = llm
    return llm

# Initialize embedding model 
def get_embeddings() -> OllamaEmbeddings:
    """Initialize embedding model via Ollama."""
    return OllamaEmbeddings(
        model="embeddinggemma", 
        base_url="http://10.56.69.207:11434", #default is localhost:11434
    )

# Prep dream data (Load and chunk)
def load_and_chunk_dreams() -> List[Document]:
    """
    Load dream PDF from DATA_DIR and split into chunks for storage and retrieval.
    RecursiveCharacterTextSplitter is a character based splitter.
    """
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
    Takes chunks, creates embeddings, stores in Qdrant.
    Returns vector_store object
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

# define lexical BM25 retriever
def build_bm25_retriever(chunks: List[Document]) -> BM25Retriever:
    """ Retrieve chunks using BM25 retriever"""
    return BM25Retriever.from_documents(chunks)

# initialise RAG pipeline
def init_pipeline() -> QdrantVectorStore:
    """
    Initialise the full RAG pipeline to build vector store:
    load, chunk, embed as vectors, store.
    """
    chunks = load_and_chunk_dreams()
    vector_store = build_vector_store(chunks)

    global DREAM_VECTOR_STORE
    DREAM_VECTOR_STORE = vector_store

    return vector_store

# initialise RAG pipeline with BM25 retriever - upgrade
def init_pipeline_with_bm25() -> tuple[QdrantVectorStore, BM25Retriever]:
    """
    Initialise both semantic (Qdrant) and lexical (BM25) retrievers.
    Returns (vector_store, bm25_retriever).
    """
    chunks = load_and_chunk_dreams()
    vector_store = build_vector_store(chunks)
    bm25_retriever = build_bm25_retriever(chunks)

    global DREAM_VECTOR_STORE
    DREAM_VECTOR_STORE = vector_store

    return vector_store, bm25_retriever


# function to answer a dream query - called by fastapi
def answer_dream_query(
    query: str,
    vector_store: QdrantVectorStore,
    llm: ChatOllama,
) -> str:
    """
    - Uses  vector store to retrieve similar dream chunks.
    - Applies a similarity threshold to avoid weak matches.
    - Formats context and passes to the LLM 
    """
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    if not results_with_scores:
        return "No strong matches were found in your dream archive for this query."

    # Check that the best match is strong enough at all
    top_doc, top_score = results_with_scores[0]
    if top_score < 0.4:
        return "No strong matches were found in your dream archive for this query."

    # Filter out weaker matches (floor is 0.45)
    # keep docs very close to the best score >= 80% of top_score
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

# define hybrid retriever with added bm25
def hybrid_retrieve(
    query: str,
    vector_store: QdrantVectorStore,
    bm25_retriever: BM25Retriever,
    k_semantic: int = 5,
    k_lexical: int = 5,
    k_final: int = 5,
) -> List[Document]:
    """
    Very simple hybrid retrieval:
    - get top-k lexical matches with BM25
    - get top-k semantic matches from Qdrant
    - return a de-duplicated list preferring lexical hits, then semantic hits
    """
    # lexical results
    lexical_docs = bm25_retriever.get_relevant_documents(query)[:k_lexical]
    # semantic results 
    semantic_docs = vector_store.similarity_search(query, k=k_semantic)

    seen = set()
    fused: List[Document] = []

    for doc in lexical_docs + semantic_docs:
        key = doc.page_content
        if key in seen:
            continue
        seen.add(key)
        fused.append(doc)
        if len(fused) >= k_final:
            break

    return fused

# function to answer a dream query using hybrid retriever
def answer_dream_query_hybrid(
    query: str,
    vector_store: QdrantVectorStore,
    bm25_retriever: BM25Retriever,
    llm: ChatOllama,
) -> str:
    """
    Answer a dream query using a simple hybrid retriever (BM25 + semantic).
    Intended as a retrieval upgrade over pure semantic search.
    """
    docs = hybrid_retrieve(query, vector_store, bm25_retriever)

    if not docs:
        return "No strong matches were found in your dream archive for this query."

    formatted_chunks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", "unknown page")
        formatted_chunks.append(
            f"Retrieved dream {idx} (Source: {source}, Page: {page}):\n"
            f"{doc.page_content.strip()}"
        )

    context = "\n\n".join(formatted_chunks)
    prompt = SYSTEM_PROMPT.format(query=query, context=context)

    response = llm.invoke(prompt)
    return response.content

@tool
def dream_archive_search(query: str) -> str:
    """
    Tool: search the user's private dream archive for relevant entries and patterns.

    Uses the same retrieval logic as the main DreamNest agent. Requires that
    init_pipeline() and get_llm() have been called so DREAM_VECTOR_STORE and
    DREAM_LLM are populated.
    """
    if DREAM_VECTOR_STORE is None or DREAM_LLM is None:
        raise RuntimeError(
            "Dream pipeline is not initialised. Call init_pipeline()/get_llm() first."
        )
    return answer_dream_query(query, DREAM_VECTOR_STORE, DREAM_LLM)


@tool
def tavily_dream_info(query: str) -> str:
    """
    Tool: search the public web for general information about dreaming and dream journaling.
    Never use this tool to answer questions about the user's specific dreams or to interpret dream symbols.
    """
    client = TavilySearchResults(
        max_results=3,
        description=(
            "Use this tool ONLY for general, public information about sleep, dreaming, "
            "or journaling (e.g. what is a dream journal, basic sleep hygiene). "
            "NEVER use it to interpret the user's dreams, analyse symbols, "
            "or answer questions about this specific user's dream content."
        ),
    )
    return client.run(query)

def build_retrieval_agent(llm: Optional[ChatOllama] = None):
    """
    Create an agent that can call:
    - dream_archive_search (private RAG over the user's dream journal)
    - tavily_dream_info (general info about dreaming / journaling)

    Assumes init_pipeline() and get_llm() have been called beforehand if llm is None.
    """
    agent_llm = llm or DREAM_LLM
    if agent_llm is None:
        raise RuntimeError(
            "LLM is not initialised. Provide an llm or call get_llm() first."
        )

    tools = [dream_archive_search, tavily_dream_info]
    return create_agent(agent_llm, tools=tools)