"""
Agentic orchestration backend for the DreamNest app.
Runs locally. Uses Ollama for LLM and embedding models. Qdrant for in-memory vector store.
Includes a baseline retriever (semantic search only) and a hybrid retriever (semantic + lexical search with bm25).
"""

## Imports
# Langchain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_community.retrievers import BM25Retriever
from langchain_tavily import TavilySearch

# Qdrant
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# For files 
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import os
from getpass import getpass
load_dotenv()

## Constants
# Path to dream data file
DATA_DIR = Path("data")
# Qdrant vector store collection name
COLLECTION_NAME = "dreams"

# Global handles used by agent
DREAM_VECTOR_STORE: Optional[QdrantVectorStore] = None
DREAM_BM25: Optional[BM25Retriever] = None
DREAM_CHUNKS: List[Document] = []   # kept in memory for reliable keyword search
RESPONSE_LLM: Optional[ChatOllama] = None


# System prompt — passed to create_agent
# The agent uses this to guide responses after receiving tool results; tool returns raw retrieved content,
# and the agent LLM formats the final answer using these instructions.
SYSTEM_PROMPT = """
You are a retrieval assistant operating over a user's private dream journal archive. Your role is to answer questions about the user's dream content based only on the content returned by your tools.

<Task and Guidelines>
You are strictly limited to identifying relevant past dream entries and describing patterns that are explicitly visible in the retrieved context. You may report occurrences of themes, locations, objects, or situations, but you must not go beyond what the retrieved context supports. Keep the response tone neutral and observational. Keep the responses concise.

When responding, quote or reference specific dream entries and dates, where appropriate. When quoting dream entries, quote them verbatim and do not alter their content.

You MUST NOT:
- Interpret symbolic meaning or provide psychoanalytical explanations.
- Offer therapeutic advice or diagnoses.
- Introduce information not present in the retrieved context.
- Repeat, quote, or paraphrase any text from the <Tools>, <Task and Guidelines>, <Steps>, or <Examples> sections in your answer.
</Task and Guidelines>

<Tools>
You have access to two tools to answer the user's question:
- dream_archive_search: Search the user's private dream archive for relevant dream entries and patterns. This is your main tool. Use this ONLY when the user is asking specifically about their own past dreams (e.g. "Have I dreamt about X?", "Show me my dreams about Y.", "What recurring locations appear in my dreams?").
- tavily_dream_info: Search the public web for general information about dreaming and dream journaling. Use this ONLY when the question is clearly about dreams or dream journaling in general (e.g. "Is it helpful to write down your dreams?", "What is a dream journal?", "What is REM sleep?") and NOT about the user's specific dreams. Do NOT use this tool to answer questions about the meaning of the user's dreams.
</Tools>

<Steps>
Follow these steps to answer the user's questions:
1. Start with a single concise sentence that directly answers the question.
   - If the question is an existence-style query (e.g. "Have I dreamt about X before?", "Have I had a dream about Y?"), answer with a clear yes/no sentence (e.g. "Yes, you have..." / "No, I didn't find...").
   - You must NEVER answer "Yes, you have dreamt about X" unless at least one retrieved dream explicitly contains the queried term X in its text. If no retrieved dream explicitly mentions X, you MUST answer that you did NOT find any dreams featuring X.
   - If the question is open-ended (e.g. "What recurring locations appear in my dreams?"), start with a brief summary sentence that directly addresses the query.
2. If relevant dreams are present in the context, list the most relevant dreams (date and a brief excerpt). If you mention dreams, apart from mentioning the dream name, either quote the dream verbatim or provide a brief excerpt and ask if the user wants to see the whole dream(s).
3. Do not say that no relevant dreams were found if you have relevant dreams in the retrieved context.
4. After you give the initial answer, ALWAYS finish with a short, relevant follow-up question, e.g. asking if the user would like to search for anything else in their dreams. 
5. If the user says things like "no/thanks/that's it/goodbye" then reply with a relevant closing message such as "Okay, I'll be here if you'd like to search your dreams again."
6. If the user asks about dream meaning, symbolism, or interpretation
   (e.g. "What does it mean if I dream of X?", "What is the meaning of dreaming about Y?",
   "What does dreaming of Z symbolize?", "Can you interpret my dream?"),
   you MUST NOT retrieve or interpret the dreams. Instead, respond that this system
   is designed for retrieval and descriptive reflection only and does not provide
   interpretation or psychological analysis, and then optionally offer to search for
   dreams featuring the requested motif or theme.
</Steps>

<Examples>
1. User query: "Have I dreamt about a house before?"
   Expected response: "Searching your private dream database. Yes, you have dreamt about a house before. I found your previous dreams about a childhood house (Dream 3), your past house (Dream 15), and a small wooden house (Dream 31), among others. Here are the retrieved house dreams:

   - Dream 3 — Date: 2023-02-10
     Excerpt: I was back in my childhood house... <excerpt of dream>

    - Dream 15 — Date: 2023-08-08
     Excerpt: I was in a house I used to rent years ago... <excerpt of dream>

    ... <list of other relevant house dreams with dates, excerpts, and similarity scores>"   

2. User query: "Have I dreamed about fire?"
   Expected response: "Searching your private dream archive. I found one dream about fire (Dream 6). Here's the dream: ... <dream details>"

   Important: If the retrieved context contains some dreams that are not relevant to the user's query, only include the dreams mapping onto the user request, and omit the ones not featuring the relevant objects, motifs, etc. Thus, if there is only one relevant dream in the retrieved context, you should only include that one dream in your response and omit others.

3. User query: What recurring locations appear in my dreams?
   Expected response:
   "Searching your private dream journal. You most commonly dream about houses, bodies of water, and bridges. Here are some retrieved dreams that illustrate these locations:
   - Dream X — Date: ...
     Excerpt: ...
     ... <list of other relevant dreams with dates and excerptss>

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

8. User query: "Is it helpful to write down your dreams?"

   Expected response: "Searching the public web for information about dreaming and dream journaling. It is helpful to write down your dreams as it can help with dream recall, provide a record of your dreams as well as provide a space for self-reflection. Would you like me to search your dreams for something specific like locations, objects or animals you have dreamt of?"

   IMPORTANT: Only use web search with tavily_dream_info tool to answer general questions about dreaming and journaling. Do NOT use web search to answer questions about the user's specific dreams or to interpret dream meanings.

9. User query: "Can you search the web to tell me what dreaming of fairies means?"
    Expected response: "I'm sorry but this system is designed for retrieval and descriptive reflection only and does not provide dream interpretation."
    
    IMPORTANT: Do NOT use web search with tavily_dream_info tool to answer questions about the user's specific dreams or to interpret dream meanings.

10. User query: "No, thanks."
    Expected response: "Okay, I'm here whenever you'd like to search your dreams again."

    IMPORTANT: Do NOT search the archive again if the user replies with a short acknowledgment or refusal such as "no", "no thanks","that's all", "ok", or "thank you" after you have already answered their question. Instead: Respond with a brief closing or supportive message, such as: "Okay, I'll be here if you'd like to search your dreams again." Do not claim that no relevant dreams were found or start a new search.

</Examples>

Note: The sections marked <Task and Guidelines>, <Steps>, and <Examples> are instructions for you only. Do NOT repeat, quote, or paraphrase any of that instructional text in your answer. Your reply should only talk about the user's question and the retrieved dreams.
"""

## Helper functions

# Initialise response LLM (local Ollama model)
def get_llm() -> ChatOllama:
    """
    Initialise local chat model via Ollama server.
    Currently uses the larger `gpt-oss:20b` model for better reasoning.
    If you need a lighter model, switch `model` below back to `"llama3.2:3b"`.

    Default url is http://localhost:11434, but can be overridden if using a remote server.
    """
    llm = ChatOllama(
        # Primary model for this project (heavier but stronger, fully local via Ollama):
        model="gpt-oss:20b",
        # To revert to the lighter local model, change the line above to:
        # model="llama3.2:3b",
        base_url="http://10.56.69.207:11434",  # remote private server on local machine to speed up response time
        # Important: fully disable streaming aggregation to avoid
        # `No data received from Ollama stream` errors in langchain-ollama.
        disable_streaming=True,
    )
    global RESPONSE_LLM
    RESPONSE_LLM = llm
    return llm


# TEMPORARY helper for debugging: use OpenAI model instead of local Ollama.
# This is ONLY to prove that the FastAPI + agent + tools wiring is correct.
# To restore the privacy-first local setup, switch FastAPI's startup back
# to `get_llm()` and stop using this function.
def get_llm_openai() -> ChatOpenAI:
    """
    Initialise an OpenAI-hosted chat model for debugging.

    This uses `langchain_openai.ChatOpenAI` and relies on the environment
    variable OPENAI_API_KEY being set. It is NOT privacy-first, since
    prompts and context leave the local machine.

    Use this only to confirm that the agent + tools stack works independently
    of Ollama's streaming behaviour, then switch back to `get_llm()`.
    """
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )
    global RESPONSE_LLM
    RESPONSE_LLM = llm
    return llm


# Initialise embedding model 
def get_embeddings() -> OllamaEmbeddings:
    """
    Initialise embedding model via Ollama.
    Default url is http://localhost:11434, but can be overridden if using a remote server.
    Returns the embeddoing model for Qdrant vector store"""
    return OllamaEmbeddings(
        model="embeddinggemma", 
        base_url="http://10.56.69.207:11434", # remote server on local machine to speed up response time
    )


# Prepare data (load and chunk)
def load_and_chunk_dreams() -> List[Document]:
    """
    Loads the dream pdf file from DATA_DIR and splits it into chunks for storage and retrieval.
    Dreams are split as coherent narrative chunks, separated by "Dream <number>" markers.
    Eeach dream is a separate chunk for response cohesion.
    Uses a RecursiveCharacterTextSplitter (character-based).
    Returns a list of Document objects, each representing a chunk.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dream data directory '{DATA_DIR}' not found. ")

    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=0,separators=["Dream "])

    return splitter.split_documents(raw_docs)


# Build Qdrant in-memory vector store
def build_vector_store(chunks: List[Document]) -> QdrantVectorStore:
    """
    Creates an in-memory Qdrant vector store and populates it with dream chunks.
    Takes dream chunks, creates embeddings, stores in Qdrant.
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


# lexical retriever (bm25)
def build_bm25_retriever(chunks: List[Document]) -> BM25Retriever:
    """ 
    Builds the lexical retriever (bm25) using the chunks.
    Lexical retriever searches for exact keyword matches in the chunks.
    k=20 so we get enough candidates before applying keyword filter.
    Returns the BM25Retriever object.    
    """
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 20
    return retriever


# initialise RAG pipeline - baseline semantic retrieval
def init_pipeline_semantic() -> QdrantVectorStore:
    """
    Initialises RAG pipeline with semantic retrieval.
    Builds and populates the vector store.
    Returns the vector store object.
    """
    chunks = load_and_chunk_dreams()
    vector_store = build_vector_store(chunks)

    global DREAM_VECTOR_STORE, DREAM_CHUNKS
    DREAM_VECTOR_STORE = vector_store
    DREAM_CHUNKS = []   # no keyword scan for semantic-only baseline

    return vector_store


# Initialise RAG pipeline - upgraded hybrid retrieval
def init_pipeline_hybrid() -> tuple[QdrantVectorStore, BM25Retriever]:
    """
    Initialises the RAG pipeline using both semantic and lexical retrievers.
    Builds and populates the vector store.
    Builds the lexical retriever (bm25) using the chunks.
    Returns the vector store and bm25 retriever objects.
    """
    chunks = load_and_chunk_dreams()
    vector_store = build_vector_store(chunks)
    bm25_retriever = build_bm25_retriever(chunks)

    # expose for tools/agent
    global DREAM_VECTOR_STORE, DREAM_BM25, DREAM_CHUNKS
    DREAM_VECTOR_STORE = vector_store
    DREAM_BM25 = bm25_retriever
    DREAM_CHUNKS = chunks

    return vector_store, bm25_retriever


# RAG answer function (semantic only - baseline)
# used by fastapi app to answer user questions
def answer_dream_query_semantic(
    query: str,
    vector_store: QdrantVectorStore,
    llm: ChatOllama,
) -> str:
    """
    Uses  vector store to retrieve similar dream chunks.
    Applies a similarity threshold to avoid weak matches.
    Formats context and passes to the LLM 
    Returns the LLM response.
    """
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    if not results_with_scores:
        return "No strong matches were found in your dream archive for this query."

    # Tresholding logic to filter out weaker matches
    # only keep chunks with a similarity score >= 0.4
    top_doc, top_score = results_with_scores[0]
    if top_score < 0.4:
        return "No strong matches were found in your dream archive for this query."
    # only keep chunks very close to the best score >= 80% of top_score
    min_score = max(0.45, top_score * 0.80)
    filtered_results = [
        (doc, score) for doc, score in results_with_scores if score >= min_score
    ]
    if not filtered_results:
        return "No strong matches were found in your dream archive for this query."

    # format chunks for the response
    formatted_chunks: list[str] = []
    for idx, (doc, score) in enumerate(filtered_results, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", "unknown page")
        formatted_chunks.append(
            f"Retrieved dream {idx} (Source: {source}, Page: {page}):\n"
            f"{doc.page_content.strip()}"
        )
    context = "\n\n".join(formatted_chunks)
    # Pass system prompt + user query + retrieved context as a message list
    from langchain_core.messages import SystemMessage, HumanMessage as HMsg
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HMsg(content=f"Query: {query}\n\nRetrieved context:\n{context}"),
    ]
    response = llm.invoke(messages)
    return response.content


# Hybrid retriever with bm25
def hybrid_retrieve(
    query: str,
    vector_store: QdrantVectorStore,
    bm25_retriever: BM25Retriever,
    k_semantic: int = 5,
    k_lexical: int = 5,
    k_final: int = 3,
) -> List[Document]:
    """
    Hybrid retrieval combining lexical and semantic search.
    - Gets top-k semantic matches from Qdrant and applies the same thresholding
      logic as the baseline semantic retriever.
    - Gets top-k lexical matches with BM25 and applies a simple keyword filter
      so that the raw query terms appear in the retrieved chunks.
    - Fuses results to return a de-duplicated list preferring lexical hits over
      semantic hits, up to k_final results.
    """
    # Semantic results with score-based thresholding
    results_with_scores = vector_store.similarity_search_with_score(query, k=k_semantic)

    top_score = results_with_scores[0][1] if results_with_scores else 0
    # Keep semantic docs if scores are above the minimum threshold.
    # Threshold is 0.2 (not 0.4) because embeddinggemma produces scores in
    # the 0.25-0.45 range on this corpus — a strict 0.4 cutoff blocks valid results.
    # Relative filter keeps only docs within 75% of the top score.
    if top_score >= 0.2:
        min_score = top_score * 0.75
        semantic_docs: List[Document] = [
            doc for doc, score in results_with_scores if score >= min_score
        ]
    else:
        # Semantic scores very low — rely on keyword scan below
        semantic_docs = []

    # Keyword scan: search all chunks for exact token matches.
    # Used as the lexical component of hybrid retrieval.
    query_tokens = [t for t in query.lower().split() if len(t) > 2 and t.isalpha()]
    if query_tokens:
        lexical_docs: List[Document] = [
            doc for doc in DREAM_CHUNKS
            if any(tok in doc.page_content.lower() for tok in query_tokens)
        ][:k_lexical]
    else:
        lexical_docs = []

    # Fuse: semantic first (higher quality), then BM25 to fill gaps
    seen = set()
    fused: List[Document] = []

    for doc in semantic_docs + lexical_docs:
        key = doc.page_content
        if key in seen:
            continue
        seen.add(key)
        fused.append(doc)
        if len(fused) >= k_final:
            break

    return fused


# RAG answer function (lexical and semantic search - hybrid)
def answer_dream_query_hybrid(
    query: str,
    vector_store: QdrantVectorStore,
    bm25_retriever: BM25Retriever,
    llm: ChatOllama,
) -> str:
    """
    Answers the user's query using the hybrid retriever.
    Gets the top-k lexical and semantic matches. 
    Formats results and passes to LLM.
    Returns the LLM response.
    """
    # fused docs
    docs = hybrid_retrieve(query, vector_store, bm25_retriever)
    if not docs:
        return "No strong matches were found in your dream archive for this query."

    # format chunks for the response
    formatted_chunks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", "unknown page")
        formatted_chunks.append(
            f"Retrieved dream {idx} (Source: {source}, Page: {page}):\n"
            f"{doc.page_content.strip()}"
        )
    context = "\n\n".join(formatted_chunks)
    from langchain_core.messages import SystemMessage, HumanMessage as HMsg
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HMsg(content=f"Query: {query}\n\nRetrieved context:\n{context}"),
    ]
    response = llm.invoke(messages)
    return response.content


## Agent's tools

class _ArchiveQuery(BaseModel):
    query: str = Field(description="Plain text search query, e.g. 'fire', 'house', 'recurring water dreams'")

class _WebQuery(BaseModel):
    query: str = Field(description="Plain text search query about dreaming or journaling")

@tool("dream_archive_search", args_schema=_ArchiveQuery)
def dream_archive_search(query: str) -> str:
    """
    Search the user's private dream archive for relevant dream entries and patterns.
    Use this tool for ANY question about the user's own past dreams — motifs, locations,
    objects, recurring themes, specific events. Always call this tool for questions like
    "Have I dreamt about X?", "Find dreams about Y", "What recurring themes appear in my dreams?".

    Returns the raw retrieved dream entries for the agent to summarise.
    """
    if DREAM_VECTOR_STORE is None:
        raise RuntimeError("Dream pipeline not initialised. Call init_pipeline_hybrid() first.")

    print(f"[TOOL] dream_archive_search called — query: {query!r}")
    # Hybrid retrieval: semantic (Qdrant) + lexical keyword scan (in-memory).
    # Combines semantic similarity with exact keyword matching so that both
    # thematic queries ("recurring locations") and specific keyword queries
    # ("fire", "whale", "clocks") are handled reliably.
    docs = hybrid_retrieve(query, DREAM_VECTOR_STORE, DREAM_BM25)

    if not docs:
        return "No relevant dreams found in the archive for this query."

    # Return raw formatted dream content — the agent LLM formats the final response
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Dream {i} — source: {source}, page: {page}]\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


# Tavily tool for external search (for general info about dreaming)
@tool("tavily_dream_info", args_schema=_WebQuery)
def tavily_dream_info(query: str) -> str:
    """
    Search the public web for general information about dreaming and dream journaling.

    Use this tool ONLY for general, public information about sleep, dreaming, or journaling
    (e.g. what is a dream journal, basic sleep hygiene, benefits of keeping a dream diary).
    Do NOT use this tool to answer questions about the user's specific dreams or to interpret
    dream meanings.
    """
    print(f"[TOOL] tavily_dream_info called — query: {query!r}")
    client = TavilySearch(max_results=3)
    return client.invoke(query)

## Agent
# Create the RAG agent with tools
def build_retrieval_agent(llm: Optional[ChatOllama] = None):
    """
    Creates an agent that can call the following tools:
    - dream_archive_search (private RAG over the user's dream journal)
    - tavily_dream_info (general info about dreaming / journaling)

    Assumes init_pipeline_semantic()/init_pipeline_hybrid() and get_llm()
    have been called beforehand if llm is None.
    """
    agent_llm = llm or RESPONSE_LLM
    if agent_llm is None:
        raise RuntimeError(
            "LLM is not initialised. Provide an llm or call get_llm() first."
        )

    tools = [dream_archive_search, tavily_dream_info]
    # One system prompt passed to create_agent — same pattern as the lesson code.
    # The tool returns raw retrieved content; the agent LLM uses SYSTEM_PROMPT to format the answer.
    return create_agent(agent_llm, tools=tools, system_prompt=SYSTEM_PROMPT)