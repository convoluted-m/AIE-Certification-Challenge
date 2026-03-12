"""
Agentic orchestration backend for the DreamNest app.
Runs locally. Uses Ollama for LLM and embedding models. Qdrant for in-memory vector store.
Includes a baseline retriever (semantic search only) and a hybrid retriever (semantic + lexical search with bm25).
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
# Path to dream data file
DATA_DIR = Path("data")
# Qdrant vector store collection name
COLLECTION_NAME = "dreams"
# global var for dream vector store
DREAM_VECTOR_STORE: Optional[QdrantVectorStore] = None
# global var for response LLM
RESPONSE_LLM: Optional[ChatOllama] = None


# System prompt for generating responses
# Sets the agent's role and provides instructions with few-shot examples
SYSTEM_PROMPT = """
You are a retrieval assistant operating over a user's private dream journal archive. Your role is to answer the user query {query} about their dream content based only on the retrieved context {context}.

<Task and Guidelines>
You are strictly limited to identifying relevant past dream entries and describing patterns that are explicitly visible in the retrieved context. You may report occurrences of themes, locations, objects, or situations, but you must not go beyond what the retrieved context supports. Keep the response tone neutral and observational. Keep the responses concise.

When responding, quote or reference specific dream entries, dates, and similarity scores, where appropriate. When quoting dream entries, quote them verbatim and do not alter their content. When you mention similarity scores, state them as "Similarity score: X.XX" for each individual dream. Do not invent or describe pairwise similarity relationships between different dreams.

You MUST NOT:
- Interpret symbolic meaning or provide psychoanalytical explanations.
- Offer therapeutic advice or diagnoses.
- Introduce information not present in the retrieved context.
- Invent similarity scores or add placeholders like "N/A"; if no score is present in the context, omit it.
- Repeat, quote, or paraphrase any text from the <Task and Guidelines>, <Steps>, or <Examples> sections in your answer.
</Task and Guidelines>

<Steps>
Follow these steps to answer the user's questions:
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

# Initialise response LLM
def get_llm() -> ChatOllama:
    """
    Initialise local llama chat model via Ollama server.
    Creates the chat llm pointing to Ollama server.
    Default url is http://localhost:11434, but can be overridden if using a remote server
    """
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://10.56.69.207:11434",  #remote private server onlocal machine to speed up response time
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
    Returns the BM25Retriever object.    
    """
    return BM25Retriever.from_documents(chunks)


# initialise RAG pipeline - baseline semantic retrieval
def init_pipeline_semantic() -> QdrantVectorStore:
    """
    Initialises RAG pipeline with semantic retrieval.
    Builds and populates the vector store.
    Returns the vector store object.
    """
    chunks = load_and_chunk_dreams()
    vector_store = build_vector_store(chunks)

    global DREAM_VECTOR_STORE
    DREAM_VECTOR_STORE = vector_store

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

    global DREAM_VECTOR_STORE # used by tools and agent
    DREAM_VECTOR_STORE = vector_store

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
            f"Retrieved dream {idx} (Source: {source}, Page: {page}, Similarity score: {score:.2f}):\n"
            f"{doc.page_content.strip()}"
        )
    # join chunks into a single context string
    context = "\n\n".join(formatted_chunks)
    # inject the query and context into the system prompt
    prompt = SYSTEM_PROMPT.format(query=query, context=context)
    # invoke the LLM with the full prompt
    response = llm.invoke(prompt)
    # return the LLM response
    return response.content


# Hybrid retriever with bm25
def hybrid_retrieve(
    query: str,
    vector_store: QdrantVectorStore,
    bm25_retriever: BM25Retriever,
    k_semantic: int = 5,
    k_lexical: int = 5,
    k_final: int = 5,
) -> List[Document]:
    """
    Hybrid retrieval combining lexical and semantic search.
    - Gets top-k lexical matches with BM25
    - Gets top-k semantic matches from Qdrant
    - Fuses results to return a de-duplicated list
    Return a de-duplicated list preferring lexical hits over semantic hits
    """
    # lexical results
    lexical_docs = bm25_retriever.invoke(query)[:k_lexical]
    # semantic results 
    semantic_docs = vector_store.similarity_search(query, k=k_semantic)
    # de-duplication set
    seen = set()
    # fused results list    
    fused: List[Document] = []
    
    # loop over results and add to fused list if not already seen
    for doc in lexical_docs + semantic_docs:
        key = doc.page_content
        # key is the page content of the document
        if key in seen:
            continue
        seen.add(key)
        fused.append(doc)
        # stop if we have reached the final number of results
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
    # join chunks into a context string
    context = "\n\n".join(formatted_chunks)
    # inject query and context into  system prompt
    prompt = SYSTEM_PROMPT.format(query=query, context=context)
    # invoke LLM with full prompt
    response = llm.invoke(prompt)
    return response.content


## Agent's tools
# RAG tool for searching dreams (uses semantic retrieval function)
@tool
def dream_archive_search(query: str) -> str:
    """
    Tool: search the user's private dream archive for relevant entries and patterns.
    Uses the same retrieval logic as the main DreamNest agent. 
    Requires that init_pipeline_semantic() and get_llm() have been called so DREAM_VECTOR_STORE and RESPONSE_LLM are populated.
    """
    if DREAM_VECTOR_STORE is None or RESPONSE_LLM is None:
        raise RuntimeError(
            "Dream pipeline is not initialised. Call init_pipeline_semantic()/get_llm() first."
        )
    return answer_dream_query_semantic(query, DREAM_VECTOR_STORE, RESPONSE_LLM)


# Tavily tool for external search (for general info about dreaming/ journaling)
@tool
def tavily_dream_info(query: str) -> str:
    """
    Tool: search the public web for general information about dreaming and dream journaling.
    Never use this tool to answer questions about the user's specific dreams or to interpret dream meanings.
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

## Agent
# Create the RAG agent with tools
def build_retrieval_agent(llm: Optional[ChatOllama] = None):
    """
    Creates an agent that can call the following tools:
    - dream_archive_search (private RAG over the user's dream journal)
    - tavily_dream_info (general info about dreaming / journaling)
    Assumes init_pipeline_semantic() and get_llm() have been called beforehand if llm is None.
    """
    agent_llm = llm or RESPONSE_LLM
    if agent_llm is None:
        raise RuntimeError(
            "LLM is not initialised. Provide an llm or call get_llm() first."
        )

    tools = [dream_archive_search, tavily_dream_info]
    return create_agent(agent_llm, tools=tools)