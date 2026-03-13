"""
Legacy fixed-flow RAG helpers kept for debugging/reference only.

This module preserves the original non-agentic answer functions so they can be
used for one-off debugging or historical comparison, while the main runtime in
`agent.py` stays focused on pure agentic RAG (create_agent + tools).
"""

from langchain_core.messages import HumanMessage as HMsg, SystemMessage
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever

from agent import SYSTEM_PROMPT, hybrid_retrieve


def answer_dream_query_semantic(
    query: str,
    vector_store: QdrantVectorStore,
    llm: ChatOllama,
) -> str:
    """
    Legacy fixed-flow semantic RAG answer helper.
    Retrieves relevant chunks from vector search and asks the LLM to answer.
    """
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    if not results_with_scores:
        return "No strong matches were found in your dream archive for this query."

    top_doc, top_score = results_with_scores[0]
    if top_score < 0.4:
        return "No strong matches were found in your dream archive for this query."
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
            f"Retrieved dream {idx} (Source: {source}, Page: {page}):\n"
            f"{doc.page_content.strip()}"
        )

    context = "\n\n".join(formatted_chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HMsg(content=f"Query: {query}\n\nRetrieved context:\n{context}"),
    ]
    response = llm.invoke(messages)
    return response.content


def answer_dream_query_hybrid(
    query: str,
    vector_store: QdrantVectorStore,
    bm25_retriever: BM25Retriever,
    llm: ChatOllama,
) -> str:
    """
    Legacy fixed-flow hybrid RAG answer helper.
    Uses hybrid retrieval, then asks the LLM to produce a response.
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
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HMsg(content=f"Query: {query}\n\nRetrieved context:\n{context}"),
    ]
    response = llm.invoke(messages)
    return response.content
