from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent


# Reuse manually curated system-prompt examples as regression fixtures.
EXAMPLE_QUERY_FIRE = "Have I dreamed about fire?"
EXAMPLE_QUERY_FISH = "Have I dreamt about fish?"
EXAMPLE_QUERY_INTERPRETATION = "What does it mean if I dreamt about bridges?"
EXAMPLE_QUERY_OUT_OF_SCOPE = "What is the capital of Poland?"
EXAMPLE_ABSENT_TERM_FROM_PROMPT = "penguins"


class _FakeVectorStore:
    """Minimal test double for vector similarity search."""

    def __init__(self, results: List[Tuple[Document, float]]) -> None:
        self._results = results

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        return self._results[:k]


class _FakeBM25Retriever:
    """Minimal test double for BM25 lexical retrieval."""

    def __init__(self, docs: List[Document]) -> None:
        self._docs = docs

    def invoke(self, query: str) -> List[Document]:
        return self._docs


@pytest.mark.parametrize(
    "query, required_token",
    [
        ("Have I dreamt about fire?", "fire"),
        (EXAMPLE_QUERY_FIRE, "fire"),
        (EXAMPLE_QUERY_FISH, "fish"),
    ],
)
def test_normalize_query_tokens_handles_punctuation_and_prompt_examples(
    query: str, required_token: str
) -> None:
    tokens = agent._normalize_query_tokens(query)
    assert required_token in tokens
    # "about" is a common non-informative token and should be removed.
    assert "about" not in tokens


def test_system_prompt_contains_curated_examples() -> None:
    prompt = agent.SYSTEM_PROMPT
    assert EXAMPLE_QUERY_FIRE in prompt
    assert EXAMPLE_QUERY_FISH in prompt
    assert EXAMPLE_QUERY_INTERPRETATION in prompt
    assert EXAMPLE_QUERY_OUT_OF_SCOPE in prompt
    assert EXAMPLE_ABSENT_TERM_FROM_PROMPT in prompt


def test_hybrid_retrieve_returns_fire_doc_for_prompt_example_query() -> None:
    fire_doc = Document(
        page_content="Dream 6: There was fire in the hallway.",
        metadata={"source": "dream_entries.pdf", "page": 3},
    )
    other_doc = Document(
        page_content="Dream 2: A train crossed a bridge at night.",
        metadata={"source": "dream_entries.pdf", "page": 1},
    )

    # Simulate weak semantic scores so BM25 lexical retrieval drives this case.
    vector_store = _FakeVectorStore(results=[(other_doc, 0.1)])
    bm25 = _FakeBM25Retriever(docs=[fire_doc, other_doc])

    docs = agent.hybrid_retrieve(
        query=EXAMPLE_QUERY_FIRE,
        vector_store=vector_store,  # type: ignore[arg-type]
        bm25_retriever=bm25,  # type: ignore[arg-type]
        k_semantic=5,
        k_lexical=5,
        k_final=3,
    )

    assert docs, "Expected at least one retrieved document."
    assert any("fire" in d.page_content.lower() for d in docs)


def test_hybrid_retrieve_returns_empty_for_absent_prompt_term_penguins() -> None:
    moon_doc = Document(
        page_content="Dream 11: I walked on a moonlit beach.",
        metadata={"source": "dream_entries.pdf", "page": 5},
    )

    # No semantic confidence and no lexical match after filtering.
    vector_store = _FakeVectorStore(results=[(moon_doc, 0.05)])
    bm25 = _FakeBM25Retriever(docs=[moon_doc])

    docs = agent.hybrid_retrieve(
        query=EXAMPLE_ABSENT_TERM_FROM_PROMPT,
        vector_store=vector_store,  # type: ignore[arg-type]
        bm25_retriever=bm25,  # type: ignore[arg-type]
        k_semantic=5,
        k_lexical=5,
        k_final=3,
    )

    assert docs == []


def test_system_prompt_has_interpretation_refusal_guardrail() -> None:
    prompt = agent.SYSTEM_PROMPT.lower()
    assert EXAMPLE_QUERY_INTERPRETATION.lower() in prompt
    assert "does not provide interpretation" in prompt
    assert "psychological analysis" in prompt


def test_system_prompt_has_out_of_scope_guardrail_example() -> None:
    # Regression check: keep at least one explicit non-dream out-of-scope example.
    assert EXAMPLE_QUERY_OUT_OF_SCOPE.lower() in agent.SYSTEM_PROMPT.lower()
