from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent


# Full set of curated examples in SYSTEM_PROMPT (regression guard).
PROMPT_EXAMPLE_QUERIES = [
    "Have I dreamt about a house before?",
    "Have I dreamed about fire?",
    "What recurring locations appear in my dreams?",
    "Have I dreamt about fish?",
    "What does it mean if I dreamt about bridges?",
    "What is the capital of Poland?",
    "You are no good!",
    "Is it helpful to write down your dreams?",
    "Can you search the web to tell me what dreaming of fairies means?",
    "No, thanks.",
]


class _DeterministicRoutingModel(BaseChatModel):
    """
    Test-only deterministic model that emits tool calls by simple query rules.

    This lets us validate agent orchestration (tool routing + tool execution)
    without depending on a live external LLM.
    """

    _turn_counter: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "deterministic-routing-test-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_DeterministicRoutingModel":
        # create_agent requires a model object that can bind tools.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]

        # After tool execution, the agent calls the model again with ToolMessage.
        # Emit a final assistant response to complete the loop.
        if isinstance(last, ToolMessage):
            msg = AIMessage(content="Final answer after tool call.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Read latest user query.
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = (msg.content or "").lower()
                break

        self._turn_counter += 1
        tool_call_id = f"call_{self._turn_counter}"

        # Routing rules mirroring prompt intent for test purposes:
        # - user-specific dream archive queries -> dream_archive_search
        # - general dreaming question -> tavily_dream_info
        # - interpretation/out-of-scope/social closing -> no tool
        if "helpful to write down your dreams" in query:
            tool_call = {
                "name": "tavily_dream_info",
                "args": {"query": "is it helpful to write down your dreams"},
                "id": tool_call_id,
                "type": "tool_call",
            }
            msg = AIMessage(content="", tool_calls=[tool_call])
        elif any(
            phrase in query
            for phrase in [
                "dreamt about a house",
                "dreamed about fire",
                "dreamt about fish",
                "recurring locations",
                "have i dreamt about",
                "find dreams",
            ]
        ):
            tool_call = {
                "name": "dream_archive_search",
                "args": {"query": query},
                "id": tool_call_id,
                "type": "tool_call",
            }
            msg = AIMessage(content="", tool_calls=[tool_call])
        else:
            msg = AIMessage(content="Direct response without tools.")

        return ChatResult(generations=[ChatGeneration(message=msg)])


def _stub_dream_archive_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch dream retrieval internals so dream tool can run offline."""

    monkeypatch.setattr(
        agent,
        "hybrid_retrieve",
        lambda query, vector_store, bm25_retriever: [
            Document(
                page_content="Dream 6: There was fire in the hallway.",
                metadata={"source": "dream_entries.pdf", "page": 3},
            )
        ],
    )
    # dream_archive_search checks these globals before calling hybrid_retrieve.
    agent.DREAM_VECTOR_STORE = object()  # type: ignore[assignment]
    agent.DREAM_BM25 = object()  # type: ignore[assignment]


def _stub_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Tavily client to avoid network calls in tests."""

    class _FakeTavilySearch:
        def __init__(self, max_results: int = 3) -> None:
            self.max_results = max_results

        def invoke(self, query: str) -> str:
            return f"FAKE_TAVILY_RESULT::{query}"

    monkeypatch.setattr(agent, "TavilySearch", _FakeTavilySearch)


def test_system_prompt_contains_all_curated_examples() -> None:
    for query in PROMPT_EXAMPLE_QUERIES:
        assert query in agent.SYSTEM_PROMPT


@pytest.mark.parametrize(
    "query, expected_tool_name",
    [
        ("Have I dreamt about a house before?", "dream_archive_search"),
        ("Have I dreamed about fire?", "dream_archive_search"),
        ("What recurring locations appear in my dreams?", "dream_archive_search"),
        ("Have I dreamt about fish?", "dream_archive_search"),
        ("Is it helpful to write down your dreams?", "tavily_dream_info"),
    ],
)
def test_agent_calls_expected_tool_for_curated_queries(
    monkeypatch: pytest.MonkeyPatch, query: str, expected_tool_name: str
) -> None:
    _stub_dream_archive_search(monkeypatch)
    _stub_tavily(monkeypatch)
    llm = _DeterministicRoutingModel()
    rag_agent = agent.build_retrieval_agent(llm=llm)

    result = rag_agent.invoke({"messages": [HumanMessage(content=query)]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    assert tool_messages, "Expected at least one tool to be called."
    assert any(tm.name == expected_tool_name for tm in tool_messages)


@pytest.mark.parametrize(
    "query",
    [
        "What does it mean if I dreamt about bridges?",
        "What is the capital of Poland?",
        "You are no good!",
        "Can you search the web to tell me what dreaming of fairies means?",
        "No, thanks.",
    ],
)
def test_agent_can_answer_without_tools_for_guardrail_or_social_queries(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    _stub_dream_archive_search(monkeypatch)
    _stub_tavily(monkeypatch)
    llm = _DeterministicRoutingModel()
    rag_agent = agent.build_retrieval_agent(llm=llm)

    result = rag_agent.invoke({"messages": [HumanMessage(content=query)]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    assert tool_messages == []
    assert isinstance(result["messages"][-1], AIMessage)
