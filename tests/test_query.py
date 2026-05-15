"""Tests for multi-source query orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from src.fin_ai.core.query import (
    SourceRetrieverConfig,
    VectorStoreFilteredRetriever,
    build_multi_source_prompt,
    build_source_retriever_configs,
    build_source_specific_prompts,
    format_source_citations,
    plan_query_sources_with_llm,
    query_with_multi_source_prompting,
    retrieve_multi_source_documents,
    route_query_to_sources,
)


@dataclass
class FakeRetriever:
    documents: list[Document]

    def invoke(self, query: str) -> list[Document]:
        return list(self.documents)


@dataclass
class FakeVectorStore:
    documents: list[Document]

    def __post_init__(self) -> None:
        self.docstore = type("Docstore", (), {"_dict": {str(index): doc for index, doc in enumerate(self.documents)}})()

    def as_retriever(self, search_type: str = "mmr", search_kwargs: dict | None = None):
        search_kwargs = search_kwargs or {}
        documents = list(self.documents)
        metadata_filter = search_kwargs.get("filter") or {}
        k = search_kwargs.get("k", len(documents))

        class _Retriever:
            def invoke(self_inner, query: str) -> list[Document]:
                filtered = documents
                if metadata_filter:
                    filtered = [
                        doc
                        for doc in documents
                        if all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
                    ]
                return filtered[:k]

        return _Retriever()


def _make_sources() -> list[SourceRetrieverConfig]:
    earnings_docs = [
        Document(
            page_content="Quarterly revenue grew 20% year over year.",
            metadata={"source": "earnings.csv", "source_type": "csv", "filename": "earnings.csv", "chunk_index": 0},
        ),
        Document(
            page_content="Operating margin improved to 18%.",
            metadata={"source": "earnings.csv", "source_type": "csv", "filename": "earnings.csv", "chunk_index": 1},
        ),
    ]
    sec_docs = [
        Document(
            page_content="The 10-K highlights AI infrastructure demand.",
            metadata={"source": "report.pdf", "source_type": "pdf", "filename": "report.pdf", "chunk_index": 0},
        )
    ]
    news_docs = [
        Document(
            page_content="Recent news mentions supply constraints in Q1.",
            metadata={"source": "news.html", "source_type": "html", "filename": "news.html", "chunk_index": 0},
        )
    ]
    return [
        SourceRetrieverConfig(
            name="earnings_csv",
            retriever=FakeRetriever(earnings_docs),
            description="Quarterly earnings and revenue metrics",
            metadata_filter={"source_type": "csv"},
            weight=2.0,
            tags=("earnings", "revenue", "csv"),
        ),
        SourceRetrieverConfig(
            name="sec_pdf",
            retriever=FakeRetriever(sec_docs),
            description="Annual filing and risk disclosures",
            metadata_filter={"source_type": "pdf"},
            weight=1.0,
            tags=("10-k", "filing", "pdf"),
        ),
        SourceRetrieverConfig(
            name="web_news",
            retriever=FakeRetriever(news_docs),
            description="HTML news and website commentary",
            metadata_filter={"source_type": "html"},
            weight=0.5,
            tags=("news", "html"),
        ),
    ]


class TestSourceConfigBuilders:
    @pytest.mark.unit
    def test_build_source_retriever_configs_group_by_source_type(self):
        vector_store = FakeVectorStore(
            [
                Document(page_content="csv row", metadata={"source_type": "csv", "filename": "data.csv"}),
                Document(page_content="pdf page", metadata={"source_type": "pdf", "filename": "report.pdf"}),
            ]
        )

        configs = build_source_retriever_configs(vector_store, base_name="mixed", group_by="source_type")

        assert [config.name for config in configs] == ["mixed:csv", "mixed:pdf"]
        assert all(isinstance(config.retriever, VectorStoreFilteredRetriever) for config in configs)


class TestQueryRouting:
    @pytest.mark.unit
    def test_route_query_prefers_matching_sources(self):
        sources = _make_sources()

        routed = route_query_to_sources("Summarize revenue from the earnings csv", sources, max_sources=2)

        assert [source.name for source in routed][0] == "earnings_csv"
        assert len(routed) == 2

    @pytest.mark.unit
    def test_plan_query_sources_with_llm_uses_json_selection(self, monkeypatch):
        class FakeResponse:
            content = '{"selected_sources": ["sec_pdf"], "reasoning": "The question asks about the filing."}'

        def fake_run_model_request(**kwargs):
            return FakeResponse()

        monkeypatch.setattr("src.fin_ai.core.query._run_model_request", fake_run_model_request)

        decision = plan_query_sources_with_llm(
            "Use the filing to summarize ai infrastructure demand",
            _make_sources(),
            provider="ollama",
            max_sources=1,
        )

        assert decision is not None
        assert decision.selected_sources == ["sec_pdf"]
        assert decision.planner_used is True


class TestMultiSourceRetrieval:
    @pytest.mark.unit
    def test_separate_retrieval_applies_global_metadata_filter(self):
        sources = _make_sources()

        result = retrieve_multi_source_documents(
            "What changed?",
            sources,
            mode="separate",
            global_metadata_filter={"source_type": "csv"},
        )

        assert result.mode == "separate"
        assert result.source_results[0].documents
        assert result.source_results[1].documents == []
        assert result.source_results[2].documents == []

    @pytest.mark.unit
    def test_routed_retrieval_limits_sources(self):
        sources = _make_sources()

        result = retrieve_multi_source_documents(
            "Use the filing to summarize ai infrastructure demand",
            sources,
            mode="routed",
            max_sources=1,
        )

        assert result.selected_sources == ["sec_pdf"]
        assert len(result.source_results) == 1
        assert result.routing_decision is not None

    @pytest.mark.unit
    def test_ensemble_retrieval_orders_by_weighted_rank(self):
        shared_doc = Document(
            page_content="Shared datapoint across sources.",
            metadata={"source": "shared", "filename": "shared.txt", "chunk_index": 0},
        )
        sources = [
            SourceRetrieverConfig(
                name="heavy_source",
                retriever=FakeRetriever([shared_doc]),
                weight=3.0,
                tags=("shared",),
            ),
            SourceRetrieverConfig(
                name="light_source",
                retriever=FakeRetriever([
                    shared_doc,
                    Document(page_content="Unique lower ranked content", metadata={"source": "light", "filename": "light.txt", "chunk_index": 1}),
                ]),
                weight=1.0,
                tags=("shared",),
            ),
        ]

        result = retrieve_multi_source_documents("shared datapoint", sources, mode="ensemble")

        assert result.combined_documents[0].page_content == "Shared datapoint across sources."


class TestPromptBuilders:
    @pytest.mark.unit
    def test_build_multi_source_prompt_groups_by_source(self):
        retrieval = retrieve_multi_source_documents("What changed?", _make_sources(), mode="separate")

        prompt = build_multi_source_prompt("What changed?", retrieval)

        assert "### Source: earnings_csv" in prompt
        assert "### Source: sec_pdf" in prompt
        assert "Quarterly revenue grew 20% year over year." in prompt
        assert "[Source: source_name]" in prompt

    @pytest.mark.unit
    def test_build_source_specific_prompts_creates_one_prompt_per_source(self):
        retrieval = retrieve_multi_source_documents("What changed?", _make_sources(), mode="separate")

        prompts = build_source_specific_prompts("What changed?", retrieval)

        assert set(prompts) == {"earnings_csv", "sec_pdf", "web_news"}
        assert "Only use evidence from source: earnings_csv." in prompts["earnings_csv"]
        assert "Quarterly revenue grew 20% year over year." in prompts["earnings_csv"]

    @pytest.mark.unit
    def test_format_source_citations_returns_markdown(self):
        retrieval = retrieve_multi_source_documents("What changed?", _make_sources(), mode="separate")

        citations = format_source_citations(retrieval, response_type="Markdown")

        assert "Sources Consulted:" in citations
        assert "**earnings_csv**" in citations
        assert "earnings.csv" in citations


class TestEndToEndPrompting:
    @pytest.mark.unit
    def test_query_with_multi_source_prompting_uses_combined_prompt(self, monkeypatch):
        recorded: dict[str, str] = {}

        class FakeResponse:
            def __init__(self, content: str) -> None:
                self.content = content

        def fake_run_model_request(**kwargs):
            recorded["provider"] = kwargs["provider"]
            recorded["format"] = kwargs["response_format"]
            recorded["prompt"] = kwargs["prompt"]
            recorded["system_prompt"] = kwargs["system_prompt"]
            if "Return valid JSON only." in kwargs["prompt"]:
                return FakeResponse('{"selected_sources": ["earnings_csv", "sec_pdf"], "reasoning": "Need both earnings and filing evidence."}')
            return FakeResponse("ok")

        monkeypatch.setattr("src.fin_ai.core.query._run_model_request", fake_run_model_request)

        result = query_with_multi_source_prompting(
            "Compare earnings and filing evidence",
            _make_sources(),
            provider="ollama",
            mode="routed",
            max_sources=2,
            use_llm_planner=True,
            planner_provider="ollama",
        )

        assert recorded["provider"] == "ollama"
        assert "Retrieved Context:" in recorded["prompt"]
        assert result.response.content == "ok"
        assert len(result.source_prompts) == 2
        assert result.retrieval.routing_decision is not None
        assert result.retrieval.routing_decision.planner_used is True