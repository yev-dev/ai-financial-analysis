"""Integration tests for the **Research_Analyst** agent.

Requires a running local Ollama instance with ``llama3.1``.
"""

from __future__ import annotations

from typing import Any

import pytest

from fin_ai.agents.scheduler import run_agent_direct


class TestResearchAnalystAgent:
    """Integration tests exercising the Research_Analyst agent profile."""

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_agent_instantiation(self, ollama_llm_config: dict[str, Any]):
        """Verify the agent can be instantiated."""
        from fin_ai.agents.workflow import SingleAssistant

        agent = SingleAssistant("Research_Analyst", llm_config=ollama_llm_config)
        assert agent is not None

    @pytest.mark.integration
    def test_agent_has_publishing_tools(self):
        """Verify the Research_Analyst has publishing capabilities."""
        from fin_ai.agents.agent_library import library

        profile = library["Research_Analyst"]
        tools = profile["tools"]
        assert "publish_research_report" in tools
        assert "publish_research_html" in tools
        assert "get_stock_data" in tools
        assert "get_analyst_recommendations" in tools

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_agent_chat_company_research(
        self,
        skip_if_no_llama: None,
        ollama_llama3_llm_config: dict[str, Any],
    ):
        """Send a company research prompt and verify the agent responds."""
        result = run_agent_direct(
            "Provide a brief research summary on Apple Inc (AAPL)."
            " Include key financial metrics.",
            agent_class="SingleAssistant",
            agent_name="Research_Analyst",
            llm_config=ollama_llama3_llm_config,
        )
        assert result is not None

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_agent_handles_missing_data_gracefully(
        self,
        skip_if_no_llama: None,
        ollama_llama3_llm_config: dict[str, Any],
    ):
        """Verify the agent handles missing ticker data gracefully."""
        result = run_agent_direct(
            "Research the company with ticker ZZZZ.",
            agent_class="SingleAssistant",
            agent_name="Research_Analyst",
            llm_config=ollama_llama3_llm_config,
        )
        assert result is not None

    @pytest.mark.integration
    def test_all_tools_resolve(self):
        """Verify every tool name in the profile resolves in _TOOL_MAP."""
        from fin_ai.agents.agent_library import library
        from fin_ai.agents.workflow import _TOOL_MAP

        profile = library["Research_Analyst"]
        for tool_name in profile["tools"]:
            assert tool_name in _TOOL_MAP, (
                f"Tool '{tool_name}' not found in _TOOL_MAP"
            )

    @pytest.mark.integration
    def test_financial_snapshot_via_service(self):
        """Verify offline financial snapshot is accessible (no LLM needed)."""
        from fin_ai.core.service import OfflineMarketDataService
        from pathlib import Path

        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        if not data_dir.is_dir() or not list(data_dir.iterdir()):
            pytest.skip("No offline data directory found — skipping.")
        service = OfflineMarketDataService(data_dir)
        bs = service.get_balance_sheet("AAPL")
        cf = service.get_cash_flow("AAPL")
        inc = service.get_income_stmt("AAPL")
        assert not bs.empty
        assert not cf.empty
        assert not inc.empty
