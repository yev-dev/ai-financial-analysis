"""Integration tests for the **Market_Analyst** agent.

Requires a running local Ollama instance with ``llama3.1``.
"""

from __future__ import annotations

from typing import Any

import pytest

from fin_ai.agents.scheduler import run_agent_direct


class TestMarketAnalystAgent:
    """Integration tests exercising the Market_Analyst agent profile."""

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_agent_instantiation(self, ollama_llm_config: dict[str, Any]):
        """Verify the agent can be instantiated with a real Ollama config."""
        from fin_ai.agents.workflow import SingleAssistant

        agent = SingleAssistant("Market_Analyst", llm_config=ollama_llm_config)
        assert agent is not None
        assert agent.assistant is not None

    @pytest.mark.integration
    def test_agent_has_market_tools(self):
        """Verify the Market_Analyst has price and recommendation tools."""
        from fin_ai.agents.agent_library import library

        profile = library["Market_Analyst"]
        tools = profile["tools"]
        assert "get_stock_data" in tools
        assert "get_analyst_recommendations" in tools
        assert "get_stock_info" in tools

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_agent_responds_with_offline_data(
        self,
        skip_if_no_llama: None,
        ollama_llama3_llm_config: dict[str, Any],
    ):
        """Exercise the agent to query stock data (offline fallback)."""
        result = run_agent_direct(
            "What is the current stock price of AAPL?",
            agent_class="SingleAssistant",
            agent_name="Market_Analyst",
            llm_config=ollama_llama3_llm_config,
        )
        assert result is not None

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_agent_handles_unknown_ticker(
        self,
        skip_if_no_llama: None,
        ollama_llama3_llm_config: dict[str, Any],
    ):
        """Verify graceful handling of an invalid ticker."""
        result = run_agent_direct(
            "What is the stock info for ZZZZ?",
            agent_class="SingleAssistant",
            agent_name="Market_Analyst",
            llm_config=ollama_llama3_llm_config,
        )
        assert result is not None

    @pytest.mark.integration
    def test_all_tools_resolve(self):
        """Verify every tool name in the profile resolves in _TOOL_MAP."""
        from fin_ai.agents.agent_library import library
        from fin_ai.agents.workflow import _TOOL_MAP

        profile = library["Market_Analyst"]
        for tool_name in profile["tools"]:
            assert tool_name in _TOOL_MAP, (
                f"Tool '{tool_name}' not found in _TOOL_MAP"
            )

    @pytest.mark.integration
    def test_offline_stock_info(
        self,
    ):
        """Verify the agent can retrieve offline stock info (no LLM needed)."""
        from fin_ai.core.service import OfflineMarketDataService
        from pathlib import Path

        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        if not data_dir.is_dir() or not list(data_dir.iterdir()):
            pytest.skip("No offline data directory found — skipping.")
        service = OfflineMarketDataService(data_dir)
        info = service.get_stock_info("AAPL")
        assert isinstance(info, dict)
        assert len(info) > 0
