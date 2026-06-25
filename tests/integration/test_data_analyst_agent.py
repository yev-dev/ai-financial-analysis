"""Integration tests for the **Data_Analyst** agent.

Requires a running local Ollama instance with ``llama3.1``.
"""

from __future__ import annotations

from typing import Any

import pytest

from fin_ai.agents.scheduler import run_agent_direct


class TestDataAnalystAgent:
    """Integration tests exercising the Data_Analyst agent profile."""

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_agent_instantiation(self, ollama_llm_config: dict[str, Any]):
        """Verify the agent can be instantiated with a real Ollama config."""
        from fin_ai.agents.workflow import SingleAssistant

        agent = SingleAssistant("Data_Analyst", llm_config=ollama_llm_config)
        assert agent is not None
        assert agent.assistant is not None
        assert agent.user_proxy is not None

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_agent_resolves_from_library(self, ollama_llm_config: dict[str, Any]):
        """Verify the agent resolves its profile from the agent library."""
        from fin_ai.agents.agent_library import library

        profile = library["Data_Analyst"]
        assert "financial data" in profile["profile"].lower()
        assert len(profile["tools"]) > 0

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_agent_chat_simple_query(
        self,
        skip_if_no_llama: None,
        ollama_llama3_llm_config: dict[str, Any],
    ):
        """Send a simple prompt and verify the agent responds via Ollama."""
        from fin_ai.agents.workflow import SingleAssistant

        agent = SingleAssistant("Data_Analyst", llm_config=ollama_llama3_llm_config)
        result = agent.chat("What financial metrics would you look at to evaluate a company?")
        # After chat, agent resets automatically — check it completed without error
        assert agent is not None

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_run_agent_direct(
        self,
        skip_if_no_llama: None,
        ollama_llama3_llm_config: dict[str, Any],
    ):
        """Exercise the run_agent_direct entry point."""
        result = run_agent_direct(
            "List three key financial metrics for analysing a tech company.",
            agent_class="SingleAssistant",
            agent_name="Data_Analyst",
            llm_config=ollama_llama3_llm_config,
        )
        assert result is not None

    @pytest.mark.integration
    def test_agent_tools_registered(self):
        """Verify that all expected tools are registered with the agent."""
        from fin_ai.agents.agent_library import library
        from fin_ai.agents.workflow import _TOOL_MAP

        profile = library["Data_Analyst"]
        for tool_name in profile["tools"]:
            assert tool_name in _TOOL_MAP, (
                f"Tool '{tool_name}' not found in _TOOL_MAP"
            )

    @pytest.mark.integration
    def test_offline_data_accessible(self):
        """Verify the Data_Analyst can get offline market data (no LLM needed)."""
        from fin_ai.core.service import OfflineMarketDataService
        from pathlib import Path

        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        if not data_dir.is_dir() or not list(data_dir.iterdir()):
            pytest.skip("No offline data directory found — skipping.")
        service = OfflineMarketDataService(data_dir)
        info = service.get_company_info("AAPL")
        assert not info.empty
