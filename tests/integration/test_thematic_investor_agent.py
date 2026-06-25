"""Integration tests for the **Thematic_Investor** agent.

Requires a running local Ollama instance with ``llama3.1``.
"""

from __future__ import annotations

from typing import Any

import pytest

from fin_ai.agents.scheduler import run_agent_direct


class TestThematicInvestorAgent:
    """Integration tests exercising the Thematic_Investor agent profile."""

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_agent_instantiation(self, ollama_llm_config: dict[str, Any]):
        """Verify the agent can be instantiated."""
        from fin_ai.agents.workflow import SingleAssistant

        agent = SingleAssistant("Thematic_Investor", llm_config=ollama_llm_config)
        assert agent is not None

    @pytest.mark.integration
    def test_agent_has_thematic_tools(self):
        """Verify the Thematic_Investor has the expected tool set."""
        from fin_ai.agents.agent_library import library

        profile = library["Thematic_Investor"]
        tools = profile["tools"]
        # Should have fundamental analysis tools but not publishing
        assert "get_income_stmt" in tools
        assert "get_analyst_recommendations" in tools
        assert "get_stock_info" in tools
        # Should NOT have publishing tools
        assert "publish_research_report" not in tools

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_agent_chat_thematic_query(
        self,
        skip_if_no_llama: None,
        ollama_llama3_llm_config: dict[str, Any],
    ):
        """Send a thematic investment question."""
        result = run_agent_direct(
            "Evaluate NVDA from an AI thematic investment perspective.",
            agent_class="SingleAssistant",
            agent_name="Thematic_Investor",
            llm_config=ollama_llama3_llm_config,
        )
        assert result is not None

    @pytest.mark.integration
    def test_all_tools_resolve(self):
        """Verify every tool name in the profile resolves in _TOOL_MAP."""
        from fin_ai.agents.agent_library import library
        from fin_ai.agents.workflow import _TOOL_MAP

        profile = library["Thematic_Investor"]
        for tool_name in profile["tools"]:
            assert tool_name in _TOOL_MAP, (
                f"Tool '{tool_name}' not found in _TOOL_MAP"
            )
