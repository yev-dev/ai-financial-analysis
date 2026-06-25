"""Integration tests for the **Research_Publisher** agent.

Requires a running local Ollama instance with ``llama3.1``.
"""

from __future__ import annotations

from typing import Any

import pytest

from fin_ai.agents.scheduler import run_agent_direct


class TestResearchPublisherAgent:
    """Integration tests exercising the Research_Publisher agent profile."""

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_agent_instantiation(self, ollama_llm_config: dict[str, Any]):
        """Verify the agent can be instantiated."""
        from fin_ai.agents.workflow import SingleAssistant

        agent = SingleAssistant("Research_Publisher", llm_config=ollama_llm_config)
        assert agent is not None

    @pytest.mark.integration
    def test_agent_has_publishing_tools(self):
        """Verify the Research_Publisher has all distribution tools."""
        from fin_ai.agents.agent_library import library

        profile = library["Research_Publisher"]
        tools = profile["tools"]
        assert "publish_research_html" in tools
        assert "publish_research_pdf" in tools
        assert "publish_research_report" in tools
        assert "send_research_email" in tools

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_agent_publishes_html_report(
        self,
        skip_if_no_llama: None,
        ollama_llama3_llm_config: dict[str, Any],
    ):
        """Ask the agent to publish a research report as HTML."""
        result = run_agent_direct(
            'Publish a short research report on AAPL. '
            'Title: "Quick Analysis: AAPL". '
            'Content: "AAPL shows strong fundamentals with steady revenue growth."',
            agent_class="SingleAssistant",
            agent_name="Research_Publisher",
            llm_config=ollama_llama3_llm_config,
        )
        assert result is not None

    @pytest.mark.integration
    def test_all_tools_resolve(self):
        """Verify every tool name in the profile resolves in _TOOL_MAP."""
        from fin_ai.agents.agent_library import library
        from fin_ai.agents.workflow import _TOOL_MAP

        profile = library["Research_Publisher"]
        for tool_name in profile["tools"]:
            assert tool_name in _TOOL_MAP, (
                f"Tool '{tool_name}' not found in _TOOL_MAP"
            )

    @pytest.mark.integration
    def test_publish_html_directly(self):
        """Verify publish_research_html works directly (no LLM needed)."""
        from fin_ai.core.tools import publish_research_html
        import json

        result = json.loads(
            publish_research_html(
                content="# Test Report\n\nThis is a **test**.",
                title="Integration Test Report",
            )
        )
        assert result["status"] == "published"
        assert result["format"] == "html"
        assert "Integration_Test_Report" in result["filename"]
