"""Integration tests for the **Test_Agent** (diagnostics / introspection).

This agent does not require an LLM — it uses tool introspection to provide
system information.  It is the simplest and fastest to test.
"""

from __future__ import annotations

from typing import Any

import pytest

from fin_ai.agents.scheduler import run_agent_direct


class TestTestAgent:
    """Integration tests for the Test_Agent diagnostic profile."""

    @pytest.mark.integration
    def test_agent_instantiation(self):
        """Verify the Test_Agent can be instantiated without an LLM."""
        from fin_ai.agents.workflow import SingleAssistant

        agent = SingleAssistant("Test_Agent", llm_config=False)
        assert agent is not None

    @pytest.mark.integration
    def test_agent_has_introspection_tools(self):
        """Verify the Test_Agent has diagnostic/introspection tools."""
        from fin_ai.agents.agent_library import library

        profile = library["Test_Agent"]
        tools = profile["tools"]
        assert "list_agent_profiles" in tools
        assert "list_vector_stores" in tools
        assert "list_available_models" in tools
        assert "get_provider_info" in tools

    @pytest.mark.integration
    def test_all_tools_resolve(self):
        """Verify every tool name in the profile resolves in _TOOL_MAP."""
        from fin_ai.agents.agent_library import library
        from fin_ai.agents.workflow import _TOOL_MAP

        profile = library["Test_Agent"]
        for tool_name in profile["tools"]:
            assert tool_name in _TOOL_MAP, (
                f"Tool '{tool_name}' not found in _TOOL_MAP"
            )

    @pytest.mark.integration
    def test_list_agent_profiles_tool_exists(self):
        """Verify the list_agent_profiles function is importable."""
        from fin_ai.agents.engine_bridge import list_agent_profiles
        result = list_agent_profiles()
        assert "Test_Agent" in result

    @pytest.mark.integration
    def test_get_provider_info_returns_string(self):
        """Verify get_provider_info returns structured output."""
        from fin_ai.agents.engine_bridge import get_provider_info
        result = get_provider_info()
        assert isinstance(result, str)
        assert len(result) > 0
