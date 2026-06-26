"""Integration tests for the **agent scheduler** (pipelines, orchestration).

Requires a running local Ollama instance with ``llama3.1`` for slow tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from fin_ai.agents.scheduler import run_agent_direct


class TestRunAgentDirect:
    """Tests for the run_agent_direct convenience function."""

    @pytest.mark.integration
    def test_validates_agent_class(self):
        """Verify run_agent_direct raises on unknown agent class."""
        with pytest.raises(ValueError, match="Unknown agent_class"):
            run_agent_direct(
                "test",
                agent_class="NonExistent",
                agent_name="Test_Agent",
                llm_config={},
            )

    @pytest.mark.integration
    def test_requires_retrieve_config_for_rag(self):
        """Verify SingleAssistantRAG requires retrieve_config."""
        with pytest.raises(ValueError, match="retrieve_config"):
            run_agent_direct(
                "test",
                agent_class="SingleAssistantRAG",
                agent_name="Data_Analyst",
                llm_config={},
            )

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_run_with_real_ollama(
        self,
        skip_if_no_llama: None,
        ollama_llama3_llm_config: dict[str, Any],
    ):
        """Verify run_agent_direct works end-to-end with real Ollama."""
        agent = run_agent_direct(
            "Say hello and reply with just 'Hello!'.",
            agent_class="SingleAssistant",
            agent_name="Test_Agent",
            llm_config=ollama_llama3_llm_config,
        )
        from fin_ai.agents.workflow import SingleAssistant
        assert isinstance(agent, SingleAssistant)


class TestAgentPipeline:
    """Integration tests for the AgentPipeline orchestration."""

    @pytest.mark.integration
    def test_pipeline_creation(self):
        """Verify a pipeline can be created with tasks."""
        from fin_ai.agents.scheduler import AgentPipeline, AgentTask

        pipeline = AgentPipeline("test_pipeline", llm_config={})
        pipeline.add_task(
            AgentTask(
                name="task1",
                prompt="Analyze AAPL",
                agent_config="Data_Analyst",
            )
        )
        pipeline.add_task(
            AgentTask(
                name="task2",
                prompt="Get market data for AAPL",
                agent_config="Market_Analyst",
                depends_on=["task1"],
            )
        )
        assert len(pipeline._tasks) == 2

    @pytest.mark.integration
    def test_pipeline_duplicate_task_raises(self):
        """Verify duplicate task names are rejected."""
        from fin_ai.agents.scheduler import AgentPipeline, AgentTask

        pipeline = AgentPipeline("test", llm_config={})
        pipeline.add_task(
            AgentTask(name="dup", prompt="test", agent_config="Test_Agent")
        )
        with pytest.raises(ValueError, match="Duplicate"):
            pipeline.add_task(
                AgentTask(name="dup", prompt="test2", agent_config="Test_Agent")
            )

    @pytest.mark.integration
    def test_build_financial_analysis_pipeline(self):
        """Verify the convenience pipeline builder creates the expected tasks."""
        from fin_ai.agents.scheduler import (
            build_financial_analysis_pipeline,
            AgentPipeline,
        )

        pipeline = build_financial_analysis_pipeline(
            tickers=["AAPL"],
            llm_config={},
        )
        assert isinstance(pipeline, AgentPipeline)
        assert len(pipeline._tasks) >= 2  # analysis tasks + synthesis
