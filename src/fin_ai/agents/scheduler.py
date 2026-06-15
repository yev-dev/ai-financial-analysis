"""
Multi-Agent Scheduler & Orchestration Framework.

Provides scheduling, direct interaction, and state-sharing capabilities
so multiple agents can run cooperatively — either on-demand or on a
configurable recurring schedule.

Key Abstractions
----------------
- ``AgentTask`` : A unit of work (prompt + target agent).
- ``AgentPipeline`` : A sequence of tasks with shared state.
- ``AgentScheduler`` : Runs pipelines on a schedule or on-demand.
- ``run_agent_direct`` : Convenience for one-shot agent execution.

Usage (Direct)::

    from fin_ai.agents.scheduler import run_agent_direct

    result = run_agent_direct(
        agent_class="SingleAssistantRAG",
        agent_name="Data_Analyst",
        prompt="Analyze AAPL's 2024 performance.",
        llm_config=llm_config,
        retrieve_config=retrieve_config,
    )

Usage (Pipeline)::

    from fin_ai.agents.scheduler import AgentPipeline, AgentTask

    pipeline = AgentPipeline("Weekly Analysis")
    pipeline.add_task(AgentTask("AAPL Analysis", agent="analyst_aapl"))
    pipeline.add_task(AgentTask("MSFT Analysis", agent="analyst_msft"))
    pipeline.run()
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentTask:
    """A single unit of work for an agent pipeline.

    Parameters
    ----------
    name : str
        Human-readable task name.
    prompt : str
        The prompt / message to send to the agent.
    agent_config : str | dict
        Agent config name or dict resolvable.
    depends_on : list[str]
        Names of tasks that must complete before this one runs.
    timeout_seconds : int
        Max wall-clock time for this task. 0 = no limit.
    retry_count : int
        Number of retries on failure.
    metadata : dict
        Arbitrary extra metadata (e.g. ``retrieve_config`` for RAG agents).
    """

    name: str
    prompt: str = ""
    agent_config: str | dict[str, Any] = "Data_Analyst"
    depends_on: list[str] = field(default_factory=list)
    timeout_seconds: int = 600
    retry_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime state (populated by scheduler)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempts: int = 0


@dataclass
class PipelineResult:
    """Aggregate result of a pipeline execution."""

    pipeline_name: str
    started_at: datetime
    completed_at: datetime
    task_results: dict[str, Any] = field(default_factory=dict)
    shared_state: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared State Store (cross-agent memory)
# ---------------------------------------------------------------------------


class SharedState:
    """Thread-safe key-value store shared across agents in a pipeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def update(self, mapping: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(mapping)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# ---------------------------------------------------------------------------
# Agent Pipeline
# ---------------------------------------------------------------------------


class AgentPipeline:
    """Ordered collection of ``AgentTask`` objects with dependency resolution.

    Parameters
    ----------
    name : str
        Pipeline name.
    llm_config : dict
        LLM configuration forwarded to every agent.
    shared_state : SharedState, optional
        Pre-existing state store.
    """

    def __init__(
        self,
        name: str,
        llm_config: dict[str, Any] | None = None,
        shared_state: SharedState | None = None,
    ):
        self.name = name
        self.llm_config = llm_config or {}
        self.shared_state = shared_state or SharedState()
        self._tasks: list[AgentTask] = []
        self._task_map: dict[str, AgentTask] = {}

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def add_task(self, task: AgentTask) -> "AgentPipeline":
        if task.name in self._task_map:
            raise ValueError(f"Duplicate task name: {task.name}")
        self._tasks.append(task)
        self._task_map[task.name] = task
        return self

    def add_tasks(self, tasks: list[AgentTask]) -> "AgentPipeline":
        for t in tasks:
            self.add_task(t)
        return self

    def remove_task(self, name: str) -> None:
        self._tasks = [t for t in self._tasks if t.name != name]
        self._task_map.pop(name, None)

    # ------------------------------------------------------------------
    # Dependency resolution (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _resolve_order(self) -> list[AgentTask]:
        in_degree: dict[str, int] = {
            t.name: len(t.depends_on) for t in self._tasks
        }
        adj: dict[str, list[str]] = defaultdict(list)

        for t in self._tasks:
            for dep in t.depends_on:
                adj[dep].append(t.name)

        for t in self._tasks:
            for dep in t.depends_on:
                if dep not in self._task_map:
                    raise ValueError(
                        f"Task '{t.name}' depends on unknown task '{dep}'"
                    )

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            name = queue.pop(0)
            order.append(name)
            for neighbor in adj[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._tasks):
            remaining = set(self._task_map) - set(order)
            raise ValueError(f"Circular dependency detected among: {remaining}")

        return [self._task_map[name] for name in order]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        agent_factory: Callable[..., Any] | None = None,
    ) -> PipelineResult:
        """Execute all pipeline tasks in dependency order."""
        started = datetime.now()
        task_results: dict[str, Any] = {}
        errors: list[str] = []

        try:
            ordered = self._resolve_order()
        except ValueError as exc:
            return PipelineResult(
                pipeline_name=self.name,
                started_at=started,
                completed_at=datetime.now(),
                success=False,
                errors=[str(exc)],
            )

        factory = agent_factory or _default_agent_factory

        for task in ordered:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.attempts += 1

            try:
                agent = factory(task, self.llm_config)
                result = _execute_task(agent, task)
                task.result = result
                task.status = TaskStatus.COMPLETED
                task_results[task.name] = result
                self.shared_state.set(task.name, result)

            except Exception as exc:
                logger.exception("Task '%s' failed: %s", task.name, exc)
                if task.attempts <= task.retry_count:
                    logger.info(
                        "Retrying task '%s' (attempt %d)",
                        task.name,
                        task.attempts + 1,
                    )
                    task.status = TaskStatus.PENDING
                    self._tasks.insert(0, task)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = str(exc)
                    errors.append(f"{task.name}: {exc}")

            task.completed_at = datetime.now()

        return PipelineResult(
            pipeline_name=self.name,
            started_at=started,
            completed_at=datetime.now(),
            task_results=task_results,
            shared_state=self.shared_state.snapshot(),
            success=len(errors) == 0,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Agent Scheduler
# ---------------------------------------------------------------------------


class AgentScheduler:
    """Runs ``AgentPipeline`` objects on a recurring schedule.

    Parameters
    ----------
    pipelines : list[AgentPipeline]
        Pipelines to manage.
    interval_seconds : int
        Seconds between pipeline runs. Default: 3600 (hourly).
    run_on_start : bool
        Execute all pipelines immediately on ``start()``.
    """

    def __init__(
        self,
        pipelines: list[AgentPipeline],
        interval_seconds: int = 3600,
        run_on_start: bool = False,
    ):
        self.pipelines = pipelines
        self.interval_seconds = interval_seconds
        self.run_on_start = run_on_start
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._results: list[PipelineResult] = []

    @property
    def results(self) -> list[PipelineResult]:
        return list(self._results)

    def start(self) -> None:
        """Begin the scheduling loop in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Scheduler is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            "Scheduler started (interval=%ds, pipelines=%d)",
            self.interval_seconds,
            len(self.pipelines),
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the scheduler to stop and wait for the thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            logger.info("Scheduler stopped.")

    def run_once(self) -> list[PipelineResult]:
        """Execute all pipelines once (synchronous)."""
        results: list[PipelineResult] = []
        for p in self.pipelines:
            logger.info("Running pipeline: %s", p.name)
            result = p.run()
            self._results.append(result)
            results.append(result)
            logger.info(
                "Pipeline '%s' complete (success=%s, tasks=%d)",
                p.name,
                result.success,
                len(result.task_results),
            )
        return results

    def _loop(self) -> None:
        """Internal scheduling loop."""
        if self.run_on_start:
            self.run_once()

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.interval_seconds)
            if not self._stop_event.is_set():
                self.run_once()


# ---------------------------------------------------------------------------
# Direct single-agent execution
# ---------------------------------------------------------------------------


def run_agent_direct(
    prompt: str,
    *,
    agent_class: str = "SingleAssistantRAG",
    agent_name: str = "Data_Analyst",
    llm_config: dict[str, Any] | None = None,
    retrieve_config: dict[str, Any] | None = None,
    rag_description: str = "",
    **agent_kwargs,
) -> Any:
    """Run a single agent directly with the given prompt.

    Parameters
    ----------
    prompt : str
        The message/question to send to the agent.
    agent_class : str
        One of ``"SingleAssistant"``, ``"SingleAssistantRAG"``,
        ``"SingleAssistantShadow"``, ``"MultiAssistant"``,
        ``"MultiAssistantWithLeader"``.
    agent_name : str
        Agent config name from the agent library (e.g. ``"Data_Analyst"``).
    llm_config : dict
        LLM configuration.
    retrieve_config : dict
        RAG retrieval config (required for ``SingleAssistantRAG``).
    rag_description : str
        Description of the RAG tool.
    """
    from fin_ai.agents.workflow import (
        SingleAssistant,
        SingleAssistantRAG,
        SingleAssistantShadow,
        MultiAssistant,
        MultiAssistantWithLeader,
    )

    _class_map = {
        "SingleAssistant": SingleAssistant,
        "SingleAssistantRAG": SingleAssistantRAG,
        "SingleAssistantShadow": SingleAssistantShadow,
        "MultiAssistant": MultiAssistant,
        "MultiAssistantWithLeader": MultiAssistantWithLeader,
    }

    cls = _class_map.get(agent_class)
    if cls is None:
        raise ValueError(
            f"Unknown agent_class '{agent_class}'. "
            f"Choose from: {list(_class_map)}"
        )

    llm_config = llm_config or {}

    if agent_class == "SingleAssistantRAG":
        if retrieve_config is None:
            raise ValueError("retrieve_config is required for SingleAssistantRAG.")
        agent = cls(
            agent_name,
            llm_config=llm_config,
            retrieve_config=retrieve_config,
            rag_description=rag_description,
            **agent_kwargs,
        )
    else:
        agent = cls(
            agent_name,
            llm_config=llm_config,
            **agent_kwargs,
        )

    agent.chat(prompt)
    return agent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_agent_factory(task: AgentTask, llm_config: dict) -> Any:
    """Default factory that builds a SingleAssistantRAG agent for a task."""
    from fin_ai.agents.workflow import SingleAssistantRAG

    retrieve_config = task.metadata.get("retrieve_config", {})
    if retrieve_config:
        return SingleAssistantRAG(
            task.agent_config,
            llm_config=llm_config,
            retrieve_config=retrieve_config,
            human_input_mode="NEVER",
        )
    else:
        from fin_ai.agents.workflow import SingleAssistant

        return SingleAssistant(
            task.agent_config,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )


def _execute_task(agent: Any, task: AgentTask) -> Any:
    """Execute a single task via the agent and return its last message."""
    agent.chat(task.prompt)
    if hasattr(agent, "user_proxy"):
        history = agent.user_proxy.chat_messages
        if history:
            last_agent = list(history.keys())[-1]
            messages = history[last_agent]
            if messages:
                return messages[-1].get("content", "")
    return None


# ---------------------------------------------------------------------------
# Pre-built pipeline builders
# ---------------------------------------------------------------------------


def build_financial_analysis_pipeline(
    tickers: list[str],
    llm_config: dict[str, Any],
    retrieve_configs: dict[str, dict[str, Any]] | None = None,
) -> AgentPipeline:
    """Build a pipeline that analyses multiple companies in sequence.

    Each company gets its own ``AgentTask``.  Tasks run in order and their
    results are stored in shared state for downstream consumption.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to analyse.
    llm_config : dict
        LLM config for all agents.
    retrieve_configs : dict, optional
        Mapping of ``{ticker: retrieve_config}``.
    """
    retrieve_configs = retrieve_configs or {}
    pipeline = AgentPipeline(
        name=f"Financial_Analysis_{'_'.join(tickers)}",
        llm_config=llm_config,
    )

    for ticker in tickers:
        task = AgentTask(
            name=f"analyze_{ticker}",
            prompt=(
                f"Analyze {ticker.upper()}'s latest financial performance. "
                f"Retrieve relevant data from SEC filings and earnings calls. "
                f"Summarize key findings: revenue trends, profitability, risks, "
                f"and forward outlook. Provide a clear, structured analysis."
            ),
            agent_config="Data_Analyst",
            metadata={
                "retrieve_config": retrieve_configs.get(ticker, {}),
            },
        )
        pipeline.add_task(task)

    # Synthesis task that depends on all company analyses
    synth_prompt = (
        "Synthesize the following individual company analyses into a "
        "comparative investment report. Highlight relative strengths, "
        "weaknesses, and rank the companies by investment attractiveness.\n\n"
    )
    for ticker in tickers:
        synth_prompt += f"Results for {ticker.upper()}: see shared state.\n"

    synth_task = AgentTask(
        name="synthesize_report",
        prompt=synth_prompt,
        agent_config="Financial_Analyst",
        depends_on=[f"analyze_{t}" for t in tickers],
    )
    pipeline.add_task(synth_task)

    return pipeline
