"""
Request Processor — handles all RAG queries and agent interactions.

This is the central processing module used by the dashboard, notebooks,
and agent framework.  It consolidates:
- RAG query execution (vector store loading, question answering)
- Agent creation and execution (single-shot agent tasks)
- Provider configuration and model resolution
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from langchain_community.vectorstores import FAISS

import openai

from fin_ai.core.request import ModelRequest, RequestPayload
from fin_ai.core.query import (
    SourceRetrieverConfig,
    query_with_multi_source_prompting,
    build_source_retriever_configs,
)

logger = logging.getLogger(__name__)

from fin_ai.core.rag import (
    create_or_load_vector_store,
    discover_vector_stores_by_source,
    get_markdown_splits,
    load_embedding_metadata,
    load_and_convert_document,
    save_embedding_metadata,
    _create_source_metadata,
)
from fin_ai.core.providers import list_models, ModelInfo
from fin_ai.core.tools import YAHOO_FINANCE_TOOLS, execute_litellm_tool_call
from dashboard.utils import (
    append_question_history,
    get_embeddings,
    load_question_history,
    purge_vector_db_assets,
)

# Config
from fin_ai.config.fin_ai import (
    VECTOR_DB_DIR,
    OLLAMA_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    GITHUB_EMBEDDING_BASE_URL,
    QUESTION_HISTORY_DIR,
    SUPPORTED_UPLOAD_TYPES,
    SUPPORTED_SOURCE_SUFFIXES,
)


# ---------------------------------------------------------------------------
# Configuration / environment
# ---------------------------------------------------------------------------


def resolve_chat_provider_env(
    provider: str,
    *,
    github_token: str = "",
    deepseek_token: str = "",
    deepseek_base_url: str = "",
    github_endpoint: str = "",
    selected_model: str = "",
) -> None:
    """Set environment variables required by the chat provider's client builder."""
    if provider == "github":
        os.environ["GITHUB_TOKEN"] = github_token
        os.environ["GITHUB_MODEL"] = selected_model
        os.environ["GITHUB_ENDPOINT"] = github_endpoint
    elif provider == "deepseek":
        os.environ["DEEPSEEK_TOKEN"] = deepseek_token
        os.environ["DEEPSEEK_MODEL"] = selected_model
        os.environ["DEEPSEEK_BASE_URL"] = deepseek_base_url
    else:
        os.environ["OLLAMA_MODEL"] = selected_model
        os.environ["OLLAMA_ENDPOINT"] = OLLAMA_BASE_URL


def resolve_embedding_provider_env(
    emb_provider: str,
    *,
    github_token: str = "",
) -> None:
    """Set environment variables required by the embedding provider."""
    if emb_provider == "github":
        os.environ["GITHUB_TOKEN"] = github_token


def fetch_models(
    provider: str,
    api_key: str = "",
    base_url: str = "",
) -> list[ModelInfo]:
    """Fetch available models for a provider.  Returns empty list on error."""
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return list_models(provider, **kwargs)


def find_source_document(vector_db_name: str) -> Path | None:
    """Locate the original source file for a vector DB."""
    for suffix in SUPPORTED_SOURCE_SUFFIXES:
        candidate = Path(VECTOR_DB_DIR) / f"{vector_db_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def get_source_vector_stores() -> dict[str, Path]:
    """Discover vector stores and return name → path mapping."""
    return discover_vector_stores_by_source(VECTOR_DB_DIR)


def get_vector_db_names(vector_stores: dict[str, Path]) -> list[str]:
    """Return only vector DB names that have a corresponding source document."""
    return [name for name in vector_stores if find_source_document(name) is not None]


# ---------------------------------------------------------------------------
# Document indexing
# ---------------------------------------------------------------------------


def process_uploaded_document(
    file_binary: bytes,
    file_name: str,
    embedding_model: str,
    embedding_base_url: str,
    emb_provider: str,
    source_type: str,
    temp_dir: Path | str | None = None,
    github_token: str | None = None,
) -> dict[str, Any]:
    """Index an uploaded document into a FAISS vector store."""
    from hashlib import md5

    upload_hash = md5(file_binary).hexdigest()
    suffix = Path(file_name).suffix.lower()
    base_name = Path(file_name).stem

    temp_upload_dir = Path(temp_dir or Path(VECTOR_DB_DIR) / "_temp_uploads")
    temp_upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_upload_dir / f"{upload_hash}{suffix}"
    if not temp_path.exists():
        temp_path.write_bytes(file_binary)

    start_time = perf_counter()

    markdown_content = load_and_convert_document(temp_path)
    document_metadata = _create_source_metadata(temp_path, source_type)
    document_metadata["source"] = file_name
    document_metadata["filename"] = file_name
    document_metadata["source_type"] = source_type
    document_metadata["file_size"] = len(file_binary)
    chunks = get_markdown_splits(markdown_content, metadata=document_metadata)

    embeddings = get_embeddings(
        embedding_model,
        embedding_base_url,
        provider=emb_provider,
        github_token=github_token if emb_provider == "github" else None,
    )
    vector_store = create_or_load_vector_store(base_name, chunks, embeddings)
    save_embedding_metadata(base_name, provider=emb_provider, model=embedding_model, base_url=embedding_base_url)

    vector_db_path = Path(VECTOR_DB_DIR) / f"{base_name}.faiss"
    vector_store.save_local(str(vector_db_path))

    source_path = Path(VECTOR_DB_DIR) / file_name
    source_path.write_bytes(file_binary)

    temp_path.unlink(missing_ok=True)

    return {
        "vector_db_name": base_name,
        "elapsed": perf_counter() - start_time,
        "doc_metadata": document_metadata,
    }


# ---------------------------------------------------------------------------
# Vector store loading
# ---------------------------------------------------------------------------


def load_vector_stores_for_query(
    selected_vector_db_names: list[str],
    source_vector_stores: dict[str, Path],
    embeddings: Any,
) -> dict[str, FAISS]:
    """Load FAISS vector stores with dimension sanity checks."""
    import faiss

    loaded: dict[str, FAISS] = {}

    # Probe embedding dimension once — shared across all stores
    _dimension: int | None = None

    for name in selected_vector_db_names:
        if name not in source_vector_stores:
            continue
        vs_path = source_vector_stores[name]
        if not vs_path.exists():
            continue
        faiss_load_dir = str(vs_path.parent) if vs_path.is_file() else str(vs_path)

        vs = FAISS.load_local(faiss_load_dir, embeddings=embeddings, allow_dangerous_deserialization=True)

        if _dimension is None:
            probe = embeddings.embed_query("dimension probe")
            _dimension = len(probe)

        if _dimension != vs.index.d:
            raise ValueError(
                f"Embedding dimension mismatch for vector store '{name}': "
                f"index={vs.index.d}, embedding={_dimension}"
            )

        loaded[name] = vs

    return loaded


def build_query_source_configs(
    loaded_stores: dict[str, FAISS],
    group_by: str = "vector_db",
) -> list[SourceRetrieverConfig]:
    """Build source retriever configs from loaded vector stores."""
    configs: list[SourceRetrieverConfig] = []
    for name, store in loaded_stores.items():
        configs.extend(
            build_source_retriever_configs(store, base_name=name, group_by=group_by, search_k=5)
        )
    return configs


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------


def answer_question(
    question: str,
    source_configs: Sequence[SourceRetrieverConfig],
    *,
    provider: str,
    system_prompt: str = "You are a concise financial analysis assistant.",
    temperature: float = 0.2,
    retrieval_mode: str = "ensemble",
    auto_truncate_prompt: bool = True,
    use_tools: bool = False,
    response_format: str = "text",
) -> dict[str, Any]:
    """Run a RAG query against the selected vector stores."""
    start_time = perf_counter()

    effective_system = _build_tool_aware_system_prompt(system_prompt) if use_tools else system_prompt

    llm_result = query_with_multi_source_prompting(
        question,
        source_configs,
        provider=provider,
        response_format=response_format,
        mode=retrieval_mode,
        system_prompt=effective_system,
        temperature=temperature,
        auto_truncate_prompt=auto_truncate_prompt,
        tools=YAHOO_FINANCE_TOOLS if use_tools else None,
    )

    llm_response = llm_result.response

    if use_tools and llm_response is not None:
        first_message = llm_response.get_metadata().raw_response.choices[0].message
        tool_calls = _extract_tool_calls(first_message)
        if tool_calls:
            follow_up_messages = _build_tool_follow_up(effective_system, llm_result.prompt, first_message, tool_calls)
            follow_up_payload = RequestPayload(
                prompt=question,
                system_prompt=effective_system,
                temperature=temperature,
                auto_truncate_prompt=auto_truncate_prompt,
                tools=YAHOO_FINANCE_TOOLS,
                messages=follow_up_messages,
            )
            requester = ModelRequest(provider=provider, format="text")
            llm_response = requester.client.send(follow_up_payload, response_class=requester.response_class)

    content = llm_response.content if llm_response else ""
    metadata = llm_response.get_metadata() if llm_response else None

    return {
        "response": content,
        "metadata": metadata,
        "llm_result": llm_result,
        "elapsed": perf_counter() - start_time,
    }


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


def build_agent_llm_config(
    provider: str,
    model: str,
    *,
    ollama_base_url: str = "",
    github_endpoint: str = "",
    github_token: str = "",
    deepseek_base_url: str = "",
    deepseek_token: str = "",
) -> dict[str, Any]:
    """Build an AutoGen-compatible llm_config from provider settings.

    Parameters
    ----------
    provider : str
        ``"ollama"``, ``"github"``, or ``"deepseek"``.
    model : str
        Model identifier (e.g. ``"llama3.1"``, ``"openai/gpt-4o"``).
    """
    if provider == "github":
        return {
            "config_list": [
                {"model": model, "base_url": github_endpoint, "api_key": github_token}
            ],
            "temperature": 0,
            "timeout": 120,
        }
    elif provider == "deepseek":
        return {
            "config_list": [
                {"model": model, "base_url": deepseek_base_url, "api_key": deepseek_token}
            ],
            "temperature": 0,
            "timeout": 120,
        }
    else:
        base = ollama_base_url or OLLAMA_BASE_URL
        return {
            "config_list": [
                {"model": model, "base_url": base, "api_key": "ollama"}
            ],
            "temperature": 0,
            "timeout": 120,
        }


def run_agent_task(
    agent_name: str,
    prompt: str,
    llm_config: dict[str, Any],
    *,
    embedding_model: str = "",
    embedding_provider: str = "",
    embedding_base_url: str = "",
    chat_provider: str = "ollama",
    is_publisher: bool = False,
    publisher_format: str = "html",
    publisher_email: str = "",
) -> dict[str, Any]:
    """Run a single agent task and return the response.

    This is the primary entry point for programmatic agent interaction
    from the dashboard, notebooks, or any script.

    Parameters
    ----------
    agent_name : str
        Agent profile name (e.g. ``"Data_Analyst"``, ``"Research_Publisher"``).
    prompt : str
        Task prompt to send to the agent.
    llm_config : dict
        AutoGen-compatible LLM config (use ``build_agent_llm_config`` to build).
    embedding_model : str
        Embedding model for initialising the engine bridge.
    embedding_provider : str
        Embedding provider (``"ollama"`` or ``"github"``).
    embedding_base_url : str
        Embedding API base URL.
    chat_provider : str
        Chat provider for RAG queries.
    is_publisher : bool
        If True, uses ``SingleAssistant`` (no RAG); otherwise ``SingleAssistantRAG``.
    publisher_format : str
        Report format for Research_Publisher (``"html"`` or ``"pdf"``).
    publisher_email : str
        Optional email address to send the report to.

    Returns
    -------
    dict with keys ``response``, ``agent_name``, ``success``, ``error`` (if failed).
    """
    from fin_ai.agents import SingleAssistantRAG, SingleAssistant, init_engine
    from fin_ai.agents.engine_bridge import publish_research_report

    # Initialise the engine bridge for local RAG
    init_engine(
        chat_provider=chat_provider,
        embedding_model=embedding_model or None,
        embedding_provider=embedding_provider or None,
        embedding_base_url=embedding_base_url or None,
        github_token=os.environ.get("GITHUB_TOKEN") or None,
    )

    _retrieve_config = {
        "task": "qa",
        "vector_db": None,
        "docs_path": [],
        "chunk_token_size": 1000,
        "get_or_create": False,
        "collection_name": "processor_agent_rag",
        "must_break_at_empty_line": False,
        "customized_prompt": (
            "Context from local stores:\n{input_context}\n\n"
            "Query: {input_question}"
        ),
    }

    try:
        if is_publisher:
            agent = SingleAssistant(
                agent_name,
                llm_config=llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=8,
                code_execution_config=False,
            )
            # Append publishing instruction
            full_prompt = prompt.strip()
            if publisher_email:
                full_prompt += (
                    f"\n\nAfter analysis, publish as {publisher_format} and email to {publisher_email}."
                )
            else:
                full_prompt += f"\n\nAfter analysis, publish as {publisher_format}."
        else:
            agent = SingleAssistantRAG(
                agent_name,
                llm_config=llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=8,
                code_execution_config=False,
                retrieve_config=_retrieve_config,
                rag_description="Query local FAISS vector stores for financial context.",
            )
            full_prompt = prompt

        agent.chat(full_prompt)

        # Extract last message
        history = agent.user_proxy.chat_messages
        response = ""
        if history:
            last_agent = list(history.keys())[-1]
            msgs = history[last_agent]
            if msgs:
                response = msgs[-1].get("content", "")

        pub_result = None
        if is_publisher:
            try:
                pub_result = publish_research_report(
                    content=response or prompt,
                    title=f"{agent_name} Report",
                    format=publisher_format,
                    email=publisher_email,
                )
            except Exception:
                pass

        return {
            "response": response,
            "agent_name": agent_name,
            "success": True,
            "publication": pub_result,
        }

    except openai.APIConnectionError as exc:
        logger.exception("API connection error for agent '%s'", agent_name)
        return {
            "response": "",
            "agent_name": agent_name,
            "success": False,
            "error": (
                f"Cannot connect to the LLM provider. "
                f"Check that your model endpoint is running and reachable. "
                f"Details: {exc}"
            ),
        }
    except Exception as exc:
        logger.exception("Agent task '%s' failed", agent_name)
        return {
            "response": "",
            "agent_name": agent_name,
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Question history
# ---------------------------------------------------------------------------


def load_history(vector_db_name: str) -> list[dict[str, Any]]:
    return load_question_history(vector_db_name, QUESTION_HISTORY_DIR)


def save_history_entry(vector_db_name: str, entry: dict[str, Any]) -> None:
    append_question_history(vector_db_name, QUESTION_HISTORY_DIR, entry)


def clear_history(vector_db_name: str) -> None:
    from dashboard.utils import clear_question_history as _clear
    _clear(vector_db_name, QUESTION_HISTORY_DIR)


def purge_vector_db(vector_db_name: str) -> list[Path]:
    return purge_vector_db_assets(vector_db_name, Path(VECTOR_DB_DIR), QUESTION_HISTORY_DIR)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_tool_aware_system_prompt(base_prompt: str | None) -> str:
    tool_names = [
        tool.get("function", {}).get("name", "")
        for tool in YAHOO_FINANCE_TOOLS
        if tool.get("type") == "function"
    ]
    tool_names = [name for name in tool_names if name]
    available = ", ".join(tool_names) if tool_names else "none"
    guidance = (
        f"You have access to function tools. Available: {available}. "
        "When asked for stock data, financials, or analyst recs, call the "
        "appropriate tool instead of guessing."
    )
    base = (base_prompt or "").strip()
    if guidance in base:
        return base
    if not base:
        return guidance
    return f"{base}\n\n{guidance}"


def _extract_tool_calls(message: object) -> list[dict]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []
    extracted: list[dict] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            name = fn.get("name")
            args_text = fn.get("arguments", "{}")
            call_id = tc.get("id")
        else:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None)
            args_text = getattr(fn, "arguments", "{}")
            call_id = getattr(tc, "id", None)
        try:
            arguments = json.loads(args_text or "{}")
        except json.JSONDecodeError:
            arguments = {}
        extracted.append({
            "id": call_id or f"call_{len(extracted)}",
            "name": name,
            "arguments": arguments,
            "arguments_text": args_text or "{}",
        })
    return extracted


def _build_tool_follow_up(
    system: str,
    prompt: str,
    first_message: Any,
    tool_calls: list[dict],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": first_message.content or "",
            "tool_calls": [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments_text"]}}
                for tc in tool_calls
            ],
        },
    ]
    for tc in tool_calls:
        tool_result = execute_litellm_tool_call(tc["name"], tc["arguments"])
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": tc["name"] or "unknown_tool",
            "content": json.dumps(tool_result),
        })
    return messages
