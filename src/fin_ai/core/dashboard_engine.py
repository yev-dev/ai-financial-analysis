"""Backend engine for the financial analyst dashboard.

Contains all business logic, data processing, query execution, and environment
management extracted from the Streamlit presentation layer.  This module is
imported by ``dashboard.financial_analyst_dashboard`` and can also be used
standalone for scripting / testing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from langchain_community.vectorstores import FAISS

from fin_ai.core.request import ModelRequest, RequestPayload
from fin_ai.core.query import (
    SourceRetrieverConfig,
    query_with_multi_source_prompting,
    build_source_retriever_configs,
    format_source_citations,
)
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
from fin_ai.agents.tools import YAHOO_FINANCE_TOOLS, execute_litellm_tool_call
from dashboard.utils import (
    append_question_history,
    get_embeddings,
    load_question_history,
    purge_vector_db_assets,
)
from dashboard import (
    DEFAULT_GITHUB_MODEL,
    DEFAULT_GITHUB_EMBEDDING_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    QUESTION_HISTORY_DIR,
    VECTOR_DB_DIR,
)


SUPPORTED_UPLOAD_TYPES = ["pdf", "csv", "json", "html", "docx"]
SUPPORTED_SOURCE_SUFFIXES = [".pdf", ".csv", ".json", ".html", ".docx"]


# ---------------------------------------------------------------------------
# Configuration / defaults
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
    """Locate the original source file for a vector DB by trying known suffixes."""
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
# Document upload / indexing
# ---------------------------------------------------------------------------


def process_uploaded_document(
    file_binary: bytes,
    file_name: str,
    embedding_model: str,
    embedding_base_url: str,
    emb_provider: str,
    source_type: str,
    temp_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Index an uploaded document into a FAISS vector store.

    Returns a dict with keys ``vector_db_name``, ``elapsed``, and
    ``doc_metadata``.
    """
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

    embeddings = get_embeddings(embedding_model, embedding_base_url, provider=emb_provider)
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
# Vector store loading for querying
# ---------------------------------------------------------------------------


def load_vector_stores_for_query(
    selected_vector_db_names: list[str],
    source_vector_stores: dict[str, Path],
    embeddings: Any,
) -> dict[str, FAISS]:
    """Load FAISS vector stores for the selected names.

    Each store is loaded with the provided ``embeddings`` instance.
    """
    loaded: dict[str, FAISS] = {}
    for name in selected_vector_db_names:
        if name not in source_vector_stores:
            continue
        vs_path = source_vector_stores[name]
        if not vs_path.exists():
            continue
        faiss_load_dir = str(vs_path.parent) if vs_path.is_file() else str(vs_path)
        loaded[name] = FAISS.load_local(faiss_load_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
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
    """Run a query against the selected vector stores and return results.

    Returns a dict with keys ``response``, ``metadata``, ``llm_result``,
    ``elapsed``.
    """
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
            follow_up_messages: list[dict[str, Any]] = [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": llm_result.prompt},
                {
                    "role": "assistant",
                    "content": first_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments_text"]},
                        }
                        for tc in tool_calls
                    ],
                },
            ]
            for tc in tool_calls:
                tool_result = execute_litellm_tool_call(tc["name"], tc["arguments"])
                follow_up_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["name"] or "unknown_tool",
                        "content": json.dumps(tool_result),
                    }
                )
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
# Helpers ported from the original presentation layer
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
        "You have access to function tools via YAHOO_FINANCE_TOOLS. "
        f"Available tools: {available}. "
        "When the user asks for stock prices, company facts, dividends, financial statements, "
        "or analyst recommendations, call the most appropriate tool instead of guessing. "
        "Use the exact tool arguments required by the schema. "
        "After tool output is returned, summarize clearly and reference the returned data."
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
