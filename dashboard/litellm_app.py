import os
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import streamlit as st

# Import dashboard first so dashboard/__init__.py runs and bootstraps paths.
from dashboard import DEFAULT_GITHUB_MODEL, VECTOR_DB_DIR
from fin_ai.core.request import ModelRequest, RequestPayload
from fin_ai.core.response import ResponseFactory


st.set_page_config(page_title="LiteLLM Chat", layout="wide")
st.title("LiteLLM Provider Chat")
st.caption("Query either GitHub Models or local Ollama using fin_ai.core request/response wrappers.")

HISTORY_FILE_PATH = Path(VECTOR_DB_DIR) / "litellm_answer_history.json"


def _to_notebook_source(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return [""]
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def _extract_python_code(text: str) -> str:
    code_block_match = re.search(r"```python\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    generic_block_match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if generic_block_match:
        return generic_block_match.group(1).strip()

    return text.strip()


def _build_notebook(cells: list[dict]) -> str:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2)


def _load_answer_history() -> list[dict]:
    if not HISTORY_FILE_PATH.exists():
        return []
    try:
        payload = json.loads(HISTORY_FILE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return payload
    except Exception:
        return []


def _save_answer_history(history: list[dict]) -> None:
    HISTORY_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE_PATH.write_text(json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8")


def _history_to_notebook_cells(history: list[dict]) -> list[dict]:
    cells: list[dict] = []
    for item in history:
        cell = item.get("notebook_cell")
        if isinstance(cell, dict):
            cells.append(cell)
    return cells


if "answer_history" not in st.session_state:
    st.session_state["answer_history"] = _load_answer_history()

provider_label_to_key = {
    "GitHub Models": "github",
    "Local Ollama": "ollama",
}

with st.sidebar:
    st.header("Settings")
    provider_label = st.selectbox("Provider", list(provider_label_to_key.keys()), index=0)
    provider = provider_label_to_key[provider_label]

    response_format = st.selectbox(
        "Response Format",
        ResponseFactory.available(),
        index=ResponseFactory.available().index("markdown") if "markdown" in ResponseFactory.available() else 0,
        help="Rendering style implemented by fin_ai.core.response wrappers.",
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    max_tokens = st.number_input("Max Tokens (optional)", min_value=0, value=0, step=32)

    st.divider()
    st.subheader("Notebook Export")
    notebook_cell_language = st.selectbox(
        "Answer Cell Type",
        ["markdown", "python"],
        index=0,
        help="Each new answer is appended to the notebook using this cell type.",
    )
    notebook_filename = st.text_input("Notebook File Name", value="litellm_answers.ipynb")
    notebook_cells = _history_to_notebook_cells(st.session_state["answer_history"])
    st.caption(f"Saved answer cells (local history): {len(notebook_cells)}")

    col_clear, col_download = st.columns(2)
    with col_clear:
        if st.button("Clear Historical Answers"):
            st.session_state["answer_history"] = []
            _save_answer_history([])
            st.rerun()
    with col_download:
        notebook_json = _build_notebook(notebook_cells)
        st.download_button(
            "Download Notebook",
            data=notebook_json,
            file_name=notebook_filename,
            mime="application/x-ipynb+json",
            disabled=len(notebook_cells) == 0,
        )

    if provider == "github":
        st.text_input("GitHub Model", value=os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL), key="github_model")
        st.text_input(
            "GitHub Endpoint",
            value=os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference"),
            key="github_endpoint",
        )
        st.text_input("GitHub Token", value=os.getenv("GITHUB_TOKEN", ""), type="password", key="github_token")
    else:
        st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "llama3.1"), key="ollama_model")
        st.text_input("Ollama Endpoint", value=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"), key="ollama_endpoint")

system_prompt = st.text_area(
    "System Prompt",
    value="You are a helpful financial analysis assistant.",
    height=100,
)
user_prompt = st.text_area(
    "User Prompt",
    value="What are the top three indicators of financial health for a public company?",
    height=140,
)

if st.button("Send", type="primary"):
    if provider == "github" and not st.session_state.get("github_token"):
        st.error("GitHub Token is required when provider is GitHub Models.")
    elif not user_prompt.strip():
        st.error("User Prompt cannot be empty.")
    else:
        if provider == "github":
            os.environ["GITHUB_MODEL"] = st.session_state["github_model"]
            os.environ["GITHUB_ENDPOINT"] = st.session_state["github_endpoint"]
            os.environ["GITHUB_TOKEN"] = st.session_state["github_token"]
        else:
            os.environ["OLLAMA_MODEL"] = st.session_state["ollama_model"]
            os.environ["OLLAMA_ENDPOINT"] = st.session_state["ollama_endpoint"]

        payload = RequestPayload(
            prompt=user_prompt.strip(),
            system_prompt=system_prompt.strip() or None,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens > 0 else None,
        )

        try:
            response = ModelRequest(provider=provider, format=response_format).request(payload)
        except Exception as exc:
            st.exception(exc)
        else:
            answer_text = response.content or ""

            if notebook_cell_language == "python":
                cell_source = _to_notebook_source(_extract_python_code(answer_text))
                cell_type = "code"
            else:
                cell_source = _to_notebook_source(answer_text)
                cell_type = "markdown"

            notebook_cell = {
                "cell_type": cell_type,
                "metadata": {
                    "language": notebook_cell_language,
                },
                "source": cell_source,
                **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
            }

            metadata = response.get_metadata()
            metadata_dict = asdict(metadata)
            metadata_dict.pop("raw_response", None)

            st.session_state["answer_history"].append(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "provider": provider,
                    "model": metadata_dict.get("model"),
                    "response_format": response_format,
                    "answer": answer_text,
                    "notebook_cell": notebook_cell,
                    "metadata": metadata_dict,
                }
            )
            _save_answer_history(st.session_state["answer_history"])

            st.subheader("Response")
            if response_format == "markdown":
                st.markdown(response.render())
            else:
                st.text(response.render())

            st.subheader("Metadata")
            st.json(metadata_dict)

            with st.expander("Raw Response", expanded=False):
                st.text(str(metadata.raw_response))
