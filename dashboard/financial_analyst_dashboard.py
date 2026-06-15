# streamlit run dashboard/financial_analyst_dashboard.py

"""
Financial Data Analysis — Streamlit presentation layer.

All business logic lives in ``fin_ai.core.dashboard_engine``.  This module
only contains Streamlit UI widgets and rendering helpers.
"""

from __future__ import annotations

import json
import os
import warnings
from hashlib import md5
from pathlib import Path
from time import perf_counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Reduce noisy dependency logs from transformers backends we do not use directly.
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

WARNING_PATTERNS = [
    r"Accessing `__path__` from `.models.vilt.image_processing_vilt`.*",
    r"Accessing `__path__` from `.models.aria.image_processing_aria`.*",
    r".*Behavior may be different and this alias will be removed in future versions\..*",
    r".*Disabling PyTorch because PyTorch >= 2\.4 is required but found 2\.2\.2.*",
    r".*PyTorch was not found\. Models won't be available and only tokenizers, configuration and file/data utilities can be used\..*",
]
for pattern in WARNING_PATTERNS:
    warnings.filterwarnings("ignore", message=pattern)

import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import FAISS

from dashboard import (
    DEFAULT_GITHUB_MODEL,
    DEFAULT_GITHUB_EMBEDDING_MODEL,
    GITHUB_EMBEDDING_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    OLLAMA_BASE_URL,
    VECTOR_DB_DIR,
)
from dashboard.utils import (
    execute_python_code,
    extract_python_code,
    get_embeddings,
    render_pdf_pages,
    render_csv_thumbnail,
    sanitize_generated_python_code,
)
from fin_ai.core.fin_ai_engine import (
    SUPPORTED_UPLOAD_TYPES,
    answer_question,
    build_query_source_configs,
    clear_history,
    fetch_models,
    find_source_document,
    get_source_vector_stores,
    get_vector_db_names,
    load_history,
    load_vector_stores_for_query,
    process_uploaded_document,
    purge_vector_db,
    resolve_chat_provider_env,
    resolve_embedding_provider_env,
    save_history_entry,
)
from fin_ai.core.providers import list_models
from fin_ai.core.query import format_source_citations
from fin_ai.core.rag import load_embedding_metadata

# ---------------------------------------------------------------------------
# Agent framework imports
# ---------------------------------------------------------------------------
from fin_ai.agents import (
    SingleAssistantRAG,
    SingleAssistant,
    init_engine,
    publish_research_report,
    library as agent_library,
)
from fin_ai.agents.workflow import register_tools
from autogen import register_function

st.set_page_config(page_title="Financial Data Analysis", layout="wide")
SIDEBAR_PREVIEW_WIDTH = 320


def _looks_like_embedding_model(model_id: str) -> bool:
    """Heuristic filter for embedding-capable model identifiers."""
    mid = (model_id or "").strip().lower()
    return "embedding" in mid or "embed" in mid

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def get_local_model_options() -> tuple[list[str], list[str], str | None]:
    return [DEFAULT_CHAT_MODEL], [DEFAULT_EMBEDDING_MODEL], None

@st.cache_data(show_spinner=False)
def get_cached_csv_thumbnail_path(csv_path: str, images_folder: str, source_mtime: float) -> str:
    del source_mtime
    return render_csv_thumbnail(csv_path, images_folder)

@st.cache_data(show_spinner=False)
def get_cached_pdf_page_paths(pdf_path: str, images_folder: str, zoom: float, source_mtime: float) -> tuple[str, ...]:
    del source_mtime
    return tuple(str(p) for p in render_pdf_pages(pdf_path, images_folder, zoom=zoom))

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_response_output(response_text: str, response_type: str, panel_key: str) -> None:
    if response_type == "Plain Text":
        st.text(response_text)
        return

    if response_type == "Python Code":
        raw_code = extract_python_code(response_text)
        code = sanitize_generated_python_code(raw_code)
        if code != raw_code:
            st.caption("Auto-cleaned malformed Python syntax from model output.")
        st.code(code, language="python")

        with st.expander("Run in Streamlit (Pyodide/WebAssembly)", expanded=False):
            _render_pyodide_runner(code, panel_key=panel_key)

        confirm_key = f"confirm_python_{panel_key}"
        confirmed = st.checkbox("I understand this will execute model-generated Python locally.", key=confirm_key)
        if st.button("Execute Python", key=f"execute_python_{panel_key}"):
            if not confirmed:
                st.warning("Confirm execution before running Python code.")
                return
            stdout_text, stderr_text = execute_python_code(code)
            if stdout_text.strip():
                st.text(stdout_text)
            if stderr_text.strip():
                st.error(stderr_text)
            elif not stdout_text.strip():
                st.success("Python code executed successfully with no output.")
        return

    st.markdown(response_text.replace('$', '\\$'))


def render_source_citations(citations_text: str | None, response_type: str) -> None:
    if not citations_text:
        return
    st.caption("Source-level citations")
    if response_type == "Plain Text":
        st.text(citations_text)
    else:
        st.markdown(citations_text.replace('$', '\\$'))


def _render_pyodide_runner(initial_code: str, panel_key: str) -> None:
    code_json = json.dumps(initial_code)
    html = f"""
<div style="font-family: ui-monospace, SFMono-Regular, Menlo, monospace; border: 1px solid #ddd; border-radius: 8px; padding: 10px;">
    <div style="font-family: system-ui, -apple-system, Segoe UI, sans-serif; font-size: 14px; margin-bottom: 8px;">
        A WebAssembly-powered Python kernel backed by Pyodide
    </div>
    <textarea id="code_{panel_key}" style="width: 100%; height: 170px;"></textarea>
    <div style="margin-top: 8px; display: flex; gap: 8px; align-items: center;">
        <button id="run_{panel_key}">Run In Browser</button>
        <span id="status_{panel_key}" style="font-family: system-ui, -apple-system, Segoe UI, sans-serif; font-size: 12px; color: #444;"></span>
    </div>
    <pre id="output_{panel_key}" style="white-space: pre-wrap; margin-top: 10px; background: #f7f7f7; padding: 10px; border-radius: 6px; max-height: 180px; overflow: auto;"></pre>
</div>
<script>
    const initialCode = {code_json};
    const codeEl = document.getElementById("code_{panel_key}");
    const runEl = document.getElementById("run_{panel_key}");
    const statusEl = document.getElementById("status_{panel_key}");
    const outputEl = document.getElementById("output_{panel_key}");
    codeEl.value = initialCode;
    async function ensurePyodide() {{
        if (!window.__streamlitPyodideReady) {{
            statusEl.textContent = "Loading Pyodide runtime...";
            if (!window.loadPyodide) {{
                await new Promise((resolve, reject) => {{
                    const script = document.createElement("script");
                    script.src = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js";
                    script.onload = resolve;
                    script.onerror = reject;
                    document.head.appendChild(script);
                }});
            }}
            window.__streamlitPyodideReady = await window.loadPyodide();
        }}
        return window.__streamlitPyodideReady;
    }}
    runEl.onclick = async () => {{
        runEl.disabled = true; outputEl.textContent = "";
        try {{
            const pyodide = await ensurePyodide();
            statusEl.textContent = "Running...";
            const stdout = [], stderr = [];
            pyodide.setStdout({{ batched: (msg) => stdout.push(msg) }});
            pyodide.setStderr({{ batched: (msg) => stderr.push(msg) }});
            const result = await pyodide.runPythonAsync(codeEl.value);
            const chunks = [];
            if (stdout.length) chunks.push(stdout.join("\\n"));
            if (result !== undefined) chunks.push(String(result));
            if (stderr.length) chunks.push("\\n[stderr]\\n" + stderr.join("\\n"));
            outputEl.textContent = chunks.join("\\n\\n") || "(no output)";
            statusEl.textContent = "Done";
        }} catch (err) {{ outputEl.textContent = String(err); statusEl.textContent = "Failed"; }}
        finally {{ runEl.disabled = false; }}
    }};
</script>
"""
    components.html(html, height=460, scrolling=True)


def display_pdf_in_sidebar(pdf_path: str | Path, file_name: str) -> None:
    try:
        images_folder = Path(VECTOR_DB_DIR) / file_name / "images"
        source_path = Path(pdf_path)
        source_mtime = source_path.stat().st_mtime if source_path.exists() else 0.0
        for page_index, img_path in enumerate(
            get_cached_pdf_page_paths(str(source_path), str(images_folder), zoom=1.5, source_mtime=source_mtime),
            start=1,
        ):
            st.sidebar.image(str(img_path), caption=f"Page {page_index}", width=SIDEBAR_PREVIEW_WIDTH)
    except Exception as e:
        st.sidebar.error(f"Error loading PDF: {str(e)}")


def display_csv_in_sidebar(csv_path: str | Path, file_name: str) -> None:
    try:
        images_folder = Path(VECTOR_DB_DIR) / file_name / "images"
        source_path = Path(csv_path)
        source_mtime = source_path.stat().st_mtime if source_path.exists() else 0.0
        img = get_cached_csv_thumbnail_path(str(source_path), str(images_folder), source_mtime)
        if img:
            st.sidebar.image(img, caption="CSV Preview", width=SIDEBAR_PREVIEW_WIDTH)
        else:
            st.sidebar.info("Could not generate CSV preview.")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV preview: {str(e)}")


# ---------------------------------------------------------------------------
# -- Page layout
# ---------------------------------------------------------------------------

st.title("Financial Data Analysis")

# Discover vector stores
source_vector_stores = get_source_vector_stores()
vector_db_names = get_vector_db_names(source_vector_stores)
vector_db_options = vector_db_names + ["Upload New Document"]
selected_vector_db = st.selectbox("Select Vector DB or Upload New Document", vector_db_options, index=0, key="vector_db_selector")
is_upload_mode = selected_vector_db == "Upload New Document"

available_chat_models, available_embedding_models, model_load_error = get_local_model_options()

# ---------------------------------------------------------------------------
# -- Reasoning -----------------------------------------------------------
# ---------------------------------------------------------------------------
if not is_upload_mode:
    st.sidebar.subheader("🧠 Reasoning")
    st.sidebar.caption(
        "Choose the LLM that will analyse the retrieved context and generate "
        "answers. Select a provider, model, response format, and optional "
        "financial data tools."
    )

    provider_label_to_key = {
        "Local Ollama": "ollama",
        "GitHub Models": "github",
        "DeepSeek Models": "deepseek",
    }
    default_provider = os.getenv("DEFAULT_PROVIDER", "ollama").strip().lower()
    provider_labels = list(provider_label_to_key.keys())
    default_provider_index = next(
        (i for i, k in enumerate(provider_labels) if provider_label_to_key[k] == default_provider), 0,
    )
    selected_provider_label = st.sidebar.selectbox("Select Provider", provider_labels, index=default_provider_index, key="chat_provider")
    selected_provider = provider_label_to_key[selected_provider_label]
    github_endpoint = os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")

    if selected_provider == "github":
        github_token = st.sidebar.text_input("GitHub Token", value=os.getenv("GITHUB_TOKEN", ""), type="password", key="github_token_input")
        with st.spinner("Fetching available GitHub models..."):
            gh_models = fetch_models("github", api_key=github_token)
        display_model_options = [m.id for m in gh_models] or [os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)]
        default_chat_model = os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)
        default_chat_index = display_model_options.index(default_chat_model) if default_chat_model in display_model_options else 0
        selected_model = st.sidebar.selectbox("Select GitHub Model", display_model_options, index=default_chat_index, key="github_model")
        github_endpoint = st.sidebar.text_input("GitHub Endpoint", value=github_endpoint, key="github_endpoint_input")

    elif selected_provider == "deepseek":
        deepseek_token = os.getenv("DEEPSEEK_TOKEN", "")
        with st.spinner("Fetching available DeepSeek models..."):
            ds_models = fetch_models("deepseek", api_key=deepseek_token)
        deepseek_model_ids = [m.id for m in ds_models]
        if deepseek_model_ids:
            default_ds = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
            default_ds_idx = deepseek_model_ids.index(default_ds) if default_ds in deepseek_model_ids else 0
            selected_model = st.sidebar.selectbox("Select DeepSeek Model", deepseek_model_ids, index=default_ds_idx, key="deepseek_model")
        else:
            selected_model = st.sidebar.text_input("Select DeepSeek Model", value=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL), key="deepseek_model_fallback")
        deepseek_base_url = st.sidebar.text_input("DeepSeek Base URL", value=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL), key="deepseek_base_url_input")

    else:
        default_chat_model = os.getenv("OLLAMA_MODEL", DEFAULT_CHAT_MODEL)
        default_chat_index = available_chat_models.index(default_chat_model) if default_chat_model in available_chat_models else 0
        selected_model = st.sidebar.selectbox("Select Local Chat Model", available_chat_models, index=default_chat_index, key="ollama_chat_model")

    response_type = st.sidebar.selectbox("Select Response Type", ["Plain Text", "Markdown", "Python Code"], index=1, key="response_type")
    auto_truncate_prompt = st.sidebar.checkbox("Auto-truncate prompt (gpt-5 guard)", value=True)
    use_tools = st.sidebar.checkbox("Enable function tools (financial data)", value=False)

    if use_tools:
        with st.sidebar.expander("Available tools", expanded=False):
            from fin_ai.core.tools import YAHOO_FINANCE_TOOLS
            for tool in YAHOO_FINANCE_TOOLS:
                if tool.get("type") == "function":
                    fn = tool["function"]
                    st.sidebar.markdown(f"**`{fn['name']}`** — {fn.get('description', '')}")

    if model_load_error:
        st.sidebar.warning(model_load_error)

    st.sidebar.divider()

# ---------------------------------------------------------------------------
# -- Agents ----------------------------------------------------------------
# ---------------------------------------------------------------------------

# Agent Smith icon from The Matrix (base64-encoded small SVG glyph)
_AGENT_SMITH_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" '
    'fill="none" stroke="#00ff41" stroke-width="1.8" stroke-linecap="round" '
    'stroke-linejoin="round">'
    # Sunglasses shape
    '<path d="M2 10l2 2 4-2 4 2 4-2 4 2 2-2"/>'
    # Lenses
    '<ellipse cx="6" cy="11" rx="2.5" ry="1.5" />'
    '<ellipse cx="18" cy="11" rx="2.5" ry="1.5" />'
    # Bridge
    '<line x1="8.5" y1="11" x2="15.5" y2="11"/>'
    # Temples
    '<path d="M2 10c0-2 1-4 2-5"/>'
    '<path d="M22 10c0-2-1-4-2-5"/>'
    # Earpieces
    '<circle cx="4" cy="5" r="1.2"/>'
    '<circle cx="20" cy="5" r="1.2"/>'
    '</svg>'
)
_AGENT_SMITH_HTML = f'<span style="vertical-align: middle;">{_AGENT_SMITH_SVG}</span>'

st.sidebar.markdown(
    f"### {_AGENT_SMITH_HTML} Agents",
    unsafe_allow_html=True,
)
st.sidebar.caption(
    "Run agentic workflows against your vector stores. Select an agent profile, "
    "enter a task prompt, and the agent will use its registered tools to analyse, "
    "research, and optionally publish a report."
)

agent_names = [n for n in agent_library]
selected_agent = st.sidebar.selectbox(
    "Select Agent Profile",
    agent_names,
    index=0,
    key="agent_profile",
)

agent_rag_query = st.sidebar.text_area(
    "Agent Task Prompt",
    placeholder='e.g. "Analyse NVDA financials and competitive position"',
    key="agent_prompt",
    height=80,
)

agent_run_col, agent_clear_col = st.sidebar.columns(2)
with agent_run_col:
    agent_submit = st.button("▶️ Run Agent", key="run_agent", use_container_width=True)
with agent_clear_col:
    if st.button("Clear Output", key="clear_agent", use_container_width=True):
        st.session_state.pop("agent_response", None)
        st.session_state.pop("agent_publication", None)
        st.rerun()

agent_email = st.sidebar.text_input(
    "Email report (optional)",
    placeholder="analyst@firm.com",
    key="agent_email",
)

agent_format = st.sidebar.selectbox(
    "Report format",
    ["html", "pdf"],
    index=0,
    key="agent_format",
)

# Show the embedding model that agents will use (set below in Embedding section)
st.sidebar.caption(
    "Agents use the embedding model configured in the 📐 Embedding section below "
    "when querying local vector stores."
)

st.sidebar.divider()

# ---------------------------------------------------------------------------
# -- Embedding -----------------------------------------------------------
# ---------------------------------------------------------------------------
st.sidebar.caption(
    "Configure the model that converts your documents into vector "
    "representations. Embeddings are used both when indexing new documents "
    "and when retrieving relevant context during querying."
)

embeddings_provider_label_to_key = {"Local Ollama": "ollama", "GitHub Models": "github"}
default_emb_provider = os.getenv("DEFAULT_EMBEDDINGS_PROVIDER", DEFAULT_EMBEDDINGS_PROVIDER).strip().lower()
emb_labels = list(embeddings_provider_label_to_key.keys())
default_emb_idx = next((i for i, k in enumerate(emb_labels) if embeddings_provider_label_to_key[k] == default_emb_provider), 0)
selected_emb_provider_label = st.sidebar.selectbox("Select Embedding Provider", emb_labels, index=default_emb_idx, key="emb_provider_select")
selected_emb_provider = embeddings_provider_label_to_key[selected_emb_provider_label]

if selected_emb_provider == "github":
    embedding_github_token = st.sidebar.text_input("GitHub Token (Embeddings)", value=os.getenv("GITHUB_TOKEN", ""), type="password", key="embedding_github_token")
    with st.spinner("Fetching GitHub embedding models..."):
        emb_models = fetch_models("github", api_key=embedding_github_token)
    embedding_model_ids = [m.id for m in emb_models if _looks_like_embedding_model(m.id)]
    available_embedding_models_display = embedding_model_ids if embedding_model_ids else [DEFAULT_GITHUB_EMBEDDING_MODEL, "openai/text-embedding-3-large"]
    if emb_models and not embedding_model_ids:
        st.sidebar.warning(
            "No embedding-capable GitHub models were detected from the catalog. "
            "Using safe defaults to avoid 400 errors from /embeddings."
        )
    default_emb_model_idx = available_embedding_models_display.index(DEFAULT_GITHUB_EMBEDDING_MODEL) if DEFAULT_GITHUB_EMBEDDING_MODEL in available_embedding_models_display else 0
    embeddings_base_url = os.getenv("GITHUB_EMBEDDING_BASE_URL", GITHUB_EMBEDDING_BASE_URL)
else:
    ollama_emb_endpoint = st.sidebar.text_input("Ollama Endpoint (Embeddings)", value=os.getenv("OLLAMA_ENDPOINT", OLLAMA_BASE_URL), key="embedding_ollama_endpoint")
    available_embedding_models_display = available_embedding_models
    default_emb_model_idx = available_embedding_models_display.index(DEFAULT_EMBEDDING_MODEL) if DEFAULT_EMBEDDING_MODEL in available_embedding_models_display else 0
    embeddings_base_url = ollama_emb_endpoint

selected_embedding_model = st.sidebar.selectbox("Select Embedding Model", available_embedding_models_display, index=default_emb_model_idx, key="embedding_model")

st.sidebar.divider()

# -- Upload New Document (inside Embedding section) -------------------------
if is_upload_mode:
    selected_source_type = st.sidebar.selectbox("Type of Source Document", SUPPORTED_UPLOAD_TYPES, index=0, key="source_type")
    uploaded_file = st.sidebar.file_uploader("Upload a document for analysis", type=SUPPORTED_UPLOAD_TYPES)
    if uploaded_file:
        binary = uploaded_file.getvalue()
        suffix = Path(uploaded_file.name).suffix.lower()
        st.sidebar.subheader("Uploaded Document")
        st.sidebar.write(uploaded_file.name)

        if suffix == ".pdf":
            display_pdf_in_sidebar(str(Path(VECTOR_DB_DIR) / "_temp_uploads" / f"{md5(binary).hexdigest()}{suffix}"), uploaded_file.name.split('.')[0])
        elif suffix == ".csv":
            display_csv_in_sidebar(str(Path(VECTOR_DB_DIR) / "_temp_uploads" / f"{md5(binary).hexdigest()}{suffix}"), uploaded_file.name.split('.')[0])
        else:
            st.sidebar.info(f"Preview is not available for {suffix} files.")

        if st.sidebar.button("Process Document and Store in Vector DB"):
            with st.spinner("Processing document..."):
                if selected_emb_provider == "github":
                    os.environ["GITHUB_TOKEN"] = embedding_github_token
                result = process_uploaded_document(
                    file_binary=binary,
                    file_name=uploaded_file.name,
                    embedding_model=selected_embedding_model,
                    embedding_base_url=embeddings_base_url,
                    emb_provider=selected_emb_provider,
                    source_type=selected_source_type,
                )
                st.success("Document processed and stored in the vector database.")
                st.caption(f"Document processing completed in {result['elapsed']:.2f} seconds.")
                st.rerun()

# -- Query widgets (defaults: initialised here for sidebar use,
# the actual widgets are rendered in the main panel below).
selected_query_vector_dbs: list[str] = []
retrieval_mode: str = "ensemble"
source_grouping: str = "vector_db"
if not is_upload_mode:
    selected_query_vector_dbs = st.session_state.get("query_vector_dbs", [selected_vector_db])
    if not selected_query_vector_dbs:
        selected_query_vector_dbs = [selected_vector_db]
    if selected_vector_db not in selected_query_vector_dbs:
        selected_query_vector_dbs = [selected_vector_db] + selected_query_vector_dbs
    source_grouping = st.session_state.get("source_grouping", "vector_db")
    retrieval_mode = st.session_state.get("retrieval_mode", "ensemble")

# ---------------------------------------------------------------------------
# -- Querying (sidebar: history, retrieval settings, embeddings) ---------
# ---------------------------------------------------------------------------
if not is_upload_mode:
    # -- History management -------------------------------------------------
    history_vector_db = selected_vector_db
    # Always reload history from disk so newly saved entries appear immediately.
    st.session_state["question_history"] = load_history(history_vector_db)
    st.session_state["history_vector_db"] = history_vector_db

    with st.sidebar.expander("Previous Questions", expanded=False):
        history = st.session_state.get("question_history", [])
        if not history:
            st.sidebar.caption("No saved question history yet for this vector DB.")
        else:
            if st.sidebar.button("Clear History", key=f"clear_history_{history_vector_db}"):
                clear_history(history_vector_db)
                st.session_state["question_history"] = []
                st.rerun()
            for idx, item in enumerate(history[:10], start=1):
                with st.sidebar.expander(f"Q{idx}: {item['question'][:80]}{'...' if len(item['question']) > 80 else ''}", expanded=False):
                    st.sidebar.caption(f"Model: {item['chat_model']} | Embedding: {item['embedding_model']} | Type: {item.get('response_type', 'Markdown')} | Mode: {item.get('retrieval_mode', 'ensemble')} | Time: {item['answer_seconds']:.2f}s")

    # -- Maintenance (purge) ------------------------------------------------
    with st.sidebar.expander("Maintenance", expanded=False):
        st.sidebar.warning("This permanently deletes the selected vector DB and related files.")
        confirm_purge = st.sidebar.checkbox(f"Confirm purge of '{selected_vector_db}'", key=f"confirm_purge_{selected_vector_db}")
        if st.sidebar.button("Purge Vector DB", type="secondary", disabled=not confirm_purge, key=f"purge_vector_db_{selected_vector_db}"):
            deleted = purge_vector_db(selected_vector_db)
            if deleted:
                st.session_state["question_history"] = []
                st.session_state.pop("latest_response", None)
                st.success(f"Purged vector DB '{selected_vector_db}'.")
                st.rerun()
            else:
                st.sidebar.warning(f"No files found to purge for '{selected_vector_db}'.")

    # -- Embeddings for query ------------------------------------------------
    saved_emb_meta = load_embedding_metadata(selected_vector_db)
    if saved_emb_meta:
        actual_emb_provider = saved_emb_meta["provider"]
        actual_emb_model = saved_emb_meta["model"]
        actual_emb_base_url = saved_emb_meta["base_url"]
        st.sidebar.caption(f"🔖 Using saved embedding config for **{selected_vector_db}**: `{actual_emb_provider}/{actual_emb_model} @ {actual_emb_base_url}`")
        if actual_emb_provider == "github":
            os.environ["GITHUB_TOKEN"] = embedding_github_token
        embeddings = get_embeddings(actual_emb_model, actual_emb_base_url, provider=actual_emb_provider)
    else:
        if selected_emb_provider == "github":
            os.environ["GITHUB_TOKEN"] = embedding_github_token
        embeddings = get_embeddings(selected_embedding_model, embeddings_base_url, provider=selected_emb_provider)
        st.sidebar.caption(f"Using current embedding selection: `{selected_emb_provider}/{selected_embedding_model} @ {embeddings_base_url}`")

    # -- Load vector stores --------------------------------------------------
    try:
        loaded_stores = load_vector_stores_for_query(selected_query_vector_dbs, source_vector_stores, embeddings)
    except ValueError as e:
        st.sidebar.error(str(e))
        loaded_stores = {}
    vector_store = loaded_stores.get(selected_vector_db)

    if vector_store is None:
        if selected_vector_db not in loaded_stores:
            st.sidebar.warning(f"Vector DB '{selected_vector_db}' not found.")
    else:
        source_doc = find_source_document(selected_vector_db)
        if source_doc and source_doc.suffix.lower() == ".pdf":
            display_pdf_in_sidebar(source_doc, selected_vector_db)
        elif source_doc:
            st.sidebar.info(f"Source document: {source_doc.name}")
        else:
            st.sidebar.warning("Source document not found for the selected vector DB.")

        query_source_configs = build_query_source_configs(loaded_stores, group_by=source_grouping)

        available_source_names = [c.name for c in query_source_configs]

# -- Query Vector DB Sources & retrieval settings (main panel) ---------------
if selected_vector_db != "Upload New Document":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.multiselect(
            "Query Vector DB Sources",
            vector_db_names,
            default=[selected_vector_db],
            key="query_vector_dbs",
        )
    with col2:
        retrieval_mode = st.selectbox("Retrieval Mode", ["ensemble", "separate", "routed"], index=0, key="retrieval_mode")
    with col3:
        source_grouping = st.selectbox("Group Sources By", ["vector_db", "filename", "source_type", "source"], index=0, key="source_grouping")

    if available_source_names:
        selected_source_names = st.multiselect(
            "Restrict to Source Groups",
            available_source_names,
            default=available_source_names,
            key="restrict_sources",
        )

# -- Question input ---------------------------------------------------------
question = ""
submit_clicked = False
if selected_vector_db != "Upload New Document":
    question = st.text_input("Enter your question:", placeholder="e.g., What is the company's revenue for the quarter?", key="question_input")
    submit_clicked = st.button("Submit Question")

# -- Latest response display (cached) ---------------------------------------
latest_response = st.session_state.get("latest_response")
if latest_response and not submit_clicked:
    st.subheader("Latest Response")
    render_response_output(latest_response["answer"], latest_response.get("response_type", "Markdown"), panel_key="latest_response")
    render_source_citations(latest_response.get("citations"), latest_response.get("response_type", "Markdown"))

# -- Submit and answer ------------------------------------------------------
if submit_clicked and question and selected_vector_db != "Upload New Document":
    active_configs = [c for c in query_source_configs if not selected_source_names or c.name in selected_source_names]
    if not active_configs:
        st.error("Select at least one source group before submitting a question.")
    else:
        with st.spinner("Answering your question..."):
            # Set chat provider env
            resolve_chat_provider_env(
                selected_provider,
                github_token=github_token if selected_provider == "github" else "",
                deepseek_token=deepseek_token if selected_provider == "deepseek" else "",
                deepseek_base_url=deepseek_base_url if selected_provider == "deepseek" else "",
                github_endpoint=github_endpoint,
                selected_model=selected_model,
            )

            result = answer_question(
                question,
                active_configs,
                provider=selected_provider,
                system_prompt="You are a concise financial analysis assistant.",
                temperature=0.2,
                retrieval_mode=retrieval_mode,
                auto_truncate_prompt=bool(auto_truncate_prompt),
                use_tools=use_tools,
            )

        llm_response = result.get("response")
        metadata = result.get("metadata")

        if not llm_response:
            st.error("Model returned no response.")
            st.stop()

        if metadata and metadata.prompt_truncated:
            st.warning(f"Prompt was truncated to fit gpt-5 input limits ({metadata.prompt_tokens_before_guard} -> {metadata.prompt_tokens_after_guard} tokens before sending).")

        st.session_state["latest_response"] = {"question": question, "answer": llm_response, "response_type": response_type}

        st.subheader("Latest Response")
        render_response_output(llm_response, response_type, panel_key="latest_response")
        st.caption(f"Answer generated in {result['elapsed']:.2f} seconds.")
        if metadata:
            st.caption(str(metadata))

        save_history_entry(selected_vector_db, {
            "question": question,
            "answer": llm_response,
            "vector_db": selected_vector_db,
            "chat_model": selected_model,
            "provider": selected_provider,
            "embedding_model": selected_embedding_model,
            "response_type": response_type,
            "answer_seconds": result["elapsed"],
        })
        st.session_state["question_history"] = load_history(selected_vector_db)


# ---------------------------------------------------------------------------
# -- Agent execution & display ---------------------------------------------
# ---------------------------------------------------------------------------

# Helper to build an LLM config from the current sidebar provider/model selections
def _build_agent_llm_config() -> dict:
    """Build an autogen-compatible llm_config from the current sidebar selections."""
    if selected_provider == "ollama":
        return {
            "config_list": [
                {
                    "model": selected_model,
                    "base_url": OLLAMA_BASE_URL,
                    "api_key": "ollama",
                }
            ],
            "temperature": 0,
            "timeout": 120,
        }
    elif selected_provider == "github":
        return {
            "config_list": [
                {
                    "model": selected_model,
                    "base_url": github_endpoint,
                    "api_key": github_token,
                }
            ],
            "temperature": 0,
            "timeout": 120,
        }
    else:  # deepseek
        return {
            "config_list": [
                {
                    "model": selected_model,
                    "base_url": deepseek_base_url,
                    "api_key": deepseek_token if selected_provider == "deepseek" else "",
                }
            ],
            "temperature": 0,
            "timeout": 120,
        }


if agent_submit and agent_rag_query.strip():
    # Initialise the engine bridge so local RAG context is available
    with st.spinner(f"🤖 Running {selected_agent} agent..."):
        # Initialise engine bridge (picks up defaults from dashboard env)
        engine_status = init_engine(
            chat_provider=selected_provider,
            embedding_model=selected_embedding_model,
            embedding_provider=selected_emb_provider,
            embedding_base_url=embeddings_base_url,
        )
        agent_llm_config = _build_agent_llm_config()

        _is_publisher = selected_agent == "Research_Publisher"

        if _is_publisher:
            # Research_Publisher: SingleAssistant (no RAG retrieve_config needed)
            agent = SingleAssistant(
                selected_agent,
                llm_config=agent_llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=8,
                code_execution_config=False,
            )
        else:
            # Other agents: SingleAssistantRAG with local RAG config
            agent = SingleAssistantRAG(
                selected_agent,
                llm_config=agent_llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=8,
                code_execution_config=False,
                retrieve_config={
                    "task": "qa",
                    "vector_db": None,
                    "docs_path": [],
                    "chunk_token_size": 1000,
                    "get_or_create": False,
                    "collection_name": "dashboard_agent_rag",
                    "must_break_at_empty_line": False,
                    "customized_prompt": (
                        "Context from local stores:\n{input_context}\n\n"
                        "Query: {input_question}"
                    ),
                },
                rag_description="Query local FAISS vector stores for financial context.",
            )

        # If the prompt mentions email/publish and Research_Publisher, append publishing instruction
        prompt = agent_rag_query.strip()
        email_to = agent_email.strip()
        if _is_publisher and email_to:
            prompt += (
                f"\n\nAfter completing your analysis, publish the report as {agent_format} "
                f"and email it to {email_to}."
            )
        elif _is_publisher:
            prompt += (
                f"\n\nAfter completing your analysis, publish the report as {agent_format}."
            )

        try:
            agent.chat(prompt)
            # Extract the last message
            history = agent.user_proxy.chat_messages
            last_content = ""
            if history:
                last_agent = list(history.keys())[-1]
                msgs = history[last_agent]
                if msgs:
                    last_content = msgs[-1].get("content", "")
            st.session_state["agent_response"] = last_content

            # Also try to publish if Research_Publisher and the agent didn't already
            if _is_publisher:
                pub_result = publish_research_report(
                    content=last_content or agent_rag_query,
                    title=f"{selected_agent} Report",
                    format=agent_format,
                    email=email_to,
                )
                st.session_state["agent_publication"] = pub_result

        except Exception as exc:
            st.session_state["agent_response"] = f"❌ Agent error: {exc}"


# Display agent output
agent_response = st.session_state.get("agent_response")
if agent_response:
    st.subheader(f"🤖 {selected_agent} Response")
    with st.container():
        render_response_output(agent_response, "Markdown", panel_key="agent_response")

agent_publication = st.session_state.get("agent_publication")
if agent_publication:
    st.subheader("📄 Publication Result")
    st.code(agent_publication, language="json")
