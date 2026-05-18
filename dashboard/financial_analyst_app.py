# streamlit run app.py

import json
import logging
import os
import warnings
from hashlib import md5
from pathlib import Path
from time import perf_counter
from urllib.error import URLError
from urllib.request import Request, urlopen

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Reduce noisy dependency logs from transformers backends we do not use directly.
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress known upstream transformers warnings emitted by transitive dependencies.
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
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    QUESTION_HISTORY_DIR,
    VECTOR_DB_DIR,
)


from fin_ai.core.query import (
    build_source_retriever_configs,
    format_source_citations,
    query_with_multi_source_prompting,
)
from fin_ai.agents.tools import YAHOO_FINANCE_TOOLS, execute_litellm_tool_call
from fin_ai.core.request import ModelRequest, RequestPayload
from fin_ai.core.rag import (
    _create_source_metadata,
    create_or_load_vector_store,
    discover_vector_stores_by_source,
    get_markdown_splits,
    load_embedding_metadata,
    load_and_convert_document,
    save_embedding_metadata,
)
from dashboard.utils import (
    append_question_history,
    clear_question_history,
    execute_python_code,
    extract_python_code,
    get_embeddings,
    load_question_history,
    purge_vector_db_assets,
    render_pdf_pages,
    render_csv_thumbnail,
    sanitize_generated_python_code,
)

st.set_page_config(page_title="Financial Data Analysis", layout="wide")

SIDEBAR_PREVIEW_WIDTH = 320

# Default GPT family models
DEFAULT_GITHUB_MODELS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
]


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
            render_pyodide_runner(code, panel_key=panel_key)

        confirmation_key = f"confirm_python_{panel_key}"
        confirmed = st.checkbox(
            "I understand this will execute model-generated Python locally.",
            key=confirmation_key,
        )
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


def _build_tool_aware_system_prompt(base_prompt: str | None) -> str:
    tool_names = [
        tool.get("function", {}).get("name", "")
        for tool in YAHOO_FINANCE_TOOLS
        if tool.get("type") == "function"
    ]
    tool_names = [name for name in tool_names if name]
    available_tools = ", ".join(tool_names) if tool_names else "none"

    tool_guidance = (
        "You have access to function tools via YAHOO_FINANCE_TOOLS. "
        f"Available tools: {available_tools}. "
        "When the user asks for stock prices, company facts, dividends, financial statements, "
        "or analyst recommendations, call the most appropriate tool instead of guessing. "
        "Use the exact tool arguments required by the schema. "
        "After tool output is returned, summarize clearly and reference the returned data."
    )

    base = (base_prompt or "").strip()
    if tool_guidance in base:
        return base
    if not base:
        return tool_guidance
    return f"{base}\n\n{tool_guidance}"


def _extract_tool_calls(message: object) -> list[dict]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []

    extracted: list[dict] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            function_payload = tool_call.get("function", {})
            name = function_payload.get("name")
            arguments_text = function_payload.get("arguments", "{}")
            call_id = tool_call.get("id")
        else:
            function_payload = getattr(tool_call, "function", None)
            name = getattr(function_payload, "name", None)
            arguments_text = getattr(function_payload, "arguments", "{}")
            call_id = getattr(tool_call, "id", None)

        try:
            arguments = json.loads(arguments_text or "{}")
        except json.JSONDecodeError:
            arguments = {}

        extracted.append(
            {
                "id": call_id or f"call_{len(extracted)}",
                "name": name,
                "arguments": arguments,
                "arguments_text": arguments_text or "{}",
            }
        )

    return extracted


def _execute_tool(name: str | None, arguments: dict) -> dict:
    if not name:
        return {"error": "Missing tool name"}
    return execute_litellm_tool_call(name, arguments)


def render_pyodide_runner(initial_code: str, panel_key: str) -> None:
    code_json = json.dumps(initial_code)
    html = f"""
<div style=\"font-family: ui-monospace, SFMono-Regular, Menlo, monospace; border: 1px solid #ddd; border-radius: 8px; padding: 10px;\">
    <div style=\"font-family: system-ui, -apple-system, Segoe UI, sans-serif; font-size: 14px; margin-bottom: 8px;\">
        A WebAssembly-powered Python kernel backed by Pyodide
    </div>
    <textarea id=\"code_{panel_key}\" style=\"width: 100%; height: 170px;\"></textarea>
    <div style=\"margin-top: 8px; display: flex; gap: 8px; align-items: center;\">
        <button id=\"run_{panel_key}\">Run In Browser</button>
        <span id=\"status_{panel_key}\" style=\"font-family: system-ui, -apple-system, Segoe UI, sans-serif; font-size: 12px; color: #444;\"></span>
    </div>
    <pre id=\"output_{panel_key}\" style=\"white-space: pre-wrap; margin-top: 10px; background: #f7f7f7; padding: 10px; border-radius: 6px; max-height: 180px; overflow: auto;\"></pre>
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
        runEl.disabled = true;
        outputEl.textContent = "";
        try {{
            const pyodide = await ensurePyodide();
            statusEl.textContent = "Running...";

            const stdout = [];
            const stderr = [];
            pyodide.setStdout({{ batched: (msg) => stdout.push(msg) }});
            pyodide.setStderr({{ batched: (msg) => stderr.push(msg) }});

            const result = await pyodide.runPythonAsync(codeEl.value);
            const chunks = [];
            if (stdout.length) chunks.push(stdout.join("\n"));
            if (result !== undefined) chunks.push(String(result));
            if (stderr.length) chunks.push("\n[stderr]\n" + stderr.join("\n"));
            outputEl.textContent = chunks.join("\n\n") || "(no output)";
            statusEl.textContent = "Done";
        }} catch (err) {{
            outputEl.textContent = String(err);
            statusEl.textContent = "Failed";
        }} finally {{
            runEl.disabled = false;
        }}
    }};
</script>
"""
    components.html(html, height=460, scrolling=True)


@st.cache_data(ttl=10)
def get_local_model_options() -> tuple[list[str], list[str], str | None]:
    return [DEFAULT_CHAT_MODEL], [DEFAULT_EMBEDDING_MODEL], None
    # return load_local_model_options(
    #     base_url=OLLAMA_BASE_URL,
    #     default_chat_model=DEFAULT_CHAT_MODEL,
    #     default_embedding_model=DEFAULT_EMBEDDING_MODEL,
    # )


@st.cache_data(ttl=300)
def get_github_model_options(github_token: str) -> tuple[list[str], str | None]:
    url = "https://models.github.ai/catalog/models"
    headers = {}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except URLError as exc:
        return [os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)], f"Unable to load GitHub models: {exc.reason}"
    except Exception as exc:
        return [os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)], f"Unable to load GitHub models: {exc}"

    model_ids = sorted(
        {
            str(item.get("id", "")).strip()
            for item in payload
            if isinstance(item, dict) and item.get("id")
        }
    )

    if not model_ids:
        return [os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)], "No GitHub models found in catalog."

    return model_ids, None

# Function to display PDF content as images in the sidebar
def display_pdf_in_sidebar(pdf_path, file_name):
    try:
        images_folder = Path(VECTOR_DB_DIR) / file_name / "images"
        source_path = Path(pdf_path)
        source_mtime = source_path.stat().st_mtime if source_path.exists() else 0.0
        image_paths = get_cached_pdf_page_paths(
            str(source_path),
            str(images_folder),
            zoom=1.5,
            source_mtime=source_mtime,
        )
        for page_index, img_path in enumerate(image_paths, start=1):
            st.sidebar.image(str(img_path), caption=f"Page {page_index}", width=SIDEBAR_PREVIEW_WIDTH)
    except Exception as e:
        st.sidebar.error(f"Error loading PDF: {str(e)}")


# Function to display CSV content as thumbnail in the sidebar
def display_csv_in_sidebar(csv_path, file_name):
    try:
        images_folder = Path(VECTOR_DB_DIR) / file_name / "images"
        source_path = Path(csv_path)
        source_mtime = source_path.stat().st_mtime if source_path.exists() else 0.0
        image_path = get_cached_csv_thumbnail_path(
            str(source_path),
            str(images_folder),
            source_mtime=source_mtime,
        )
        if image_path:
            st.sidebar.image(image_path, caption="CSV Preview", width=SIDEBAR_PREVIEW_WIDTH)
        else:
            st.sidebar.info("Could not generate CSV preview.")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV preview: {str(e)}")


@st.cache_data(show_spinner=False)
def get_cached_csv_thumbnail_path(
    csv_path: str,
    images_folder: str,
    source_mtime: float,
) -> str:
    # source_mtime is intentionally included to invalidate the cache on file changes.
    del source_mtime
    image_path = render_csv_thumbnail(csv_path, images_folder)
    return image_path


@st.cache_data(show_spinner=False)
def get_cached_pdf_page_paths(
    pdf_path: str,
    images_folder: str,
    zoom: float,
    source_mtime: float,
) -> tuple[str, ...]:
    # source_mtime is intentionally included to invalidate the cache on file changes.
    del source_mtime
    image_paths = render_pdf_pages(pdf_path, images_folder, zoom=zoom)
    return tuple(str(path) for path in image_paths)


SUPPORTED_UPLOAD_TYPES = ["pdf", "csv", "json", "html", "docx"]
SUPPORTED_SOURCE_SUFFIXES = [".pdf", ".csv", ".json", ".html", ".docx"]


def find_source_document(vector_db_name: str) -> Path | None:
    for suffix in SUPPORTED_SOURCE_SUFFIXES:
        candidate = Path(VECTOR_DB_DIR) / f"{vector_db_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


# Streamlit title and layout
st.title("Financial Data Analysis")

# Discover vector stores grouped by source early so we can switch the UI for upload mode.
source_vector_stores = discover_vector_stores_by_source(VECTOR_DB_DIR)
# Only show entries that have a corresponding source document file (pdf, csv, etc.)
vector_db_names = [name for name in source_vector_stores if find_source_document(name) is not None]
vector_db_options = vector_db_names + ["Upload New Document"]
selected_vector_db = st.selectbox("Select Vector DB or Upload New Document", vector_db_options, index=0)
is_upload_mode = selected_vector_db == "Upload New Document"

available_chat_models, available_embedding_models, model_load_error = get_local_model_options()

# Provider selection for both chat and embeddings
provider_label_to_key = {
    "GitHub Models": "github",
    "Local Ollama": "ollama",
    
}
selected_provider_label = st.selectbox("Select Provider", list(provider_label_to_key.keys()))
selected_provider = provider_label_to_key[selected_provider_label]
github_endpoint = os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")
github_model_error = None

if selected_provider == "github":
    github_token = st.text_input(
        "GitHub Token",
        value=os.getenv("GITHUB_TOKEN", ""),
        type="password",
    )
    display_model_options, github_model_error = get_github_model_options(github_token)
    default_chat_model = os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)
    default_chat_index = (
        display_model_options.index(default_chat_model)
        if default_chat_model in display_model_options
        else 0
    )
    selected_model = st.selectbox(
        "Select GitHub Model",
        display_model_options,
        index=default_chat_index,
    )
    if github_model_error:
        st.warning(github_model_error)
        selected_model = st.text_input(
            "Or Enter GitHub Model Manually",
            value=selected_model,
        )

    github_endpoint = st.text_input(
        "GitHub Endpoint",
        value=github_endpoint,
    )
    github_embedding_endpoint = st.text_input(
        "Embedding Endpoint",
        value=os.getenv("GITHUB_EMBEDDING_BASE_URL", GITHUB_EMBEDDING_BASE_URL),
    )
    available_embedding_models_display = [
        DEFAULT_GITHUB_EMBEDDING_MODEL,
        "openai/text-embedding-3-large",
    ]
    default_embedding_index = 0
else:
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    default_chat_model = os.getenv("OLLAMA_MODEL", DEFAULT_CHAT_MODEL)
    default_chat_index = (
        available_chat_models.index(default_chat_model)
        if default_chat_model in available_chat_models
        else 0
    )
    selected_model = st.selectbox(
        "Select Local Chat Model",
        available_chat_models,
        index=default_chat_index,
    )
    github_embedding_endpoint = OLLAMA_BASE_URL
    available_embedding_models_display = available_embedding_models
    default_embedding_index = (
        available_embedding_models.index(DEFAULT_EMBEDDING_MODEL)
        if DEFAULT_EMBEDDING_MODEL in available_embedding_models
        else 0
    )

if is_upload_mode:
    selected_source_type = st.selectbox(
        "Type of Source Document",
        SUPPORTED_UPLOAD_TYPES,
        index=0,
    )
embedding_label = "Select Embedding Model" if selected_provider == "github" else "Select Local Embedding Model"
selected_embedding_model = st.selectbox(
    embedding_label,
    available_embedding_models_display,
    index=default_embedding_index,
)

response_type = st.selectbox(
    "Select Response Type",
    ["Plain Text", "Markdown", "Python Code"],
    index=1,
)

auto_truncate_prompt = st.checkbox(
    "Auto-truncate prompt (gpt-5 guard)",
    value=True,
    help="Disable only for debugging request-size failures with strict-input models.",
)

use_tools = st.checkbox(
    "Enable function tools (financial data)",
    value=False,
    help=(
        "Allow the model to call yfinance tools for live stock data, company info, "
        "dividends, financial statements, and analyst recommendations."
    ),
)

if use_tools:
    with st.expander("Available tools", expanded=False):
        for tool in YAHOO_FINANCE_TOOLS:
            if tool.get("type") == "function":
                fn = tool["function"]
                st.markdown(f"**`{fn['name']}`** — {fn.get('description', '')}")

if model_load_error:
    st.warning(model_load_error)

st.caption(
    "Use the same embedding model that was used when the vector DB was created."
)


if not is_upload_mode:
    with st.expander("Maintenance", expanded=False):
        st.warning("This permanently deletes the selected vector DB and related files.")
        confirm_purge = st.checkbox(
            f"Confirm purge of '{selected_vector_db}'",
            key=f"confirm_purge_{selected_vector_db}",
        )
        if st.button(
            "Purge Vector DB",
            type="secondary",
            disabled=not confirm_purge,
            key=f"purge_vector_db_{selected_vector_db}",
        ):
            deleted_paths = purge_vector_db_assets(
                vector_db_name=selected_vector_db,
                vector_db_dir=Path(VECTOR_DB_DIR),
                history_dir=Path(QUESTION_HISTORY_DIR),
            )
            if deleted_paths:
                st.session_state["question_history"] = []
                st.session_state.pop("latest_response", None)
                st.success(f"Purged vector DB '{selected_vector_db}'.")
                st.rerun()
            else:
                st.warning(f"No files found to purge for '{selected_vector_db}'.")

history_vector_db = selected_vector_db if not is_upload_mode else "__upload__"
if st.session_state.get("history_vector_db") != history_vector_db:
    st.session_state["history_vector_db"] = history_vector_db
    st.session_state["question_history"] = load_question_history(history_vector_db, QUESTION_HISTORY_DIR)

if is_upload_mode:
    uploaded_file = st.file_uploader(
        "Upload a document for analysis",
        type=SUPPORTED_UPLOAD_TYPES,
    )

    if uploaded_file:
        document_binary = uploaded_file.getvalue()
        upload_hash = md5(document_binary).hexdigest()
        document_suffix = Path(uploaded_file.name).suffix.lower()
        st.sidebar.subheader("Uploaded Document")
        st.sidebar.write(uploaded_file.name)

        temp_upload_dir = Path(VECTOR_DB_DIR) / "_temp_uploads"
        temp_upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_upload_dir / f"{upload_hash}{document_suffix}"
        if not temp_path.exists():
            temp_path.write_bytes(document_binary)

        if document_suffix == ".pdf":
            display_pdf_in_sidebar(str(temp_path), uploaded_file.name.split('.')[0])
        elif document_suffix == ".csv":
            display_csv_in_sidebar(str(temp_path), uploaded_file.name.split('.')[0])
        else:
            st.sidebar.info(f"Preview is not available for {document_suffix} files.")

        if st.button("Process Document and Store in Vector DB"):
            with st.spinner("Processing document..."):
                start_time = perf_counter()
                markdown_content = load_and_convert_document(temp_path)
                document_metadata = _create_source_metadata(temp_path, selected_source_type)
                document_metadata["source"] = uploaded_file.name
                document_metadata["filename"] = uploaded_file.name
                document_metadata["source_type"] = selected_source_type
                document_metadata["file_size"] = len(document_binary)
                chunks = get_markdown_splits(markdown_content, metadata=document_metadata)

                embeddings_url = github_embedding_endpoint if selected_provider == "github" else OLLAMA_BASE_URL
                embeddings = get_embeddings(selected_embedding_model, embeddings_url, provider=selected_provider)

                vector_store = create_or_load_vector_store(uploaded_file.name.split(".")[0], chunks, embeddings)
                save_embedding_metadata(
                    uploaded_file.name.split(".")[0],
                    provider=selected_provider,
                    model=selected_embedding_model,
                    base_url=embeddings_url,
                )

                # Ensure vector DB and PDF are stored correctly
                vector_db_path = Path(VECTOR_DB_DIR) / f"{uploaded_file.name.split('.')[0]}.faiss"
                vector_store.save_local(str(vector_db_path))

                source_path = Path(VECTOR_DB_DIR) / uploaded_file.name
                with open(source_path, "wb") as f:
                    f.write(document_binary)

                st.success("Document processed and stored in the vector database.")
                st.caption(
                    f"Document processing completed in {perf_counter() - start_time:.2f} seconds."
                )

                temp_path.unlink(missing_ok=True)
                st.rerun()


selected_query_vector_dbs: list[str] = []
retrieval_mode = "ensemble"
source_grouping = "vector_db"
use_llm_query_planner = False
max_routed_sources = 1
vector_store = None
loaded_vector_stores: dict[str, FAISS] = {}
query_source_configs = []
selected_source_names: list[str] = []

history_vector_db = selected_vector_db if selected_vector_db != "Upload New Document" else "__upload__"
if st.session_state.get("history_vector_db") != history_vector_db:
    st.session_state["history_vector_db"] = history_vector_db
    st.session_state["question_history"] = load_question_history(history_vector_db, QUESTION_HISTORY_DIR)

if selected_vector_db != "Upload New Document":
    with st.expander("Previous Questions", expanded=False):
        history = st.session_state.get("question_history", [])
        if not history:
            st.caption("No saved question history yet for this vector DB.")
        else:
            if st.button("Clear History", key=f"clear_history_{history_vector_db}"):
                clear_question_history(history_vector_db, QUESTION_HISTORY_DIR)
                st.session_state["question_history"] = []
                st.rerun()

            for index, item in enumerate(history[:10], start=1):
                st.markdown(f"**Q:** {item['question']}")
                st.caption(
                    f"Model: {item['chat_model']} | Embedding: {item['embedding_model']} | "
                    f"Type: {item.get('response_type', 'Markdown')} | "
                    f"Mode: {item.get('retrieval_mode', 'ensemble')} | Time: {item['answer_seconds']:.2f}s"
                )
                if item.get("answer"):
                    render_response_output(
                        item["answer"],
                        item.get("response_type", "Markdown"),
                        panel_key=f"history_{history_vector_db}_{index}",
                    )
                    render_source_citations(item.get("citations"), item.get("response_type", "Markdown"))
                if st.button("Reuse Question", key=f"reuse_question_{history_vector_db}_{index}"):
                    st.session_state["question_input"] = item["question"]
                    st.rerun()

if selected_vector_db != "Upload New Document":
    selected_query_vector_dbs = st.multiselect(
        "Query Vector DB Sources",
        vector_db_names,
        default=[selected_vector_db],
    )
    if not selected_query_vector_dbs:
        selected_query_vector_dbs = [selected_vector_db]
    if selected_vector_db not in selected_query_vector_dbs:
        selected_query_vector_dbs = [selected_vector_db] + selected_query_vector_dbs

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["ensemble", "separate", "routed"],
        index=0,
    )
    source_grouping = st.selectbox(
        "Group Sources By",
        ["vector_db", "filename", "source_type", "source"],
        index=0,
    )
    use_llm_query_planner = st.checkbox(
        "Use LLM query planner for routed mode",
        value=True,
        disabled=retrieval_mode != "routed",
    )
    max_routed_sources = st.slider(
        "Max Routed Sources",
        min_value=1,
        max_value=max(1, len(selected_query_vector_dbs) * 4),
        value=min(max(1, len(selected_query_vector_dbs)), max(1, len(selected_query_vector_dbs) * 4)),
        disabled=retrieval_mode != "routed",
    )

    embeddings_url = github_embedding_endpoint if selected_provider == "github" else OLLAMA_BASE_URL
    embeddings = get_embeddings(selected_embedding_model, embeddings_url, provider=selected_provider)

    for query_vector_db in selected_query_vector_dbs:
        if query_vector_db in source_vector_stores:
            query_vector_db_path = source_vector_stores[query_vector_db]
            if query_vector_db_path.exists():
                # FAISS.load_local expects a directory; if the discovered path is the
                # .faiss file itself, use its parent directory instead.
                faiss_load_dir = (
                    str(query_vector_db_path.parent)
                    if query_vector_db_path.is_file()
                    else str(query_vector_db_path)
                )
                loaded_vector_stores[query_vector_db] = FAISS.load_local(
                    faiss_load_dir,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True,
                )

    vector_store = loaded_vector_stores.get(selected_vector_db)
    if vector_store is None:
        st.sidebar.warning(f"Vector DB '{selected_vector_db}' not found.")
    else:
        source_document = find_source_document(selected_vector_db)
        if source_document and source_document.suffix.lower() == ".pdf":
            display_pdf_in_sidebar(source_document, selected_vector_db)
        elif source_document:
            st.sidebar.info(f"Source document: {source_document.name}")
        else:
            st.sidebar.warning("Source document not found for the selected vector DB.")

        for query_vector_db, query_vector_store in loaded_vector_stores.items():
            query_source_configs.extend(
                build_source_retriever_configs(
                    query_vector_store,
                    base_name=query_vector_db,
                    group_by=source_grouping,
                    search_k=5,
                )
            )

        if not query_source_configs:
            query_source_configs = build_source_retriever_configs(
                vector_store,
                base_name=selected_vector_db,
                group_by="vector_db",
                search_k=5,
            )

        available_source_names = [config.name for config in query_source_configs]
        selected_source_names = st.multiselect(
            "Restrict to Source Groups",
            available_source_names,
            default=available_source_names,
        )

# Question input section (hidden while uploading a new document)
question = ""
submit_clicked = False
if selected_vector_db != "Upload New Document":
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the company's revenue for the quarter?",
        key="question_input",
    )

    submit_clicked = st.button("Submit Question")

latest_response = st.session_state.get("latest_response")
if latest_response and not submit_clicked:
    st.subheader("Latest Response")
    render_response_output(
        latest_response["answer"],
        latest_response.get("response_type", "Markdown"),
        panel_key="latest_response",
    )
    render_source_citations(
        latest_response.get("citations"),
        latest_response.get("response_type", "Markdown"),
    )

# Button to process and generate answers
if submit_clicked and question and selected_vector_db != "Upload New Document":
    active_source_configs = [
        config for config in query_source_configs
        if not selected_source_names or config.name in selected_source_names
    ]
    if not active_source_configs:
        st.error("Select at least one source group before submitting a question.")
    else:
        with st.spinner("Answering your question..."):
            start_time = perf_counter()

            if selected_provider == "github":
                if not github_token:
                    st.error("GitHub Token is required when GitHub Models provider is selected.")
                    st.stop()
                os.environ["GITHUB_TOKEN"] = github_token
                os.environ["GITHUB_MODEL"] = selected_model
                os.environ["GITHUB_ENDPOINT"] = github_endpoint
            else:
                os.environ["OLLAMA_MODEL"] = selected_model
                os.environ["OLLAMA_ENDPOINT"] = OLLAMA_BASE_URL

            base_system_prompt = "You are a concise financial analysis assistant."
            effective_system_prompt = (
                _build_tool_aware_system_prompt(base_system_prompt)
                if use_tools
                else base_system_prompt
            )

            llm_result = query_with_multi_source_prompting(
                question,
                active_source_configs,
                provider=selected_provider,
                response_format="text",
                mode=retrieval_mode,
                system_prompt=effective_system_prompt,
                temperature=0.2,
                auto_truncate_prompt=bool(auto_truncate_prompt),
                tools=YAHOO_FINANCE_TOOLS if use_tools else None,
            )

            llm_response = llm_result.response
            if use_tools and llm_response is not None:
                first_message = llm_response.get_metadata().raw_response.choices[0].message
                tool_calls = _extract_tool_calls(first_message)
                if tool_calls:
                    follow_up_messages = [
                        {"role": "system", "content": effective_system_prompt},
                        {"role": "user", "content": llm_result.prompt},
                        {
                            "role": "assistant",
                            "content": first_message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments_text"],
                                    },
                                }
                                for tc in tool_calls
                            ],
                        },
                    ]
                    for tc in tool_calls:
                        tool_result = _execute_tool(tc["name"], tc["arguments"])
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
                        system_prompt=effective_system_prompt,
                        temperature=0.2,
                        auto_truncate_prompt=bool(auto_truncate_prompt),
                        tools=YAHOO_FINANCE_TOOLS,
                        messages=follow_up_messages,
                    )
                    requester = ModelRequest(provider=selected_provider, format="text")
                    llm_response = requester.client.send(
                        follow_up_payload,
                        response_class=requester.response_class,
                    )

        if llm_response is None:
            st.error("Model returned no response.")
            st.stop()

        response = llm_response.content
        metadata = llm_response.get_metadata()
        if metadata.prompt_truncated:
            before = metadata.prompt_tokens_before_guard
            after = metadata.prompt_tokens_after_guard
            st.warning(
                f"Prompt was truncated to fit gpt-5 input limits"
                f" ({before} -> {after} tokens before sending)."
            )

        st.session_state["latest_response"] = {
            "question": question,
            "answer": response,
            "response_type": response_type,
        }

        st.subheader("Latest Response")
        render_response_output(response, response_type, panel_key="latest_response")

        st.caption(
            f"Answer generated in {perf_counter() - start_time:.2f} seconds."
        )
        st.caption(str(metadata))

        history_entry = {
            "question": question,
            "answer": response,
            "vector_db": selected_vector_db,
            "chat_model": selected_model,
            "provider": selected_provider,
            "embedding_model": selected_embedding_model,
            "response_type": response_type,
            "answer_seconds": perf_counter() - start_time,
        }
        append_question_history(selected_vector_db, QUESTION_HISTORY_DIR, history_entry)
        st.session_state["question_history"] = load_question_history(selected_vector_db, QUESTION_HISTORY_DIR)

