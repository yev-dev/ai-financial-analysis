import ast
import os
import json
import math
import re
import statistics
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import requests
from requests.exceptions import RequestException
import matplotlib.pyplot as plt
import pandas as pd
import pymupdf
import seaborn as sns
from langchain_ollama import OllamaEmbeddings


def get_pdf_text(file_path):
    doc = pymupdf.open(file_path)
    texts = []
    for page in doc:
        temp = page.get_text()
        texts.append(temp.strip())
    doc.close()
    return texts


def render_pdf_pages(file_path, output_dir, zoom=1.5):
    doc = pymupdf.open(file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = []

    try:
        for index, page in enumerate(doc, start=1):
            image_path = output_path / f"page_{index}.png"
            if not image_path.exists():
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                image_path.write_bytes(pixmap.tobytes("png"))
            image_paths.append(image_path)
    finally:
        doc.close()

    return image_paths


def get_question_history_path(vector_db_name: str, history_dir: Path) -> Path:
    sanitized_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", vector_db_name).strip("_")
    if not sanitized_name:
        sanitized_name = "default"
    return history_dir / f"{sanitized_name}.json"


def load_question_history(vector_db_name: str, history_dir: Path) -> list[dict[str, Any]]:
    history_path = get_question_history_path(vector_db_name, history_dir)
    if not history_path.exists():
        return []

    try:
        return json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_question_history(vector_db_name: str, history_dir: Path, history: list[dict[str, Any]]) -> None:
    history_path = get_question_history_path(vector_db_name, history_dir)
    history_path.write_text(
        json.dumps(history, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def append_question_history(
    vector_db_name: str,
    history_dir: Path,
    entry: dict[str, Any],
    limit: int = 50,
) -> None:
    history = load_question_history(vector_db_name, history_dir)
    history.insert(0, entry)
    save_question_history(vector_db_name, history_dir, history[:limit])


def clear_question_history(vector_db_name: str, history_dir: Path) -> None:
    save_question_history(vector_db_name, history_dir, [])


def extract_python_code(response_text: str) -> str:
    code_block_match = re.search(
        r"```python\s*(.*?)```",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    if code_block_match:
        return code_block_match.group(1).strip()

    generic_block_match = re.search(r"```\s*(.*?)```", response_text, re.DOTALL)
    if generic_block_match:
        return generic_block_match.group(1).strip()

    return response_text.strip()


def sanitize_generated_python_code(code: str) -> str:
    cleaned_lines = []
    for line in code.splitlines():
        normalized_line = line
        if re.match(r"^\s*(from\s+\S+\s+import\s+.+|import\s+\S+.*)\s+to\s+specify\b", line):
            normalized_line = re.sub(r"\s+to\s+specify\b.*$", "", line)
        cleaned_lines.append(normalized_line)
    return "\n".join(cleaned_lines).strip()


def validate_python_code(code: str) -> None:
    allowed_modules = {
        "math",
        "statistics",
        "json",
        "matplotlib",
        "matplotlib.pyplot",
        "seaborn",
        "pandas",
        "yfinance",
    }
    blocked_names = {
        "eval",
        "exec",
        "open",
        "compile",
        "input",
        "__import__",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "shutil",
        "socket",
    }

    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed_modules:
                    raise ValueError(f"Import '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name not in allowed_modules:
                raise ValueError(f"Import from '{module_name}' is not allowed.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in blocked_names:
                raise ValueError(f"Function '{node.func.id}' is not allowed.")
        elif isinstance(node, ast.Name):
            if node.id in blocked_names:
                raise ValueError(f"Name '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("Dunder attribute access is not allowed.")


def execute_python_code(code: str) -> tuple[str, str]:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    validate_python_code(code)

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    execution_globals = {
        "__name__": "__main__",
        "__builtins__": safe_builtins,
        "plt": plt,
        "sns": sns,
        "pd": pd,
        "math": math,
        "statistics": statistics,
        "json": json,
    }

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, execution_globals, execution_globals)
    except Exception as exc:
        error_output = stderr_buffer.getvalue()
        if error_output:
            error_output = f"{error_output}\n{exc}"
        else:
            error_output = str(exc)
        return stdout_buffer.getvalue(), error_output

    return stdout_buffer.getvalue(), stderr_buffer.getvalue()


def is_embedding_model(model: dict) -> bool:
    name = str(model.get("name", "")).lower()
    details = model.get("details") or {}
    family = str(details.get("family", "")).lower()
    families = [str(item).lower() for item in details.get("families", [])]

    embedding_markers = (
        "embed",
        "embedding",
        "bge",
        "e5",
        "nomic-bert",
        "snowflake-arctic-embed",
    )

    return any(marker in name for marker in embedding_markers) or any(
        marker in family or marker in families for marker in embedding_markers
    )


def load_local_model_options(
    base_url: str,
    default_chat_model: str,
    default_embedding_model: str,
) -> tuple[list[str], list[str], str | None]:
    try:
        with urlopen(f"{base_url}/api/tags") as response:
            payload = json.load(response)
    except URLError as exc:
        return [default_chat_model], [default_embedding_model], (
            f"Unable to load local Ollama models: {exc.reason}"
        )
    except Exception as exc:
        return [default_chat_model], [default_embedding_model], (
            f"Unable to load local Ollama models: {exc}"
        )

    models = payload.get("models", [])

    chat_model_names = sorted(
        {
            model["name"]
            for model in models
            if model.get("name") and not is_embedding_model(model)
        }
    )

    embedding_model_names = sorted(
        {
            model["name"]
            for model in models
            if model.get("name") and is_embedding_model(model)
        }
    )

    if not chat_model_names:
        chat_model_names = [default_chat_model]

    if not embedding_model_names:
        embedding_model_names = [default_embedding_model]

    if not models:
        return chat_model_names, embedding_model_names, "No local Ollama models were found."

    return chat_model_names, embedding_model_names, None


def get_ollama_embeddings(model_name: str, base_url: str) -> OllamaEmbeddings:
    """Return an OllamaEmbeddings instance connected to the given base_url.

    Raises RuntimeError with actionable guidance if the model is not available or
    Ollama can't be reached.
    """
    try:
        return OllamaEmbeddings(model=model_name, base_url=base_url)
    except Exception as exc:
        error_text = str(exc)
        if "not found" in error_text.lower() and model_name in error_text:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not available in Ollama. Run: ollama pull {model_name}"
            ) from exc
        raise RuntimeError(
            f"Failed to initialize Ollama embeddings: {error_text}.\nQuick check: curl {base_url}/api/tags"
        ) from exc


def get_github_embeddings(model_name: str, base_url: str | None = None) -> object:
    """Return a lightweight GitHub embeddings adapter. GITHUB_TOKEN is optional.
    base_url takes precedence over the GITHUB_EMBEDDING_BASE_URL env variable,
    which in turn takes precedence over GITHUB_ENDPOINT.
    """
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    github_model = model_name or os.getenv("GITHUB_EMBEDDING_MODEL", "openai/text-embedding-3-small").strip()
    github_endpoint = (
        base_url
        or os.getenv("GITHUB_EMBEDDING_BASE_URL", "").strip()
        or os.getenv("GITHUB_ENDPOINT", "").strip()
        or "https://models.github.ai/inference"
    )

    if not github_model:
        raise RuntimeError("A model must be set to use GitHub embeddings.")

    class GitHubEmbeddings:
        """Minimal embeddings adapter calling the GitHub Models embeddings endpoint.

        POSTs JSON {"model": "...", "input": [...]} to `<endpoint>/embeddings`
        following the OpenAI-compatible REST API used by GitHub Models.
        """

        def __init__(self, model: str, endpoint: str, token: str):
            self.model = model
            self.endpoint = endpoint.rstrip("/")
            self.token = token
            self.headers = {
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
            }
            if self.token:
                self.headers["Authorization"] = f"Bearer {self.token}"

        def _request(self, inputs: list[str]) -> list[list[float]]:
            url = f"{self.endpoint}/embeddings"
            payload = {"model": self.model, "input": inputs}
            try:
                resp = requests.post(url, json=payload, headers=self.headers, timeout=30)
                resp.raise_for_status()
            except RequestException as re:
                raise RuntimeError(f"Failed to call GitHub embeddings endpoint {url}: {re}") from re

            data = resp.json()

            # Common response shapes: OpenAI-like {data: [{embedding: [...]}, ...]}
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    embeddings = []
                    for item in data["data"]:
                        if isinstance(item, dict) and "embedding" in item:
                            embeddings.append(item["embedding"])
                        elif isinstance(item, dict) and "embeddings" in item:
                            embeddings.append(item["embeddings"])
                        else:
                            raise RuntimeError(f"Unexpected embedding item shape: {item}")
                    return embeddings
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    return data["embeddings"]

            # If API returned a bare list-of-lists
            if isinstance(data, list) and all(isinstance(i, list) for i in data):
                return data

            raise RuntimeError(f"Unable to parse embeddings response from GitHub endpoint: {data}")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._request(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._request([text])[0]

        def __call__(self, text: str) -> list[float]:
            return self.embed_query(text)

    return GitHubEmbeddings(model=github_model, endpoint=github_endpoint, token=github_token)


def get_embeddings(model_name: str, base_url: str, provider: str = "ollama") -> object:
    """
    Return embeddings instance based on provider selection.
    provider: "ollama" or "github"
    """
    if provider == "ollama":
        return get_ollama_embeddings(model_name, base_url)
    elif provider == "github":
        return get_github_embeddings(model_name, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider}")
