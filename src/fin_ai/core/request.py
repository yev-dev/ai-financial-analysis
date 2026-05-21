from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Type

import litellm
import tiktoken
from litellm import completion
from fin_ai.core.response import ConsoleModelResponse, ModelResponse, Provider, ResponseFactory, ResponseMetadata

litellm.drop_params = True

MODEL_INPUT_TOKEN_LIMITS = {
    "gpt-5": 4000,
    "gpt-4.1-mini": 8000,
}

MODEL_SAFE_INPUT_BUDGETS = {
    "gpt-5": 3600,
    "gpt-4.1-mini": 7200,
}


def resolve_model_name(provider: Provider) -> str:
    if provider == "ollama":
        return f"ollama/{os.getenv('OLLAMA_MODEL', 'llama3.1')}"
    if provider == "github":
        return os.getenv("GITHUB_MODEL", "openai/gpt-4o")
    raise ValueError(f"Unknown provider {provider!r}.")


def get_model_input_token_limit(model: str) -> int | None:
    normalized = model.lower()
    for model_key, limit in MODEL_INPUT_TOKEN_LIMITS.items():
        if model_key in normalized:
            return limit
    return None


def get_model_safe_input_budget(model: str) -> int | None:
    normalized = model.lower()
    for model_key, budget in MODEL_SAFE_INPUT_BUDGETS.items():
        if model_key in normalized:
            return budget
    return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback heuristic: roughly 4 characters per token for English prose.
        return max(1, len(text) // 4)


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(count_tokens(str(message.get("content", ""))) for message in messages)


def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    left = 0
    right = len(text)
    best = ""

    while left <= right:
        mid = (left + right) // 2
        candidate = text[:mid]
        token_count = count_tokens(candidate)
        if token_count <= max_tokens:
            best = candidate
            left = mid + 1
        else:
            right = mid - 1

    return best.rstrip()


@dataclass(slots=True)
class RequestPayload:
    prompt: str
    system_prompt: str | None = "You are a helpful financial analysis assistant."
    temperature: float = 0.2
    max_tokens: int | None = None
    proxy_port: int | None = None
    auto_truncate_prompt: bool = True
    tools: list[dict[str, Any]] | None = None
    messages: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class PromptGuardResult:
    messages: list[dict[str, Any]]
    truncated: bool = False
    prompt_tokens_before: int | None = None
    prompt_tokens_after: int | None = None


class LiteLLMClient:
    """Low-level wrapper over LiteLLM completion calls."""

    def __init__(
        self,
        provider: Provider,
        model: str,
        api_base: str,
        api_key: str | None = None,
        proxy_host: str | None = None,
        proxy_port: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.proxy_host = proxy_host or os.getenv("PX_PROXY_HOST", "127.0.0.1")
        self.proxy_port = proxy_port

    @contextmanager
    def _proxy_env(self, proxy_port: int | None):
        """Temporarily set proxy env vars for LiteLLM HTTP calls. Useful for per-request proxy configuration within a corporate network."""
        if not proxy_port:
            yield
            return

        proxy_url = f"http://{self.proxy_host}:{proxy_port}"
        keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
        previous = {key: os.environ.get(key) for key in keys}

        for key in keys:
            os.environ[key] = proxy_url

        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def send(
        self,
        request: RequestPayload,
        response_class: Type[ModelResponse] = ConsoleModelResponse,
    ) -> ModelResponse:
        if request.messages is not None:
            messages = list(request.messages)
        else:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

        if request.auto_truncate_prompt:
            guard = _apply_model_input_guard(messages, self.model)
        else:
            current_tokens = count_message_tokens(messages)
            guard = PromptGuardResult(
                messages=messages,
                truncated=False,
                prompt_tokens_before=current_tokens,
                prompt_tokens_after=current_tokens,
            )
        messages = guard.messages

        call_args: dict[str, Any] = {
            "model": self.model,
            "api_base": self.api_base,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.tools is not None:
            call_args["tools"] = request.tools
        if self.api_key is not None:
            call_args["api_key"] = self.api_key
        if request.max_tokens is not None:
            call_args["max_tokens"] = request.max_tokens

        effective_proxy_port = request.proxy_port if request.proxy_port is not None else self.proxy_port
        with self._proxy_env(effective_proxy_port):
            raw = completion(**call_args)
        usage = getattr(raw, "usage", None)
        choice = raw.choices[0]

        metadata = ResponseMetadata(
            provider=self.provider,
            model=self.model,
            api_base=self.api_base,
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            prompt_truncated=guard.truncated,
            prompt_tokens_before_guard=guard.prompt_tokens_before,
            prompt_tokens_after_guard=guard.prompt_tokens_after,
            raw_response=raw,
        )
        return response_class(content=choice.message.content or "", metadata=metadata)


def _apply_model_input_guard(messages: list[dict[str, str]], model: str) -> PromptGuardResult:
    """Trim oversized prompts for strict-input models before API call."""
    current_tokens = count_message_tokens(messages)
    safe_budget = get_model_safe_input_budget(model)
    if safe_budget is None:
        return PromptGuardResult(messages=messages, prompt_tokens_before=current_tokens, prompt_tokens_after=current_tokens)

    if current_tokens <= safe_budget:
        return PromptGuardResult(messages=messages, prompt_tokens_before=current_tokens, prompt_tokens_after=current_tokens)

    adjusted = deepcopy(messages)
    user_indexes = [idx for idx, msg in enumerate(adjusted) if msg.get("role") == "user"]
    if not user_indexes:
        return PromptGuardResult(messages=adjusted, prompt_tokens_before=current_tokens, prompt_tokens_after=current_tokens)

    target_idx = user_indexes[-1]
    other_tokens = count_message_tokens(
        [msg for idx, msg in enumerate(adjusted) if idx != target_idx]
    )
    max_user_tokens = max(128, safe_budget - other_tokens)

    original_user_text = str(adjusted[target_idx].get("content", ""))
    trimmed_user_text = trim_text_to_tokens(original_user_text, max_user_tokens)
    truncated = trimmed_user_text != original_user_text
    if truncated:
        limit = get_model_input_token_limit(model)
        limit_note = f"{limit} token" if limit is not None else "model"
        trimmed_user_text += f"\n\n[Prompt truncated to fit {limit_note} input limit.]"
    adjusted[target_idx]["content"] = trimmed_user_text
    after_tokens = count_message_tokens(adjusted)
    return PromptGuardResult(
        messages=adjusted,
        truncated=truncated,
        prompt_tokens_before=current_tokens,
        prompt_tokens_after=after_tokens,
    )


class ModelRequest:
    """Consolidated wrapper for model requests across providers."""

    def __init__(self, provider: Provider, format: str = "console") -> None:
        self.provider = provider
        self.response_class = ResponseFactory.get(format)
        self.client = ModelRequestFactory.create(provider)

    def request(self, payload: RequestPayload) -> ModelResponse:
        return self.client.send(payload, response_class=self.response_class)


def _build_ollama_client() -> LiteLLMClient:
    endpoint = _normalize_ollama_endpoint(os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"))
    default_proxy_port = _read_proxy_port_from_env()
    return LiteLLMClient(
        provider="ollama",
        model=resolve_model_name("ollama"),
        api_base=endpoint,
        proxy_port=default_proxy_port,
    )


def _build_github_client() -> LiteLLMClient:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("Missing GITHUB_TOKEN in environment.")
    endpoint = os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")
    default_proxy_port = _read_proxy_port_from_env()
    return LiteLLMClient(
        provider="github",
        model=resolve_model_name("github"),
        api_base=endpoint,
        api_key=token,
        proxy_port=default_proxy_port,
    )


class ModelRequestFactory:
    """Factory that creates provider-specific LiteLLM clients from env config."""

    _registry: dict[str, Any] = {
        "ollama": _build_ollama_client,
        "github": _build_github_client,
    }

    @classmethod
    def register(cls, name: str, builder: Any) -> None:
        cls._registry[name] = builder

    @classmethod
    def get(cls, provider: str) -> Any:
        try:
            return cls._registry[provider]
        except KeyError:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown provider {provider!r}. Available: {available}.")

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, provider: str) -> LiteLLMClient:
        return cls.get(provider)()


def _normalize_ollama_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint[: -len("/v1")]
    return endpoint


def _read_proxy_port_from_env() -> int | None:
    """Read px proxy port from env, if configured."""
    raw = (os.getenv("PX_PROXY_PORT") or os.getenv("PROXY_PORT") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
