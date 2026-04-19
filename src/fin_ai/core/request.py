from __future__ import annotations

import os
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Type

from litellm import completion
from fin_ai.core.response import ConsoleModelResponse, ModelResponse, Provider, ResponseFactory, ResponseMetadata


@dataclass(slots=True)
class RequestPayload:
    prompt: str
    system_prompt: str | None = "You are a helpful financial analysis assistant."
    temperature: float = 0.2
    max_tokens: int | None = None
    proxy_port: int | None = None


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
        """Temporarily set proxy env vars for LiteLLM HTTP calls."""
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
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        call_args: dict[str, Any] = {
            "model": self.model,
            "api_base": self.api_base,
            "messages": messages,
            "temperature": request.temperature,
        }
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
            raw_response=raw,
        )
        return response_class(content=choice.message.content or "", metadata=metadata)


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
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    default_proxy_port = _read_proxy_port_from_env()
    return LiteLLMClient(
        provider="ollama",
        model=f"ollama/{model_name}",
        api_base=endpoint,
        proxy_port=default_proxy_port,
    )


def _build_github_client() -> LiteLLMClient:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("Missing GITHUB_TOKEN in environment.")
    endpoint = os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")
    model_name = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
    default_proxy_port = _read_proxy_port_from_env()
    return LiteLLMClient(
        provider="github",
        model=model_name,
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
