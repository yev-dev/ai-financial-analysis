from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Type

Provider = Literal["ollama", "github"]


@dataclass(slots=True)
class ResponseMetadata:
    """Token usage and model provenance returned alongside every completion."""

    provider: Provider
    model: str
    api_base: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw_response: Any = None

    def __str__(self) -> str:
        col = 20
        lines = [
            f"{'Provider':<{col}}{self.provider}",
            f"{'Model':<{col}}{self.model}",
            f"{'API Base':<{col}}{self.api_base}",
        ]
        if self.finish_reason:
            lines.append(f"{'Finish Reason':<{col}}{self.finish_reason}")
        if self.prompt_tokens is not None:
            lines.append(f"{'Prompt Tokens':<{col}}{self.prompt_tokens}")
        if self.completion_tokens is not None:
            lines.append(f"{'Completion Tokens':<{col}}{self.completion_tokens}")
        if self.total_tokens is not None:
            lines.append(f"{'Total Tokens':<{col}}{self.total_tokens}")
        return "\n".join(lines)


@dataclass
class ModelResponse(ABC):
    """Abstract model response; subclasses choose the rendering format."""

    content: str
    metadata: ResponseMetadata

    @abstractmethod
    def render(self) -> str:
        """Return the full formatted response string."""

    def get_metadata(self) -> ResponseMetadata:
        return self.metadata

    def to_console(self) -> None:
        print(self.render())

    def __str__(self) -> str:
        return self.render()


@dataclass
class TextModelResponse(ModelResponse):
    def render(self) -> str:
        return "\n".join([
            str(self.metadata),
            "",
            "Response:",
            self.content or "<empty response>",
        ])


@dataclass
class ConsoleModelResponse(ModelResponse):
    def render(self) -> str:
        sep = "-" * 52
        return "\n".join([
            sep,
            str(self.metadata),
            sep,
            self.content or "<empty response>",
            sep,
        ])


@dataclass
class MarkdownModelResponse(ModelResponse):
    def render(self) -> str:
        meta = self.metadata
        lines = [
            "## Model Response",
            f"- **Provider:** {meta.provider}",
            f"- **Model:** {meta.model}",
            f"- **API Base:** {meta.api_base}",
        ]
        if meta.finish_reason:
            lines.append(f"- **Finish Reason:** {meta.finish_reason}")
        if meta.prompt_tokens is not None:
            lines.append(f"- **Prompt Tokens:** {meta.prompt_tokens}")
        if meta.completion_tokens is not None:
            lines.append(f"- **Completion Tokens:** {meta.completion_tokens}")
        if meta.total_tokens is not None:
            lines.append(f"- **Total Tokens:** {meta.total_tokens}")
        lines.extend(["", "### Response", self.content or "<empty response>"])
        return "\n".join(lines)


RESPONSE_TYPES: dict[str, Type[ModelResponse]] = {
    "text": TextModelResponse,
    "console": ConsoleModelResponse,
    "markdown": MarkdownModelResponse,
}


class ResponseFactory:
    """Factory for selecting and instantiating the correct response class."""

    _registry: dict[str, Type[ModelResponse]] = dict(RESPONSE_TYPES)

    @classmethod
    def register(cls, name: str, response_class: Type[ModelResponse]) -> None:
        if not issubclass(response_class, ModelResponse):
            raise TypeError(f"{response_class!r} must be a subclass of ModelResponse.")
        cls._registry[name] = response_class

    @classmethod
    def get(cls, format: str) -> Type[ModelResponse]:
        try:
            return cls._registry[format]
        except KeyError:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown response format {format!r}. Available: {available}.")

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, format: str, content: str, metadata: ResponseMetadata) -> ModelResponse:
        return cls.get(format)(content=content, metadata=metadata)
