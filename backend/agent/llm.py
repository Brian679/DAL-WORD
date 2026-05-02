from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from . import gemini, grok

_ACTIVE_MODEL: ContextVar[str] = ContextVar("active_model", default="gemini")


def _normalize_model_choice(model_choice: str | None) -> str:
    choice = (model_choice or "gemini").strip().lower()
    if choice in {"grok", "xai"}:
        return "grok"
    return "gemini"


def set_active_model(model_choice: str | None) -> str:
    normalized = _normalize_model_choice(model_choice)
    _ACTIVE_MODEL.set(normalized)
    return normalized


def get_active_model() -> str:
    return _ACTIVE_MODEL.get()


def _provider_module():
    if get_active_model() == "grok":
        return grok
    return gemini


def get_model_label() -> str:
    return _provider_module().get_model_label()


def classify_intent(message: str, doc_context: str) -> dict[str, Any]:
    return _provider_module().classify_intent(message, doc_context)


def chat_with_document(message: str, doc_context: str) -> str:
    return _provider_module().chat_with_document(message, doc_context)


def generate_text(prompt: str) -> str:
    return _provider_module().generate_text(prompt)


def enhance_text(text: str, topic: str, instruction: str = "") -> str:
    return _provider_module().enhance_text(text, topic, instruction)


def generate_outline_sections(topic: str) -> list[dict[str, Any]]:
    return _provider_module().generate_outline_sections(topic)


def generate_section_content(
    title: str,
    topic: str,
    context: str = "",
    word_count: int = 220,
) -> str:
    return _provider_module().generate_section_content(
        title=title,
        topic=topic,
        context=context,
        word_count=word_count,
    )


def create_execution_plan(intent: str) -> list[str]:
    return gemini.create_execution_plan(intent)
