"""Grok API wrappers for the autonomous document agent.
Reads GROK_API_KEY from Django settings (loaded from .env).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests
from django.conf import settings

logger = logging.getLogger(__name__)
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = getattr(settings, "GROK_MODEL", "llama-3.3-70b-versatile")


def get_model_label() -> str:
    return "Grok"


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                part = part[4:].strip()
            if part:
                return part
    return text


def _parse_json(text: str) -> Any:
    return json.loads(_strip_fences(text))


def _request_text(prompt: str) -> str:
    api_key = getattr(settings, "GROK_API_KEY", "")
    if not api_key:
        raise ValueError("GROK_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Empty response from Grok API")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def generate_text(prompt: str) -> str:
    return _request_text(prompt)


def _json_response(prompt: str) -> Any:
    text = _request_text(
        prompt
        + "\n\nIMPORTANT: Respond with valid JSON ONLY. No markdown, no extra text."
    )
    return _parse_json(text)


def classify_intent(message: str, doc_context: str) -> dict[str, Any]:
    prompt = f"""You are an AI document assistant. Classify the user's intent.

Document excerpt:
{doc_context[:2000]}

User message: \"{message}\"

Choose ONE intent:
- enhance_document
- enhance_section
- write_section
- write_dissertation
- create_outline
- add_chart
- add_image
- write_report
- write_assignment
- write_presentation
- write_spreadsheet
- chat

Guidance:
- If user says "correct", "fix", "improve" for a specific part -> enhance_section.
- If user says "redo chapter X" or "rewrite chapter X" -> write_section with target_section.
- If user says "write full dissertation", "write thesis", or "write project on <topic>" -> write_dissertation.
- If user asks for report/assignment/powerpoint/excel -> map to the matching write_* intent.
- If user asks for a full/complete/entire project deliverable with chapters, treat it as write_dissertation.
- If user asks to generate substantial new document content, do NOT return chat.

Return JSON exactly:
{{"intent": "<intent>", "target_section": "<section name or null>", "topic": "<main topic or null>"}}"""
    try:
        result = _json_response(prompt)
        if isinstance(result, dict) and "intent" in result:
            return {
                "intent": result.get("intent", "chat"),
                "target_section": result.get("target_section"),
                "topic": result.get("topic"),
            }
    except Exception as exc:
        logger.warning("grok classify_intent failed: %s", exc)
    return {"intent": "chat", "target_section": None, "topic": None}


def chat_with_document(message: str, doc_context: str) -> str:
    prompt = (
        "You are an expert academic writing assistant embedded in a word processor.\n"
        f"Document:\n{doc_context[:5000]}\n\n"
        f"User: {message}\n\n"
        "Give a helpful, direct response. Be concise and professional."
    )
    return generate_text(prompt)


def enhance_text(text: str, topic: str, instruction: str = "") -> str:
    prompt = (
        f"Improve this academic text about '{topic}'.\n"
        f"{'Instruction: ' + instruction + chr(10) if instruction else ''}"
        "Requirements: clearer sentences, better structure, professional tone, "
        "no jargon, keep original meaning. Avoid generic AI disclaimers and robotic phrasing. "
        "Use varied sentence rhythm and specific details where appropriate.\n\n"
        f"TEXT:\n{text[:3000]}\n\n"
        "Return ONLY the improved text, nothing else."
    )
    return generate_text(prompt)


def generate_section_content(
    title: str,
    topic: str,
    context: str = "",
    word_count: int = 220,
) -> str:
    prompt = (
        f"Write a detailed academic paragraph (~{word_count} words) for the section "
        f"titled '{title}' in a research paper about: '{topic}'.\n"
        f"{'Additional context: ' + context[:700] if context else ''}\n"
        "Use clear, formal academic language. Be specific and substantive. "
        "Write with natural human flow: varied sentence lengths, precise claims, and grounded examples. "
        "Avoid filler, repeated sentence templates, and AI-sounding phrases such as 'in today's world', "
        "'it is important to note', or meta references to being an AI."
    )
    return generate_text(prompt)


def generate_outline_sections(topic: str) -> list[dict[str, Any]]:
    prompt = f"""Generate a complete dissertation chapter outline for: '{topic}'.

Return a JSON array with 8 items. Each item must have:
- title: string
- subsections: array of strings

Example item:
{{"title": "Chapter 1: Introduction", "subsections": ["1.1 Background", "1.2 Problem Statement"]}}

Include these major chapters:
Introduction, Literature Review, Methodology, Results and Analysis,
Discussion, Conclusion, References, Appendices."""
    try:
        result = _json_response(prompt)
        if isinstance(result, list) and result:
            normalized: list[dict[str, Any]] = []
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    normalized.append(
                        {
                            "title": str(item.get("title") or f"Chapter {i + 1}"),
                            "subsections": item.get("subsections", []),
                        }
                    )
                elif isinstance(item, str):
                    normalized.append({"title": item, "subsections": []})
            if normalized:
                return normalized[:8]
    except Exception as exc:
        logger.warning("grok generate_outline_sections failed: %s", exc)

    defaults = [
        "Chapter 1: Introduction",
        "Chapter 2: Literature Review",
        "Chapter 3: Methodology",
        "Chapter 4: Results and Analysis",
        "Chapter 5: Discussion",
        "Chapter 6: Conclusion",
        "References",
        "Appendices",
    ]
    return [{"title": title, "subsections": []} for title in defaults]
