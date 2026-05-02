"""Gemini API wrappers for the autonomous document agent.
Reads GEMINI_API_KEY from Django settings (loaded from .env).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import google.generativeai as genai
from django.conf import settings

logger = logging.getLogger(__name__)
MODEL_NAME = "gemini-1.5-flash"


def _model() -> genai.GenerativeModel:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def get_model_label() -> str:
    return "Gemini 1.5 Flash"


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


def generate_text(prompt: str) -> str:
    response = _model().generate_content(prompt)
    return (response.text or "").strip()


def _json_response(prompt: str) -> Any:
    response = _model().generate_content(
        prompt
        + "\n\nIMPORTANT: Respond with valid JSON ONLY. No markdown, no extra text."
    )
    return _parse_json(response.text or "")


def classify_intent(message: str, doc_context: str) -> dict[str, Any]:
    """Classify a user message into one autonomous action intent."""
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
- If user asks for a full/complete/entire project or long-form deliverable, do NOT return chat; choose the closest write_* intent.

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
        logger.warning("classify_intent failed: %s", exc)
    return {"intent": "chat", "target_section": None, "topic": None}


def create_execution_plan(intent: str) -> list[str]:
    plans = {
        "enhance_document": [
            "Reading and analysing document structure",
            "Identifying weak sections",
            "Enhancing each section for clarity and academic tone",
            "Checking language flow and consistency",
            "Saving updated document",
        ],
        "enhance_section": [
            "Locating target section",
            "Analysing existing content",
            "Rewriting with improved clarity and academic tone",
            "Saving updated section",
        ],
        "write_section": [
            "Locating or creating target section",
            "Generating academic content",
            "Inserting content into document",
            "Saving changes",
        ],
        "write_dissertation": [
            "Generating dissertation outline",
            "Writing Introduction",
            "Writing Literature Review",
            "Writing Methodology",
            "Writing Results and Analysis",
            "Writing Discussion",
            "Writing Conclusion",
            "Saving complete dissertation",
        ],
        "create_outline": [
            "Generating chapter structure for topic",
            "Creating section entries",
            "Saving outline to document",
        ],
        "write_report": [
            "Building report structure",
            "Writing each report section",
            "Saving report document",
        ],
        "write_assignment": [
            "Building assignment structure",
            "Writing each assignment section",
            "Saving assignment document",
        ],
        "write_presentation": [
            "Building slide outline",
            "Writing slide-by-slide content",
            "Saving presentation draft",
        ],
        "write_spreadsheet": [
            "Designing worksheet structure",
            "Generating table-ready content",
            "Saving spreadsheet draft",
        ],
        "add_chart": [
            "Locating target section",
            "Generating chart from data",
            "Inserting chart into section",
            "Saving document",
        ],
        "add_image": [
            "Locating target section",
            "Generating image",
            "Inserting image into section",
            "Saving document",
        ],
        "chat": [
            "Reading document context",
            "Understanding user intent",
            "Reviewing relevant document sections",
            "Drafting response strategy",
            "Generating response",
            "Final quality pass",
            "Returning chat summary",
        ],
    }
    return plans.get(intent, ["Analysing request", "Executing", "Saving"])


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
    """Generate structured dissertation chapters with optional subsections."""
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
        logger.warning("generate_outline_sections failed: %s", exc)

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
