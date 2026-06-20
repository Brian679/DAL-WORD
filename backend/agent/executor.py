from typing import Any

from documents.models import Document, DocumentVersion

from .gemini import generate_outline_sections, generate_section_content
from .planner import create_dissertation_outline
from .tools import find_section, generate_chart, generate_image, insert_after, insert_image_block, update_section

import re


def _sanitize_body(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _save(document: Document, note: str) -> Document:
    document.save(update_fields=["content", "updated_at"])
    DocumentVersion.objects.create(document=document, content=document.content, note=note)
    return document


def enhance_section(document: Document, query: str, instruction: str) -> Document:
    idx = find_section(document.content, query)
    if idx is None:
        raise ValueError(f"section not found for query: {query}")

    section = document.content["sections"][idx]
    original = section.get("content", "")
    topic = document.content.get("topic", document.title)
    try:
        enhanced = generate_section_content(
            title=section.get("title", query),
            topic=topic,
            context=f"{instruction}\n\n{original}",
        )
    except Exception:
        enhanced = f"{original}\n\nRefined draft: {instruction}."
    enhanced = _sanitize_body(enhanced)
    document.content = update_section(document.content, idx, enhanced)
    return _save(document, f"enhance:{query}")


def create_outline(document: Document, topic: str) -> Document:
    try:
        titles = generate_outline_sections(topic)
        sections = [
            {
                "title": t.get("title", "Untitled Section") if isinstance(t, dict) else str(t),
                "content": "",
            }
            for t in titles
        ]
    except Exception:
        # Fallback to static outline if Gemini is unavailable
        outline = create_dissertation_outline(topic)
        sections = [{"title": item.title, "content": ""} for item in outline]
    document.content = {"topic": topic, "sections": sections}
    return _save(document, "outline-generated")


def insert_chart(
    document: Document,
    section_query: str,
    series: list[float],
    chart_type: str = "line",
    title: str | None = None,
) -> Document:
    idx = find_section(document.content, section_query)
    if idx is None:
        raise ValueError(f"section not found for query: {section_query}")

    section_title = document.content["sections"][idx].get("title", "")
    chart_path = generate_chart(
        series=series,
        chart_type=chart_type,
        title=title or f"{section_title or section_query} overview",
    )
    document.content = insert_image_block(
        document.content,
        idx,
        src=chart_path,
        caption=f"Chart for {section_title}",
    )
    return _save(document, f"chart:{section_query}")


def insert_concept_image(document: Document, section_query: str, prompt: str) -> Document:
    idx = find_section(document.content, section_query)
    if idx is None:
        raise ValueError(f"section not found for query: {section_query}")

    image_path = generate_image(prompt)
    section_title = document.content["sections"][idx].get("title", "")
    document.content = insert_image_block(
        document.content,
        idx,
        src=image_path,
        caption=f"Image for {section_title}",
    )
    return _save(document, f"image:{section_query}")


def insert_section(document: Document, after_query: str, title: str, content: str) -> Document:
    idx = find_section(document.content, after_query)
    if idx is None:
        raise ValueError(f"section not found for query: {after_query}")

    document.content = insert_after(document.content, idx, title=title, content=content)
    return _save(document, f"insert-after:{after_query}")


def update_table_of_contents(document: Document) -> Document:
    """Recompute the Table of Contents / List of Figures / List of Tables from
    the document's current sections, mirroring Word's "Update Table" action."""
    from .autonomous import _refresh_preliminary_pages

    sections = (document.content or {}).get("sections", [])
    _refresh_preliminary_pages(sections)
    document.content = {**(document.content or {}), "sections": sections}
    return _save(document, "update-table-of-contents")


def run_action(document: Document, action: str, payload: dict[str, Any]) -> Document:
    if action == "enhance_section":
        return enhance_section(document, payload["query"], payload["instruction"])
    if action == "generate_outline":
        return create_outline(document, payload["topic"])
    if action == "generate_chart":
        return insert_chart(
            document,
            payload["query"],
            payload.get("series", [1, 2, 3]),
            payload.get("chart_type", "line"),
            payload.get("title"),
        )
    if action == "generate_image":
        return insert_concept_image(document, payload["query"], payload["prompt"])
    if action == "insert_section":
        return insert_section(document, payload["after_query"], payload["title"], payload["content"])
    if action == "update_table_of_contents":
        return update_table_of_contents(document)
    raise ValueError(f"unsupported action: {action}")
