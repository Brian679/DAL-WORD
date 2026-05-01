from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
from django.conf import settings


def _sections(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return doc.get("sections", [])


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_number_token(text: str) -> str | None:
    match = re.search(r"\b(\d+(?:\.\d+)*)\b", text or "")
    if match:
        return match.group(1)
    return None


def find_section(doc: dict[str, Any], query: str) -> int | None:
    query_lower = _norm(query)
    query_num = _extract_number_token(query_lower)

    if not query_lower:
        return None

    # First pass: exact number token match (e.g., "4.2", "chapter 1")
    if query_num:
        for index, section in enumerate(_sections(doc)):
            title = _norm(section.get("title", ""))
            title_num = _extract_number_token(title)
            if title_num == query_num:
                return index

    # Second pass: normalized title inclusion
    for index, section in enumerate(_sections(doc)):
        title = _norm(section.get("title", ""))
        if query_lower in title:
            return index

    # Third pass: best-effort keyword overlap for natural prompts
    stop_words = {"chapter", "section", "and", "the", "of", "for", "in", "to"}
    query_words = {
        w
        for w in re.split(r"[^a-z0-9.]+", query_lower)
        if len(w) > 2 and w not in stop_words
    }
    if not query_words:
        return None
    best_index = None
    best_score = 0
    for index, section in enumerate(_sections(doc)):
        title_words = {
            w
            for w in re.split(r"[^a-z0-9.]+", _norm(section.get("title", "")))
            if len(w) > 2 and w not in stop_words
        }
        score = len(query_words & title_words)
        if score > best_score:
            best_score = score
            best_index = index
    required_score = 2 if len(query_words) >= 2 else 1
    if best_index is not None and best_score >= required_score:
        return best_index

    return None


def update_section(doc: dict[str, Any], section_idx: int, new_content: str) -> dict[str, Any]:
    sections = _sections(doc)
    if section_idx < 0 or section_idx >= len(sections):
        raise IndexError("section index out of range")
    sections[section_idx]["content"] = new_content
    return doc


def insert_after(doc: dict[str, Any], section_idx: int, title: str, content: str) -> dict[str, Any]:
    sections = _sections(doc)
    entry = {"title": title, "content": content}
    sections.insert(section_idx + 1, entry)
    return doc


def insert_image_block(doc: dict[str, Any], section_idx: int, src: str, caption: str) -> dict[str, Any]:
    sections = _sections(doc)
    blocks = sections[section_idx].setdefault("blocks", [])
    blocks.append({"type": "image", "src": src, "caption": caption})
    return doc


def generate_image(prompt: str) -> str:
    slug = prompt.lower().replace(" ", "-")[:40]
    return f"/media/images/generated-{slug}-{uuid4().hex[:8]}.png"


def generate_chart(series: list[float], chart_type: str = "line", title: str = "Generated Chart") -> str:
    charts_dir = Path(settings.MEDIA_ROOT) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"chart-{uuid4().hex[:10]}.png"
    output_path = charts_dir / file_name

    plt.figure(figsize=(8, 4.5))
    if chart_type == "bar":
        plt.bar(range(len(series)), series)
    else:
        plt.plot(series)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return f"/media/charts/{file_name}"
