from pathlib import Path
import re
from textwrap import fill
from typing import Any
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
    images_dir = Path(settings.MEDIA_ROOT) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"generated-{uuid4().hex[:10]}.png"
    output_path = images_dir / file_name

    tokens = [
        token.capitalize()
        for token in re.split(r"[^a-z0-9]+", (prompt or "").lower())
        if len(token) > 3 and token not in {"with", "from", "that", "this", "into", "diagram", "image", "chart"}
    ]
    keywords: list[str] = []
    for token in tokens:
        if token not in keywords:
            keywords.append(token)
        if len(keywords) == 3:
            break
    while len(keywords) < 3:
        keywords.append(["Context", "Analysis", "Outcome"][len(keywords)])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    fig.patch.set_facecolor("#f6efe5")
    ax.set_facecolor("#fffaf5")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.06,
        0.93,
        "Concept Illustration",
        fontsize=20,
        fontweight="bold",
        color="#6b3f1d",
        ha="left",
        va="center",
    )
    ax.text(
        0.06,
        0.86,
        fill(prompt.strip() or "Generated concept image", width=52),
        fontsize=11,
        color="#5a524a",
        ha="left",
        va="top",
        linespacing=1.4,
    )

    positions = [
        (0.07, 0.18, 0.23, 0.24),
        (0.385, 0.18, 0.23, 0.24),
        (0.70, 0.18, 0.23, 0.24),
    ]
    colors = ["#f4b183", "#a9d18e", "#9dc3e6"]
    descriptions = [
        "Key context and source inputs",
        "Processing, controls, or evaluation",
        "Resulting impact or recommendation",
    ]

    for index, (x, y, width, height) in enumerate(positions):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=1.5,
            edgecolor="#7a6a58",
            facecolor=colors[index],
            alpha=0.92,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + height * 0.64,
            fill(keywords[index], width=14),
            fontsize=14,
            fontweight="bold",
            color="#2d241c",
            ha="center",
            va="center",
        )
        ax.text(
            x + width / 2,
            y + height * 0.28,
            fill(descriptions[index], width=18),
            fontsize=9.5,
            color="#3d342c",
            ha="center",
            va="center",
            linespacing=1.3,
        )

    ax.annotate("", xy=(0.385, 0.30), xytext=(0.30, 0.30), arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#6b3f1d"})
    ax.annotate("", xy=(0.70, 0.30), xytext=(0.615, 0.30), arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#6b3f1d"})

    central = FancyBboxPatch(
        (0.28, 0.56),
        0.44,
        0.14,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.8,
        edgecolor="#6b3f1d",
        facecolor="#fff2cc",
    )
    ax.add_patch(central)
    ax.text(
        0.50,
        0.63,
        fill(prompt.strip() or "Generated concept image", width=34),
        fontsize=12,
        fontweight="bold",
        color="#4a3421",
        ha="center",
        va="center",
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/media/images/{file_name}"


def generate_chart(series: list[float], chart_type: str = "line", title: str = "Generated Chart") -> str:
    charts_dir = Path(settings.MEDIA_ROOT) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"chart-{uuid4().hex[:10]}.png"
    output_path = charts_dir / file_name

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    fig.patch.set_facecolor("#fff8f0")
    ax.set_facecolor("#fffdf9")
    x_positions = list(range(1, len(series) + 1))

    if chart_type == "bar":
        bars = ax.bar(x_positions, series, color="#c26d2c", edgecolor="#7a3e14", linewidth=1.0)
        for bar, value in zip(bars, series):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#5b2d0c",
            )
    else:
        ax.plot(x_positions, series, color="#176087", linewidth=2.6, marker="o", markersize=6)
        ax.fill_between(x_positions, series, color="#9cd4f5", alpha=0.35)
        for x_value, y_value in zip(x_positions, series):
            ax.text(x_value, y_value, f" {y_value:.1f}", fontsize=9, color="#0f3e57", va="bottom")

    ax.set_title(title, fontsize=15, fontweight="bold", color="#4a3421", pad=14)
    ax.set_xlabel("Observation", color="#5a524a")
    ax.set_ylabel("Value", color="#5a524a")
    ax.set_xticks(x_positions)
    ax.grid(axis="y", color="#d9c7b8", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9d8d7f")
    ax.spines["bottom"].set_color("#9d8d7f")
    ax.tick_params(colors="#5a524a")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return f"/media/charts/{file_name}"
