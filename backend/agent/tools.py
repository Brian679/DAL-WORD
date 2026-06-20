from pathlib import Path
import json
import re
from statistics import mean
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


# ── per-subject matplotlib drawers ─────────────────────────────────────────

def _draw_apple(ax: Any) -> None:
    from matplotlib.patches import Ellipse
    import numpy as np
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#e8f5e9")
    ax.add_patch(Ellipse((0.50, 0.46), 0.46, 0.43, color="#e63946", zorder=3))
    ax.add_patch(Ellipse((0.39, 0.55), 0.13, 0.09, color="#ff8fa3", alpha=0.52, zorder=4))
    ax.add_patch(Ellipse((0.50, 0.675), 0.055, 0.032, color="#c1121f", zorder=4))
    sx = [0.500, 0.508, 0.518, 0.528]; sy = [0.675, 0.730, 0.780, 0.815]
    ax.plot(sx, sy, color="#5c3a1e", lw=5, zorder=5, solid_capstyle="round")
    ax.add_patch(Ellipse((0.585, 0.815), 0.18, 0.065, angle=-32, color="#40916c", zorder=4))
    ax.plot([0.528, 0.618], [0.815, 0.823], color="#2d6a4f", lw=1.2, zorder=5, alpha=0.75)


def _draw_tree(ax: Any) -> None:
    from matplotlib.patches import Rectangle, Circle
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#e3f2fd")
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.14, color="#8fbc5c", zorder=2))
    ax.add_patch(Rectangle((0.44, 0.14), 0.12, 0.28, color="#795548", zorder=3))
    for cy, r, col in [(0.62, 0.21, "#66bb6a"), (0.73, 0.19, "#43a047"), (0.82, 0.155, "#2e7d32")]:
        ax.add_patch(Circle((0.50, cy), r, color=col, zorder=4))


def _draw_sun(ax: Any) -> None:
    from matplotlib.patches import Circle
    import numpy as np
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#fffde7")
    ax.add_patch(Circle((0.50, 0.50), 0.22, color="#fdd835", zorder=3))
    ax.add_patch(Circle((0.50, 0.50), 0.19, color="#ffee58", zorder=4))
    for angle in range(0, 360, 30):
        rad = angle * 3.14159 / 180
        x0 = 0.50 + 0.25 * np.cos(rad); y0 = 0.50 + 0.25 * np.sin(rad)
        x1 = 0.50 + 0.36 * np.cos(rad); y1 = 0.50 + 0.36 * np.sin(rad)
        ax.plot([x0, x1], [y0, y1], color="#f9a825", lw=4, solid_capstyle="round", zorder=2)


def _draw_heart(ax: Any) -> None:
    import numpy as np
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#fce4ec")
    t = np.linspace(0, 2 * np.pi, 400)
    x = 0.50 + 0.22 * (16 * np.sin(t) ** 3) / 16
    y = 0.50 + 0.22 * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)) / 13
    ax.fill(x, y, color="#e91e63", zorder=3, alpha=0.9)


def _draw_star(ax: Any) -> None:
    import numpy as np
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#fffde7")
    angles_outer = np.linspace(np.pi / 2, 5 * np.pi / 2, 6)[:-1]
    angles_inner = angles_outer + np.pi / 5
    pts = []
    for ao, ai in zip(angles_outer, angles_inner):
        pts.append((0.50 + 0.35 * np.cos(ao), 0.50 + 0.35 * np.sin(ao)))
        pts.append((0.50 + 0.15 * np.cos(ai), 0.50 + 0.15 * np.sin(ai)))
    xs, ys = zip(*pts)
    ax.fill(list(xs), list(ys), color="#fdd835", zorder=3)
    ax.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], color="#f57f17", lw=2, zorder=4)


def _draw_house(ax: Any) -> None:
    from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch as FBP
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#e3f2fd")
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.12, color="#a5d6a7", zorder=2))
    ax.add_patch(Rectangle((0.18, 0.12), 0.64, 0.40, color="#ef9a9a", zorder=3, ec="#c62828", lw=1.5))
    ax.add_patch(Rectangle((0.44, 0.12), 0.12, 0.20, color="#5d4037", zorder=4))
    ax.add_patch(plt.Circle((0.462, 0.22), 0.012, color="#ffd54f", zorder=5))
    for wx in [0.24, 0.66]:
        ax.add_patch(Rectangle((wx, 0.27), 0.11, 0.11, color="#b3e5fc", zorder=4, ec="#0277bd", lw=1.2))
    ax.add_patch(Polygon([[0.12, 0.52], [0.50, 0.80], [0.88, 0.52]], closed=True,
                          color="#b71c1c", zorder=3, ec="#7f0000", lw=1.5))


def _draw_flower(ax: Any) -> None:
    from matplotlib.patches import Circle, Ellipse
    import numpy as np
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#f3e5f5")
    ax.plot([0.50, 0.50], [0.10, 0.44], color="#388e3c", lw=5, solid_capstyle="round", zorder=2)
    ax.add_patch(Ellipse((0.38, 0.30), 0.16, 0.07, angle=25, color="#66bb6a", zorder=3))
    ax.add_patch(Ellipse((0.62, 0.35), 0.16, 0.07, angle=-25, color="#66bb6a", zorder=3))
    petal_colors = ["#ce93d8", "#ab47bc", "#9c27b0", "#7b1fa2", "#e040fb", "#ea80fc"]
    for i, angle in enumerate(range(0, 360, 60)):
        rad = angle * np.pi / 180
        cx = 0.50 + 0.155 * np.cos(rad); cy = 0.50 + 0.155 * np.sin(rad)
        ax.add_patch(Ellipse((cx, cy), 0.17, 0.10, angle=angle, color=petal_colors[i % 6], zorder=4))
    ax.add_patch(Circle((0.50, 0.50), 0.085, color="#fdd835", zorder=5))


def _draw_book(ax: Any) -> None:
    from matplotlib.patches import Rectangle, FancyBboxPatch as FBP
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#fff8e1")
    ax.add_patch(Rectangle((0.18, 0.18), 0.58, 0.62, color="#4527a0", zorder=2))
    for i in range(8):
        shade = 220 + i * 4
        ax.add_patch(Rectangle((0.76 + i * 0.008, 0.21), 0.006, 0.56,
                                color=(shade/255, shade/255, shade/255), zorder=3))
    ax.add_patch(FBP((0.20, 0.20), 0.54, 0.58, boxstyle="round,pad=0.008",
                     color="#7c43bd", zorder=4, ec="#4527a0", lw=1.5))
    ax.add_patch(Rectangle((0.20, 0.20), 0.055, 0.58, color="#5e35b1", zorder=5))
    for y_pos, w_frac in [(0.61, 0.32), (0.55, 0.26), (0.45, 0.18)]:
        ax.add_patch(Rectangle((0.34, y_pos), w_frac, 0.03, color="#ede7f6", zorder=6))


def _draw_cloud(ax: Any) -> None:
    from matplotlib.patches import Circle
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#e3f2fd")
    for cx, cy, r in [(0.50, 0.52, 0.18), (0.35, 0.46, 0.14), (0.65, 0.46, 0.14),
                      (0.42, 0.40, 0.12), (0.58, 0.40, 0.12)]:
        ax.add_patch(Circle((cx, cy), r, color="white", zorder=3, ec="#b0bec5", lw=1.0))


def _draw_banana(ax: Any) -> None:
    import numpy as np
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#fffde7")
    t = np.linspace(0, np.pi, 300)
    xo = 0.50 + 0.34 * np.cos(t); yo = 0.50 + 0.28 * np.sin(t)
    xi = 0.50 + 0.20 * np.cos(t); yi = 0.50 + 0.16 * np.sin(t)
    px = list(xo) + list(xi[::-1]); py = list(yo) + list(yi[::-1])
    ax.fill(px, py, color="#fdd835", zorder=3)
    ax.plot(xo, yo, color="#f9a825", lw=3, zorder=4)
    for tip_x, tip_y in [(xo[0], yo[0]), (xo[-1], yo[-1])]:
        ax.plot(tip_x, tip_y, "o", color="#795548", ms=7, zorder=5)


def _draw_fish(ax: Any) -> None:
    from matplotlib.patches import Ellipse
    import numpy as np
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#e0f7fa")
    ax.add_patch(Ellipse((0.46, 0.50), 0.50, 0.26, color="#29b6f6", zorder=3))
    tail = np.array([[0.21, 0.50], [0.08, 0.65], [0.06, 0.50], [0.08, 0.35]])
    ax.fill(tail[:, 0], tail[:, 1], color="#0288d1", zorder=3)
    ax.add_patch(plt.Circle((0.66, 0.52), 0.030, color="white", zorder=5))
    ax.add_patch(plt.Circle((0.665, 0.52), 0.015, color="#212121", zorder=6))


def _draw_car(ax: Any) -> None:
    from matplotlib.patches import Rectangle, FancyBboxPatch as FBP, Ellipse
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#eceff1")
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.22, color="#607d8b", zorder=2))
    ax.add_patch(FBP((0.08, 0.22), 0.84, 0.22, boxstyle="round,pad=0.02",
                     color="#ef5350", zorder=3, ec="#b71c1c", lw=1.5))
    ax.add_patch(FBP((0.22, 0.44), 0.56, 0.20, boxstyle="round,pad=0.015",
                     color="#e53935", zorder=4, ec="#b71c1c", lw=1.5))
    ax.add_patch(FBP((0.27, 0.47), 0.20, 0.12, boxstyle="round,pad=0.01",
                     color="#b3e5fc", zorder=5, ec="#0277bd", lw=1))
    ax.add_patch(FBP((0.53, 0.47), 0.20, 0.12, boxstyle="round,pad=0.01",
                     color="#b3e5fc", zorder=5, ec="#0277bd", lw=1))
    for wx in [0.24, 0.76]:
        ax.add_patch(plt.Circle((wx, 0.21), 0.10, color="#212121", zorder=6))
        ax.add_patch(plt.Circle((wx, 0.21), 0.055, color="#bdbdbd", zorder=7))


_SUBJECT_DRAWERS: dict[str, Any] = {
    "apple": _draw_apple,
    "tree": _draw_tree,
    "sun": _draw_sun,
    "heart": _draw_heart,
    "star": _draw_star,
    "house": _draw_house,
    "flower": _draw_flower,
    "book": _draw_book,
    "cloud": _draw_cloud,
    "banana": _draw_banana,
    "fish": _draw_fish,
    "car": _draw_car,
}


def _draw_concept_diagram(ax: Any, prompt: str, keywords: list[str]) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.06, 0.93, "Concept Illustration", fontsize=20, fontweight="bold",
            color="#6b3f1d", ha="left", va="center")
    ax.text(0.06, 0.86, fill(prompt.strip() or "Generated concept image", width=52),
            fontsize=11, color="#5a524a", ha="left", va="top", linespacing=1.4)
    positions = [(0.07, 0.18, 0.23, 0.24), (0.385, 0.18, 0.23, 0.24), (0.70, 0.18, 0.23, 0.24)]
    colors = ["#f4b183", "#a9d18e", "#9dc3e6"]
    descs = ["Key context and source inputs", "Processing, controls, or evaluation",
             "Resulting impact or recommendation"]
    for i, (x, y, w, h) in enumerate(positions):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.018,rounding_size=0.03",
                                    linewidth=1.5, edgecolor="#7a6a58",
                                    facecolor=colors[i], alpha=0.92))
        ax.text(x + w / 2, y + h * 0.64, fill(keywords[i], width=14), fontsize=14,
                fontweight="bold", color="#2d241c", ha="center", va="center")
        ax.text(x + w / 2, y + h * 0.28, fill(descs[i], width=18), fontsize=9.5,
                color="#3d342c", ha="center", va="center", linespacing=1.3)
    ax.annotate("", xy=(0.385, 0.30), xytext=(0.30, 0.30),
                arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#6b3f1d"})
    ax.annotate("", xy=(0.70, 0.30), xytext=(0.615, 0.30),
                arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#6b3f1d"})
    ax.add_patch(FancyBboxPatch((0.28, 0.56), 0.44, 0.14,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1.8, edgecolor="#6b3f1d", facecolor="#fff2cc"))
    ax.text(0.50, 0.63, fill(prompt.strip() or "Generated concept image", width=34),
            fontsize=12, fontweight="bold", color="#4a3421", ha="center", va="center")


def _as_list_text(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return items[:4]
    return fallback


def _draw_framework_diagram(ax: Any, spec: dict[str, Any]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = str(spec.get("title") or "Conceptual Framework")
    left_label = str(spec.get("left_label") or "Independent Variables")
    middle_label = str(spec.get("middle_label") or "Mediating Variables")
    right_label = str(spec.get("right_label") or "Dependent Variable")
    control_label = str(spec.get("control_label") or "Control Variables")
    notes = str(spec.get("notes") or "").strip()

    left_items = _as_list_text(spec.get("left_items"), ["Input 1", "Input 2", "Input 3"])
    middle_items = _as_list_text(spec.get("middle_items"), ["Process factor"])
    right_items = _as_list_text(spec.get("right_items"), ["Outcome"])
    control_items = _as_list_text(spec.get("control_items"), ["Context", "Time"])

    ax.add_patch(FancyBboxPatch(
        (0.03, 0.88),
        0.94,
        0.09,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor="#1f3a5f",
        facecolor="#e9f2ff",
    ))
    ax.text(0.50, 0.925, fill(title, width=58), fontsize=15, fontweight="bold", color="#0f2742", ha="center", va="center")

    panels = [
        (0.06, 0.33, 0.24, 0.44, "#f4f9f1", left_label, left_items),
        (0.38, 0.33, 0.24, 0.44, "#fff7ea", middle_label, middle_items),
        (0.70, 0.33, 0.24, 0.44, "#edf6ff", right_label, right_items),
    ]
    for x, y, w, h, color, label, items in panels:
        ax.add_patch(FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.3,
            edgecolor="#2f3d4a",
            facecolor=color,
        ))
        ax.text(x + w / 2, y + h - 0.06, fill(label, width=24), fontsize=11, fontweight="bold", color="#1d2b36", ha="center", va="center")

        item_text = "\n".join(f"- {item}" for item in items)
        ax.text(x + 0.02, y + h - 0.11, fill(item_text, width=26), fontsize=9.2, color="#2d3b46", ha="left", va="top", linespacing=1.35)

    ax.annotate("", xy=(0.38, 0.55), xytext=(0.30, 0.55), arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#2b4d6f"})
    ax.annotate("", xy=(0.70, 0.55), xytext=(0.62, 0.55), arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#2b4d6f"})

    ax.add_patch(FancyBboxPatch(
        (0.12, 0.12),
        0.76,
        0.14,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#586e80",
        facecolor="#f7fbff",
    ))
    ax.text(0.50, 0.225, fill(control_label, width=50), fontsize=10.5, fontweight="bold", color="#244054", ha="center", va="center")
    ax.text(
        0.50,
        0.165,
        fill(" | ".join(control_items), width=84),
        fontsize=9,
        color="#3d5466",
        ha="center",
        va="center",
    )

    if notes:
        ax.text(0.03, 0.04, fill(f"Note: {notes}", width=120), fontsize=8.3, color="#516472", ha="left", va="bottom")


def _draw_process_flow(ax: Any, steps: list[str], title: str = "") -> None:
    """Render a horizontal process-flow / methodology flowchart."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("#f7f9fc")
    n = min(len(steps), 5) or 1
    steps = steps[:n]
    colors = ["#4e89e0", "#5aac6f", "#e0884e", "#9c59d1", "#d95f5f"]
    box_w = 0.14
    gap = (1.0 - n * box_w) / (n + 1)
    box_h = 0.30
    box_y = 0.32
    if title:
        ax.text(0.50, 0.94, fill(title, width=60), fontsize=14, fontweight="bold",
                color="#1a2a3a", ha="center", va="center")
    for i, step in enumerate(steps):
        x = gap + i * (box_w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, box_y), box_w, box_h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            linewidth=1.5, edgecolor=colors[i % len(colors)],
            facecolor=colors[i % len(colors)] + "33",
        ))
        step_num = f"Step {i + 1}"
        ax.text(x + box_w / 2, box_y + box_h * 0.70, step_num, fontsize=9,
                fontweight="bold", color=colors[i % len(colors)], ha="center", va="center")
        ax.text(x + box_w / 2, box_y + box_h * 0.38,
                fill(step, width=16), fontsize=8.5, color="#1a2a3a",
                ha="center", va="center", linespacing=1.25)
        if i < n - 1:
            arrow_x = x + box_w + 0.008
            ax.annotate("", xy=(arrow_x + gap - 0.016, box_y + box_h / 2),
                        xytext=(arrow_x, box_y + box_h / 2),
                        arrowprops={"arrowstyle": "-|>", "lw": 1.8,
                                    "color": "#5a7a9a"})


def _draw_timeline(ax: Any, phases: list[str], title: str = "") -> None:
    """Render a vertical research-phases timeline."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    n = min(len(phases), 6) or 1
    phases = phases[:n]
    colors = ["#2b5eb6", "#1a8f5a", "#c26d2c", "#8b4fcf", "#b51c1c", "#0d7a7a"]
    if title:
        ax.text(0.50, 0.95, fill(title, width=62), fontsize=14, fontweight="bold",
                color="#1a2a3a", ha="center", va="center")
    ax.plot([0.38, 0.38], [0.06, 0.88], color="#aab6c8", linewidth=2, zorder=1)
    step_h = 0.82 / max(n, 1)
    for i, phase in enumerate(phases):
        y = 0.88 - i * step_h - step_h * 0.5
        ax.plot(0.38, y, "o", markersize=11, color=colors[i % len(colors)], zorder=3)
        ax.plot(0.38, y, "o", markersize=6, color="white", zorder=4)
        ax.text(0.41, y, fill(phase, width=42), fontsize=9.5, color="#1a2a3a",
                ha="left", va="center", linespacing=1.3)
        ax.text(0.35, y, f"Phase {i + 1}", fontsize=8.5, fontweight="bold",
                color=colors[i % len(colors)], ha="right", va="center")


def generate_image(prompt: str, framework_spec: dict[str, Any] | None = None) -> str:
    images_dir = Path(settings.MEDIA_ROOT) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"generated-{uuid4().hex[:10]}.png"
    output_path = images_dir / file_name

    words = re.split(r"[^a-z]+", (prompt or "").lower())
    drawer = next((fn for w in words if (fn := _SUBJECT_DRAWERS.get(w))), None)

    # ── Detect academic diagram type from prompt keywords ────────────────────
    _is_process = any(k in (prompt or "").lower() for k in [
        "process", "step", "procedure", "methodology", "flowchart", "flow chart",
        "workflow", "pipeline", "stages of", "steps of", "phases of", "how to",
    ])
    _is_timeline = any(k in (prompt or "").lower() for k in [
        "timeline", "gantt", "schedule", "phases", "research phase", "time frame",
        "milestones", "roadmap", "sequence",
    ])

    def _ai_generate_process_steps(n: int = 5) -> list[str]:
        """Ask the LLM to suggest steps/phases for the given prompt."""
        from .llm import generate_text as _gen
        try:
            raw = _gen(
                f"List {n} short step/phase labels (3-5 words each) for: '{prompt}'. "
                "Return JSON array of strings only, e.g. [\"Data Collection\", \"Analysis\"]."
            )
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                items = json.loads(m.group(0))
                items = [str(i).strip() for i in items if str(i).strip()]
                if items:
                    return items[:n]
        except Exception:
            pass
        return [f"Step {i + 1}" for i in range(n)]

    if framework_spec:
        fig, ax = plt.subplots(figsize=(12.4, 7.2), dpi=180)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        _draw_framework_diagram(ax, framework_spec)
    elif _is_timeline:
        phases = _ai_generate_process_steps(6)
        fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
        fig.patch.set_facecolor("#f8f9fa")
        ax.set_facecolor("#f8f9fa")
        _draw_timeline(ax, phases, title=prompt.strip()[:72])
    elif _is_process:
        steps = _ai_generate_process_steps(5)
        fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
        fig.patch.set_facecolor("#f7f9fc")
        ax.set_facecolor("#f7f9fc")
        _draw_process_flow(ax, steps, title=prompt.strip()[:72])
    elif drawer:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
        drawer(ax)
        fig.patch.set_facecolor(ax.get_facecolor())
        subject_label = next(w.capitalize() for w in words if w in _SUBJECT_DRAWERS)
        caption = prompt.strip()
        ax.text(0.50, 0.955, subject_label, ha="center", va="top", fontsize=20,
                fontweight="bold", color="#212121", transform=ax.transAxes)
        if caption.lower() != subject_label.lower():
            ax.text(0.50, 0.025, fill(caption, width=55), ha="center", va="bottom",
                    fontsize=9, color="#555", transform=ax.transAxes, linespacing=1.3)
    else:
        tokens = [
            t.capitalize() for t in re.split(r"[^a-z0-9]+", (prompt or "").lower())
            if len(t) > 3 and t not in {"with", "from", "that", "this", "into", "diagram", "image", "chart"}
        ]
        kws: list[str] = []
        for t in tokens:
            if t not in kws:
                kws.append(t)
            if len(kws) == 3:
                break
        while len(kws) < 3:
            kws.append(["Context", "Analysis", "Outcome"][len(kws)])
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        fig.patch.set_facecolor("#f6efe5")
        ax.set_facecolor("#fffaf5")
        _draw_concept_diagram(ax, prompt, kws)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/media/images/{file_name}"


def _sanitize_series(series: list[float]) -> list[float]:
    cleaned: list[float] = []
    for value in series or []:
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            continue
    return cleaned


def _y_limits(values: list[float]) -> tuple[float, float]:
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        pad = max(abs(y_min) * 0.12, 1.0)
        return y_min - pad, y_max + pad
    spread = y_max - y_min
    pad = spread * 0.15
    return y_min - pad, y_max + pad


def generate_chart(
    series: list[float],
    chart_type: str = "line",
    title: str = "Generated Chart",
    x_labels: list[str] | None = None,
    unit: str | None = None,
) -> str:
    charts_dir = Path(settings.MEDIA_ROOT) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"chart-{uuid4().hex[:10]}.png"
    output_path = charts_dir / file_name

    values = _sanitize_series(series)
    if not values:
        values = [0.0]

    selected_chart = (chart_type or "line").strip().lower()

    # Normalise x_labels: strip blanks, ensure list length matches values
    clean_labels: list[str] | None = None
    if x_labels:
        stripped = [str(lbl).strip() for lbl in x_labels if str(lbl).strip()]
        if stripped:
            # Pad or truncate to match value count
            while len(stripped) < len(values):
                stripped.append(str(len(stripped) + 1))
            clean_labels = stripped[: len(values)]

    y_axis_label = (unit.strip() if unit and unit.strip() else "Value")

    # ── Pie chart: separate code path (no x/y axes) ─────────────────────────
    if selected_chart == "pie":
        _generate_pie(values, title, output_path, x_labels=clean_labels)
        return f"/media/charts/{file_name}"

    # ── All axis-based charts ─────────────────────────────────────────────────
    x_positions = list(range(1, len(values) + 1))
    y_min, y_max = _y_limits(values)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=170)
    fig.patch.set_facecolor("#fff8f0")
    ax.set_facecolor("#fffdf9")

    if selected_chart == "bar":
        bars = ax.bar(
            x_positions,
            values,
            color="#c26d2c",
            edgecolor="#7a3e14",
            linewidth=1.0,
            width=0.62,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            offset = (y_max - y_min) * 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (offset if value >= 0 else -offset),
                f"{value:.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
                color="#5b2d0c",
                zorder=5,
            )

    elif selected_chart == "scatter":
        # colour-map points by value: low=cool, high=warm
        import numpy as np
        scatter_colors = plt.cm.RdYlGn(
            [(v - min(values)) / max(max(values) - min(values), 1e-9) for v in values]
        )
        ax.scatter(
            x_positions,
            values,
            c=scatter_colors,
            s=90,
            edgecolors="#444",
            linewidths=0.7,
            zorder=4,
        )
        # regression trend line
        coeffs = np.polyfit(x_positions, values, 1)
        trend_y = [coeffs[0] * x + coeffs[1] for x in x_positions]
        ax.plot(
            x_positions,
            trend_y,
            color="#888",
            linewidth=1.4,
            linestyle="--",
            zorder=3,
            label=f"Trend  y={coeffs[0]:+.2f}x",
        )
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
        offset = (y_max - y_min) * 0.025
        for x_value, y_value in zip(x_positions, values):
            ax.text(x_value, y_value + offset, f"{y_value:.1f}",
                    fontsize=8, ha="center", va="bottom", color="#333", zorder=6)

    elif selected_chart == "area":
        ax.fill_between(x_positions, values, color="#9cd4f5", alpha=0.42, zorder=2)
        ax.plot(
            x_positions,
            values,
            color="#176087",
            linewidth=2.6,
            marker="o",
            markersize=5.5,
            markerfacecolor="#e8f6ff",
            markeredgewidth=1.2,
            markeredgecolor="#0f4e6e",
            zorder=4,
        )
        offset = (y_max - y_min) * 0.025
        for x_value, y_value in zip(x_positions, values):
            ax.text(x_value, y_value + offset, f"{y_value:.1f}",
                    fontsize=8.5, color="#0f3e57", va="bottom", ha="center", zorder=6)

    else:  # default: line
        ax.plot(
            x_positions,
            values,
            color="#176087",
            linewidth=2.8,
            marker="o",
            markersize=6.2,
            markerfacecolor="#e8f6ff",
            markeredgewidth=1.4,
            markeredgecolor="#0f4e6e",
            zorder=4,
        )
        ax.fill_between(x_positions, values, color="#9cd4f5", alpha=0.18, zorder=2)
        offset = (y_max - y_min) * 0.025
        for x_value, y_value in zip(x_positions, values):
            ax.text(
                x_value,
                y_value + (offset if y_value >= 0 else -offset),
                f"{y_value:.1f}",
                fontsize=8.7,
                color="#0f3e57",
                va="bottom" if y_value >= 0 else "top",
                ha="center",
                zorder=6,
            )

    # peak / low markers (shared for bar, scatter, area, line)
    if selected_chart != "scatter":  # scatter already colour-codes them
        peak_index = max(range(len(values)), key=lambda i: values[i])
        low_index  = min(range(len(values)), key=lambda i: values[i])
        ax.scatter([x_positions[peak_index]], [values[peak_index]], s=70, color="#1b5e20", zorder=8)
        ax.scatter([x_positions[low_index]],  [values[low_index]],  s=70, color="#b71c1c", zorder=8)
        ax.annotate(
            f"Peak {values[peak_index]:.1f}",
            xy=(x_positions[peak_index], values[peak_index]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=8.5, color="#1b5e20",
        )
        ax.annotate(
            f"Low {values[low_index]:.1f}",
            xy=(x_positions[low_index], values[low_index]),
            xytext=(0, -16), textcoords="offset points",
            ha="center", fontsize=8.5, color="#b71c1c",
        )

    # summary stats box
    summary_text = (
        f"n={len(values)}\n"
        f"avg={mean(values):.2f}\n"
        f"min={min(values):.2f}\n"
        f"max={max(values):.2f}"
    )
    ax.text(
        0.985, 0.965, summary_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.4, color="#5a524a",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#fff3e6", "ec": "#d8b89a", "alpha": 0.95},
    )

    ax.set_title(title, fontsize=16, fontweight="bold", color="#4a3421", pad=14)
    ax.set_xlabel("Observation", color="#5a524a")
    ax.set_ylabel(y_axis_label, color="#5a524a")
    ax.set_xticks(x_positions)
    if clean_labels:
        ax.set_xticklabels(clean_labels, rotation=30 if max(len(lbl) for lbl in clean_labels) > 6 else 0, ha="right" if max(len(lbl) for lbl in clean_labels) > 6 else "center", fontsize=8.5)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color="#d9c7b8", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9d8d7f")
    ax.spines["bottom"].set_color("#9d8d7f")
    ax.tick_params(colors="#5a524a")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/media/charts/{file_name}"


def save_dataset_json(dataset: dict[str, Any], prefix: str = "dataset") -> str:
    charts_dir = Path(settings.MEDIA_ROOT) / "charts" / "datasets"
    charts_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{prefix}-{uuid4().hex[:10]}.json"
    output_path = charts_dir / file_name
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(dataset, fp, ensure_ascii=True, indent=2)
    return f"/media/charts/datasets/{file_name}"


def _generate_pie(values: list[float], title: str, output_path: "Path", x_labels: list[str] | None = None) -> None:
    """Render a styled pie / donut chart."""
    import numpy as np

    # keep only positive values
    indexed = [(v, (x_labels[i] if x_labels and i < len(x_labels) else None)) for i, v in enumerate(values) if v > 0]
    if not indexed:
        indexed = [(1.0, None)]
    pos_values = [v for v, _ in indexed]
    raw_labels = [lbl for _, lbl in indexed]
    labels = [
        f"{lbl}\n{v:.1f}" if lbl else f"Slice {i + 1}\n{v:.1f}"
        for i, (v, lbl) in enumerate(indexed)
    ]

    palette = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
        "#264653", "#8ecae6", "#a8dadc", "#95d5b2", "#b5838d",
    ]
    colors = [palette[i % len(palette)] for i in range(len(pos_values))]

    # explode the largest slice slightly
    largest = pos_values.index(max(pos_values))
    explode = [0.05 if i == largest else 0.0 for i in range(len(pos_values))]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=170)
    fig.patch.set_facecolor("#fff8f0")

    wedges, texts, autotexts = ax.pie(
        pos_values,
        labels=labels,
        explode=explode,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 4 else "",
        pctdistance=0.72,
        startangle=140,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
    )

    # donut hole
    hole = plt.Circle((0, 0), 0.46, fc="#fff8f0")
    ax.add_patch(hole)

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color("#2d241c")
        autotext.set_fontweight("bold")

    total = sum(pos_values)
    ax.text(0, 0, f"Total\n{total:.1f}", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#4a3421")

    ax.set_title(title, fontsize=16, fontweight="bold", color="#4a3421", pad=16)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Agent document tools (Copilot-style) ──────────────────────────────────────

def doc_list_sections(document) -> list[dict[str, Any]]:
    """Return document outline: [{index, title, word_count, has_content}]."""
    sections = (document.content or {}).get("sections", [])
    return [
        {
            "index": i,
            "title": s.get("title", f"Section {i}"),
            "word_count": len((s.get("content") or "").split()),
            "has_content": bool((s.get("content") or "").strip()),
        }
        for i, s in enumerate(sections)
    ]


def doc_read_section(document, query: str) -> dict[str, Any] | None:
    """Read a specific section by title match. Returns {index, title, content}."""
    idx = find_section(document.content, query)
    if idx is None:
        return None
    sections = (document.content or {}).get("sections", [])
    section = sections[idx]
    return {
        "index": idx,
        "title": section.get("title", ""),
        "content": section.get("content", ""),
    }


def doc_search(document, query: str) -> list[dict[str, Any]]:
    """Search for text across all sections. Returns [{index, title, snippet}]."""
    results = []
    query_lower = query.lower()
    for i, section in enumerate((document.content or {}).get("sections", [])):
        content = section.get("content") or ""
        if query_lower in content.lower():
            pos = content.lower().find(query_lower)
            start = max(0, pos - 100)
            end = min(len(content), pos + 200)
            results.append({
                "index": i,
                "title": section.get("title", ""),
                "snippet": content[start:end],
            })
    return results


def doc_edit_section(document, query: str, new_content: str) -> bool:
    """Replace a section's content in-place. Returns True on success."""
    idx = find_section(document.content, query)
    if idx is None:
        return False
    sections = (document.content or {}).get("sections", [])
    sections[idx]["content"] = new_content
    return True


def doc_insert_in_section(
    document, query: str, after_text: str, new_paragraph: str
) -> bool:
    """
    Insert new_paragraph after after_text within a section.
    Appends at end if after_text is not found.
    """
    idx = find_section(document.content, query)
    if idx is None:
        return False
    sections = (document.content or {}).get("sections", [])
    content = sections[idx].get("content") or ""
    pos = content.lower().find(after_text.lower())
    if pos == -1:
        sections[idx]["content"] = content.rstrip() + "\n\n" + new_paragraph
    else:
        end_pos = content.find("\n\n", pos)
        if end_pos == -1:
            end_pos = len(content)
        sections[idx]["content"] = (
            content[:end_pos] + "\n\n" + new_paragraph + content[end_pos:]
        )
    return True
