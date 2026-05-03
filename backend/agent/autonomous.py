"""
Autonomous document agent.
Accepts a free-form user message, classifies intent, plans, and executes
direct edits on the document using a selectable LLM provider.
"""
from __future__ import annotations

import logging
import json
import re
from typing import Any

from documents.models import Document, DocumentVersion

from .llm import (
    chat_with_document,
    classify_intent,
    create_execution_plan,
    enhance_text,
    set_active_model,
    get_model_label,
    generate_outline_sections,
    generate_section_content,
    generate_text,
)
from .tools import (
    find_section,
    generate_chart,
    generate_image,
    update_section,
)

logger = logging.getLogger(__name__)


DISSERTATION_TEMPLATE: list[dict[str, Any]] = [
    {
        "title": "Chapter 1: Introduction",
        "subsections": [
            "1.1 Background of the Study",
            "1.2 Statement of the Problem",
            "1.3 Research Objectives",
            "1.4 Research Questions",
            "1.5 Research Hypotheses",
            "1.6 Significance of the Study",
            "1.7 Scope and Delimitations",
            "1.8 Definition of Key Terms",
        ],
    },
    {
        "title": "Chapter 2: Literature Review",
        "subsections": [
            {
                "title": "2.1 Introduction",
                "children": [],
            },
            {
                "title": "2.2 Conceptual Review",
                "children": [
                    {"title": "2.2.1 Definition and Conceptualisation of Key Terms", "children": []},
                    {"title": "2.2.2 Core Concepts and Theoretical Constructs", "children": []},
                    {"title": "2.2.3 Dimensions, Indicators, and Measurement", "children": []},
                    {"title": "2.2.4 Relationships Between Variables", "children": []},
                ],
            },
            {
                "title": "2.3 Theoretical Framework",
                "children": [
                    {"title": "2.3.1 Overview of Relevant Theories", "children": []},
                    {"title": "2.3.2 Foundational and Classical Theories", "children": []},
                    {"title": "2.3.3 Contemporary and Emerging Theories", "children": []},
                    {"title": "2.3.4 Applicability and Justification to the Study", "children": []},
                ],
            },
            {
                "title": "2.4 Empirical Review",
                "children": [
                    {"title": "2.4.1 Global Evidence and Trends", "children": []},
                    {"title": "2.4.2 Evidence from Developed Economies", "children": []},
                    {"title": "2.4.3 Evidence from Emerging Economies", "children": []},
                    {"title": "2.4.4 Evidence from Developing Economies and Africa", "children": []},
                    {"title": "2.4.5 Sectoral and Industry-Specific Evidence", "children": []},
                    {"title": "2.4.6 Synthesis and Critical Appraisal of Empirical Studies", "children": []},
                ],
            },
            {
                "title": "2.5 Research Gap",
                "children": [
                    {"title": "2.5.1 Identified Gaps in the Literature", "children": []},
                    {"title": "2.5.2 Contribution and Justification of the Present Study", "children": []},
                ],
            },
            {
                "title": "2.6 Chapter Summary",
                "children": [],
            },
        ],
    },
    {
        "title": "Chapter 3: Methodology",
        "subsections": [
            "3.1 Introduction",
            "3.2 Research Design",
            "3.3 Target Population",
            "3.4 Sampling Techniques and Sample Size",
            "3.5 Data Collection Methods",
            "3.6 Data Analysis Techniques",
            "3.7 Reliability and Validity",
            "3.8 Ethical Considerations",
            "3.9 Chapter Summary",
        ],
    },
    {
        "title": "Chapter 4: Results and Discussion",
        "subsections": [],
    },
    {
        "title": "Chapter 5: Conclusion and Recommendations",
        "subsections": [
            "5.1 Introduction",
            "5.2 Summary of Findings",
            "5.3 Conclusions",
            "5.4 Recommendations",
            "5.5 Limitations of the Study",
            "5.6 Areas for Further Research",
        ],
    },
    {
        "title": "Chapter 6: References and Appendices",
        "subsections": [
            "6.1 References",
            "6.2 Appendices",
        ],
    },
]


def _normalize_subsection_node(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"title": raw, "children": [], "kind": "text"}
    if isinstance(raw, dict):
        children = [_normalize_subsection_node(ch) for ch in raw.get("children", [])]
        return {
            "title": str(raw.get("title") or "Untitled subsection"),
            "children": children,
            "kind": raw.get("kind", "text"),
            "meta": raw.get("meta", {}),
        }
    return {"title": str(raw), "children": [], "kind": "text"}


def _research_design(message: str, topic: str, document: Document) -> str:
    source = " ".join(
        [
            message or "",
            topic or "",
            _flatten_doc(document),
        ]
    ).lower()
    if "mixed method" in source or "mixed-method" in source or "mixed methods" in source:
        return "mixed"
    if "quantitative" in source:
        return "quantitative"
    if "qualitative" in source:
        return "qualitative"
    if any(k in source for k in ["systematic review", "scoping review", "conceptual paper", "theoretical"]):
        return "non_empirical"
    return "quantitative"


def _extract_objectives(document: Document, topic: str) -> list[str]:
    sections = (document.content or {}).get("sections", [])
    chapter_1_idx = find_section(document.content, "Chapter 1")
    chapter_1 = sections[chapter_1_idx] if chapter_1_idx is not None and chapter_1_idx < len(sections) else {}
    chapter_1_text = (chapter_1.get("content") or "").strip()

    lines = [ln.strip(" -\t") for ln in chapter_1_text.splitlines() if ln.strip()]
    hits: list[str] = []
    for line in lines:
        if len(hits) >= 5:
            break
        if re.search(r"\bobjective\b", line.lower()) and len(line) > 24:
            hits.append(line[:140])

    if hits:
        return hits

    short_topic = (topic or "the study topic").strip()
    return [
        f"Determine the current state of {short_topic}",
        f"Evaluate key drivers and constraints affecting {short_topic}",
        f"Propose evidence-based recommendations for improving outcomes in {short_topic}",
    ]


def _objective_section_title(objective: str) -> str:
    """Derive a short readable section title from a full objective statement."""
    text = re.sub(
        r"^(to\s+|the\s+study\s+aims?\s+to\s+|this\s+study\s+(aims?\s+to|seeks?\s+to|will)\s+|"
        r"to\s+examine\s+|to\s+determine\s+|to\s+assess\s+|to\s+evaluate\s+|"
        r"to\s+investigate\s+|to\s+analyse\s+|to\s+analyze\s+)",
        "",
        objective.strip(),
        flags=re.IGNORECASE,
    )
    text = text[0].upper() + text[1:] if text else objective
    if len(text) > 65:
        cut = text[:65].rsplit(" ", 1)[0]
        return cut
    return text


def _chapter4_subsections(research_design: str, objectives: list[str]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = [
        {"title": "4.1 Introduction", "children": []},
    ]

    if research_design in {"quantitative", "qualitative", "mixed"}:
        demo_children: list[dict[str, Any]] = [
            {
                "title": "4.2.1 Demographic Distribution of Respondents",
                "kind": "table",
                "children": [],
                "meta": {"table_type": "demographics"},
            }
        ]
        if research_design in {"quantitative", "mixed"}:
            demo_children.append(
                {
                    "title": "4.2.2 Demographic Distribution Chart",
                    "kind": "chart",
                    "children": [],
                    "meta": {"chart_type": "demographics"},
                }
            )
        nodes.append({"title": "4.2 Respondent Profile", "children": demo_children})
        obj_start = 3
    else:
        obj_start = 2

    # Each objective gets its own top-level section: 4.X <Short Objective Title>
    for i, objective in enumerate(objectives):
        sec_num = obj_start + i
        short_title = _objective_section_title(objective)
        if research_design == "qualitative":
            table_title = f"4.{sec_num}.1 Theme Matrix"
        else:
            table_title = f"4.{sec_num}.1 Summary Table"

        obj_node: dict[str, Any] = {
            "title": f"4.{sec_num} {short_title}",
            "kind": "objective_findings",
            "meta": {"objective": objective},
            "children": [
                {
                    "title": table_title,
                    "kind": "table",
                    "children": [],
                    "meta": {"objective": objective},
                },
                {
                    "title": f"4.{sec_num}.2 Data Visualization",
                    "kind": "chart",
                    "children": [],
                    "meta": {"objective": objective},
                },
            ],
        }
        nodes.append(obj_node)

    last_idx = obj_start + len(objectives)
    nodes.append({"title": f"4.{last_idx} Discussion of Findings", "children": []})
    nodes.append({"title": f"4.{last_idx + 1} Chapter Summary", "children": []})
    return [_normalize_subsection_node(n) for n in nodes]


def _table_text_for_node(node_title: str, research_design: str, topic: str, objective: str | None = None) -> str:
    seed_text = f"{node_title}|{research_design}|{topic}|{objective or ''}"
    seed = sum(ord(c) for c in seed_text)

    def pct_triplet(total: int = 100) -> tuple[int, int, int]:
        a = 24 + (seed % 19)
        b = 29 + ((seed // 3) % 17)
        c = total - a - b
        if c < 18:
            c = 18
            b = total - a - c
        return a, b, c

    if "demographic" in node_title.lower():
        male = 45 + (seed % 31)
        female = 120 - male
        age1, age2, age3 = pct_triplet(120)
        return (
            "| Variable | Category | Frequency | Percentage |\n"
            "|---|---|---:|---:|\n"
            f"| Gender | Male | {male} | {male / 120 * 100:.1f}% |\n"
            f"| Gender | Female | {female} | {female / 120 * 100:.1f}% |\n"
            f"| Age | 18-29 | {age1} | {age1 / 120 * 100:.1f}% |\n"
            f"| Age | 30-39 | {age2} | {age2 / 120 * 100:.1f}% |\n"
            f"| Age | 40+ | {age3} | {age3 / 120 * 100:.1f}% |"
        )
    if research_design == "qualitative":
        t1 = 2 + (seed % 4)
        t2 = 2 + ((seed // 5) % 4)
        t3 = 2 + ((seed // 11) % 4)
        return (
            "| Theme | Supporting Mentions | Representative Insight | Interpretation |\n"
            "|---|---|---|\n"
            f"| Theme 1 | {t1} | Participants highlighted concerns around {topic[:40]} | Indicates persistent implementation barriers |\n"
            f"| Theme 2 | {t2} | Respondents reported uneven institutional readiness | Suggests need for governance alignment |\n"
            f"| Theme 3 | {t3} | Stakeholders requested stronger policy direction | Supports a coordinated reform approach |"
        )
    objective_text = objective or "the objective"
    a = round(2.6 + (seed % 16) * 0.11, 2)
    b = round(3.1 + ((seed // 7) % 14) * 0.12, 2)
    c = round(2.4 + ((seed // 13) % 15) * 0.1, 2)
    return (
        "| Metric | Observation | Interpretation |\n"
        "|---|---:|---|\n"
        f"| Indicator A ({objective_text[:30]}) | {a} | Moderate performance with room for improvement |\n"
        f"| Indicator B | {b} | Stronger outcome where controls were applied |\n"
        f"| Indicator C | {c} | Weakest dimension and major constraint area |"
    )


def _ai_chart_series(context: str, n_points: int = 8) -> dict[str, Any]:
    """Ask the LLM to produce realistic numeric data for a chart.

    Returns a dict with keys:
      - series: list[float]   — the data points
      - chart_type: str       — suggested chart type (bar/line/scatter/area/pie)
      - x_labels: list[str]   — short label for each point
      - unit: str             — measurement unit, e.g. "%", "score", "count"
    Falls back to seed-based data on any error so the pipeline never breaks.
    """
    prompt = (
        f"You are a data analyst. Generate realistic numeric data for a chart about:\n"
        f"  \"{context}\"\n\n"
        f"Rules:\n"
        f"- Return ONLY a JSON object, no markdown, no extra text.\n"
        f"- Exactly {n_points} data points. Values must be realistic for the topic.\n"
        f"- Pick the best chart_type from: bar, line, scatter, area, pie.\n"
        f"- x_labels: short (1-3 word) label for each data point (e.g. year, category, group).\n"
        f"- unit: the measurement unit (e.g. '%%', 'score', 'count', 'USD', 'years').\n"
        f"- Do NOT use random-looking numbers. Make them believable for the topic.\n\n"
        f"Format:\n"
        f"{{\"series\":[...],\"chart_type\":\"bar\",\"x_labels\":[...],\"unit\":\"%%\"}}"
    )

    try:
        raw = generate_text(prompt)
        # strip any accidental fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        data = json.loads(raw)
        series = [float(v) for v in (data.get("series") or []) if v is not None]
        if not series:
            raise ValueError("empty series")
        return {
            "series": series,
            "chart_type": str(data.get("chart_type") or "bar").lower(),
            "x_labels": [str(label) for label in (data.get("x_labels") or [])],
            "unit": str(data.get("unit") or ""),
        }
    except Exception as exc:
        logger.warning("_ai_chart_series fallback (%s): %s", context[:60], exc)
        seed = sum(ord(c) for c in context)
        base = 10.0 + (seed % 30)
        return {
            "series": [round(base + i * 3.5 + (seed + i) % 7, 1) for i in range(n_points)],
            "chart_type": "bar",
            "x_labels": [f"Item {i+1}" for i in range(n_points)],
            "unit": "",
        }


def _extract_json_obj(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object found")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("json payload is not an object")
    return parsed


def _default_framework_spec(topic: str, prompt: str) -> dict[str, Any]:
    short_topic = (topic or "the study").strip()
    return {
        "title": f"Conceptual Framework: {short_topic}",
        "left_label": "Independent Variables",
        "left_items": ["Policy factors", "Resource capacity", "Stakeholder readiness"],
        "middle_label": "Mediating Factors",
        "middle_items": ["Implementation quality", "Operational efficiency"],
        "right_label": "Dependent Variable",
        "right_items": [short_topic],
        "control_label": "Control Variables",
        "control_items": ["Context", "Institution size", "Time"],
        "notes": (prompt or "").strip()[:180],
    }


def _framework_target_index(document: Document, target: str | None, prompt: str) -> int | None:
    sections = (document.content or {}).get("sections", [])
    if not sections:
        return None

    if target:
        direct = find_section(document.content, target)
        if direct is not None:
            return direct

    text = f"{(target or '')} {(prompt or '')}".lower()
    looks_framework = any(k in text for k in ["conceptual", "theoretical", "framework", "model"])
    if looks_framework:
        preferred_titles = [
            "2.3 Theoretical Framework",
            "Theoretical Framework",
            "2.2 Conceptual Review",
            "Conceptual Review",
            "Literature Review",
            "Chapter 2",
        ]
        for query in preferred_titles:
            idx = find_section(document.content, query)
            if idx is not None:
                return idx

        for i, section in enumerate(sections):
            combined = f"{section.get('title', '')}\n{section.get('content', '')}".lower()
            if any(k in combined for k in ["conceptual framework", "theoretical framework", "framework"]):
                return i

    return len(sections) - 1


def _build_framework_spec(document: Document, target: str | None, prompt: str) -> dict[str, Any]:
    sections = (document.content or {}).get("sections", [])
    idx = _framework_target_index(document, target, prompt)
    local_title = ""
    local_content = ""
    if idx is not None and 0 <= idx < len(sections):
        local_title = str(sections[idx].get("title") or "")
        local_content = str(sections[idx].get("content") or "")

    full_context = _full_context_for_generation(document)[:5000]
    topic = str((document.content or {}).get("topic") or document.title or "Study")
    objectives = _extract_objectives(document, topic)
    objective_text = "\n".join(f"- {item}" for item in objectives[:4])

    llm_prompt = (
        "You are an academic research assistant. Build a professional conceptual framework specification "
        "for a dissertation figure.\n\n"
        f"Document title: {document.title}\n"
        f"Topic: {topic}\n"
        f"Target section title: {local_title or 'N/A'}\n"
        f"User request: {prompt}\n\n"
        "Research objectives:\n"
        f"{objective_text or '- N/A'}\n\n"
        "Relevant section content:\n"
        f"{local_content[:1400]}\n\n"
        "Whole document context:\n"
        f"{full_context}\n\n"
        "Return JSON only with this exact shape:\n"
        "{"
        '"title":"...",'
        '"left_label":"Independent Variables",'
        '"left_items":["...","..."],'
        '"middle_label":"Mediating/Moderating Variables",'
        '"middle_items":["...","..."],'
        '"right_label":"Dependent Variable",'
        '"right_items":["..."],'
        '"control_label":"Control Variables",'
        '"control_items":["...","..."],'
        '"notes":"Short rationale under 180 chars"'
        "}\n"
        "Rules: keep labels short, items concrete and topic-aware, no markdown."
    )

    try:
        data = _extract_json_obj(generate_text(llm_prompt))
        return {
            "title": str(data.get("title") or f"Conceptual Framework: {topic}"),
            "left_label": str(data.get("left_label") or "Independent Variables"),
            "left_items": [str(x) for x in (data.get("left_items") or [])][:4],
            "middle_label": str(data.get("middle_label") or "Mediating Variables"),
            "middle_items": [str(x) for x in (data.get("middle_items") or [])][:4],
            "right_label": str(data.get("right_label") or "Dependent Variable"),
            "right_items": [str(x) for x in (data.get("right_items") or [])][:4],
            "control_label": str(data.get("control_label") or "Control Variables"),
            "control_items": [str(x) for x in (data.get("control_items") or [])][:4],
            "notes": str(data.get("notes") or "")[0:180],
        }
    except Exception as exc:
        logger.warning("_build_framework_spec fallback: %s", exc)
        return _default_framework_spec(topic, prompt)


def _insert_block_marker(section_text: str, block_id: str, prompt: str) -> str:
    text = section_text or ""
    marker = f"[[BLOCK:{block_id}]]"
    if marker in text:
        return text

    lines = text.splitlines()
    lowered_prompt = (prompt or "").lower()
    framework_request = any(k in lowered_prompt for k in ["conceptual", "theoretical", "framework", "model"])

    if framework_request:
        for i, line in enumerate(lines):
            lower_line = line.lower().strip()
            if any(k in lower_line for k in ["conceptual framework", "theoretical framework", "framework"]):
                lines.insert(i + 1, marker)
                return "\n".join(lines)

    if not text.strip():
        return marker
    return text.rstrip() + "\n\n" + marker


def _chart_series_for_node(node_title: str, objective: str | None = None) -> list[float]:
    context = node_title + (f" — {objective}" if objective else "")
    result = _ai_chart_series(context, n_points=8)
    return result["series"]


def _next_caption_number(document: Document, kind: str) -> int:
    token = "figure" if kind == "figure" else "table"
    max_seen = 0

    sections = (document.content or {}).get("sections", [])
    for section in sections:
        content = section.get("content", "") or ""
        for match in re.finditer(rf"\b{token}\s+(\d+)\b", content, flags=re.IGNORECASE):
            max_seen = max(max_seen, int(match.group(1)))

        for block in section.get("blocks", []) or []:
            caption = (block.get("caption") or "")
            for match in re.finditer(rf"\b{token}\s+(\d+)\b", caption, flags=re.IGNORECASE):
                max_seen = max(max_seen, int(match.group(1)))

    return max_seen + 1


def _table_discussion_text(node_title: str, research_design: str, objective: str | None = None) -> str:
    if "demographic" in node_title.lower():
        return (
            "Interpretation: The respondent profile indicates a reasonably balanced distribution across key demographic categories, "
            "which supports broad coverage of participant perspectives.\n"
            "Discussion: This distribution improves confidence that subsequent objective-level findings are not overly driven by a single subgroup."
        )

    if research_design == "qualitative":
        return (
            "Interpretation: The theme matrix shows recurring viewpoints across participants, with some themes appearing more frequently than others.\n"
            "Discussion: The pattern suggests consistent experiential concerns and provides evidence for targeted policy and implementation recommendations."
        )

    obj = (objective or "the objective").strip()
    return (
        f"Interpretation: The metric pattern for {obj[:70]} shows uneven performance across indicators, with stronger outcomes in selected dimensions.\n"
        "Discussion: The spread across indicators highlights where focused interventions are required to improve overall study outcomes."
    )


def _chart_discussion_text(series: list[float], objective: str | None = None) -> str:
    avg = round(sum(series) / len(series), 2) if series else 0.0
    high = max(series) if series else 0.0
    low = min(series) if series else 0.0
    trend = "upward" if len(series) > 1 and series[-1] >= series[0] else "mixed"
    objective_label = (objective or "the objective").strip()
    return (
        f"Interpretation: Figure trend is {trend}, with values ranging from {low:.2f} to {high:.2f} and an average of {avg:.2f}.\n"
        f"Discussion: For {objective_label[:70]}, the visual pattern reinforces the numerical evidence and clarifies priority areas for action."
    )


def _append_node_plan_steps(plan: list[dict[str, Any]], nodes: list[dict[str, Any]], depth: int = 1) -> None:
    indent = "  " * depth
    for node in nodes:
        kind = node.get("kind", "text")
        verb = "Writing"
        if kind == "table":
            verb = "Creating table for"
        elif kind == "chart":
            verb = "Creating chart for"
        plan.append({"step": f"{indent}{verb} {node.get('title', 'Untitled subsection')}", "status": "pending"})
        _append_node_plan_steps(plan, node.get("children", []), depth + 1)


def _execute_subsection_nodes(
    nodes: list[dict[str, Any]],
    section_title: str,
    topic: str,
    research_design: str,
    rolling_context: str,
    plan: list[dict[str, Any]],
    plan_cursor: list[int],
    figure_counter: list[int],
    table_counter: list[int],
    on_node_completed: Any | None = None,
    default_word_count: int = 220,
) -> tuple[str, str, list[dict[str, str]]]:
    chunks: list[str] = []
    blocks: list[dict[str, str]] = []
    local_context = rolling_context

    for node in nodes:
        step_idx = plan_cursor[0]
        plan_cursor[0] += 1

        title = node.get("title", "Untitled subsection")
        kind = node.get("kind", "text")
        meta = node.get("meta", {}) if isinstance(node.get("meta", {}), dict) else {}

        if kind == "table":
            objective = str(meta.get("objective") or "") or None
            table_no = table_counter[0]
            table_counter[0] += 1
            table_caption = f"Table {table_no}: {title}"
            body = (
                f"{table_caption}\n"
                f"{_table_text_for_node(title, research_design, topic, objective)}\n\n"
                f"{_table_discussion_text(title, research_design, objective)}"
            )
        elif kind == "chart":
            objective = str(meta.get("objective") or "") or None
            figure_no = figure_counter[0]
            figure_counter[0] += 1
            figure_caption = f"Figure {figure_no}: {title}"
            context_str = title + (f" — {objective}" if objective else "")
            ai_data = _ai_chart_series(context_str, n_points=8)
            chart_path = generate_chart(
                series=ai_data["series"],
                chart_type=ai_data["chart_type"],
                title=title,
            )
            block_id = f"fig-{figure_no}-{len(blocks) + 1}"
            blocks.append({
                "type": "chart",
                "src": chart_path,
                "caption": figure_caption,
                "block_id": block_id,
            })
            body = (
                f"{figure_caption}\n"
                f"[[BLOCK:{block_id}]]\n"
                f"{_chart_discussion_text(ai_data['series'], objective)}"
            )
        elif kind == "objective_findings":
            objective = str(meta.get("objective") or title)
            try:
                body = generate_section_content(
                    title=title,
                    topic=topic,
                    context=(
                        f"Research objective: {objective}\n"
                        f"Research design: {research_design}\n"
                        f"Parent chapter: {section_title}\n"
                        f"Context from previous sections:\n{local_context[-2000:]}\n\n"
                        "Write 2-3 paragraphs presenting findings specifically for this objective. "
                        "Discuss patterns, trends, and how findings directly address the objective. "
                        "Be specific and academically rigorous."
                    ),
                    word_count=280,
                )
            except Exception:
                body = _fallback_subsection_text(topic, section_title, title)
        else:
                is_lit_review = "literature review" in section_title.lower() or "chapter 2" in section_title.lower()
                lowered_title = title.lower()
                is_pointform = any(
                    k in lowered_title
                    for k in ["research objective", "objectives", "research question", "hypothes"]
                )
                if is_pointform:
                    try:
                        body = generate_text(
                            f"Write the '{title}' subsection for a research paper about: '{topic}'.\n"
                            f"Research design: {research_design}\n"
                            "Format as a numbered list ONLY (1. ... 2. ... 3. ...). "
                            "Write 3-5 clear, specific, measurable points. "
                            "Each item must be a complete standalone sentence. "
                            "Do NOT write any prose paragraph. Do NOT add an introductory sentence before the list."
                        )
                    except Exception:
                        body = _fallback_subsection_text(topic, section_title, title)
                elif is_lit_review:
                    try:
                        body = generate_section_content(
                            title=title,
                            topic=topic,
                            context=(
                                f"Parent chapter: {section_title}\n"
                                f"Research design: {research_design}\n"
                                f"Context from previous sections:\n{local_context[-2200:]}\n\n"
                                "Requirements: Write 3-5 full paragraphs (~{wc} words total). "
                                "Reference 3-5 relevant scholars, studies, or theoretical positions by name and year (e.g., Smith, 2019; Jones & Patel, 2021). "
                                "Be analytical, not merely descriptive. "
                                "Discuss contrasting views, debates, and how different authors' positions relate to {topic}. "
                                "Maintain formal academic tone throughout.".format(wc=default_word_count, topic=topic)
                            ),
                            word_count=default_word_count,
                        )
                    except Exception:
                        body = _fallback_subsection_text(topic, section_title, title)
                else:
                    try:
                        body = generate_section_content(
                            title=title,
                            topic=topic,
                            context=(
                                f"Parent chapter: {section_title}\n"
                                f"Research design: {research_design}\n"
                                f"Context from previous sections:\n{local_context[-2200:]}"
                            ),
                            word_count=default_word_count,
                        )
                    except Exception:
                        body = _fallback_subsection_text(topic, section_title, title)

        chunks.append(f"{title}\n{body}")
        local_context = f"{local_context}\n\n{title}\n{body}".strip()

        child_nodes = [_normalize_subsection_node(n) for n in node.get("children", [])]
        if child_nodes:
            child_text, local_context, child_blocks = _execute_subsection_nodes(
                child_nodes,
                section_title,
                topic,
                research_design,
                local_context,
                plan,
                plan_cursor,
                figure_counter,
                table_counter,
                on_node_completed,
                default_word_count,
            )
            if child_text:
                chunks.append(child_text)
            if child_blocks:
                blocks.extend(child_blocks)

        _done(plan, step_idx)
        if callable(on_node_completed):
            try:
                on_node_completed("\n\n".join(chunks), list(blocks), title)
            except Exception as exc:
                logger.warning("Subsection progress callback failed for '%s': %s", title, exc)

    return "\n\n".join(chunks), local_context, blocks


def _heuristic_intent(message: str) -> dict[str, Any]:
    text = (message or "").strip().lower()

    def _extract_target() -> str | None:
        chapter_match = re.search(r"chapter\s*\d+(?:\.\d+)?", text)
        if chapter_match:
            return chapter_match.group(0).title()
        sec_match = re.search(r"\b\d+(?:\.\d+)+\b", text)
        if sec_match:
            return sec_match.group(0)
        return None

    target = _extract_target()
    topic = None

    if any(
        phrase in text
        for phrase in [
            "summary of the document",
            "summarize the document",
            "summarise the document",
            "document summary",
            "summary of this document",
            "summarize this document",
            "summarise this document",
            "analyse the document",
            "analyze the document",
            "analyse this document",
            "analyze this document",
            "review the document",
            "examine the document",
            "analyse document",
            "analyze document",
        ]
    ):
        return {"intent": "summarize_document", "target_section": None, "topic": None}

    dissertation_triggers = [
        "full dissertation",
        "write dissertation",
        "write a dissertation",
        "write me a dissertation",
        "create dissertation",
        "create a dissertation",
        "do a dissertation",
        "do dissertation",
        "generate dissertation",
        "produce dissertation",
        "full thesis",
        "write thesis",
        "write a thesis",
        "write me a thesis",
        "create thesis",
        "create a thesis",
        "do a thesis",
        "do thesis",
        "generate thesis",
        "write project",
        "write a project",
        "write me a project",
        "full project",
        "complete project",
        "entire project",
    ]
    if any(t in text for t in dissertation_triggers) or (
        "dissertation" in text
        and any(k in text for k in ["write", "full", "complete", "entire", "create", "do", "generate"])
    ) or (
        "thesis" in text
        and any(k in text for k in ["write", "full", "complete", "entire", "create", "do", "generate"])
    ):
        topic_match = re.search(r"\b(?:on|about)\b\s+(.+)$", text)
        topic = topic_match.group(1).strip().rstrip(".") if topic_match else None
        return {"intent": "write_dissertation", "target_section": None, "topic": topic}

    if (
        "project" in text
        and any(k in text for k in ["full", "complete", "entire", "whole", "write", "create", "build", "do", "generate"])
    ):
        topic_match = re.search(r"\b(?:on|about)\b\s+(.+)$", text)
        topic = topic_match.group(1).strip().rstrip(".") if topic_match else None
        return {"intent": "write_dissertation", "target_section": None, "topic": topic}

    if "outline" in text:
        topic_match = re.search(r"\bon\b\s+(.+)$", text)
        topic = topic_match.group(1).strip().rstrip(".") if topic_match else None
        return {"intent": "create_outline", "target_section": None, "topic": topic}

    if "chart" in text or "graph" in text:
        return {"intent": "add_chart", "target_section": target, "topic": None}

    if "image" in text or "figure" in text:
        return {"intent": "add_image", "target_section": target, "topic": None}

    if "report" in text:
        return {"intent": "write_report", "target_section": None, "topic": None}

    if "assignment" in text:
        return {"intent": "write_assignment", "target_section": None, "topic": None}

    if "powerpoint" in text or "presentation" in text or "slides" in text:
        return {"intent": "write_presentation", "target_section": None, "topic": None}

    if "excel" in text or "spreadsheet" in text:
        return {"intent": "write_spreadsheet", "target_section": None, "topic": None}

    if any(k in text for k in [
        "correct", "improve", "enhance", "fix",
        "put it again", "add it again", "add it back",
        "put back", "put the", "include the",
        "write it again", "write it back", "write again",
        "generate the", "regenerate the",
    ]):
        return {"intent": "enhance_section", "target_section": target, "topic": None}

    if any(k in text for k in [
        "redo", "rewrite", "write chapter", "replace chapter",
        "write the", "write me the", "write a new",
    ]):
        return {"intent": "write_section", "target_section": target, "topic": None}

    return {"intent": "chat", "target_section": None, "topic": None}


def _fallback_subsection_text(topic: str, section_title: str, subsection: str) -> str:
    return (
        f"This section discusses {subsection.lower()} in relation to {topic}. "
        "It provides practical context, highlights key issues, and links the discussion to the overall study objectives. "
        "The content is structured to maintain logical flow and support evidence-based academic writing."
    )


def _chapter_number_from_title(title: str) -> int | None:
    match = re.search(r"chapter\s*(\d+)", (title or "").lower())
    if match:
        return int(match.group(1))
    return None


def _full_context_for_generation(document: Document, upto_index: int | None = None) -> str:
    sections = (document.content or {}).get("sections", [])
    if upto_index is not None:
        sections = sections[: upto_index + 1]
    parts: list[str] = [f"Title: {document.title}"]
    for sec in sections:
        st = sec.get("title", "")
        sc = sec.get("content", "")
        if st:
            parts.append(f"\n## {st}")
        if sc:
            parts.append(sc[:1200])
    return "\n".join(parts)


def _build_section_content(
    section_title: str,
    subsection_titles: list[str],
    topic: str,
    context: str,
) -> str:
    blocks: list[str] = []
    rolling_context = context or ""
    for subsection in subsection_titles:
        try:
            subsection_text = generate_section_content(
                title=subsection,
                topic=topic,
                context=(
                    f"Section: {section_title}\n{subsection}\n\n"
                    "Previously written content:\n"
                    f"{rolling_context[-2200:]}"
                ),
                word_count=220,
            )
        except Exception:
            subsection_text = _fallback_subsection_text(topic, section_title, subsection)
        blocks.append(f"{subsection}\n{subsection_text}")
        rolling_context = f"{rolling_context}\n\n{subsection}\n{subsection_text}".strip()
    return "\n\n".join(blocks)


def _heading_positions(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    positions: list[tuple[int, int, str]] = []
    cursor = 0
    for line in lines:
        stripped = line.strip()
        if stripped and (
            re.match(r"^\d+(?:\.\d+)*\s+", stripped)
            or stripped.lower().startswith("chapter ")
        ):
            positions.append((cursor, cursor + len(line), stripped))
        cursor += len(line)
    return positions


def _replace_subsection_if_present(section_text: str, subsection_query: str, new_block: str) -> str | None:
    positions = _heading_positions(section_text)
    if not positions:
        return None

    query = subsection_query.lower().strip()
    query_num_match = re.search(r"\b\d+(?:\.\d+)*\b", query)
    query_num = query_num_match.group(0) if query_num_match else None

    hit_index = None
    for idx, (_, _, heading) in enumerate(positions):
        heading_l = heading.lower()
        heading_num_match = re.search(r"\b\d+(?:\.\d+)*\b", heading_l)
        heading_num = heading_num_match.group(0) if heading_num_match else None
        if (query_num and heading_num == query_num) or (query in heading_l):
            hit_index = idx
            break

    if hit_index is None:
        return None

    start = positions[hit_index][0]
    end = positions[hit_index + 1][0] if hit_index + 1 < len(positions) else len(section_text)
    return section_text[:start] + new_block.strip() + "\n\n" + section_text[end:]


def _extract_subsection_phrase(instruction: str) -> str:
    text = (instruction or "").lower()
    subsection_num = re.search(r"\b\d+\.\d+(?:\.\d+)*\b", text)
    if subsection_num:
        return subsection_num.group(0)

    # Ordered from most specific to least so longer matches win
    known = [
        "background of the study",
        "background of study",
        "statement of the problem",
        "research objectives",
        "research questions",
        "research hypotheses",
        "significance of the study",
        "scope and delimitations",
        "definition of key terms",
        "key terms",
        "conceptual review",
        "empirical review",
        "theoretical framework",
        "research design",
        "target population",
        "sampling techniques",
        "data collection",
        "data analysis",
        "reliability and validity",
        "ethical considerations",
        "data presentation",
        "discussion of findings",
        "summary of findings",
        "recommendations",
        "limitations of the study",
        "areas for further research",
        "further research",
        "literature review",
        "methodology",
        "results and discussion",
        "conclusion and recommendations",
        "references and appendices",
        "introduction",
        "conclusion",
        "abstract",
        "references",
        "appendices",
    ]
    for phrase in known:
        if phrase in text:
            return phrase
    # Fallback: extract what comes after action verbs
    verb_match = re.search(
        r"(?:correct|fix|enhance|improve|rewrite|write|add|put|include|generate)\s+(?:the\s+|a\s+|an\s+)?(.+)",
        text,
    )
    if verb_match:
        candidate = verb_match.group(1).strip().rstrip(".!?")
        if candidate and len(candidate) < 80:
            return candidate
    return instruction


def _extract_chapter_numbers(text: str) -> list[int]:
    source = (text or "").lower()
    numbers: set[int] = set()

    def _add_from_chunk(chunk: str) -> None:
        for rng in re.finditer(r"(\d+)\s*[-to]+\s*(\d+)", chunk):
            start = int(rng.group(1))
            end = int(rng.group(2))
            lo, hi = min(start, end), max(start, end)
            for n in range(lo, hi + 1):
                numbers.add(n)
        for num in re.findall(r"\b\d+\b", chunk):
            numbers.add(int(num))

    for match in re.finditer(r"chapter\s*(\d+)\s*[-to]+\s*(\d+)", source):
        start = int(match.group(1))
        end = int(match.group(2))
        lo, hi = min(start, end), max(start, end)
        for n in range(lo, hi + 1):
            numbers.add(n)

    for match in re.finditer(r"chapters?\s*([\d\s,\-andto]+)", source):
        _add_from_chunk(match.group(1))

    for match in re.finditer(r"chapter\s*(\d+)", source):
        numbers.add(int(match.group(1)))

    return sorted(n for n in numbers if 1 <= n <= 20)


def _chapter_title_from_number(chapter_number: int) -> str:
    for item in DISSERTATION_TEMPLATE:
        number = _chapter_number_from_title(item.get("title", ""))
        if number == chapter_number:
            return item["title"]
    return f"Chapter {chapter_number}"


def _step_label(step: str) -> str:
    return (step or "").strip().lstrip("- ").strip()


def _build_chat_summary(plan: list[dict[str, Any]], intent: str, document_updated: bool) -> dict[str, Any]:
    items = [_step_label(step.get("step", "")) for step in plan if step.get("step")]
    total = len(items)
    completed = sum(1 for step in plan if step.get("status") == "done")
    pending = total - completed
    completion_percent = int(round((completed / total) * 100)) if total else 0
    next_tasks = [
        _step_label(step.get("step", ""))
        for step in plan
        if step.get("status") in {"pending", "error"}
    ][:6]

    done_items = [
        _step_label(step.get("step", ""))
        for step in plan
        if step.get("status") == "done"
    ]
    done_preview = ", ".join(done_items[:3]) if done_items else "No steps completed yet"

    if pending == 0:
        stage = "All planned tasks completed"
    elif completed == 0:
        stage = "Execution queued"
    else:
        stage = "Execution in progress"

    if document_updated:
        stage = f"{stage}; document updated"

    # Chapter-level plan (top-level items only — indent = 0)
    chapter_plan = [
        {"step": _step_label(s.get("step", "")), "status": s.get("status", "pending")}
        for s in plan
        if not (s.get("step") or "").startswith(" ")
    ]

    return {
        "stage": stage,
        "intent": intent,
        "todo_list": items,
        "chapter_plan": chapter_plan,
        "completion_percent": completion_percent,
        "tasks_completed": completed,
        "tasks_pending": pending,
        "next_tasks": next_tasks,
        "done_brief": done_preview,
    }


def _find_section_index_by_subsection(document: Document, query: str) -> int | None:
    query_l = (query or "").lower().strip()
    if not query_l:
        return None

    subsection_num = re.search(r"\b(\d+)\.(\d+)(?:\.\d+)*\b", query_l)
    if subsection_num:
        chapter_num = subsection_num.group(1)
        by_chapter = find_section(document.content, f"Chapter {chapter_num}")
        if by_chapter is not None:
            return by_chapter

    sections = (document.content or {}).get("sections", [])
    for idx, section in enumerate(sections):
        title = (section.get("title") or "").lower()
        content = (section.get("content") or "").lower()
        if query_l in title or query_l in content:
            return idx
    return None


def _chat_summary_text(summary: dict[str, Any]) -> str:
    next_tasks = summary.get("next_tasks") or []
    next_label = ", ".join(next_tasks[:3]) if next_tasks else "None (all tasks completed)"
    return (
        f"Stage: {summary.get('stage', 'Update available')}\n"
        f"Done (brief): {summary.get('done_brief', 'N/A')}\n"
        f"To-do count: {len(summary.get('todo_list', []))}\n"
        f"Completion: {summary.get('completion_percent', 0)}%\n"
        f"Completed tasks: {summary.get('tasks_completed', 0)}\n"
        f"Pending tasks: {summary.get('tasks_pending', 0)}\n"
        f"Next tasks: {next_label}"
    )


def _needs_todo_workflow(intent: str, message: str) -> bool:
    text = (message or "").strip().lower()
    if intent == "write_dissertation":
        return True

    if intent in {
        "chat",
        "summarize_document",
        "create_outline",
        "write_report",
        "write_assignment",
        "write_presentation",
        "write_spreadsheet",
        "enhance_document",
        "enhance_section",
        "write_section",
        "add_chart",
        "add_image",
    }:
        return any(
            k in text
            for k in [
                "plan",
                "todo",
                "to-do",
                "step by step",
                "workflow",
                "stages",
                "execute",
            ]
        )

    return False


def _compact_doc_summary(document: Document) -> str:
    sections = (document.content or {}).get("sections", [])
    if not sections:
        return "This document is currently empty."

    lines: list[str] = [f"Summary of {document.title}:"]
    for sec in sections[:6]:
        title = sec.get("title", "Untitled section")
        content = (sec.get("content") or "").strip().replace("\n", " ")
        snippet = content[:170] + ("..." if len(content) > 170 else "")
        lines.append(f"- {title}: {snippet or 'No content yet.'}")
    if len(sections) > 6:
        lines.append(f"- Additional sections not shown: {len(sections) - 6}")
    return "\n".join(lines)


def _summarize_document(document: Document, user_message: str, plan: list) -> tuple[str, bool]:
    if plan:
        _done(plan, 0)
    doc_context = _flatten_doc(document, truncate=False)
    msg_lower = user_message.lower()
    if any(w in msg_lower for w in ["analys", "analyze", "review", "examine"]):
        task_instruction = (
            "Provide a thorough academic analysis of this document. Cover:\n"
            "1. Overview of the topic and main argument\n"
            "2. Structure and organisation (chapters/sections present, logical flow)\n"
            "3. Strength of content in each chapter\n"
            "4. Gaps, weaknesses, or areas needing improvement\n"
            "5. Language and academic writing quality\n"
            "6. Overall assessment and specific recommendations\n"
            "Be specific — reference actual section titles and content from the document."
        )
    else:
        task_instruction = (
            "Write a clear, concise summary of this document. Include:\n"
            "1. The main topic and purpose\n"
            "2. Key points covered in each chapter or section\n"
            "3. Main findings or conclusions\n"
            "Be specific — reference actual section titles and content from the document."
        )
    summary_prompt = (
        f"{task_instruction}\n\n"
        f"User request: \"{user_message}\"\n"
        "Do NOT modify the document. Respond in plain prose."
    )
    try:
        reply = chat_with_document(summary_prompt, doc_context)
    except Exception:
        reply = _compact_doc_summary(document)
    if plan:
        _all_done(plan)
    return reply, False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _flatten_doc(document: Document, truncate: bool = False) -> str:
    content = document.content or {}
    parts = [f"Title: {document.title}"]
    for section in content.get("sections", []):
        title = section.get("title", "")
        body = section.get("content", "")
        if title:
            parts.append(f"\n## {title}")
        if body:
            parts.append(body[:600] if truncate else body)
    return "\n".join(parts)


def _save(document: Document, note: str) -> None:
    document.save(update_fields=["content", "updated_at"])
    DocumentVersion.objects.create(
        document=document, content=document.content, note=note
    )


def _done(plan: list[dict], idx: int) -> None:
    if idx < len(plan):
        plan[idx]["status"] = "done"


def _all_done(plan: list[dict]) -> None:
    for step in plan:
        step["status"] = "done"


# ── Main entry point ─────────────────────────────────────────────────────────

def run_agent(document: Document, message: str, model_choice: str | None = None) -> dict[str, Any]:
    """
    Returns:
    {
        "reply":             str,
        "plan":              [{"step": str, "status": "done"|"pending"|"error"}],
        "document_updated":  bool,
        "intent":            str,
    }
    """
    set_active_model(model_choice)
    doc_context = _flatten_doc(document)
    lowered_message = (message or "").strip().lower()

    # 1. Classify intent
    intent_data = _heuristic_intent(message)
    if intent_data.get("intent") == "chat":
        intent_data = classify_intent(message, doc_context)
    if intent_data.get("intent") == "chat":
        heur = _heuristic_intent(message)
        if heur.get("intent") != "chat":
            intent_data = heur
    intent = intent_data.get("intent", "chat")
    target_section = intent_data.get("target_section")
    topic = (
        intent_data.get("topic")
        or (document.content or {}).get("topic")
        or document.title
    )
    topic = re.sub(r"^on\s+", "", (topic or "").strip(), flags=re.IGNORECASE)

    # 2. Build plan
    if intent == "summarize_document":
        steps = ["Reading current document", "Preparing summary response"]
    else:
        steps = create_execution_plan(intent)
    plan = [{"step": s, "status": "pending"} for s in steps]
    todo_required = _needs_todo_workflow(intent, lowered_message)

    had_error = False
    error_detail = ""

    # 3. Execute
    try:
        chapter_numbers = _extract_chapter_numbers(message)

        if intent == "summarize_document":
            reply, updated = _summarize_document(document, message, plan)
        elif intent == "enhance_document":
            reply, updated = _enhance_document(document, topic, plan)
        elif intent == "enhance_section" and len(chapter_numbers) > 1:
            reply, updated = _enhance_chapter_batch(document, chapter_numbers, topic, message, plan)
        elif intent == "write_section" and len(chapter_numbers) > 1:
            reply, updated = _rewrite_chapter_batch(document, chapter_numbers, topic, message, plan)
        elif intent == "enhance_section":
            reply, updated = _enhance_section(document, target_section, topic, message, plan)
        elif intent == "write_section":
            reply, updated = _write_section(document, target_section, topic, message, plan)
        elif intent == "write_dissertation":
            reply, updated = _write_dissertation(document, topic, message, plan)
        elif intent == "create_outline":
            reply, updated = _create_outline(document, topic, plan)
        elif intent == "write_report":
            reply, updated = _write_structured_document(document, topic, "report", plan)
        elif intent == "write_assignment":
            reply, updated = _write_structured_document(document, topic, "assignment", plan)
        elif intent == "write_presentation":
            reply, updated = _write_structured_document(document, topic, "presentation", plan)
        elif intent == "write_spreadsheet":
            reply, updated = _write_structured_document(document, topic, "spreadsheet", plan)
        elif intent == "add_chart":
            reply, updated = _add_chart(document, target_section, plan)
        elif intent == "add_image":
            reply, updated = _add_image(document, target_section, message, plan)
        else:
            try:
                reply = chat_with_document(message, doc_context)
            except Exception as exc:
                had_error = True
                error_detail = str(exc)
                reply = (
                    "I could not complete that response due to a network or model issue. "
                    f"Details: {exc}."
                )
            updated = False
            _all_done(plan)
    except Exception as exc:
        logger.error("Agent execution error: %s", exc, exc_info=True)
        had_error = True
        error_detail = str(exc)
        reply = f"Something went wrong: {exc}"
        updated = False
        for step in plan:
            if step["status"] == "pending":
                step["status"] = "error"

    summary = _build_chat_summary(plan, intent, updated) if todo_required else None
    orchestration = {
        "mode": "todo" if todo_required else "direct",
        "todo_required": todo_required,
        "execution": "stepwise" if todo_required else "single_pass",
        "status": "failed" if had_error or any(s.get("status") == "error" for s in plan) else "ok",
    }
    if error_detail:
        orchestration["error"] = error_detail

    return {
        "reply": _chat_summary_text(summary) if summary else reply,
        "plan": plan if todo_required else [],
        "chat_summary": summary,
        "orchestration": orchestration,
        "document_updated": updated,
        "intent": intent,
        "model": get_model_label(),
    }



# ── Action handlers ───────────────────────────────────────────────────────────

def _enhance_document(document: Document, topic: str, plan: list) -> tuple[str, bool]:
    sections = (document.content or {}).get("sections", [])
    if not sections:
        _all_done(plan)
        return (
            "The document has no sections yet. Try: 'Create an outline for [topic]' first.",
            False,
        )

    _done(plan, 0)
    _done(plan, 1)
    count = 0
    for i, section in enumerate(sections):
        original = section.get("content", "").strip()
        if not original:
            continue
        try:
            sections[i]["content"] = enhance_text(original, topic)
            count += 1
        except Exception as exc:
            logger.warning("Enhance section %d failed: %s", i, exc)
        _done(plan, 2)

    _all_done(plan)
    if count:
        document.content["sections"] = sections
        _save(document, "enhance-document")
        return (
            f"Enhanced {count} section(s) across the document — improved clarity, "
            "academic tone, and readability.",
            True,
        )
    return "No existing content found to enhance. Add text to sections first.", False


def _enhance_section(
    document: Document,
    target: str | None,
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    _done(plan, 0)
    query = target or _extract_subsection_phrase(instruction)
    idx = find_section(document.content, query)
    if idx is None:
        idx = _find_section_index_by_subsection(document, query)

    # Section does not exist — write/generate it instead of erroring
    if idx is None:
        return _write_section(document, query, topic, instruction, plan)

    section = document.content["sections"][idx]
    original = (section.get("content") or "").strip()

    # Section exists but is empty — write it fresh
    if not original:
        return _write_section(document, query, topic, instruction, plan)

    _done(plan, 1)

    enhance_instruction = (
        f"{instruction}\n\n"
        "Improve this section: fix grammar, strengthen academic tone, improve argument clarity "
        "and structure. Preserve all factual claims and headings. "
        "Return ONLY the improved text with no meta-commentary."
    )
    try:
        enhanced = enhance_text(original, topic, enhance_instruction)
    except Exception:
        enhanced = _fallback_subsection_text(topic, section.get("title", query), query)

    subsection_block = _replace_subsection_if_present(
        original,
        subsection_query=query,
        new_block=f"{query}\n{enhanced}",
    )
    final_content = subsection_block if subsection_block else enhanced

    document.content = update_section(document.content, idx, final_content)
    _save(document, f"enhance-section:{query}")
    _all_done(plan)
    return (
        f"Corrected and enhanced section '{section.get('title', query)}' — "
        "improved clarity, structure, and academic tone.",
        True,
    )


def _write_section(
    document: Document,
    target: str | None,
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    _done(plan, 0)
    section_name = target or instruction or "New Section"
    idx = find_section(document.content, section_name)
    _done(plan, 1)

    existing_context = _full_context_for_generation(document)
    try:
        content = generate_section_content(
            section_name,
            topic,
            context=f"User request: {instruction}\n\nDocument context:\n{existing_context[:1800]}",
            word_count=320,
        )
    except Exception:
        content = _fallback_subsection_text(topic, section_name, section_name)
    _done(plan, 2)

    if idx is not None:
        document.content = update_section(document.content, idx, content)
        title = document.content["sections"][idx].get("title", section_name)
    else:
        sections = document.content.setdefault("sections", [])
        sections.append({"title": section_name, "content": content})
        title = section_name

    _save(document, f"write-section:{section_name}")
    _all_done(plan)
    return f"Written and saved section '{title}' (existing content replaced where applicable).", True


def _enhance_chapter_batch(
    document: Document,
    chapter_numbers: list[int],
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    plan.clear()
    plan.append({"step": "Preparing chapter correction queue", "status": "pending"})
    for num in chapter_numbers:
        plan.append({"step": f"Correcting Chapter {num}", "status": "pending"})

    _done(plan, 0)
    updated = False
    touched: list[str] = []
    sections = (document.content or {}).get("sections", [])

    for idx, chapter_number in enumerate(chapter_numbers, start=1):
        chapter_query = f"Chapter {chapter_number}"
        sec_idx = find_section(document.content, chapter_query)
        if sec_idx is None or sec_idx >= len(sections):
            plan[idx]["status"] = "error"
            continue

        original = sections[sec_idx].get("content", "")
        if not original.strip():
            original = f"Discuss {chapter_query} for {topic}."

        try:
            rewritten = enhance_text(original, topic, instruction)
        except Exception:
            rewritten = _fallback_subsection_text(topic, chapter_query, chapter_query)

        document.content = update_section(document.content, sec_idx, rewritten)
        _save(document, f"enhance-section:{chapter_query}")
        plan[idx]["status"] = "done"
        updated = True
        touched.append(chapter_query)

    if touched:
        return f"Corrected {len(touched)} chapter(s): {', '.join(touched)}.", True
    return "No target chapters were found to correct.", updated


def _rewrite_chapter_batch(
    document: Document,
    chapter_numbers: list[int],
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    plan.clear()
    plan.append({"step": "Preparing chapter rewrite queue", "status": "pending"})
    for num in chapter_numbers:
        plan.append({"step": f"Rewriting Chapter {num}", "status": "pending"})

    _done(plan, 0)
    updated = False
    rewritten_titles: list[str] = []

    for idx, chapter_number in enumerate(chapter_numbers, start=1):
        chapter_title = _chapter_title_from_number(chapter_number)
        chapter_template = next(
            (
                item for item in DISSERTATION_TEMPLATE
                if _chapter_number_from_title(item.get("title", "")) == chapter_number
            ),
            None,
        )
        subsection_titles = (chapter_template or {}).get("subsections") or [f"{chapter_number}.1 Overview"]
        try:
            chapter_content = _build_section_content(
                section_title=chapter_title,
                subsection_titles=subsection_titles,
                topic=topic,
                context=f"User request: {instruction}\n\n{_full_context_for_generation(document)[:1600]}",
            )
        except Exception:
            chapter_content = _fallback_subsection_text(topic, chapter_title, chapter_title)

        sec_idx = find_section(document.content, chapter_title)
        if sec_idx is None:
            sec_idx = find_section(document.content, f"Chapter {chapter_number}")

        if sec_idx is not None:
            document.content = update_section(document.content, sec_idx, chapter_content)
        else:
            document.content.setdefault("sections", []).append(
                {"title": chapter_title, "content": chapter_content}
            )

        _save(document, f"rewrite-section:{chapter_title}")
        plan[idx]["status"] = "done"
        updated = True
        rewritten_titles.append(chapter_title)

    if rewritten_titles:
        return f"Rewrote {len(rewritten_titles)} chapter(s): {', '.join(rewritten_titles)}.", True
    return "No chapters were rewritten.", updated


def _write_dissertation(
    document: Document,
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    plan.clear()
    plan.append({"step": "Creating dissertation to-do list", "status": "pending"})

    design = _research_design(instruction, topic, document)
    objectives = _extract_objectives(document, topic)

    chapter_blueprints: list[dict[str, Any]] = []
    for template in DISSERTATION_TEMPLATE:
        title = template["title"]
        if "chapter 4" in title.lower():
            nodes = _chapter4_subsections(design, objectives)
        else:
            nodes = [_normalize_subsection_node(s) for s in template.get("subsections", [])]
        chapter_blueprints.append({"title": title, "nodes": nodes})

    for chapter in chapter_blueprints:
        plan.append({"step": f"Writing {chapter['title']}", "status": "pending"})
        _append_node_plan_steps(plan, chapter["nodes"], depth=1)

    _done(plan, 0)

    sections: list[dict[str, Any]] = []
    plan_cursor = [1]
    figure_counter = [_next_caption_number(document, "figure")]
    table_counter = [_next_caption_number(document, "table")]

    for chapter in chapter_blueprints:
        chapter_title = chapter["title"]
        ch_num = _chapter_number_from_title(chapter_title)

        # After Chapter 1 is written, re-extract objectives for Chapter 4
        if ch_num == 4 and sections:
            real_objectives = _extract_objectives(document, topic)
            candidate_nodes = _chapter4_subsections(design, real_objectives)
            # Only swap if top-level node count is unchanged (preserves plan cursor alignment)
            if len(candidate_nodes) == len(chapter["nodes"]):
                chapter["nodes"] = candidate_nodes

        _done(plan, plan_cursor[0])
        plan_cursor[0] += 1

        # Add chapter placeholder first so subsection updates are persisted progressively.
        section_payload: dict[str, Any] = {"title": chapter_title, "content": chapter_title}
        sections.append(section_payload)
        document.content = {
            "topic": topic,
            "research_design": design,
            "research_objectives": objectives,
            "sections": sections,
        }
        _save(document, f"dissertation-step:{chapter_title}:start")

        def _persist_subsection_progress(partial_text: str, partial_blocks: list[dict[str, str]], node_title: str) -> None:
            safe_node = re.sub(r"[^a-zA-Z0-9_.-]+", "-", node_title).strip("-")[:60] or "subsection"
            content = f"{chapter_title}\n\n{partial_text}" if partial_text.strip() else chapter_title
            section_payload["content"] = content
            if partial_blocks:
                section_payload["blocks"] = partial_blocks
            elif "blocks" in section_payload:
                section_payload.pop("blocks", None)

            document.content = {
                "topic": topic,
                "research_design": design,
                "research_objectives": objectives,
                "sections": sections,
            }
            _save(document, f"dissertation-step:{chapter_title}:{safe_node}")

        current_context = _full_context_for_generation(document)
        ch_word_count = 500 if ch_num == 2 else 220
        chapter_text, _, chapter_blocks = _execute_subsection_nodes(
            nodes=chapter["nodes"],
            section_title=chapter_title,
            topic=topic,
            research_design=design,
            rolling_context=current_context,
            plan=plan,
            plan_cursor=plan_cursor,
            figure_counter=figure_counter,
            table_counter=table_counter,
            on_node_completed=_persist_subsection_progress,
            default_word_count=ch_word_count,
        )

        # Ensure chapter heading appears before chapter content in the editor body.
        chapter_content = f"{chapter_title}\n\n{chapter_text}" if chapter_text.strip() else chapter_text
        section_payload = {"title": chapter_title, "content": chapter_content}
        if chapter_blocks:
            section_payload["blocks"] = chapter_blocks
        elif "blocks" in section_payload:
            section_payload.pop("blocks", None)

        document.content = {
            "topic": topic,
            "research_design": design,
            "research_objectives": objectives,
            "sections": sections,
        }
        _save(document, f"dissertation-step:{chapter_title}")

    document.title = f"Dissertation: {topic}"
    document.save(update_fields=["title", "updated_at"])
    _all_done(plan)

    reply = (
        f"Dissertation generation complete for '{topic}' using a {design.replace('_', ' ')} design flow. "
        "The agent executed chapter-level and nested subsection to-do lists with individual prompts per step."
    )
    return reply, True


def _create_outline(document: Document, topic: str, plan: list) -> tuple[str, bool]:
    _done(plan, 0)
    chapters = generate_outline_sections(topic)
    _done(plan, 1)

    sections = [
        {"title": c.get("title", f"Chapter {i + 1}"), "content": ""}
        for i, c in enumerate(chapters)
    ]
    document.content = {"topic": topic, "sections": sections}
    _save(document, "outline-created")
    _all_done(plan)

    titles = "\n".join(f"• {s['title']}" for s in sections)
    return f"Created outline with {len(sections)} chapters:\n{titles}", True


def _write_structured_document(
    document: Document,
    topic: str,
    kind: str,
    plan: list,
) -> tuple[str, bool]:
    kind_map = {
        "report": ["Executive Summary", "Introduction", "Findings", "Recommendations", "Conclusion"],
        "assignment": ["Introduction", "Main Discussion", "Analysis", "Conclusion", "References"],
        "presentation": ["Slide 1: Title", "Slide 2: Problem", "Slide 3: Approach", "Slide 4: Findings", "Slide 5: Conclusion"],
        "spreadsheet": ["Dataset Overview", "Key Metrics", "Summary Table", "Insights", "Recommendations"],
    }
    structure = kind_map.get(kind, kind_map["report"])

    plan.clear()
    plan.append({"step": f"Creating {kind} to-do list", "status": "pending"})
    for title in structure:
        plan.append({"step": f"Writing {title}", "status": "pending"})

    _done(plan, 0)

    sections: list[dict[str, str]] = []
    for idx, title in enumerate(structure, start=1):
        try:
            text = generate_section_content(
                title=title,
                topic=topic,
                context=(
                    f"Document type: {kind}.\n"
                    f"Current draft context:\n{_full_context_for_generation(document)[:2200]}"
                ),
                word_count=170,
            )
        except Exception:
            text = _fallback_subsection_text(topic, kind.capitalize(), title)
        sections.append({"title": title, "content": text})

        # Persist after each step so generation behaves like an executing agent.
        document.content = {"topic": topic, "sections": sections, "document_type": kind}
        _save(document, f"{kind}-step:{title}")
        _done(plan, idx)

    document.content = {"topic": topic, "sections": sections, "document_type": kind}
    document.title = f"{kind.capitalize()}: {topic}"
    document.save(update_fields=["content", "title", "updated_at"])
    DocumentVersion.objects.create(document=document, content=document.content, note=f"{kind}-generated")

    reply = (
        f"Generated a complete {kind} for '{topic}'. "
        "The content has been written directly into the current document."
    )
    return reply, True


def _add_chart(document: Document, target: str | None, plan: list) -> tuple[str, bool]:
    _done(plan, 0)
    sections = (document.content or {}).get("sections", [])
    if not sections:
        _all_done(plan)
        return "No sections in document. Create an outline first.", False

    idx = find_section(document.content, target) if target else len(sections) - 1
    if idx is None:
        idx = 0
    section = sections[idx]
    _done(plan, 1)

    _section_title = section.get("title", "Data")
    _ai = _ai_chart_series(_section_title, n_points=8)
    chart_path = generate_chart(
        series=_ai["series"],
        chart_type=_ai["chart_type"],
        title=_section_title,
    )
    section.setdefault("blocks", []).append(
        {"type": "chart", "src": chart_path, "caption": f"Chart for {section.get('title', 'section')}"}
    )
    _save(document, f"chart:{target or 'section'}")
    _all_done(plan)
    return f"Added a chart to '{section.get('title', 'the section')}'.", True


def _add_image(
    document: Document, target: str | None, prompt: str, plan: list
) -> tuple[str, bool]:
    _done(plan, 0)
    sections = (document.content or {}).get("sections", [])

    idx = _framework_target_index(document, target, prompt)
    if idx is None:
        _all_done(plan)
        return "No sections available. Create an outline first.", False

    framework_request = any(
        key in (prompt or "").lower()
        for key in ["conceptual", "theoretical", "framework", "research model", "model"]
    )

    framework_spec = _build_framework_spec(document, target, prompt) if framework_request else None

    try:
        image_path = generate_image(prompt, framework_spec=framework_spec)
    except Exception as exc:
        _all_done(plan)
        return f"Image generation failed: {exc}", False

    _done(plan, 1)

    if sections:
        section = sections[idx]
        blocks = section.setdefault("blocks", [])
        block_id = f"img-{idx + 1}-{len(blocks) + 1}"
        caption = (
            (framework_spec or {}).get("title")
            or prompt[:80]
            or f"Image for {section.get('title', 'section')}"
        )
        blocks.append(
            {"type": "image", "src": image_path, "caption": str(caption)[:120], "block_id": block_id}
        )
        section["content"] = _insert_block_marker(section.get("content", ""), block_id, prompt)
        _save(document, f"image:{target or 'section'}")

    _all_done(plan)
    section_name = sections[idx].get("title", "the section") if sections else "the section"
    return f"Added an image to '{section_name}' using full-document context.", True
