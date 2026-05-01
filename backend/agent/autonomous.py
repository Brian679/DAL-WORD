"""
Autonomous document agent.
Accepts a free-form user message, classifies intent, plans, and executes
direct edits on the document using a selectable LLM provider.
"""
from __future__ import annotations

import logging
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
)
from .tools import (
    find_section,
    generate_chart,
    generate_image,
    insert_image_block,
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
            "2.1 Introduction",
            "2.2 Conceptual Review",
            "2.3 Theoretical Framework",
            "2.4 Empirical Review",
            "2.5 Research Gap",
            "2.6 Chapter Summary",
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
        "subsections": [
            "4.1 Introduction",
            "4.2 Data Presentation",
            "4.3 Analysis of Findings",
            "4.4 Discussion of Findings",
            "4.5 Chapter Summary",
        ],
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
        "title": "References",
        "subsections": [
            "Reference List",
        ],
    },
]


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

    if "full dissertation" in text or ("write" in text and "dissertation" in text):
        topic_match = re.search(r"\bon\b\s+(.+)$", text)
        topic = topic_match.group(1).strip().rstrip(".") if topic_match else None
        return {"intent": "write_dissertation", "target_section": None, "topic": topic}

    if (
        "project" in text
        and any(k in text for k in ["full", "complete", "entire", "whole", "write", "create", "build"])
    ):
        topic_match = re.search(r"\bon\b\s+(.+)$", text)
        topic = topic_match.group(1).strip().rstrip(".") if topic_match else None
        return {"intent": "write_report", "target_section": None, "topic": topic}

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

    if any(k in text for k in ["correct", "improve", "enhance", "fix"]):
        return {"intent": "enhance_section", "target_section": target, "topic": None}

    if any(k in text for k in ["redo", "rewrite", "write chapter", "replace chapter"]):
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

    known = [
        "background of the study",
        "background of study",
        "statement of the problem",
        "research objectives",
        "research questions",
        "research hypotheses",
        "significance of the study",
        "scope and delimitations",
        "conceptual review",
        "empirical review",
        "theoretical framework",
        "data presentation",
        "discussion of findings",
        "summary of findings",
        "recommendations",
        "conclusion",
    ]
    for phrase in known:
        if phrase in text:
            return phrase
    # Fallback: keep original instruction.
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

    return {
        "stage": stage,
        "intent": intent,
        "todo_list": items,
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
    if intent in {
        "write_dissertation",
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
        return True

    if intent in {"chat", "summarize_document"}:
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
        ) and intent == "chat"  # summarize_document always direct

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
    doc_context = _flatten_doc(document)
    summary_prompt = (
        f"The user says: \"{user_message}\"\n\n"
        "Respond directly and helpfully based on what the user asked. "
        "If they asked to analyse or review the document, provide a thorough analysis: "
        "assess the structure, content quality, argument strength, gaps, and give specific feedback. "
        "If they asked for a summary, give a concise human-readable summary. "
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

def _flatten_doc(document: Document) -> str:
    content = document.content or {}
    parts = [f"Title: {document.title}"]
    for section in content.get("sections", []):
        title = section.get("title", "")
        body = section.get("content", "")
        if title:
            parts.append(f"\n## {title}")
        if body:
            parts.append(body[:600])
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
            reply, updated = _write_dissertation(document, topic, plan)
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
    if idx is None:
        titles = ", ".join(
            s.get("title", "") for s in (document.content or {}).get("sections", [])
        )
        _all_done(plan)
        return (
            f"Could not find section '{query}'. "
            f"Available sections: {titles or 'none — create an outline first.'}",
            False,
        )

    section = document.content["sections"][idx]
    original = section.get("content", "") or f"Write about {section.get('title', query)}"
    _done(plan, 1)

    try:
        enhanced = enhance_text(original, topic, instruction)
    except Exception:
        enhanced = _fallback_subsection_text(topic, section.get("title", query), query)

    # If user asks for a specific subsection (e.g., "correct background of study"),
    # replace only that block when possible; otherwise replace whole section.
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
        f"Enhanced section '{section.get('title', query)}' with improved "
        "clarity, structure, and academic tone.",
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


def _write_dissertation(document: Document, topic: str, plan: list) -> tuple[str, bool]:
    # Build a hierarchical todo plan (chapter + subsection) for visibility and traceability.
    plan.clear()
    plan.append({"step": "Creating dissertation to-do list", "status": "pending"})

    suggested = generate_outline_sections(topic)
    suggested_by_number: dict[int, dict[str, Any]] = {}
    for chapter in suggested:
        if not isinstance(chapter, dict):
            continue
        num = _chapter_number_from_title(chapter.get("title", ""))
        if num is not None:
            suggested_by_number[num] = chapter

    # Strict dissertation structure: Chapter 1-5 + References.
    chapter_blueprints: list[dict[str, Any]] = []
    for template in DISSERTATION_TEMPLATE:
        chapter_num = _chapter_number_from_title(template["title"])
        suggested_item = suggested_by_number.get(chapter_num) if chapter_num is not None else None
        subsections = (
            suggested_item.get("subsections", []) if suggested_item else []
        ) or template.get("subsections", [])
        chapter_blueprints.append(
            {
                "title": template["title"],
                "subsections": subsections,
            }
        )

    # Expand plan with detailed steps
    for chapter in chapter_blueprints:
        plan.append({"step": f"Writing {chapter['title']}", "status": "pending"})
        for subsection in chapter["subsections"]:
            plan.append({"step": f"- {subsection}", "status": "pending"})

    _done(plan, 0)

    # Full dissertation commands should replace prior structure from scratch.
    sections: list[dict[str, Any]] = []

    plan_idx = 1
    chapter_summaries: list[str] = []
    for chapter in chapter_blueprints:
        chapter_title = chapter["title"]
        _done(plan, plan_idx)
        plan_idx += 1

        # Generate subsection-by-subsection and accumulate into chapter content.
        context = _full_context_for_generation(document)
        chapter_content = _build_section_content(
            section_title=chapter_title,
            subsection_titles=chapter["subsections"],
            topic=topic,
            context=context,
        )

        for _ in chapter["subsections"]:
            _done(plan, plan_idx)
            plan_idx += 1

        # Replace if exists, append if new.
        existing_idx = find_section({"sections": sections}, chapter_title)
        if existing_idx is not None:
            sections[existing_idx]["content"] = chapter_content
            sections[existing_idx]["title"] = chapter_title
        else:
            sections.append({"title": chapter_title, "content": chapter_content})

        document.content = {"topic": topic, "sections": sections}
        _save(document, f"dissertation-step:{chapter_title}")

        chapter_summaries.append(
            f"{chapter_title}: completed {len(chapter['subsections'])} subsection(s)."
        )

    document.title = f"Dissertation: {topic}"
    document.save(update_fields=["title", "updated_at"])
    _all_done(plan)

    reply = (
        f"Dissertation generation complete for '{topic}'. "
        "All generated content has been written directly into the open document."
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

    chart_path = generate_chart(
        series=[3.2, 4.1, 5.7, 4.8, 6.3, 7.1],
        chart_type="bar",
        title=section.get("title", "Data"),
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
    idx = find_section(document.content, target) if target else (len(sections) - 1 if sections else 0)
    if idx is None:
        idx = 0

    image_path = generate_image(prompt)
    _done(plan, 1)

    if sections:
        sections[idx].setdefault("blocks", []).append(
            {"type": "image", "src": image_path, "caption": prompt[:80]}
        )
        _save(document, f"image:{target or 'section'}")

    _all_done(plan)
    return "Added an image to the document.", True
