"""
Multi-agent planner layer.

Exposes three cooperating classes:

  IntentSpec   — structured representation of what the user wants
  TaskSpec     — a single atomic generation task (one subsection / one node)
  TaskQueue    — ordered list of TaskSpec items with status tracking
  PlannerAgent — orchestrates intent parsing → task decomposition → spec generation
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with executor.py)
# ---------------------------------------------------------------------------

@dataclass
class PlanItem:
    order: int
    title: str


def create_dissertation_outline(topic: str) -> list[PlanItem]:
    defaults = [
        "Introduction",
        "Literature Review",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
    ]
    return [PlanItem(order=i + 1, title=f"{i + 1}. {name}") for i, name in enumerate(defaults)]


def to_json(items: list[PlanItem]) -> list[dict[str, str | int]]:
    return [{"order": item.order, "title": item.title} for item in items]


# ---------------------------------------------------------------------------
# Pipeline data models
# ---------------------------------------------------------------------------

@dataclass
class IntentSpec:
    """Parsed, structured representation of the user's request."""
    intent: str                         # e.g. "write_dissertation"
    topic: str                          # research topic
    research_design: str                # quantitative / qualitative / mixed / non_empirical
    objectives: list[str]               # extracted research objectives
    target_section: str | None          # for section-level intents
    raw_message: str                    # original user text


@dataclass
class TaskSpec:
    """A single atomic generation task – one subsection, table, chart, or image."""
    id: str                             # unique identifier e.g. "ch1-node3"
    title: str                          # section/subsection title
    kind: str                           # "text" | "table" | "chart" | "image" | "objective_findings"
    word_count: int                     # target word count for text tasks
    guidelines: str                     # section-specific writing instructions
    context_hint: str                   # snippet of doc context passed to the generator
    chapter_title: str                  # parent chapter title
    chapter_num: int                    # e.g. 1, 2, 3 …
    topic: str                          # research topic (duplicated for convenience)
    research_design: str                # quantitative / qualitative / mixed / non_empirical
    meta: dict[str, Any] = field(default_factory=dict)  # arbitrary extra data (objective, table_type …)


# ---------------------------------------------------------------------------
# Task Queue
# ---------------------------------------------------------------------------

@dataclass
class _QueueEntry:
    task: TaskSpec
    status: str = "pending"     # "pending" | "running" | "done" | "failed" | "skipped"
    attempts: int = 0


class TaskQueue:
    """Ordered list of TaskSpec items with status tracking."""

    def __init__(self) -> None:
        self._entries: list[_QueueEntry] = []

    # ── Building ────────────────────────────────────────────────────────────

    def push(self, task: TaskSpec) -> None:
        self._entries.append(_QueueEntry(task=task))

    # ── Querying ─────────────────────────────────────────────────────────────

    def all(self) -> list[TaskSpec]:
        return [e.task for e in self._entries]

    def pending(self) -> list[TaskSpec]:
        return [e.task for e in self._entries if e.status == "pending"]

    def done(self) -> list[TaskSpec]:
        return [e.task for e in self._entries if e.status == "done"]

    def __len__(self) -> int:
        return len(self._entries)

    # ── Status updates ───────────────────────────────────────────────────────

    def mark_running(self, task_id: str) -> None:
        for e in self._entries:
            if e.task.id == task_id:
                e.status = "running"
                e.attempts += 1
                return

    def mark_done(self, task_id: str) -> None:
        for e in self._entries:
            if e.task.id == task_id:
                e.status = "done"
                return

    def mark_failed(self, task_id: str) -> None:
        for e in self._entries:
            if e.task.id == task_id:
                e.status = "failed"
                return

    def mark_skipped(self, task_id: str) -> None:
        for e in self._entries:
            if e.task.id == task_id:
                e.status = "skipped"
                return

    def attempts(self, task_id: str) -> int:
        for e in self._entries:
            if e.task.id == task_id:
                return e.attempts
        return 0

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._entries:
            counts[e.status] = counts.get(e.status, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------

# Heuristic keyword → research design mapping (mirrors autonomous.py)
_DESIGN_KEYWORDS: dict[str, list[str]] = {
    "mixed":        ["mixed method", "mixed-method", "mixed methods"],
    "quantitative": ["quantitative"],
    "qualitative":  ["qualitative"],
    "non_empirical": ["systematic review", "scoping review", "conceptual paper", "theoretical"],
}

# Sections that naturally call for a numbered/list format
_POINTFORM_KEYWORDS = frozenset([
    "research objective", "objectives", "research question", "hypothes",
    "recommendation", "further research", "areas for future", "definition of key",
])


class PlannerAgent:
    """
    Orchestrates three phases for every user request:

      1. parse_intent  — extract intent, topic, design, objectives from raw message
      2. decompose     — flatten chapter blueprints into a TaskQueue
      3. generate_spec — enrich each TaskSpec with guidelines and context

    Usage (inside _write_dissertation):

        planner = PlannerAgent()
        intent_spec = planner.parse_intent(message, topic, design, objectives)
        queue = planner.decompose(intent_spec, chapter_blueprints, doc_brief)
    """

    # ── Phase 1: Intent Parsing ──────────────────────────────────────────────

    def parse_intent(
        self,
        message: str,
        topic: str,
        research_design: str,
        objectives: list[str],
        intent: str = "write_dissertation",
        target_section: str | None = None,
    ) -> IntentSpec:
        """Build a structured IntentSpec from raw inputs."""
        return IntentSpec(
            intent=intent,
            topic=topic,
            research_design=research_design,
            objectives=list(objectives),
            target_section=target_section,
            raw_message=message,
        )

    # ── Phase 2: Task Decomposition ──────────────────────────────────────────

    def decompose(
        self,
        intent_spec: IntentSpec,
        chapter_blueprints: list[dict[str, Any]],
        doc_brief: str = "",
    ) -> TaskQueue:
        """
        Walk the chapter blueprint tree and create one TaskSpec per leaf node.
        Each TaskSpec is immediately enriched with writing guidelines.
        """
        queue = TaskQueue()
        for ch_idx, chapter in enumerate(chapter_blueprints):
            ch_title = chapter.get("title", f"Chapter {ch_idx + 1}")
            ch_num = self._chapter_number(ch_title)
            nodes = chapter.get("nodes", [])
            self._flatten_nodes(
                nodes=nodes,
                queue=queue,
                intent_spec=intent_spec,
                chapter_title=ch_title,
                chapter_num=ch_num,
                doc_brief=doc_brief,
                path_prefix=f"ch{ch_num}",
            )
        return queue

    # ── Phase 3: Spec Generation ─────────────────────────────────────────────

    def generate_spec(
        self,
        task: TaskSpec,
        intent_spec: IntentSpec,
        doc_context: str = "",
    ) -> TaskSpec:
        """
        Enrich a TaskSpec with section-specific guidelines and context snippet.
        Returns a new (enriched) TaskSpec.
        """
        guidelines = self._build_guidelines(task.title, intent_spec.topic, task.research_design)
        is_pointform = any(k in task.title.lower() for k in _POINTFORM_KEYWORDS)
        wc = 120 if is_pointform else task.word_count

        return TaskSpec(
            id=task.id,
            title=task.title,
            kind=task.kind,
            word_count=wc,
            guidelines=guidelines,
            context_hint=doc_context[-2000:] if doc_context else "",
            chapter_title=task.chapter_title,
            chapter_num=task.chapter_num,
            topic=task.topic,
            research_design=task.research_design,
            meta=task.meta,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _flatten_nodes(
        self,
        nodes: list[dict[str, Any]],
        queue: TaskQueue,
        intent_spec: IntentSpec,
        chapter_title: str,
        chapter_num: int,
        doc_brief: str,
        path_prefix: str,
    ) -> None:
        for n_idx, node in enumerate(nodes):
            task_id = f"{path_prefix}-n{n_idx}"
            kind = node.get("kind", "text")
            title = node.get("title", "Untitled")
            meta = node.get("meta") or {}
            default_wc = 500 if chapter_num == 2 else 220

            task = TaskSpec(
                id=task_id,
                title=title,
                kind=kind,
                word_count=default_wc,
                guidelines="",          # filled by generate_spec
                context_hint=doc_brief,
                chapter_title=chapter_title,
                chapter_num=chapter_num,
                topic=intent_spec.topic,
                research_design=intent_spec.research_design,
                meta=dict(meta),
            )
            # Enrich inline so the queue holds fully-specified tasks
            task = self.generate_spec(task, intent_spec, doc_context=doc_brief)
            queue.push(task)

            # Recurse into children
            children = node.get("children", [])
            if children:
                self._flatten_nodes(
                    nodes=children,
                    queue=queue,
                    intent_spec=intent_spec,
                    chapter_title=chapter_title,
                    chapter_num=chapter_num,
                    doc_brief=doc_brief,
                    path_prefix=task_id,
                )

    @staticmethod
    def _chapter_number(title: str) -> int:
        m = re.search(r"chapter\s+(\d+)", title, re.IGNORECASE)
        if m:
            return int(m.group(1))
        if "preliminary" in title.lower():
            return 0
        return 99

    @staticmethod
    def _build_guidelines(section_title: str, topic: str, research_design: str = "") -> str:
        """
        Return section-specific writing instructions.
        Mirrors the logic in autonomous._subsection_guidelines but kept
        here so the Planner is self-contained.
        """
        lowered = section_title.lower()

        if "abstract" in lowered:
            return ("Write a concise academic abstract (200–300 words): background, problem statement, "
                    "objectives, methodology, key findings, conclusion. Single flowing paragraph, no headings.")
        if "dedication" in lowered:
            return "Brief heartfelt dedication (3–6 lines). Begin with 'To…' or 'Dedicated to…'."
        if "acknowledgement" in lowered or "acknowledgment" in lowered:
            return "Formal acknowledgements (150–250 words): supervisors, institution, participants, family."
        if "table of contents" in lowered:
            return "Placeholder note that a full ToC will be compiled on final assembly; list main chapter titles."
        if "list of figure" in lowered:
            return "Placeholder note for List of Figures with a sample format line."
        if "list of table" in lowered:
            return "Placeholder note for List of Tables with a sample format line."
        if "abbreviation" in lowered or "acronym" in lowered:
            return (f"List of Abbreviations for '{topic}'. Format: ABBR — Full meaning. "
                    "At least 10–15 entries, field-specific.")
        if "chapter summary" in lowered:
            return ("Chapter Summary (200–300 words): recap key points, highlight findings, "
                    "explain link to next chapter. Past tense, specific to this chapter.")
        if "research objective" in lowered or (lowered.endswith("objectives") and "background" not in lowered):
            return ("Numbered list (4–6 items). Each starts with an action verb "
                    "(To examine / To investigate / To assess …). One sentence per objective.")
        if "research question" in lowered or (lowered.endswith("questions") and "background" not in lowered):
            return ("Numbered list (4–6 interrogative sentences) tied to objectives. "
                    "Answerable through the stated methodology.")
        if "hypothes" in lowered:
            return ("Numbered null/alternative pairs: N. H0: … H1: … (3–5 pairs). "
                    "Formal statistical language.")
        if "significance" in lowered and "statistical" not in lowered:
            return ("Structured significance: theoretical contribution, then practical sub-sections "
                    "(Researcher / Institution / Organisation / Policy). 2–4 sentences each.")
        if "background" in lowered and "theoretical" not in lowered:
            return ("Background (350–500 words): global context, country/sector situation, "
                    "problem identification, study justification.")
        if "statement of the problem" in lowered or "problem statement" in lowered:
            return ("Problem statement (250–350 words): articulate the problem, cite evidence, "
                    "consequences of non-action, what this study will do.")
        if "scope" in lowered or "delimitation" in lowered:
            return ("Scope & Delimitations (200–300 words): geography, population, time frame, "
                    "variables studied, exclusions with justification.")
        if "definition of key terms" in lowered or "key terms" in lowered:
            return ("Alphabetical list of 6–10 key terms. Each: bold term or term + colon, "
                    "2–3 sentence academic definition with citation.")
        if "recommendation" in lowered:
            return "Numbered practical recommendations (5–8 items) derived from findings."
        if "limitation" in lowered:
            return "Academic limitations (200–250 words): scope, sample, data, methodology."
        if "conclusion" in lowered:
            return "Conclusions (200–300 words): synthesise findings, restate significance, close."
        if "reference" in lowered:
            return (
                "No verified external sources were retrievable for this document (the literature search "
                "returned nothing, e.g. due to no network access). Write a short note stating plainly that "
                "automatic source retrieval failed and that the entries below are ILLUSTRATIVE PLACEHOLDERS, "
                "not verified citations — they must be replaced with real literature before submission. "
                "Then provide 8–12 example references in APA 7th edition format, each prefixed with "
                "'[Placeholder]' so they cannot be mistaken for verified sources."
            )
        if "appendix" in lowered or "appendices" in lowered:
            return "Appendix: label each item (Appendix A, B …), title, and brief description."

        # ── System-build (SDLC) methodology subsections — Chapter 3 of a
        # software/web/information-system build dissertation. These titles never
        # occur in a survey-based study, so the title alone is a safe signal.
        if "research design" in lowered:
            if research_design == "mixed":
                design_clause = (
                    "State plainly that this is a MIXED-METHODS, design-and-build study: a quantitative "
                    "engineering strand (iterative prototyping, controlled performance testing, reproducible "
                    "metrics) combined with a qualitative strand (real or simulated end users engaged to "
                    "validate the system meets their actual needs). Do not describe it as purely engineering/"
                    "experimental — name both strands explicitly."
                )
            elif research_design == "qualitative":
                design_clause = (
                    "State plainly that this is a QUALITATIVE, design-and-build study: emphasis is on engaging "
                    "real or simulated end users to evaluate usability and fitness for purpose, with feedback "
                    "analysed thematically, rather than on quantitative performance thresholds."
                )
            else:
                design_clause = (
                    "Describe the design-and-build, experimental research approach (iterative prototyping, "
                    "controlled performance testing, reproducible quantifiable results) consistent with this "
                    "study's engineering stance."
                )
            return (
                f"Write the Research Design for THIS specific system-build study. {design_clause} Describe the "
                "phases the research proceeded through (requirements, design/implementation, integration, "
                "testing) and why this approach fits the stated research objectives."
            )
        if "existing system" in lowered:
            return ("Critically review the existing approach to this specific topic (manual process, legacy "
                    "software, or competing tools) — name concrete limitations, not a generic 'manual records "
                    "are slow' claim — and end by linking each limitation to a requirement covered next.")
        if "system requirement" in lowered or "functional requirement" in lowered or lowered.strip() == "requirements":
            return ("Split into Functional Requirements (what the system must DO, specific to this topic's "
                    "actual use cases) and Non-Functional Requirements (quality attributes that matter for THIS "
                    "system — performance, security, usability, etc). Tie each requirement to a research "
                    "objective.")
        if "system design" in lowered or "architecture" in lowered:
            return ("Describe the system's architecture (layers/components and how they interact), naming the "
                    "actual technologies used and why they fit this topic. Justify each design decision against "
                    "a requirement defined earlier rather than describing a generic architecture in the "
                    "abstract.")
        if "database" in lowered and ("design" in lowered or "model" in lowered):
            return ("Describe the data model: entities specific to this topic, their attributes, and "
                    "relationships, plus key design decisions (keys, indexing, normalisation, or spatial data "
                    "structures if relevant) and why they fit this system's expected access patterns.")
        if "tools and technolog" in lowered:
            return ("Name and justify the actual front-end, back-end, database, and any specialised tools/"
                    "libraries used to build this specific system, explaining why each was chosen over "
                    "alternatives. Avoid a generic 'modern web technologies' description with no concrete "
                    "tool names.")
        if "testing strategy" in lowered or "system testing" in lowered:
            if research_design == "mixed":
                design_clause = (
                    "Because this study uses a MIXED-METHODS design, combine a quantitative strand (unit, "
                    "integration, and performance/load testing against measurable thresholds) WITH a "
                    "qualitative strand: user-acceptance testing where real or simulated end users exercise the "
                    "system and give feedback (interviews, think-aloud sessions, or open-ended questions) "
                    "analysed thematically. Report both strands and how they were triangulated — do not drop "
                    "the qualitative arm."
                )
            elif research_design == "qualitative":
                design_clause = (
                    "Because this study uses a QUALITATIVE design, centre the testing strategy on "
                    "user-acceptance testing: real or simulated end users exercising the system, feedback "
                    "captured via interviews or open-ended questions and analysed thematically, rather than "
                    "statistical performance thresholds."
                )
            else:
                design_clause = (
                    "Describe unit testing, integration testing, and system/user-acceptance testing against "
                    "the functional and non-functional requirements, with concrete test cases (including edge "
                    "cases) and measurable pass/fail criteria specific to this topic."
                )
            return (f"Write the testing strategy for THIS specific system (name actual modules/features being "
                    f"tested, not a placeholder). {design_clause} End by stating which Chapter 4 sections "
                    "report the results of this testing.")

        # Generic fallback
        return (f"Write the '{section_title}' section using formal academic prose. "
                "Be specific, substantive, and analytical. Do NOT use generic filler content.")

