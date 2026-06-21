"""
Multi-agent generation pipeline.

Pipeline stages (executed in order for every text-kind TaskSpec):

  User Request
      ↓
  PlannerAgent          ← planner.py — intent parsing, task decomposition, spec generation
      ↓
  TaskQueue             ← planner.py — ordered list of TaskSpec items
      ↓
  ContentGenerator      ← this file — calls the LLM with task-specific prompt + guidelines
      ↓
  RuntimeSandbox        ← this file — safe executor; catches ALL exceptions
      ↓
  ErrorAnalyzer         ← this file — categorises what went wrong
      ↓
  RepairAgent           ← this file — retries with a different strategy
      ↓
  Full dissertation     ← assembled by autonomous._write_dissertation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .planner import TaskSpec, TaskQueue

logger = logging.getLogger(__name__)

# ── Filler phrases that indicate the LLM echoed the prompt instead of writing ──
_FILLER_PHRASES: tuple[str, ...] = (
    "this section addresses",
    "this subsection addresses",
    "the analysis will be developed",
    "will be discussed in this section",
    "writing instructions for this section",
    "current document (read this",
    "section-specific instructions",
    "format instruction",
    "critical rules",
)

# ── Phrases indicating the LLM reviewed/critiqued the document instead of writing
# the requested chapter content (e.g. asked to write a "Chapter Summary" and
# instead produced a strengths/weaknesses assessment of the whole document) ──
_META_COMMENTARY_PHRASES: tuple[str, ...] = (
    "this document currently",
    "the document currently",
    "key strength is that",
    "main weakness is that",
    "main gap is that",
    "area of improvement",
    "areas of improvement",
    "areas for improvement",
    "i have reviewed",
    "upon reviewing",
    "as a reviewer",
    "as an academic reviewer",
    "overall, the document",
    "the writer should",
    "could be strengthened by",
    "i recommend revising",
    "this section is currently",
    "in summary, this document",
)


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

class ErrorCategory(Enum):
    EMPTY_RESPONSE    = "empty_response"       # LLM returned nothing
    CONTENT_TOO_SHORT = "content_too_short"    # body < 80 chars
    FILLER_DETECTED   = "filler_detected"      # LLM echoed the prompt back
    FORMAT_ERROR      = "format_error"         # Wrong structure (e.g. hypothesis missing H0/H1)
    LLM_ERROR         = "llm_error"            # Exception raised during generation
    RATE_LIMIT        = "rate_limit"           # API quota / rate-limit error
    UNKNOWN           = "unknown"              # Catch-all


class RepairStrategy(Enum):
    RETRY_SIMPLER   = "retry_simpler"    # Re-send with a stripped-down direct prompt
    RETRY_SHORTER   = "retry_shorter"    # Re-send requesting fewer words
    RETRY_TEMPLATE  = "retry_template"   # Use a structured template prompt
    USE_FALLBACK    = "use_fallback"     # Caller-supplied static fallback text
    SKIP            = "skip"             # Give up (non-critical node)


# ---------------------------------------------------------------------------
# Pipeline data objects
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result produced by the RuntimeSandbox for one TaskSpec."""
    task_id:        str
    content:        str
    blocks:         list[dict[str, Any]] = field(default_factory=list)
    success:        bool = True
    error:          str | None = None
    attempts:       int = 1
    error_category: ErrorCategory | None = None
    strategy_used:  RepairStrategy | None = None


@dataclass
class ErrorReport:
    """Diagnosis produced by the ErrorAnalyzer."""
    category: ErrorCategory
    strategy: RepairStrategy
    detail:   str


# ---------------------------------------------------------------------------
# ContentGenerator
# ---------------------------------------------------------------------------

class ContentGenerator:
    """
    Generates text content for a single TaskSpec by calling the LLM.

    The caller supplies the actual LLM call via `generate_fn` so this class
    stays independent of the model provider.

    generate_fn signature:
        (title: str, topic: str, context: str, word_count: int) -> str
    """

    def generate(
        self,
        task: TaskSpec,
        document_brief: str,
        rolling_context: str,
        generate_fn: Callable[[str, str, str, int], str],
        user_instruction: str = "",
    ) -> str:
        """Build the full prompt and call the LLM, returning raw text."""
        lowered = task.title.lower()
        is_pointform = any(
            k in lowered for k in [
                "research objective", "objectives", "research question", "hypothes",
                "recommendation", "further research", "areas for future", "definition of key",
            ]
        )
        wc = min(task.word_count, 120) if is_pointform else task.word_count

        instr_lower = user_instruction.lower()
        explicit_list = any(k in instr_lower for k in [
            "point form", "bullet point", "bullet list", "numbered list",
            "number form", "in points", "as points", "list form",
        ])
        explicit_prose = any(k in instr_lower for k in ["paragraph", "prose", "flowing", "narrative"])

        if explicit_prose and not explicit_list:
            format_override = (
                "FORMAT INSTRUCTION (from user): Write as flowing academic paragraphs. "
                "Do NOT use bullet points or numbered lists.\n\n"
            )
        elif explicit_list or (is_pointform and not explicit_prose):
            format_override = (
                "FORMAT INSTRUCTION: Present content as a numbered list. "
                "Each item on its own line starting with '1.', '2.', etc. "
                "May include 1–2 introductory sentences. Do NOT write as continuous prose.\n\n"
            )
        else:
            format_override = ""

        prompt_context = (
            f"{document_brief}\n\n"
            f"You are writing the section: '{task.title}'\n"
            f"Parent chapter: {task.chapter_title}\n\n"
            "--- DOCUMENT CONTENT ALREADY WRITTEN ---\n"
            f"{rolling_context[-3000:]}\n"
            "--- END OF EXISTING DOCUMENT ---\n\n"
            + format_override
            + (
                f"SECTION-SPECIFIC INSTRUCTIONS:\n{task.guidelines}\n\n"
                if task.guidelines else
                "Write in formal academic prose. Be specific, substantive, and analytical.\n\n"
            )
            + "CRITICAL RULES:\n"
            "1. Every sentence must be specifically about the research topic in the STUDY BRIEF above.\n"
            "2. Reference the actual topic, objectives, methodology, and context of this specific study.\n"
            "3. Do NOT write generic academic content that could apply to any study.\n"
            "4. Do NOT include the section heading in your response.\n"
            "5. Do NOT use filler phrases such as 'this section will discuss', 'in today's world', "
            "'it is important to note', 'delve into', 'navigate the complexities of', "
            "'in the realm of', 'a testament to'.\n"
            "6. Write actual academic content — grounded in THIS specific research.\n"
            "7. Vary paragraph openers and sentence rhythm — never start consecutive paragraphs with "
            "the same word, and mix short analytical sentences with longer elaborations.\n"
            "8. Ground every claim in a concrete example, statistic, or citation — never leave an "
            "assertion as an unsupported generalisation.\n"
        )

        return generate_fn(task.title, task.topic, prompt_context, wc)

    def generate_simple(
        self,
        task: TaskSpec,
        document_brief: str,
        generate_fn: Callable[[str, str, str, int], str],
    ) -> str:
        """Stripped-down direct prompt used by RepairAgent (RETRY_SIMPLER)."""
        prompt_context = (
            f"{document_brief}\n\n"
            f"Write the '{task.title}' section for a dissertation on: '{task.topic}'.\n"
            f"Chapter: {task.chapter_title} | Design: {task.research_design}\n\n"
            + (f"Instructions: {task.guidelines}\n\n" if task.guidelines else "")
            + "Write 2–4 focused academic paragraphs. Do NOT repeat the section heading. "
            "Be specific and grounded in the research topic."
        )
        wc = max(80, task.word_count // 2)
        return generate_fn(task.title, task.topic, prompt_context, wc)

    def generate_template(
        self,
        task: TaskSpec,
        document_brief: str,
        generate_fn: Callable[[str, str, str, int], str],
    ) -> str:
        """Highly structured template prompt used by RepairAgent (RETRY_TEMPLATE)."""
        prompt_context = (
            f"{document_brief}\n\n"
            f"Complete the following structured template for the '{task.title}' section "
            f"of a dissertation on '{task.topic}':\n\n"
            f"Topic: {task.topic}\n"
            f"Chapter: {task.chapter_title}\n"
            f"Research design: {task.research_design}\n\n"
            "TEMPLATE TO COMPLETE (replace <...> with real content):\n"
            f"<Opening sentence about {task.topic}>\n"
            "<Key academic point 1 with supporting rationale>\n"
            "<Key academic point 2 linking to objectives>\n"
            "<Closing sentence summarising the section's contribution>\n\n"
            "Write in formal academic prose. Be concise and topic-specific."
        )
        wc = max(60, task.word_count // 3)
        return generate_fn(task.title, task.topic, prompt_context, wc)


# ---------------------------------------------------------------------------
# RuntimeSandbox
# ---------------------------------------------------------------------------

class RuntimeSandbox:
    """
    Safely executes a ContentGenerator call.

    Catches all exceptions and converts them into a failed TaskResult,
    so the pipeline never crashes on a single node failure.
    """

    def execute(
        self,
        task: TaskSpec,
        generator: ContentGenerator,
        document_brief: str,
        rolling_context: str,
        generate_fn: Callable[[str, str, str, int], str],
        user_instruction: str = "",
    ) -> TaskResult:
        try:
            logger.info("▶ [Sandbox] Generating: %s | %s", task.chapter_title, task.title)
            body = generator.generate(
                task=task,
                document_brief=document_brief,
                rolling_context=rolling_context,
                generate_fn=generate_fn,
                user_instruction=user_instruction,
            )
            return TaskResult(task_id=task.id, content=body or "", success=True)
        except Exception as exc:
            err_str = str(exc)
            logger.warning("▶ [Sandbox] ERROR for '%s': %s", task.title, err_str)
            return TaskResult(
                task_id=task.id,
                content="",
                success=False,
                error=err_str,
                attempts=1,
            )


# ---------------------------------------------------------------------------
# ErrorAnalyzer
# ---------------------------------------------------------------------------

class ErrorAnalyzer:
    """
    Inspects a TaskResult (or raw content string) and returns a diagnosis
    with a recommended repair strategy.
    """

    # Minimum meaningful body length
    MIN_LENGTH = 80

    def analyze(self, result: TaskResult, task: TaskSpec) -> ErrorReport:
        """Categorise the failure and recommend the best repair strategy."""
        # ── Exception during generation ───────────────────────────────────
        if not result.success and result.error:
            err_lower = result.error.lower()
            if any(k in err_lower for k in ["rate limit", "quota", "429", "resource_exhausted"]):
                return ErrorReport(
                    category=ErrorCategory.RATE_LIMIT,
                    strategy=RepairStrategy.RETRY_SHORTER,
                    detail=f"API quota/rate limit: {result.error[:120]}",
                )
            return ErrorReport(
                category=ErrorCategory.LLM_ERROR,
                strategy=RepairStrategy.RETRY_TEMPLATE,
                detail=f"LLM exception: {result.error[:120]}",
            )

        body = (result.content or "").strip()

        # ── Empty response ────────────────────────────────────────────────
        if not body:
            return ErrorReport(
                category=ErrorCategory.EMPTY_RESPONSE,
                strategy=RepairStrategy.RETRY_SIMPLER,
                detail="LLM returned an empty response.",
            )

        # ── Content too short ─────────────────────────────────────────────
        if len(body) < self.MIN_LENGTH:
            return ErrorReport(
                category=ErrorCategory.CONTENT_TOO_SHORT,
                strategy=RepairStrategy.RETRY_SIMPLER,
                detail=f"Response too short ({len(body)} chars, minimum {self.MIN_LENGTH}).",
            )

        # ── Filler / prompt echo ──────────────────────────────────────────
        body_lower = body.lower()
        if any(phrase in body_lower for phrase in _FILLER_PHRASES):
            return ErrorReport(
                category=ErrorCategory.FILLER_DETECTED,
                strategy=RepairStrategy.RETRY_SIMPLER,
                detail="LLM echoed the prompt or returned a filler placeholder.",
            )

        # ── Meta-commentary (reviewed the document instead of writing it) ──
        if any(phrase in body_lower for phrase in _META_COMMENTARY_PHRASES):
            return ErrorReport(
                category=ErrorCategory.FILLER_DETECTED,
                strategy=RepairStrategy.RETRY_SIMPLER,
                detail="LLM produced a review/critique of the document instead of section content.",
            )

        # ── Structural format error ───────────────────────────────────────
        if "hypoth" in task.title.lower():
            missing_h0 = "h0" not in body_lower and "null hypothesis" not in body_lower
            missing_h1 = "h1" not in body_lower and "alternative hypothesis" not in body_lower
            if missing_h0 or missing_h1 or len(body.strip().splitlines()) < 2:
                return ErrorReport(
                    category=ErrorCategory.FORMAT_ERROR,
                    strategy=RepairStrategy.RETRY_SIMPLER,
                    detail="Hypothesis section missing H0/H1 pairs or is single-line.",
                )

        # ── All checks passed ─────────────────────────────────────────────
        return ErrorReport(
            category=ErrorCategory.UNKNOWN,
            strategy=RepairStrategy.USE_FALLBACK,
            detail="Content looks valid.",
        )

    def is_acceptable(self, content: str, task: TaskSpec) -> bool:
        """Quick check: is this content good enough to use?"""
        body = (content or "").strip()
        if len(body) < self.MIN_LENGTH:
            return False
        body_lower = body.lower()
        if any(phrase in body_lower for phrase in _FILLER_PHRASES):
            return False
        if any(phrase in body_lower for phrase in _META_COMMENTARY_PHRASES):
            return False
        if "hypoth" in task.title.lower():
            missing = ("h0" not in body_lower and "null hypothesis" not in body_lower) or \
                      ("h1" not in body_lower and "alternative hypothesis" not in body_lower)
            if missing:
                return False
        return True


# ---------------------------------------------------------------------------
# RepairAgent
# ---------------------------------------------------------------------------

class RepairAgent:
    """
    Retries a failed task using the strategy recommended by ErrorAnalyzer.

    fallback_fn signature:
        (topic: str, chapter_title: str, section_title: str, word_count: int) -> str
    """

    def repair(
        self,
        task: TaskSpec,
        error_report: ErrorReport,
        document_brief: str,
        rolling_context: str,
        generate_fn: Callable[[str, str, str, int], str],
        fallback_fn: Callable[[str, str, str, int], str],
    ) -> TaskResult:
        strategy = error_report.strategy
        generator = ContentGenerator()

        logger.info(
            "▶ [Repair] Strategy=%s for '%s' (reason: %s)",
            strategy.value, task.title, error_report.detail,
        )

        if strategy == RepairStrategy.RETRY_SIMPLER:
            return self._try(
                task=task,
                fn=lambda: generator.generate_simple(task, document_brief, generate_fn),
                strategy=strategy,
            )

        if strategy == RepairStrategy.RETRY_SHORTER:
            short_task = TaskSpec(
                id=task.id, title=task.title, kind=task.kind,
                word_count=max(60, task.word_count // 3),
                guidelines=task.guidelines, context_hint=task.context_hint,
                chapter_title=task.chapter_title, chapter_num=task.chapter_num,
                topic=task.topic, research_design=task.research_design, meta=task.meta,
            )
            return self._try(
                task=short_task,
                fn=lambda: generator.generate_simple(short_task, document_brief, generate_fn),
                strategy=strategy,
            )

        if strategy == RepairStrategy.RETRY_TEMPLATE:
            return self._try(
                task=task,
                fn=lambda: generator.generate_template(task, document_brief, generate_fn),
                strategy=strategy,
            )

        if strategy == RepairStrategy.SKIP:
            logger.warning("▶ [Repair] Skipping '%s'", task.title)
            return TaskResult(
                task_id=task.id, content="", success=False,
                error="Skipped after repeated failures.",
                strategy_used=RepairStrategy.SKIP,
            )

        # USE_FALLBACK (default)
        content = fallback_fn(task.topic, task.chapter_title, task.title, task.word_count)
        return TaskResult(
            task_id=task.id, content=content, success=True,
            attempts=2, strategy_used=RepairStrategy.USE_FALLBACK,
        )

    @staticmethod
    def _try(task: TaskSpec, fn: Callable[[], str], strategy: RepairStrategy) -> TaskResult:
        try:
            content = fn()
            return TaskResult(
                task_id=task.id, content=content or "", success=bool(content),
                attempts=2, strategy_used=strategy,
            )
        except Exception as exc:
            logger.warning("▶ [Repair] Failed with strategy=%s: %s", strategy.value, exc)
            return TaskResult(
                task_id=task.id, content="", success=False,
                error=str(exc), attempts=2, strategy_used=strategy,
            )


# ---------------------------------------------------------------------------
# Pipeline  (orchestrator)
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Orchestrates the full generation pipeline for text-kind nodes.

    For each TaskSpec in the queue:
      1. ContentGenerator generates text
      2. RuntimeSandbox catches exceptions
      3. ErrorAnalyzer diagnoses quality
      4. If bad → RepairAgent retries with alternate strategy
      5. If repair still fails → USE_FALLBACK

    Non-text nodes (table, chart, image) are skipped by the pipeline —
    they are handled by the caller's existing specialised logic.
    """

    def __init__(self) -> None:
        self._generator = ContentGenerator()
        self._sandbox   = RuntimeSandbox()
        self._analyzer  = ErrorAnalyzer()
        self._repair    = RepairAgent()

    def run_node(
        self,
        task: TaskSpec,
        document_brief: str,
        rolling_context: str,
        generate_fn: Callable[[str, str, str, int], str],
        fallback_fn: Callable[[str, str, str, int], str],
        user_instruction: str = "",
    ) -> TaskResult:
        """
        Run the full pipeline for a single text TaskSpec.
        Returns a TaskResult (always — never raises).
        """
        # ── Stage 1: ContentGenerator + RuntimeSandbox ───────────────────
        result = self._sandbox.execute(
            task=task,
            generator=self._generator,
            document_brief=document_brief,
            rolling_context=rolling_context,
            generate_fn=generate_fn,
            user_instruction=user_instruction,
        )

        # ── Stage 2: ErrorAnalyzer ────────────────────────────────────────
        if result.success and self._analyzer.is_acceptable(result.content, task):
            logger.debug("▶ [Pipeline] '%s' — accepted on first attempt.", task.title)
            return result

        error_report = self._analyzer.analyze(result, task)
        logger.info(
            "▶ [Pipeline] '%s' — %s → %s",
            task.title, error_report.category.value, error_report.strategy.value,
        )

        # If the analyzer says content is actually fine (UNKNOWN → USE_FALLBACK but content exists)
        if (
            error_report.category == ErrorCategory.UNKNOWN
            and result.success
            and result.content.strip()
        ):
            return result

        # ── Stage 3: RepairAgent ──────────────────────────────────────────
        repaired = self._repair.repair(
            task=task,
            error_report=error_report,
            document_brief=document_brief,
            rolling_context=rolling_context,
            generate_fn=generate_fn,
            fallback_fn=fallback_fn,
        )

        # ── Stage 4: Final quality gate ───────────────────────────────────
        if repaired.content and self._analyzer.is_acceptable(repaired.content, task):
            return repaired

        # Last resort: static fallback
        logger.warning(
            "▶ [Pipeline] '%s' — repair did not produce acceptable content. Using static fallback.",
            task.title,
        )
        fallback_content = fallback_fn(task.topic, task.chapter_title, task.title, task.word_count)
        return TaskResult(
            task_id=task.id,
            content=fallback_content,
            success=True,
            attempts=3,
            strategy_used=RepairStrategy.USE_FALLBACK,
        )

    def run(
        self,
        queue: TaskQueue,
        document_brief: str,
        get_rolling_context: Callable[[], str],
        generate_fn: Callable[[str, str, str, int], str],
        fallback_fn: Callable[[str, str, str, int], str],
        user_instruction: str = "",
        on_node_done: Callable[[TaskSpec, TaskResult], None] | None = None,
    ) -> list[TaskResult]:
        """
        Run the pipeline for all text-kind tasks in the queue.
        Non-text tasks are skipped (handled by specialised logic in the caller).
        """
        results: list[TaskResult] = []
        for task in queue.all():
            if task.kind != "text":
                continue  # charts / tables handled externally

            queue.mark_running(task.id)
            rolling_context = get_rolling_context()
            result = self.run_node(
                task=task,
                document_brief=document_brief,
                rolling_context=rolling_context,
                generate_fn=generate_fn,
                fallback_fn=fallback_fn,
                user_instruction=user_instruction,
            )
            results.append(result)

            if result.success:
                queue.mark_done(task.id)
            else:
                queue.mark_failed(task.id)

            if on_node_done:
                on_node_done(task, result)

        summary = queue.summary()
        logger.info(
            "▶ [Pipeline] Completed. done=%d failed=%d skipped=%d",
            summary.get("done", 0), summary.get("failed", 0), summary.get("skipped", 0),
        )
        return results
