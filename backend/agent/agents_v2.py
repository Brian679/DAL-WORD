from __future__ import annotations

import logging
import re
import importlib
from dataclasses import dataclass, field
from typing import Any

from .research_layer import (
    RetrievalResult,
    build_research_brief,
    retrieval_pipeline,
    summarize_verification,
    verify_generated_citations,
)

logger = logging.getLogger(__name__)


def _llm_generate(prompt: str, *, model: str = "gemini") -> str:
    """Call configured LLM provider for specialist agents."""
    try:
        provider_name = "grok" if model == "grok" else "gemini"
        provider = importlib.import_module(f"agent.{provider_name}")
        return (provider.generate_text(prompt) or "").strip()
    except Exception as exc:
        logger.warning("_llm_generate failed (%s): %s", model, exc)
        return f"[LLM error: {exc}]"


@dataclass
class PromptContract:
    role: str
    objective: str
    constraints: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    style: str = "academic"
    required_outputs: list[str] = field(default_factory=list)

    def render(self) -> str:
        return (
            "{\n"
            f'  "role": "{self.role}",\n'
            f'  "objective": "{self.objective}",\n'
            f'  "constraints": {self.constraints},\n'
            f'  "sources": {self.sources},\n'
            f'  "style": "{self.style}",\n'
            f'  "required_outputs": {self.required_outputs}\n'
            "}"
        )


@dataclass
class AgentContext:
    topic: str
    instruction: str
    document_id: int | None = None
    retrieval: RetrievalResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SpecialistAgent:
    name: str = "SpecialistAgent"
    model: str = "gemini"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        return {"agent": self.name, "status": "noop"}


class PlannerAgent(SpecialistAgent):
    name = "PlannerAgent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        prompt = (
            f"You are a PlannerAgent for dissertation writing.\n"
            f"Topic: {ctx.topic}\n"
            f"Instruction: {ctx.instruction}\n\n"
            "Return a concise plan with labels in this format:\n"
            "PHASE: proposal|literature_review|methods|results|discussion|conclusion|full_dissertation\n"
            "RQ: 2-4 research questions\n"
            "STRUCTURE: chapter outline\n"
            "METHOD: expected methodological fit\n"
        )
        output = _llm_generate(prompt, model=self.model)
        phase = "full_dissertation"
        for line in output.splitlines():
            if line.strip().upper().startswith("PHASE:"):
                phase = line.split(":", 1)[-1].strip().lower()
                break
        return {
            "agent": self.name,
            "status": "ok",
            "phase": phase,
            "plan_output": output,
        }


class ResearchAgent(SpecialistAgent):
    name = "ResearchAgent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        result = ctx.retrieval or retrieval_pipeline(ctx.topic, document_id=ctx.document_id)
        ctx.retrieval = result

        if result.top_papers:
            paper_lines = "\n".join(
                f"- {p.title} ({p.year}) DOI:{p.doi or 'N/A'}"
                for p in result.top_papers[:10]
            )
            prompt = (
                "You are a ResearchAgent specialising in academic literature synthesis.\n"
                f"Topic: {ctx.topic}\n\n"
                f"Top papers:\n{paper_lines}\n\n"
                "Write a 200-word landscape paragraph that identifies dominant themes, methods, and gaps. "
                "Use (Author, Year) style references only for listed papers."
            )
            landscape = _llm_generate(prompt, model=self.model)
        else:
            landscape = "No papers retrieved for this topic."

        return {
            "agent": self.name,
            "status": "ok",
            "papers": len(result.papers),
            "top_papers": len(result.top_papers),
            "embedding_path": result.embedding_path,
            "landscape_summary": landscape,
        }


class EvidenceAgent(SpecialistAgent):
    name = "EvidenceAgent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        if not ctx.retrieval or not ctx.retrieval.top_papers:
            return {"agent": self.name, "status": "skipped", "reason": "no retrieval result"}

        papers_text = "\n".join(
            f"- {p.title} ({p.year}) DOI:{p.doi or 'N/A'} � {(p.abstract or '')[:140]}"
            for p in ctx.retrieval.top_papers[:10]
        )
        prompt = (
            "You are an EvidenceAgent for academic writing.\n"
            f"Topic: {ctx.topic}\n\n"
            f"Available verified papers:\n{papers_text}\n\n"
            "Extract 5-8 specific factual claims with citation and DOI where available. "
            "Only use facts grounded in the listed papers."
        )
        evidence = _llm_generate(prompt, model=self.model)

        return {
            "agent": self.name,
            "status": "ok",
            "evidence_claims": evidence,
        }


class CitationAgent(SpecialistAgent):
    name = "CitationAgent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        from .research_layer import repair_citations

        verification_report = verify_generated_citations(ctx.instruction)
        summary = summarize_verification(verification_report)

        pool = list(ctx.retrieval.top_papers) if ctx.retrieval else []
        repair = repair_citations(ctx.instruction, pool, min_confidence=60)

        notes = ""
        if repair.repaired_count > 0 or repair.removed_count > 0:
            notes = _llm_generate(
                "Summarize these citation repairs in 2 sentences:\n"
                + "\n".join(
                    f"- {item.get('action')}: {item.get('title') or item.get('original_title') or 'citation'}"
                    for item in repair.repair_log
                ),
                model=self.model,
            )

        return {
            "agent": self.name,
            "status": "ok",
            "verification": summary,
            "repair": {
                "total": repair.total_citations,
                "repaired": repair.repaired_count,
                "removed": repair.removed_count,
                "unchanged": repair.unchanged_count,
                "repaired_text": repair.repaired_text if (repair.repaired_count + repair.removed_count) > 0 else None,
                "notes": notes,
                "log": repair.repair_log,
            },
        }


class MethodologyAgent(SpecialistAgent):
    name = "MethodologyAgent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        prompt = (
            "You are a MethodologyAgent for academic research design.\n"
            f"Topic: {ctx.topic}\n"
            f"Instruction: {ctx.instruction}\n\n"
            "Return:\n"
            "DESIGN:\nAPPROACH:\nDATA_COLLECTION:\nANALYSIS:\nVALIDITY:\nLIMITATIONS:\n"
        )
        output = _llm_generate(prompt, model=self.model)

        mode = "mixed"
        for line in output.splitlines():
            if line.strip().upper().startswith("DESIGN:"):
                val = line.split(":", 1)[-1].strip().lower()
                if "qualitative" in val:
                    mode = "qualitative"
                elif "quantitative" in val:
                    mode = "quantitative"
                break

        return {
            "agent": self.name,
            "status": "ok",
            "recommended_design": mode,
            "methodology_output": output,
        }


class StatisticianAgent(SpecialistAgent):
    name = "StatisticianAgent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        prompt = (
            "You are a StatisticianAgent for academic research.\n"
            f"Topic: {ctx.topic}\n"
            f"Instruction: {ctx.instruction}\n\n"
            "Return:\n"
            "PRIMARY_TEST:\nSECONDARY_TESTS:\nEFFECT_SIZE:\nSAMPLE_SIZE:\nSOFTWARE:\nASSUMPTIONS:\n"
        )
        output = _llm_generate(prompt, model=self.model)

        primary_test = "descriptive_stats"
        for line in output.splitlines():
            if line.strip().upper().startswith("PRIMARY_TEST:"):
                primary_test = line.split(":", 1)[-1].strip()
                break

        return {
            "agent": self.name,
            "status": "ok",
            "primary_test": primary_test,
            "statistics_output": output,
        }


class VisualizationAgent(SpecialistAgent):
    name = "VisualizationAgent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        prompt = (
            "You are a VisualizationAgent for academic data presentation.\n"
            f"Topic: {ctx.topic}\n"
            f"Instruction: {ctx.instruction}\n\n"
            "Recommend 2-3 visualisations with:\n"
            "CHART_TYPE:\nTITLE:\nX_AXIS:\nY_AXIS:\nPURPOSE:\n"
        )
        output = _llm_generate(prompt, model=self.model)

        chart = "bar"
        for line in output.splitlines():
            if line.strip().upper().startswith("CHART_TYPE:"):
                chart = line.split(":", 1)[-1].strip().lower().split()[0]
                break

        return {
            "agent": self.name,
            "status": "ok",
            "primary_chart": chart,
            "visualization_output": output,
        }


class CriticAgent(SpecialistAgent):
    name = "CriticAgent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        excerpt = (ctx.instruction or "")[:3000]
        prompt = (
            "You are a CriticAgent performing peer review.\n"
            f"Topic: {ctx.topic}\n\n"
            f"Text to critique:\n{excerpt}\n\n"
            "Return:\n"
            "STRENGTHS:\nWEAKNESSES:\nMISSING_CITATIONS:\nLOGICAL_GAPS:\nIMPROVEMENTS:\nQUALITY_SCORE:(1-10)\n"
        )
        output = _llm_generate(prompt, model=self.model)

        score = None
        for line in output.splitlines():
            if line.strip().upper().startswith("QUALITY_SCORE:"):
                m = re.search(r"(\d+)", line)
                if m:
                    score = int(m.group(1))
                break

        return {
            "agent": self.name,
            "status": "ok",
            "quality_score": score,
            "critique_output": output,
        }


class FormattingAgent(SpecialistAgent):
    name = "FormattingAgent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        prompt = (
            "You are a FormattingAgent for academic output formatting.\n"
            f"Topic: {ctx.topic}\n"
            f"Instruction: {ctx.instruction}\n\n"
            "Return:\n"
            "CITATION_STYLE:\nHEADING_LEVELS:\nFONT:\nMARGINS:\nLINE_SPACING:\nABSTRACT_REQUIRED:\nREFERENCE_LIST_FORMAT:\nSPECIAL_REQUIREMENTS:\n"
        )
        output = _llm_generate(prompt, model=self.model)

        citation_style = "APA 7th"
        for line in output.splitlines():
            if line.strip().upper().startswith("CITATION_STYLE:"):
                citation_style = line.split(":", 1)[-1].strip()
                break

        return {
            "agent": self.name,
            "status": "ok",
            "citation_style": citation_style,
            "formatting_output": output,
        }


class SupervisorAgent(SpecialistAgent):
    name = "SupervisorAgent"
    model = "gemini"

    _PIPELINE: list[type[SpecialistAgent]] = [
        PlannerAgent,
        ResearchAgent,
        EvidenceAgent,
        CitationAgent,
        MethodologyAgent,
        StatisticianAgent,
        VisualizationAgent,
        CriticAgent,
        FormattingAgent,
    ]

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        agent_outputs: dict[str, Any] = {}

        for AgentClass in self._PIPELINE:
            agent = AgentClass()
            try:
                result = agent.run(ctx)
                trace.append(result)
                agent_outputs[agent.name] = result
            except Exception as exc:
                logger.error("%s failed: %s", agent.name, exc, exc_info=True)
                trace.append({"agent": agent.name, "status": "error", "error": str(exc)})

        synthesis = self._synthesise(ctx, agent_outputs)
        brief = build_research_brief(ctx.retrieval) if ctx.retrieval else ""
        citation_result = agent_outputs.get("CitationAgent", {})

        return {
            "agent": self.name,
            "status": "ok",
            "trace": trace,
            "synthesis": synthesis,
            "research_brief": brief,
            "retrieval": {
                "top_papers": [
                    {"title": p.title, "doi": p.doi, "year": p.year, "source": p.source}
                    for p in (ctx.retrieval.top_papers[:10] if ctx.retrieval else [])
                ],
                "embedding_path": (ctx.retrieval.embedding_path if ctx.retrieval else None),
                "total_candidates": len(ctx.retrieval.papers) if ctx.retrieval else 0,
            },
            "citation_verification": citation_result.get("verification", {}),
            "citation_repair": citation_result.get("repair", {}),
            "contract": PromptContract(
                role="dissertation_supervisor",
                objective="Coordinate specialist agents to produce grounded academic content",
                constraints=[
                    "No fabricated citations",
                    "Only DOI-verified sources",
                    "All claims evidence-backed",
                ],
                sources=[p.title for p in (ctx.retrieval.top_papers[:5] if ctx.retrieval else [])],
                style="APA academic",
                required_outputs=[
                    "grounded synthesis",
                    "verified citations",
                    "methodology plan",
                    "critique",
                ],
            ).render(),
        }

    def _synthesise(self, ctx: AgentContext, outputs: dict[str, Any]) -> str:
        planner_out = outputs.get("PlannerAgent", {}).get("plan_output", "")
        landscape = outputs.get("ResearchAgent", {}).get("landscape_summary", "")
        evidence = outputs.get("EvidenceAgent", {}).get("evidence_claims", "")
        methodology = outputs.get("MethodologyAgent", {}).get("methodology_output", "")
        statistics = outputs.get("StatisticianAgent", {}).get("statistics_output", "")
        critique = outputs.get("CriticAgent", {}).get("critique_output", "")
        repair_notes = outputs.get("CitationAgent", {}).get("repair", {}).get("notes", "")

        parts: list[str] = []
        if planner_out:
            parts.append(f"=== PLAN ===\\n{planner_out[:600]}")
        if landscape:
            parts.append(f"=== RESEARCH LANDSCAPE ===\\n{landscape[:600]}")
        if evidence:
            parts.append(f"=== EVIDENCE ===\\n{evidence[:600]}")
        if methodology:
            parts.append(f"=== METHODOLOGY ===\\n{methodology[:400]}")
        if statistics:
            parts.append(f"=== STATISTICS ===\\n{statistics[:300]}")
        if critique:
            parts.append(f"=== CRITIQUE ===\\n{critique[:400]}")
        if repair_notes:
            parts.append(f"=== CITATION REPAIRS ===\\n{repair_notes[:300]}")

        if not parts:
            return "Supervisor: no agent output to synthesise."

        combined_parts = "\n\n".join(parts)

        prompt = (
            f"You are a SupervisorAgent overseeing dissertation writing on: {ctx.topic}\\n\\n"
            "Below are findings from specialist agents:\\n\\n"
            f"{combined_parts}\\n\\n"
            "Write a 250-word Master Context that integrates findings, calls out citation priorities, "
            "flags major quality risks, and gives 3 concrete writing instructions."
        )
        return _llm_generate(prompt, model=self.model)


def run_multi_agent_supervision(topic: str, instruction: str, document_id: int | None = None) -> dict[str, Any]:
    ctx = AgentContext(topic=topic, instruction=instruction, document_id=document_id)
    return SupervisorAgent().run(ctx)
