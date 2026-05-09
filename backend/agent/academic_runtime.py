from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class ClaimNode:
    claim: str
    citation: str | None = None
    source_passage: str | None = None
    source_location: str | None = None
    confidence: float = 0.0


@dataclass
class QualityMetrics:
    citation_density: float
    coherence_score: float
    redundancy_score: float
    argument_consistency: float
    methodology_alignment: float


class HierarchicalMemory:
    """Layered memory for long documents: section/chapter/global."""

    def __init__(self, document_id: int):
        self.base = Path(settings.MEDIA_ROOT) / "research" / "memory"
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / f"doc-{document_id}.json"
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "section_memory": {},
            "chapter_memory": {},
            "global_memory": {
                "thesis_statement": "",
                "hypotheses": [],
                "key_claims": [],
                "definitions": {},
            },
            "entity_graph": {},
        }

    def save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def set_section_memory(self, section_id: str, payload: dict[str, Any]) -> None:
        self.state["section_memory"][section_id] = payload

    def set_chapter_memory(self, chapter_id: str, payload: dict[str, Any]) -> None:
        self.state["chapter_memory"][chapter_id] = payload

    def update_global(self, **kwargs: Any) -> None:
        self.state.setdefault("global_memory", {}).update(kwargs)


class CoherenceChecker:
    """Heuristic critic loop for contradictions, redundancy, unsupported claims."""

    def check(self, text: str) -> dict[str, Any]:
        lines = [l.strip() for l in re.split(r"[\n\.]+", text or "") if l.strip()]
        low = (text or "").lower()

        unsupported = []
        for sent in lines:
            if any(k in sent.lower() for k in ["proves", "always", "never", "undeniable"]):
                if "doi" not in sent.lower() and "(" not in sent:
                    unsupported.append(sent)

        tokens = re.findall(r"\b[a-z]{4,}\b", low)
        redundancy = 0.0
        if tokens:
            uniq = len(set(tokens))
            redundancy = max(0.0, 1.0 - (uniq / len(tokens)))

        transitions = ["however", "therefore", "moreover", "in contrast", "furthermore"]
        transition_hits = sum(low.count(t) for t in transitions)
        coherence = min(1.0, 0.2 + transition_hits / 10.0)

        return {
            "contradictions": [],
            "unsupported_claims": unsupported[:8],
            "redundancy_ratio": round(redundancy, 4),
            "transition_score": round(coherence, 4),
            "missing_citations": sum(1 for s in lines if any(k in s.lower() for k in ["study", "research", "evidence"]) and "(" not in s),
        }


class ClaimGraphBuilder:
    def build(self, text: str, citations: list[str] | None = None) -> list[ClaimNode]:
        citations = citations or []
        claims: list[ClaimNode] = []
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
        for sent in sents:
            if len(sent.split()) < 8:
                continue
            cit = None
            for c in citations:
                if c.lower().split("(")[0].strip() in sent.lower():
                    cit = c
                    break
            claims.append(
                ClaimNode(
                    claim=sent,
                    citation=cit,
                    source_passage=None,
                    source_location=None,
                    confidence=0.9 if cit else 0.55,
                )
            )
        return claims


class WorkflowEngine:
    phases = [
        "proposal",
        "literature_mapping",
        "approval",
        "chapter_generation",
        "review_defense_prep",
    ]

    def choose_phase(self, prompt: str) -> str:
        low = (prompt or "").lower()
        if any(k in low for k in ["proposal", "objective", "methodology"]):
            return "proposal"
        if any(k in low for k in ["literature", "source", "gap"]):
            return "literature_mapping"
        if any(k in low for k in ["approve", "approval"]):
            return "approval"
        if any(k in low for k in ["defense", "viva", "review"]):
            return "review_defense_prep"
        return "chapter_generation"


class AcademicIntegrityGuard:
    def check(self, text: str, synthetic_mode: bool = False) -> dict[str, Any]:
        low = (text or "").lower()
        risks = []
        if "fake data" in low or "fabricated" in low:
            risks.append("fabricated_data")
        if "ghostwrite" in low or "write my whole thesis" in low:
            risks.append("ghostwriting_risk")
        if "medical" in low and "without citation" in low:
            risks.append("medical_misinformation_risk")
        return {
            "synthetic_mode": synthetic_mode,
            "risks": risks,
            "allowed": len(risks) == 0,
        }


class EvaluationEngine:
    def score(self, text: str, methodology: str = "mixed") -> QualityMetrics:
        sentences = [s for s in re.split(r"[.!?]", text or "") if s.strip()]
        citations = len(re.findall(r"\([^)]+,\s*(19|20)\d{2}\)", text or "")) + len(re.findall(r"doi", (text or "").lower()))
        citation_density = citations / max(len(sentences), 1)

        checker = CoherenceChecker().check(text)
        coherence_score = checker["transition_score"]
        redundancy_score = checker["redundancy_ratio"]

        methodology_alignment = 0.65
        low = (text or "").lower()
        if methodology == "quantitative" and any(k in low for k in ["regression", "p-value", "hypothesis"]):
            methodology_alignment = 0.9
        elif methodology == "qualitative" and any(k in low for k in ["thematic", "interview", "coding"]):
            methodology_alignment = 0.9

        argument_consistency = max(0.4, 1.0 - checker["missing_citations"] / max(len(sentences), 1))

        return QualityMetrics(
            citation_density=round(citation_density, 4),
            coherence_score=round(coherence_score, 4),
            redundancy_score=round(redundancy_score, 4),
            argument_consistency=round(argument_consistency, 4),
            methodology_alignment=round(methodology_alignment, 4),
        )
