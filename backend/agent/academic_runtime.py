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

    # Patterns for strong assertions that should be backed by citations
    _STRONG_CLAIM_RE = re.compile(
        r"\b(proves?|it is (clear|evident|obvious)|clearly (shows?|demonstrates?)"
        r"|always|never|undeniable|without (a )?doubt|scientifically proven"
        r"|has been proven|universally (accepted|agreed)|all studies|no studies"
        r"|every (researcher|study|expert))\b",
        re.IGNORECASE,
    )
    # Patterns that constitute an in-text citation
    _CITATION_RE = re.compile(
        r"\([A-Z][a-z]+(?:[,&]\s*[A-Z][a-z]+)*(?:\s+et\s+al\.?)?,\s*(?:19|20)\d{2}[a-z]?\)"  # APA
        r"|\[\d+(?:,\s*\d+)*\]"  # Numeric [1] or [1,2]
        r"|doi:\s*10\.\d{4}"  # DOI
        r"|\b10\.\d{4}/\S+",  # bare DOI
        re.IGNORECASE,
    )
    _TRANSITIONS = [
        "however", "therefore", "moreover", "in contrast", "furthermore",
        "consequently", "nevertheless", "in addition", "as a result",
        "on the other hand", "although", "despite", "whereas", "thus",
        "nonetheless", "accordingly", "subsequently",
    ]
    _RESEARCH_TERMS = re.compile(
        r"\b(stud(y|ies)|research|evidence|find(ing)?s|data|result|"
        r"analysis|survey|experiment|investigat|literature)\b",
        re.IGNORECASE,
    )

    def _has_citation(self, sent: str) -> bool:
        return bool(self._CITATION_RE.search(sent))

    def check(self, text: str) -> dict[str, Any]:
        if not (text or "").strip():
            return {
                "contradictions": [],
                "unsupported_claims": [],
                "redundancy_ratio": 0.0,
                "transition_score": 0.0,
                "missing_citations": 0,
            }

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]
        low = text.lower()

        # Unsupported claims: strong assertions with no citation in the same sentence
        unsupported = [
            s for s in sentences
            if self._STRONG_CLAIM_RE.search(s) and not self._has_citation(s)
        ]

        # Redundancy: type-token ratio on content words (4+ chars)
        tokens = re.findall(r"\b[a-z]{4,}\b", low)
        redundancy = 0.0
        if tokens:
            redundancy = max(0.0, 1.0 - len(set(tokens)) / len(tokens))

        # Coherence: transition-word density relative to word count
        word_count = max(len(low.split()), 1)
        transition_hits = sum(low.count(t) for t in self._TRANSITIONS)
        coherence = min(1.0, 0.15 + (transition_hits / word_count) * 40)

        # Missing citations: research-term sentences with no citation
        missing_cit = sum(
            1 for s in sentences
            if self._RESEARCH_TERMS.search(s) and not self._has_citation(s)
        )

        return {
            "contradictions": [],
            "unsupported_claims": unsupported[:8],
            "redundancy_ratio": round(redundancy, 4),
            "transition_score": round(coherence, 4),
            "missing_citations": missing_cit,
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
        _cit_re = re.compile(
            r"\([A-Z][a-z]+(?:[,&]\s*[A-Z][a-z]+)*(?:\s+et\s+al\.?)?,\s*(?:19|20)\d{2}[a-z]?\)"
            r"|\[\d+(?:,\s*\d+)*\]"
            r"|doi:\s*10\.\d{4}"
            r"|\b10\.\d{4}/\S+",
            re.IGNORECASE,
        )
        citations = len(_cit_re.findall(text or ""))
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
