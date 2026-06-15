"""
AI content detection engine.

Approximates Turnitin's two-signal methodology:
  1. Perplexity  – how predictable is each sentence to a language model?
     (approximated via known-phrase + structural signals)
  2. Burstiness  – humans vary their sentence complexity; LLMs stay uniform.
     B = σ(per-sentence score) / μ(per-sentence score)

Overall document AI probability combines the mean sentence score (55 %)
with a burstiness inversion signal (45 %).
"""
from __future__ import annotations

import re
import statistics
from typing import Any

# ── Known AI-generated phrase fingerprints ─────────────────────────────────
# Sourced from published research on GPT/Claude/LLaMA output characteristics.
_AI_PHRASE_PATTERNS: list[str] = [
    r"\bit is (?:important|crucial|essential|vital|worth noting|worth mentioning)\b",
    r"\bin today[''']?s (?:world|society|era|age|landscape)\b",
    r"\bit is evident that\b",
    r"\bthis (?:essay|paper|study|section|chapter|report) (?:will|aims to|seeks to|endeavours? to)\b",
    r"\bplays? (?:a )?(?:crucial|pivotal|significant|vital|key|important) role\b",
    r"\bholistic approach\b",
    r"\bparadigm shift\b",
    r"\bdelve(?:s|d)? into\b",
    r"\bcomprehensive (?:overview|understanding|analysis|examination)\b",
    r"\bunderscores? the (?:importance|need|significance|necessity)\b",
    r"\bit goes without saying\b",
    r"\bthe (?:realm|landscape|domain|field) of\b",
    r"\bfoster(?:s|ed|ing)? (?:a )?(?:deeper|better|greater|more nuanced) understanding\b",
    r"\bthis (?:highlights?|underscores?|demonstrates?|showcases?|illuminates?) the\b",
    r"\bin (?:light|the context) of the (?:above|foregoing|aforementioned)\b",
    r"\bthe aforementioned\b",
    r"\bhaving said that,?\b",
    r"\bneedless to say,?\b",
    r"\binextricably linked\b",
    r"\bsignificantly impacts?\b",
    r"\bin (?:conclusion|summary),? (?:it is|we can|this study|this paper)\b",
    r"\bthroughout (?:history|the ages|time)\b",
    r"\bthe importance of .{3,40} cannot be (?:overstated|understated|emphasised|emphasized)\b",
    r"\bpotential (?:benefits?|drawbacks?|implications?|challenges?) (?:include|are|may|of)\b",
    r"\bwithout (?:a )?(?:doubt|question|reservation)\b",
    r"\bit (?:is|can be) (?:argued|said|posited|noted|observed) that\b",
    r"\bby (?:and large|no means|all accounts)\b",
    r"\beveryone.{0,20}knows?\b",
    r"\bfacilitat(?:e|es|ing|ed)\b.{0,40}(?:understanding|learning|growth|progress)\b",
    r"\bseamlessly integrat\b",
    r"\brobust (?:framework|methodology|approach|solution|system)\b",
    r"\bemphasise?s? the (?:importance|need|significance)\b",
    r"\bintricacies? of\b",
    r"\bever-?(?:evolving|changing|growing|increasing)\b",
    r"\bmyriad(?:\s+of)? (?:ways?|factors?|reasons?|benefits?|challenges?)\b",
]

_AI_PHRASE_RES = [re.compile(p, re.IGNORECASE) for p in _AI_PHRASE_PATTERNS]

# Passive voice: "is/are/was/were/been/be/being + past participle"
_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+[a-z]+ed\b",
    re.IGNORECASE,
)

# Hedging/filler words that inflate AI text
_HEDGE_WORDS = frozenset([
    "additionally", "furthermore", "moreover", "consequently",
    "therefore", "thus", "notably", "importantly", "significantly",
    "essentially", "fundamentally", "ultimately", "effectively",
    "particularly", "specifically", "generally", "typically",
    "broadly", "extensively", "comprehensively", "notably",
    "substantially", "considerably", "predominantly", "primarily",
])

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


def _split_sentences(text: str) -> list[str]:
    sents = _SENT_RE.split(text.strip())
    return [s.strip() for s in sents if len(s.split()) >= 5]


def _ai_phrase_score(sent: str) -> float:
    hits = sum(1 for r in _AI_PHRASE_RES if r.search(sent))
    return min(1.0, hits * 0.65)


def _passive_score(sent: str) -> float:
    return 0.6 if _PASSIVE_RE.search(sent) else 0.0


def _hedge_score(sent: str) -> float:
    words = sent.lower().split()
    hits = sum(1 for w in words if w.rstrip(".,;:") in _HEDGE_WORDS)
    return min(1.0, hits / max(len(words), 1) * 7)


def _uniformity_score(sent: str) -> float:
    """Low word-length variance → more AI-like."""
    words = re.findall(r"\b\w+\b", sent)
    if len(words) < 5:
        return 0.0
    lengths = [len(w) for w in words]
    try:
        cv = statistics.stdev(lengths) / max(statistics.mean(lengths), 1)
        return max(0.0, 1.0 - cv / 0.55)
    except statistics.StatisticsError:
        return 0.0


def _sentence_ai_score(sent: str) -> float:
    return min(
        1.0,
        _ai_phrase_score(sent) * 0.50
        + _passive_score(sent) * 0.10
        + _hedge_score(sent) * 0.15
        + _uniformity_score(sent) * 0.25,
    )


def detect_ai_content(text: str) -> dict[str, Any]:
    """
    Returns per-sentence AI scores and an overall document probability.

    verdict: "likely_ai" | "mixed" | "likely_human" | "insufficient_text"
    """
    if not (text or "").strip():
        return {
            "overall_ai_percentage": 0.0,
            "burstiness": 1.0,
            "verdict": "insufficient_text",
            "sentences": [],
        }

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return {
            "overall_ai_percentage": 0.0,
            "burstiness": 1.0,
            "verdict": "insufficient_text",
            "sentences": [],
        }

    scores = [_sentence_ai_score(s) for s in sentences]
    mean_s = statistics.mean(scores)
    std_s = statistics.stdev(scores) if len(scores) > 1 else 0.0

    # Burstiness: higher → more human-like variation
    burstiness = std_s / mean_s if mean_s > 0.01 else 0.0
    # Map low burstiness → high AI signal (< 0.3 is very flat = AI)
    burstiness_ai = max(0.0, 1.0 - burstiness / 0.8)

    overall = mean_s * 0.55 + burstiness_ai * 0.45
    overall_pct = round(min(100.0, overall * 100), 1)

    sentence_results: list[dict[str, Any]] = []
    for sent, score in zip(sentences, scores):
        label = (
            "likely_ai" if score >= 0.65
            else "uncertain" if score >= 0.32
            else "likely_human"
        )
        sentence_results.append({
            "text": sent,
            "ai_probability": round(score, 3),
            "label": label,
        })

    verdict = (
        "likely_ai" if overall_pct >= 65
        else "mixed" if overall_pct >= 30
        else "likely_human"
    )

    return {
        "overall_ai_percentage": overall_pct,
        "burstiness": round(burstiness, 3),
        "verdict": verdict,
        "sentences": sentence_results,
    }
