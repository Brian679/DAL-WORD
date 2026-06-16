"""
Plagiarism detection engine using shingled fingerprint matching.

Turnitin's OriginalityCheck does not work via "AI judgement" — it breaks every
submitted paper into overlapping word n-grams ("shingles"), fingerprints them,
and looks the fingerprints up in an index built from its repository (web pages,
journals, and previously submitted student papers). Any submission whose
fingerprints collide heavily with another source is flagged, and the overall
similarity score is the percentage of the submission's text that falls inside
a matched span.

This module mirrors that approach using the only corpus an offline editor
actually has access to: every other document already stored in this workspace.
That is also exactly what catches the single most common real-world case
Turnitin exists for — resubmitted or copy-pasted work from another paper.

Pipeline:
  1. Split the target text into sentences (>= 6 words — short fragments
     produce too many false-positive shingle collisions to be useful).
  2. Shingle each sentence into overlapping k-word windows and hash each
     shingle (sha1, truncated) for cheap set comparisons.
  3. Build an inverted index (shingle hash -> source sentences) from every
     other document in the database once per call.
  4. For each target sentence, use the inverted index to find candidate
     source sentences that share at least one shingle, then score the best
     candidate with Jaccard similarity over the full shingle sets.
  5. A sentence is "matched" (near-verbatim) above MATCH_THRESHOLD, "similar"
     (likely paraphrased) above SIMILAR_THRESHOLD, otherwise original.
  6. Overall similarity % = the share of total words that fall in matched or
     similar sentences, banded into Turnitin's published colour scale.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .ai_detector import _split_sentences

SHINGLE_SIZE = 8
MATCH_THRESHOLD = 0.5
SIMILAR_THRESHOLD = 0.25
MIN_SENTENCE_WORDS = 6

_WORD_RE = re.compile(r"[a-z0-9']+")


def _normalised_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _shingle_hashes(words: list[str], k: int = SHINGLE_SIZE) -> set[str]:
    if len(words) < k:
        if not words:
            return set()
        k = len(words)
    out = set()
    for i in range(len(words) - k + 1):
        gram = " ".join(words[i:i + k])
        out.add(hashlib.sha1(gram.encode("utf-8")).hexdigest()[:16])
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _verdict_for(pct: float) -> tuple[str, str]:
    """Turnitin's published similarity colour bands."""
    if pct <= 0:
        return "no_match", "#3b82f6"
    if pct < 25:
        return "minor", "#16a34a"
    if pct < 50:
        return "moderate", "#ca8a04"
    if pct < 75:
        return "high", "#ea580c"
    return "severe", "#ef4444"


def _build_corpus_index(source_docs: list[tuple[int, str, str]]):
    """
    source_docs: list of (document_id, document_title, full_text)
    Returns (inverted_index, sentence_records) where:
      inverted_index: shingle_hash -> set of sentence_record indices
      sentence_records: list of dicts with doc_id, doc_title, text, shingles
    """
    inverted_index: dict[str, set[int]] = {}
    sentence_records: list[dict[str, Any]] = []
    for doc_id, doc_title, full_text in source_docs:
        for sent in _split_sentences(full_text):
            words = _normalised_words(sent)
            if len(words) < MIN_SENTENCE_WORDS:
                continue
            shingles = _shingle_hashes(words)
            if not shingles:
                continue
            idx = len(sentence_records)
            sentence_records.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "text": sent,
                "shingles": shingles,
            })
            for h in shingles:
                inverted_index.setdefault(h, set()).add(idx)
    return inverted_index, sentence_records


def check_plagiarism(
    text: str,
    source_docs: list[tuple[int, str, str]] | None = None,
) -> dict[str, Any]:
    """
    Compare `text` against `source_docs` (other documents in the workspace)
    using shingle fingerprint matching, Turnitin-OriginalityCheck-style.

    source_docs: list of (document_id, document_title, full_text) tuples,
                 excluding the document being checked.
    """
    if not (text or "").strip():
        return {
            "overall_similarity_percentage": 0.0,
            "verdict": "insufficient_text",
            "color": "#9ca3af",
            "sentences": [],
            "sources": [],
            "checked_against": 0,
        }

    source_docs = source_docs or []
    inverted_index, sentence_records = _build_corpus_index(source_docs)

    target_sentences = _split_sentences(text)
    results: list[dict[str, Any]] = []
    matched_words = 0
    total_words = 0
    source_hits: dict[int, dict[str, Any]] = {}

    for sent in target_sentences:
        words = _normalised_words(sent)
        word_count = len(sent.split())
        total_words += word_count

        if len(words) < MIN_SENTENCE_WORDS or not sentence_records:
            results.append({
                "text": sent,
                "similarity": 0.0,
                "label": "original",
                "source_title": None,
                "source_document_id": None,
            })
            continue

        shingles = _shingle_hashes(words)
        candidate_idxs: set[int] = set()
        for h in shingles:
            candidate_idxs |= inverted_index.get(h, set())

        best_sim = 0.0
        best_record = None
        for idx in candidate_idxs:
            rec = sentence_records[idx]
            sim = _jaccard(shingles, rec["shingles"])
            if sim > best_sim:
                best_sim = sim
                best_record = rec

        if best_sim >= MATCH_THRESHOLD:
            label = "matched"
        elif best_sim >= SIMILAR_THRESHOLD:
            label = "similar"
        else:
            label = "original"

        if label != "original":
            matched_words += word_count
            if best_record is not None:
                doc_id = best_record["doc_id"]
                entry = source_hits.setdefault(doc_id, {
                    "document_id": doc_id,
                    "title": best_record["doc_title"],
                    "matched_words": 0,
                })
                entry["matched_words"] += word_count

        results.append({
            "text": sent,
            "similarity": round(best_sim, 3),
            "label": label,
            "source_title": best_record["doc_title"] if best_record and label != "original" else None,
            "source_document_id": best_record["doc_id"] if best_record and label != "original" else None,
        })

    overall_pct = round((matched_words / total_words) * 100, 1) if total_words else 0.0
    verdict, color = _verdict_for(overall_pct)

    sources = sorted(
        (
            {
                "document_id": s["document_id"],
                "title": s["title"],
                "match_percentage": round((s["matched_words"] / total_words) * 100, 1) if total_words else 0.0,
            }
            for s in source_hits.values()
        ),
        key=lambda s: s["match_percentage"],
        reverse=True,
    )

    return {
        "overall_similarity_percentage": overall_pct,
        "verdict": verdict,
        "color": color,
        "sentences": results,
        "sources": sources,
        "checked_against": len(source_docs),
    }
