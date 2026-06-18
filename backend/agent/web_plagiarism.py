"""
External-source plagiarism check.

check_plagiarism() in plagiarism_detector.py only ever sees documents already
stored in this workspace. This module extends that net outward to the open
scholarly web — Crossref, arXiv, PubMed, SSRN, and Semantic Scholar, all
official/key-free APIs — and scores whatever it finds with the exact same
shingle/Jaccard machinery, so a "matched" or "similar" verdict means the same
thing whether the source was a local document or a paper out on the internet.

Google Scholar has no official public API. research_layer.py already gates
raw scraping of it behind ALLOW_GOOGLE_SCHOLAR_SCRAPE=1 to stay ToS-compliant
and avoid brittle behaviour; that existing gate is reused as-is here rather
than overridden. With it unset, this still searches the same literature
Google Scholar itself indexes, just through the legitimate APIs above.

Every network call below already fails soft (the search_* functions in
research_layer.py catch their own exceptions and return [] on failure), so a
sandboxed or offline deployment just gets zero external matches rather than
an error.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from .ai_detector import _split_sentences
from .plagiarism_detector import (
    MATCH_THRESHOLD,
    MIN_SENTENCE_WORDS,
    SIMILAR_THRESHOLD,
    _jaccard,
    _normalised_words,
    _shingle_hashes,
    _verdict_for,
)
from .research_layer import (
    PaperRecord,
    _cached_search,
    fetch_pdf_text,
    search_arxiv,
    search_crossref,
    search_google_scholar_scrape,
    search_pubmed,
    search_semantic_scholar,
    search_ssrn,
)

logger = logging.getLogger(__name__)

# Bounds on network fan-out: one external check fires up to
# MAX_QUERY_SENTENCES queries across 6 sources, and downloads full PDF text
# for at most MAX_PDF_FETCHES of the resulting papers.
MAX_QUERY_SENTENCES = 4
MAX_PAPERS_SCORED = 20
MAX_PDF_FETCHES = 6


def external_check_enabled() -> bool:
    return os.getenv("PLAGIARISM_EXTERNAL_CHECK", "1") == "1"


def _candidate_queries(text: str, limit: int = MAX_QUERY_SENTENCES) -> list[str]:
    """Use the longest, most distinctive sentences as search queries."""
    sentences = [
        s.strip() for s in _split_sentences(text)
        if len(_normalised_words(s)) >= MIN_SENTENCE_WORDS
    ]
    sentences.sort(key=lambda s: len(s.split()), reverse=True)
    return sentences[:limit]


def _gather_papers(queries: list[str]) -> list[PaperRecord]:
    papers: list[PaperRecord] = []
    for q in queries:
        papers.extend(_cached_search(search_crossref, f"plag:cr:{q}", q, rows=5))
        papers.extend(_cached_search(search_arxiv, f"plag:ax:{q}", q, limit=5))
        papers.extend(_cached_search(search_semantic_scholar, f"plag:ss:{q}", q, limit=5))
        papers.extend(_cached_search(search_pubmed, f"plag:pm:{q}", q, limit=3))
        papers.extend(_cached_search(search_ssrn, f"plag:ssrn:{q}", q, limit=3))
        papers.extend(search_google_scholar_scrape(q, limit=5))  # no-op unless ALLOW_GOOGLE_SCHOLAR_SCRAPE=1

    dedup: list[PaperRecord] = []
    seen: set[str] = set()
    for p in papers:
        key = (p.doi or "").lower() or re.sub(r"\W+", "", (p.title or "").lower())
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup[:MAX_PAPERS_SCORED]


def _paper_sentence_shingles(paper: PaperRecord, pdf_budget: list[int]) -> list[set[str]]:
    """Best-effort text for a paper: full PDF if openly accessible (budget-limited), else title+abstract."""
    paper_text = ""
    if paper.pdf_url and pdf_budget[0] > 0:
        pdf_budget[0] -= 1
        paper_text = fetch_pdf_text(paper.pdf_url)
    if not paper_text.strip():
        paper_text = " ".join(filter(None, [paper.title, paper.abstract]))
    return [
        _shingle_hashes(_normalised_words(s))
        for s in _split_sentences(paper_text)
        if len(_normalised_words(s)) >= MIN_SENTENCE_WORDS
    ]


def check_external_plagiarism(text: str, max_sentences: int = MAX_QUERY_SENTENCES) -> dict[str, Any]:
    """
    Search the open scholarly web for matches to `text` and score them with
    the same shingle/Jaccard machinery as check_plagiarism().

    Returns {"hits": [...], "papers_checked": int} where each hit has the
    same shape as a check_plagiarism() sentence record, plus source_url and
    source_type identifying which external source it came from.
    """
    if not (text or "").strip():
        return {"hits": [], "papers_checked": 0}

    queries = _candidate_queries(text, limit=max_sentences)
    if not queries:
        return {"hits": [], "papers_checked": 0}

    try:
        papers = _gather_papers(queries)
    except Exception as exc:
        logger.warning("External plagiarism source gathering failed: %s", exc)
        return {"hits": [], "papers_checked": 0}

    target_sentences = [s.strip() for s in _split_sentences(text)]
    pdf_budget = [MAX_PDF_FETCHES]

    hits: dict[str, dict[str, Any]] = {}
    for paper in papers:
        try:
            paper_shingle_sets = _paper_sentence_shingles(paper, pdf_budget)
        except Exception as exc:
            logger.warning("Scoring external paper %r failed: %s", paper.title, exc)
            continue
        if not paper_shingle_sets:
            continue

        for sent in target_sentences:
            words = _normalised_words(sent)
            if len(words) < MIN_SENTENCE_WORDS:
                continue
            shingles = _shingle_hashes(words)
            best_sim = max((_jaccard(shingles, ps) for ps in paper_shingle_sets), default=0.0)
            if best_sim < SIMILAR_THRESHOLD:
                continue
            prior = hits.get(sent)
            if prior is not None and prior["similarity"] >= best_sim:
                continue
            hits[sent] = {
                "text": sent,
                "similarity": round(best_sim, 3),
                "label": "matched" if best_sim >= MATCH_THRESHOLD else "similar",
                "source_title": paper.title,
                "source_url": paper.url,
                "source_type": paper.source,
            }

    return {"hits": list(hits.values()), "papers_checked": len(papers)}


def merge_into_check_result(result: dict[str, Any], external: dict[str, Any]) -> dict[str, Any]:
    """
    Fold check_external_plagiarism() hits into a check_plagiarism() result,
    upgrading any sentence the web search matched more strongly than the
    workspace did, and recomputing the overall percentage/verdict/sources.
    """
    result["external_checked"] = True
    result["external_papers_checked"] = external.get("papers_checked", 0)

    hits_by_text = {h["text"]: h for h in external.get("hits", [])}
    if not hits_by_text:
        return result

    total_words = sum(len(s["text"].split()) for s in result["sentences"]) or 1
    matched_words = 0
    web_source_words: dict[str, int] = {}
    web_source_titles: dict[str, str] = {}

    for sent_result in result["sentences"]:
        ext_hit = hits_by_text.get(sent_result["text"].strip())
        if ext_hit and ext_hit["similarity"] > sent_result["similarity"]:
            sent_result["similarity"] = ext_hit["similarity"]
            sent_result["label"] = ext_hit["label"]
            sent_result["source_title"] = ext_hit["source_title"]
            sent_result["source_document_id"] = None
            sent_result["source_url"] = ext_hit["source_url"]
            sent_result["source_type"] = ext_hit["source_type"]

        if sent_result["label"] == "original":
            continue
        word_count = len(sent_result["text"].split())
        matched_words += word_count

        url = sent_result.get("source_url")
        if url:
            web_source_words[url] = web_source_words.get(url, 0) + word_count
            web_source_titles[url] = sent_result["source_title"]

    overall_pct = round((matched_words / total_words) * 100, 1)
    verdict, color = _verdict_for(overall_pct)
    result["overall_similarity_percentage"] = overall_pct
    result["verdict"] = verdict
    result["color"] = color

    web_sources = sorted(
        (
            {
                "document_id": None,
                "title": web_source_titles[url],
                "url": url,
                "match_percentage": round((words / total_words) * 100, 1),
            }
            for url, words in web_source_words.items()
        ),
        key=lambda s: s["match_percentage"],
        reverse=True,
    )
    result["sources"] = result["sources"] + web_sources
    return result
