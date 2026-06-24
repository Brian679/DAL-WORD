from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from django.conf import settings

logger = logging.getLogger(__name__)

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_CACHE_TTL_SECONDS = 60 * 60 * 6  # 6-hour cache per query


def _cache_path(key: str) -> Path:
    base = Path(settings.MEDIA_ROOT) / "research" / "cache"
    base.mkdir(parents=True, exist_ok=True)
    safe = hashlib.md5(key.encode()).hexdigest()
    return base / f"{safe}.json"


def _cache_get(key: str) -> list[dict] | None:
    if os.getenv("RESEARCH_CACHE_ENABLED", "1") != "1":
        return None
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
        if time.time() - payload.get("ts", 0) > _CACHE_TTL_SECONDS:
            p.unlink(missing_ok=True)
            return None
        return payload.get("data")
    except Exception:
        return None


def _cache_set(key: str, data: list[dict]) -> None:
    if os.getenv("RESEARCH_CACHE_ENABLED", "1") != "1":
        return
    try:
        _cache_path(key).write_text(json.dumps({"ts": time.time(), "data": data}))
    except Exception:
        pass


def _cached_search(source_fn, cache_key: str, *args, **kwargs) -> list[PaperRecord]:
    """Wrap any search function with transparent disk caching."""
    cached = _cache_get(cache_key)
    if cached is not None:
        try:
            return [PaperRecord(**item) for item in cached]
        except Exception:
            pass
    results = source_fn(*args, **kwargs)
    _cache_set(cache_key, [r.__dict__ for r in results])
    return results


@dataclass
class PaperRecord:
    source: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    abstract: str | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CitationVerification:
    title: str | None
    doi: str | None
    confidence: int
    status: str
    matched_fields: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class RetrievalResult:
    topic: str
    expanded_queries: list[str]
    papers: list[PaperRecord]
    top_papers: list[PaperRecord]
    embedding_path: str | None = None


class EmbeddingStore:
    """Simple JSON-backed embedding store for traceability and memory retrieval."""

    def __init__(self, document_id: int | None = None):
        base = Path(settings.MEDIA_ROOT) / "research" / "embeddings"
        base.mkdir(parents=True, exist_ok=True)
        name = f"doc-{document_id}.json" if document_id is not None else "global.json"
        self.path = base / name

    @staticmethod
    def embed_text(text: str, dim: int = 128) -> list[float]:
        vector = [0.0] * dim
        tokens = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
        if not tokens:
            return vector
        for token in tokens:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
            vector[idx] += 1.0
        norm = max(sum(v * v for v in vector) ** 0.5, 1.0)
        return [v / norm for v in vector]

    def save_records(self, records: list[PaperRecord]) -> str:
        payload = []
        for rec in records:
            rec_dict = asdict(rec)
            rec_dict["embedding"] = self.embed_text(
                " ".join(
                    [
                        rec.title or "",
                        " ".join(rec.authors or []),
                        rec.abstract or "",
                        rec.journal or "",
                    ]
                )
            )
            payload.append(rec_dict)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(self.path)


def _http_get_json(url: str, timeout: int = 12, headers: dict[str, str] | None = None) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout, headers=headers or {})
    r.raise_for_status()
    return r.json()


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip())


def expand_query(topic: str, max_items: int = 6) -> list[str]:
    base = _clean_title(topic)
    variants = [
        base,
        f"{base} literature review",
        f"{base} systematic review",
        f"{base} empirical study",
        f"{base} methodology outcomes",
        f"{base} DOI",
        f"{base} academic journal",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for q in variants:
        k = q.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
        if len(out) >= max_items:
            break
    return out


def _parse_crossref_item(item: dict[str, Any]) -> PaperRecord:
    title = ""
    if isinstance(item.get("title"), list) and item["title"]:
        title = str(item["title"][0])
    authors = []
    for a in item.get("author") or []:
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        full = " ".join([p for p in [given, family] if p])
        if full:
            authors.append(full)
    year = None
    published = item.get("published-print") or item.get("published-online") or {}
    parts = (published.get("date-parts") or [[None]])[0]
    if parts and parts[0]:
        year = int(parts[0])
    return PaperRecord(
        source="crossref",
        title=_clean_title(title),
        authors=authors,
        year=year,
        journal=((item.get("container-title") or [None])[0] if isinstance(item.get("container-title"), list) else None),
        doi=item.get("DOI"),
        url=item.get("URL"),
        abstract=item.get("abstract"),
        metadata={"raw_type": item.get("type")},
    )


def search_crossref(query: str, rows: int = 10) -> list[PaperRecord]:
    try:
        url = f"https://api.crossref.org/works?query={quote_plus(query)}&rows={rows}&select=DOI,title,author,container-title,published-print,published-online,URL,type"
        data = _http_get_json(url)
        items = (data.get("message") or {}).get("items") or []
        return [_parse_crossref_item(i) for i in items if i.get("title")]
    except Exception as exc:
        logger.warning("Crossref search failed: %s", exc)
        return []


def search_semantic_scholar(query: str, limit: int = 8) -> list[PaperRecord]:
    try:
        fields = "title,year,authors,journal,abstract,url,externalIds"
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote_plus(query)}&limit={limit}&fields={quote_plus(fields)}"
        headers: dict[str, str] = {}
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if api_key:
            headers["x-api-key"] = api_key

        for attempt in range(3):
            try:
                data = _http_get_json(url, headers=headers)
                break
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    if not api_key:
                        # Without an API key we are at the public limit — don't waste time retrying.
                        logger.info("Semantic Scholar rate-limited (no API key) — skipping")
                        return []
                    wait = 5 * (2 ** attempt)
                    logger.info("Semantic Scholar rate-limited, retrying in %ds", wait)
                    time.sleep(wait)
                    continue
                raise
        else:
            logger.warning("Semantic Scholar permanently rate-limited for query: %s", query)
            return []

        papers = []
        for row in data.get("data") or []:
            papers.append(
                PaperRecord(
                    source="semantic_scholar",
                    title=_clean_title(row.get("title") or ""),
                    authors=[a.get("name") for a in row.get("authors") or [] if a.get("name")],
                    year=row.get("year"),
                    journal=(row.get("journal") or {}).get("name") if isinstance(row.get("journal"), dict) else None,
                    doi=((row.get("externalIds") or {}).get("DOI")),
                    url=row.get("url"),
                    abstract=row.get("abstract"),
                )
            )
        return [p for p in papers if p.title]
    except Exception as exc:
        logger.warning("Semantic Scholar search failed: %s", exc)
        return []


def search_arxiv(query: str, limit: int = 8) -> list[PaperRecord]:
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}&start=0&max_results={limit}"
        req = Request(url, headers={"User-Agent": "dal-word-research-bot/1.0"})
        with urlopen(req, timeout=12) as resp:
            xml = resp.read().decode("utf-8", errors="ignore")
        root = ElementTree.fromstring(xml)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out: list[PaperRecord] = []
        for entry in root.findall("a:entry", ns):
            title = _clean_title((entry.findtext("a:title", default="", namespaces=ns) or ""))
            if not title:
                continue
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            link = entry.find("a:link[@type='text/html']", ns)
            pdf_link = entry.find("a:link[@title='pdf']", ns)
            published = entry.findtext("a:published", default="", namespaces=ns)
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [
                (a.findtext("a:name", default="", namespaces=ns) or "").strip()
                for a in entry.findall("a:author", ns)
            ]
            out.append(
                PaperRecord(
                    source="arxiv",
                    title=title,
                    authors=[a for a in authors if a],
                    year=year,
                    journal="arXiv",
                    url=(link.get("href") if link is not None else None),
                    pdf_url=(pdf_link.get("href") if pdf_link is not None else None),
                    abstract=summary,
                )
            )
        return out
    except Exception as exc:
        logger.warning("arXiv search failed: %s", exc)
        return []


def search_pubmed(query: str, limit: int = 8) -> list[PaperRecord]:
    try:
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&retmode=json&retmax={limit}&term={quote_plus(query)}"
        )
        search_data = _http_get_json(search_url)
        ids = (search_data.get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return []
        summary_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&retmode=json&id={','.join(ids)}"
        )
        summary = _http_get_json(summary_url)
        out: list[PaperRecord] = []
        for pid in ids:
            row = (summary.get("result") or {}).get(pid) or {}
            title = _clean_title(row.get("title") or "")
            if not title:
                continue
            author_list = [a.get("name") for a in row.get("authors") or [] if a.get("name")]
            pubdate = str(row.get("pubdate") or "")
            year_match = YEAR_RE.search(pubdate)
            year = int(year_match.group(0)) if year_match else None
            article_ids = row.get("articleids") or []
            doi = None
            for aid in article_ids:
                if str(aid.get("idtype", "")).lower() == "doi":
                    doi = aid.get("value")
                    break
            out.append(
                PaperRecord(
                    source="pubmed",
                    title=title,
                    authors=author_list,
                    year=year,
                    journal=row.get("fulljournalname") or row.get("source"),
                    doi=doi,
                    url=(f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"),
                )
            )
        return out
    except Exception as exc:
        logger.warning("PubMed search failed: %s", exc)
        return []


def search_ssrn(query: str, limit: int = 8) -> list[PaperRecord]:
    # SSRN is indexed by Crossref, so this is a targeted Crossref fallback.
    rows = search_crossref(f"{query} SSRN", rows=limit * 2)
    out = [p for p in rows if "ssrn" in ((p.url or "") + " " + (p.journal or "")).lower()]
    return out[:limit]


def search_google_scholar_scrape(query: str, limit: int = 5) -> list[PaperRecord]:
    """
    Best-effort scraping fallback.
    Disabled by default to stay compliant and avoid brittle behavior.
    Set ALLOW_GOOGLE_SCHOLAR_SCRAPE=1 to enable.
    """
    if os.getenv("ALLOW_GOOGLE_SCHOLAR_SCRAPE", "0") != "1":
        return []
    try:
        url = f"https://scholar.google.com/scholar?hl=en&q={quote_plus(query)}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        out: list[PaperRecord] = []
        for card in soup.select("div.gs_ri")[:limit]:
            h3 = card.select_one("h3.gs_rt")
            if not h3:
                continue
            a = h3.find("a")
            title = _clean_title(h3.get_text(" ", strip=True))
            if not title:
                continue
            meta = (card.select_one("div.gs_a") or {}).get_text(" ", strip=True) if card.select_one("div.gs_a") else ""
            year_match = YEAR_RE.search(meta)
            year = int(year_match.group(0)) if year_match else None
            out.append(
                PaperRecord(
                    source="google_scholar",
                    title=title,
                    year=year,
                    url=(a.get("href") if a else None),
                    journal=None,
                    abstract=(card.select_one("div.gs_rs").get_text(" ", strip=True) if card.select_one("div.gs_rs") else None),
                    metadata={"meta": meta},
                )
            )
        # Be gentle if enabled; avoid aggressive scraping.
        time.sleep(1.5)
        return out
    except Exception as exc:
        logger.warning("Google Scholar scrape failed: %s", exc)
        return []


def rank_papers(query: str, papers: list[PaperRecord], top_k: int = 20) -> list[PaperRecord]:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored: list[PaperRecord] = []
    for p in papers:
        text = " ".join([p.title or "", p.abstract or "", p.journal or "", " ".join(p.authors or [])]).lower()
        terms = set(re.findall(r"[a-z0-9]+", text))
        overlap = len(query_terms & terms)
        doi_bonus = 2 if p.doi else 0
        journal_bonus = 1 if p.journal else 0
        year_bonus = 1 if (p.year and p.year >= 2018) else 0
        p.score = float(overlap + doi_bonus + journal_bonus + year_bonus)
        scored.append(p)
    scored.sort(key=lambda x: x.score, reverse=True)

    dedup: list[PaperRecord] = []
    seen: set[str] = set()
    for p in scored:
        key = re.sub(r"\W+", "", (p.title or "").lower())
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(p)
        if len(dedup) >= top_k:
            break
    return dedup


def citation_string(p: PaperRecord) -> str:
    author = (p.authors[0].split()[-1] if p.authors else "Unknown")
    year = str(p.year) if p.year else "n.d."
    title = p.title or "Untitled"
    venue = p.journal or p.source
    doi = p.doi or "N/A"
    return f"{author} ({year}). {title}. {venue}. DOI: {doi}"


def build_citation_context(papers: list[PaperRecord], max_items: int = 12, style: str = "APA") -> str:
    if not papers:
        return ""
    style_key = (style or "APA").strip().upper()
    if style_key in {"IEEE", "VANCOUVER"}:
        guidance = "Cite these in-text using the matching bracket number only, e.g. [1], [2]."
    else:
        guidance = "Cite these in-text using author-date format only, e.g. (Smith, 2020) — never the bracket numbers below."
    lines = [f"Use only the following verifiable sources when citing claims. {guidance}"]
    for i, p in enumerate(papers[:max_items], start=1):
        lines.append(f"[{i}] {citation_string(p)}")
    return "\n".join(lines)


def _split_author_name(name: str) -> tuple[str, str]:
    """Split a full name into (surname, rest) — "Jane A. Doe" -> ("Doe", "Jane A.")."""
    parts = name.split()
    if len(parts) >= 2:
        return parts[-1], " ".join(parts[:-1])
    return name, ""


def _authors_apa_style(authors: list[str]) -> str:
    names = []
    for a in authors[:6]:
        surname, rest = _split_author_name(a)
        if rest:
            initials = "".join(f"{n[0]}." for n in rest.split())
            names.append(f"{surname}, {initials}")
        else:
            names.append(surname)
    if not names:
        return "Unknown Author"
    if len(authors) > 6:
        return ", ".join(names) + ", et al."
    if len(names) > 1:
        return ", ".join(names[:-1]) + ", & " + names[-1]
    return names[0]


def format_reference_entry(paper: PaperRecord, style: str = "APA", index: int = 1) -> str:
    """Format a single reference-list entry from a verified PaperRecord in the
    requested citation style. Supported styles: APA, Harvard, MLA, Chicago, IEEE,
    Vancouver (case-insensitive; unrecognised values fall back to APA). Only ever
    called on real, retrieved papers — there is nothing here for an LLM to invent."""
    style_key = (style or "APA").strip().upper()
    authors = list(paper.authors or [])
    year = str(paper.year) if paper.year else "n.d."
    title = (paper.title or "Untitled").rstrip(".")
    venue = paper.journal or (paper.source or "").replace("_", " ").title()
    locator = f"https://doi.org/{paper.doi}" if paper.doi else (paper.url or "")

    if style_key in {"IEEE", "VANCOUVER"}:
        # Numbered style: "[n] F. Surname, F. Surname, "Title," Venue, Year."
        names = []
        for a in authors[:6]:
            surname, rest = _split_author_name(a)
            initials = "".join(f"{n[0]}." for n in rest.split()) if rest else ""
            names.append(f"{initials} {surname}".strip())
        author_str = ", ".join(names) if names else "Unknown Author"
        if len(authors) > 6:
            author_str += ", et al."
        entry = f'[{index}] {author_str}, "{title}," {venue}, {year}.'
        if locator:
            entry += f" {locator}"
        return entry

    if style_key == "MLA":
        # "Surname, First, et al. "Title." Venue, Year."
        if authors:
            surname, rest = _split_author_name(authors[0])
            first_fmt = f"{surname}, {rest}" if rest else surname
            author_str = f"{first_fmt}, et al" if len(authors) > 1 else first_fmt
        else:
            author_str = "Unknown Author"
        entry = f'{author_str}. "{title}." {venue}, {year}.'
        if locator:
            entry += f" {locator}."
        return entry

    if style_key == "CHICAGO":
        # Author-date: "Surname, First, and Surname, First. Year. "Title." Venue."
        if authors:
            surname, rest = _split_author_name(authors[0])
            first_fmt = f"{surname}, {rest}" if rest else surname
            if len(authors) > 1:
                author_str = f"{first_fmt}, et al" if len(authors) > 2 else f"{first_fmt}, and {authors[1]}"
            else:
                author_str = first_fmt
        else:
            author_str = "Unknown Author"
        entry = f'{author_str}. {year}. "{title}." {venue}.'
        if locator:
            entry += f" {locator}."
        return entry

    if style_key == "HARVARD":
        # Close to APA, but "and" before the final author rather than "&".
        names = []
        for a in authors[:6]:
            surname, rest = _split_author_name(a)
            if rest:
                initials = "".join(f"{n[0]}." for n in rest.split())
                names.append(f"{surname}, {initials}")
            else:
                names.append(surname)
        if not names:
            author_str = "Unknown Author"
        elif len(authors) > 6:
            author_str = ", ".join(names) + ", et al."
        elif len(names) > 1:
            author_str = ", ".join(names[:-1]) + " and " + names[-1]
        else:
            author_str = names[0]
        entry = f"{author_str}, {year}. {title}. {venue}."
        if locator:
            entry += f" Available at: {locator}."
        return entry

    # Default: APA
    author_str = _authors_apa_style(authors)
    entry = f"{author_str} ({year}). {title}. {venue}."
    if locator:
        entry += f" {locator}"
    return entry


def _title_similarity(a: str, b: str) -> float:
    """Return word-overlap Jaccard similarity between two titles."""
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower())) - {"the", "a", "an", "of", "in", "and", "for", "on"}
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower())) - {"the", "a", "an", "of", "in", "and", "for", "on"}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _author_overlap(api_authors: list[str], claim_authors: list[str]) -> bool:
    """Return True if at least one author last name appears in both lists."""
    api_surnames = {a.split()[-1].lower() for a in api_authors if a}
    cla_surnames = {a.split()[-1].lower() for a in (claim_authors or []) if a}
    return bool(api_surnames & cla_surnames)


def verify_against_crossref(
    title: str | None = None,
    doi: str | None = None,
    authors: list[str] | None = None,
    year: int | None = None,
    journal: str | None = None,
) -> CitationVerification:
    """
    Full 5-field citation verification against Crossref.

    Confidence table:
      DOI verified + all metadata match → 100
      DOI verified, title matched       → 95
      DOI verified only                 → 85
      Title exact + metadata ≥2 fields  → 90
      Title exact, no other match       → 70
      Title partial (Jaccard ≥ 0.5)     → 60
      Title partial (Jaccard ≥ 0.3)     → 40
      No match                          → 0  (reject)
    """

    # ── Path 1: verify by DOI first ──────────────────────────────────────────
    if doi:
        try:
            url = f"https://api.crossref.org/works/{quote_plus(doi.strip())}"
            data = _http_get_json(url, timeout=10)
            item: dict[str, Any] = data.get("message") or {}
            matched: list[str] = ["doi"]

            api_title = ((item.get("title") or [""])[0] if isinstance(item.get("title"), list) else "")
            if title and _title_similarity(api_title, title) >= 0.8:
                matched.append("title")

            api_authors: list[str] = []
            for a in (item.get("author") or []):
                family = (a.get("family") or "").strip()
                given = (a.get("given") or "").strip()
                api_authors.append(" ".join(p for p in [given, family] if p))
            if authors and _author_overlap(api_authors, authors):
                matched.append("authors")

            # Year check
            published = item.get("published-print") or item.get("published-online") or {}
            parts = (published.get("date-parts") or [[None]])[0]
            api_year = int(parts[0]) if (parts and parts[0]) else None
            if year and api_year and abs(api_year - year) <= 1:
                matched.append("year")

            # Journal / container
            container = (item.get("container-title") or [""])[0] if isinstance(item.get("container-title"), list) else ""
            if journal and container and (
                _title_similarity(journal, container) >= 0.5 or
                journal.lower() in container.lower() or
                container.lower() in journal.lower()
            ):
                matched.append("journal")

            field_count = len([f for f in matched if f != "doi"])
            if field_count >= 3:
                confidence = 100
            elif "title" in matched and field_count >= 2:
                confidence = 95
            elif "title" in matched:
                confidence = 85
            else:
                confidence = 75  # DOI found but title mismatch — suspicious

            return CitationVerification(
                title=title or api_title,
                doi=doi,
                confidence=confidence,
                status="verified",
                matched_fields=matched,
                notes=f"Crossref DOI verified. Matched: {matched}",
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return CitationVerification(
                    title=title, doi=doi, confidence=0, status="reject",
                    notes=f"DOI not found in Crossref (404)",
                )
            return CitationVerification(
                title=title, doi=doi, confidence=0, status="reject",
                notes=f"Crossref DOI lookup error: {exc}",
            )
        except Exception as exc:
            return CitationVerification(
                title=title, doi=doi, confidence=0, status="reject",
                notes=f"DOI lookup failed: {exc}",
            )

    # ── Path 2: title-only search ────────────────────────────────────────────
    if not title:
        return CitationVerification(
            title=title, doi=doi, confidence=0, status="reject", notes="No title or DOI provided",
        )

    candidates = search_crossref(title, rows=5)
    if not candidates:
        return CitationVerification(
            title=title, doi=doi, confidence=0, status="reject", notes="No Crossref results for title",
        )

    best = candidates[0]
    sim = _title_similarity(best.title or "", title)

    if sim >= 0.85:
        # Strong title match — check secondary fields
        matched = ["title"]
        if authors and best.authors and _author_overlap(best.authors, authors):
            matched.append("authors")
        if year and best.year and abs(best.year - year) <= 1:
            matched.append("year")
        if journal and best.journal and _title_similarity(journal, best.journal) >= 0.5:
            matched.append("journal")

        field_count = len(matched)
        if field_count >= 3:
            confidence = 90
        elif field_count >= 2:
            confidence = 80
        else:
            confidence = 70

        return CitationVerification(
            title=title, doi=best.doi, confidence=confidence,
            status="metadata_matched", matched_fields=matched,
            notes=f"Title similarity {sim:.2f}. Matched: {matched}",
        )

    if sim >= 0.5:
        return CitationVerification(
            title=title, doi=best.doi, confidence=60, status="partial_match",
            matched_fields=["title_partial"],
            notes=f"Partial title match (Jaccard {sim:.2f})",
        )

    if sim >= 0.3:
        return CitationVerification(
            title=title, doi=None, confidence=40, status="partial_match",
            matched_fields=["title_weak"],
            notes=f"Weak title match (Jaccard {sim:.2f})",
        )

    return CitationVerification(
        title=title, doi=None, confidence=0, status="reject",
        notes=f"No meaningful match found (best similarity {sim:.2f}) — likely hallucinated",
    )


def extract_citation_candidates(text: str) -> list[dict[str, Any]]:
    """
    Extract potential citations from generated text.

    Detects:
    - Bare DOI patterns: 10.xxxx/...
    - Quoted title strings (≥ 3 words)
    - APA inline citations: (Author, Year) or (Author et al., Year)
    """
    out: list[dict[str, Any]] = []

    # DOIs embedded anywhere in text
    for doi in DOI_RE.findall(text or ""):
        out.append({"title": None, "doi": doi, "authors": None, "year": None, "journal": None})

    # Quoted strings that look like titles (≥ 3 words, ≤ 200 chars)
    for m in re.finditer(r'"([^"]{12,200})"', text or ""):
        title = m.group(1).strip()
        if len(title.split()) >= 3:
            out.append({"title": title, "doi": None, "authors": None, "year": None, "journal": None})

    # APA inline citations: (Surname, Year) or (Surname et al., Year)
    apa_re = re.compile(
        r'\(([A-Z][a-zA-Zé\-]+(?:\s+et\s+al\.?)?(?:\s*&\s*[A-Z][a-zA-Z\-]+)*),\s*((?:19|20)\d{2})\)',
    )
    for m in apa_re.finditer(text or ""):
        author_raw = m.group(1).strip()
        yr = int(m.group(2))
        out.append({"title": None, "doi": None, "authors": [author_raw], "year": yr, "journal": None})

    # Deduplicate by (doi or title or author+year key)
    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for c in out:
        key = "|".join([
            (c.get("doi") or "").lower(),
            _clean_title(c.get("title") or "").lower(),
            str(c.get("year") or ""),
            ",".join(c.get("authors") or []).lower(),
        ])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    return dedup


def verify_generated_citations(text: str) -> list[CitationVerification]:
    candidates = extract_citation_candidates(text)
    results: list[CitationVerification] = []
    for c in candidates:
        results.append(verify_against_crossref(
            title=c.get("title"),
            doi=c.get("doi"),
            authors=c.get("authors"),
            year=c.get("year"),
            journal=c.get("journal"),
        ))
    return results


def summarize_verification(report: list[CitationVerification]) -> dict[str, Any]:
    if not report:
        return {"total": 0, "verified": 0, "rejected": 0, "partial": 0, "avg_confidence": 0, "details": []}
    total = len(report)
    verified = sum(1 for r in report if r.status in {"verified", "metadata_matched"})
    rejected = sum(1 for r in report if r.status == "reject")
    partial = sum(1 for r in report if r.status == "partial_match")
    avg = round(sum(r.confidence for r in report) / total, 1)
    tiers = {"high": 0, "medium": 0, "low": 0, "rejected": 0}
    for r in report:
        if r.confidence >= 85:
            tiers["high"] += 1
        elif r.confidence >= 60:
            tiers["medium"] += 1
        elif r.confidence > 0:
            tiers["low"] += 1
        else:
            tiers["rejected"] += 1
    return {
        "total": total,
        "verified": verified,
        "rejected": rejected,
        "partial": partial,
        "avg_confidence": avg,
        "tiers": tiers,
        "details": [asdict(r) for r in report],
    }


@dataclass
class CitationRepairResult:
    original_text: str
    repaired_text: str
    total_citations: int
    repaired_count: int
    removed_count: int
    unchanged_count: int
    repair_log: list[dict[str, Any]] = field(default_factory=list)


def _find_best_replacement(
    rejected_title: str | None,
    rejected_doi: str | None,
    pool: list[PaperRecord],
) -> PaperRecord | None:
    """Find the best real paper from the retrieval pool to replace a rejected citation."""
    if not pool:
        return None
    query_terms = set(re.findall(r"[a-z0-9]+", (rejected_title or "").lower()))
    scored = []
    for p in pool:
        if not p.doi:
            continue  # Only suggest DOI-backed replacements
        p_terms = set(re.findall(r"[a-z0-9]+", (p.title or "").lower()))
        sim = len(query_terms & p_terms) / max(len(query_terms | p_terms), 1)
        scored.append((sim, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0.15 else None


def repair_citations(
    text: str,
    pool: list[PaperRecord],
    min_confidence: int = 60,
) -> CitationRepairResult:
    """
    Repair hallucinated citations in generated text.

    For each citation found:
    - If confidence >= min_confidence: keep as-is
    - If confidence < min_confidence: attempt to replace with a real paper from pool
    - If no replacement found: remove the citation from text

    Returns a CitationRepairResult with the repaired text and repair log.
    """
    verifications = verify_generated_citations(text)
    if not verifications:
        return CitationRepairResult(
            original_text=text, repaired_text=text,
            total_citations=0, repaired_count=0, removed_count=0, unchanged_count=0,
        )

    repaired_text = text
    repair_log: list[dict[str, Any]] = []
    repaired_count = 0
    removed_count = 0
    unchanged_count = 0

    for v in verifications:
        if v.confidence >= min_confidence:
            unchanged_count += 1
            repair_log.append({
                "action": "kept",
                "title": v.title,
                "doi": v.doi,
                "confidence": v.confidence,
                "status": v.status,
            })
            continue

        # Low confidence — try to find a replacement
        replacement = _find_best_replacement(v.title, v.doi, pool)

        if replacement:
            # Replace the offending DOI or quoted title with the real paper's citation
            replacement_str = citation_string(replacement)
            if v.doi and v.doi in repaired_text:
                repaired_text = repaired_text.replace(v.doi, replacement.doi or replacement_str)
            elif v.title and v.title in repaired_text:
                repaired_text = repaired_text.replace(
                    f'"{v.title}"',
                    f'"{replacement.title}"',
                )
            repaired_count += 1
            repair_log.append({
                "action": "replaced",
                "original_title": v.title,
                "original_doi": v.doi,
                "original_confidence": v.confidence,
                "replacement_title": replacement.title,
                "replacement_doi": replacement.doi,
                "replacement_source": replacement.source,
            })
        else:
            # No replacement available — mark/remove the bad citation
            if v.doi and v.doi in repaired_text:
                repaired_text = repaired_text.replace(v.doi, "[citation removed — unverifiable]")
            removed_count += 1
            repair_log.append({
                "action": "removed",
                "title": v.title,
                "doi": v.doi,
                "confidence": v.confidence,
                "reason": v.notes,
            })

    return CitationRepairResult(
        original_text=text,
        repaired_text=repaired_text,
        total_citations=len(verifications),
        repaired_count=repaired_count,
        removed_count=removed_count,
        unchanged_count=unchanged_count,
        repair_log=repair_log,
    )
def fetch_pdf_text(url: str, max_pages: int = 20) -> str:
    try:
        resp = requests.get(url, timeout=18)
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            return ""
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(resp.content))  # type: ignore[name-defined]
            chunks = []
            for page in reader.pages[:max_pages]:
                chunks.append(page.extract_text() or "")
            return "\n".join(chunks).strip()
        except Exception:
            return ""
    except Exception:
        return ""


def retrieval_pipeline(
    topic: str, query: str | None = None, document_id: int | None = None, top_k: int = 20
) -> RetrievalResult:
    """
    Complete retrieval pipeline: query expansion → multi-source search → ranking → embedding storage.

    Returns: RetrievalResult with ranked papers and embedding path for traceability.
    """
    effective_query = query or topic
    expanded = expand_query(topic, max_items=3)

    all_papers: list[PaperRecord] = []

    # Semantic Scholar: only query with primary topic (rate-limit friendly).
    all_papers.extend(_cached_search(search_semantic_scholar, f"ss:{effective_query}", effective_query, limit=10))

    # Crossref, arXiv, PubMed, SSRN: search expanded queries (no auth required).
    for q in expanded:
        all_papers.extend(_cached_search(search_crossref, f"cr:{q}", q, rows=10))
        all_papers.extend(_cached_search(search_arxiv, f"ax:{q}", q, limit=8))
        all_papers.extend(_cached_search(search_pubmed, f"pm:{q}", q, limit=6))
        all_papers.extend(_cached_search(search_ssrn, f"ssrn:{q}", q, limit=5))

    # Optionally search Google Scholar if enabled.
    if os.getenv("ALLOW_GOOGLE_SCHOLAR_SCRAPE", "0") == "1":
        all_papers.extend(search_google_scholar_scrape(effective_query, limit=5))

    # Rank and deduplicate.
    ranked = rank_papers(effective_query, all_papers, top_k=top_k)

    # Store embeddings for later retrieval/memory.
    store = EmbeddingStore(document_id)
    embedding_path = store.save_records(ranked)

    return RetrievalResult(
        topic=topic,
        expanded_queries=expanded,
        papers=all_papers,
        top_papers=ranked,
        embedding_path=embedding_path,
    )


def build_research_brief(result: RetrievalResult) -> str:
    """Build a structured research context for injection into generation prompt."""
    if not result.top_papers:
        return ""
    lines = [f"## Research Context: {result.topic}", ""]
    lines.append(f"Retrieved from {len(result.papers)} candidate papers, ranking top {len(result.top_papers)}:")
    lines.append("")
    for i, p in enumerate(result.top_papers[:10], start=1):
        venue = p.journal or p.source
        year_str = f"({p.year})" if p.year else ""
        authors_str = ", ".join(p.authors[:2]) if p.authors else "Unknown"
        lines.append(f"  {i}. {p.title} — {authors_str} {year_str}, {venue}")
        if p.abstract:
            snippet = p.abstract[:120].strip() + "..." if len(p.abstract) > 120 else p.abstract
            lines.append(f"     Abstract: {snippet}")
    return "\n".join(lines)



