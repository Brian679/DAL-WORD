"""
Plagiarism detection engine using shingled fingerprint matching, plus a
rule-based similarity-reduction rewriter for the passages it flags.

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
import random
import re
from typing import Any

from .ai_detector import _SENT_RE, _match_case, _split_sentences

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


# ---------------------------------------------------------------------------
# Similarity reduction — rewrite only the sentences check_plagiarism flagged
# ---------------------------------------------------------------------------
#
# Shingle matching scores a sentence on the Jaccard overlap of its full
# 8-word-window set against a source sentence. To pull that overlap below
# SIMILAR_THRESHOLD we don't need a perfect paraphrase — we need enough of the
# word sequence disturbed that few 8-word windows survive unchanged. Two
# complementary moves do that:
#   1. Lexical substitution: swap common words for synonyms. Every substitution
#      invalidates every shingle window that crosses it (up to 8 windows per
#      change), so even moderate substitution density collapses the overlap.
#   2. Clause reordering: swapping the two halves of an "X, and/but Y" sentence
#      changes the word sequence wholesale rather than locally, catching the
#      cases substitution alone misses (e.g. sentences with few swappable words).
# A verification re-check after rewriting (mirroring how AI-humanisation
# reports a before/after score) catches the rare sentence that needs a second
# pass with a fresh random seed.

_PARAPHRASE_SYNONYMS: dict[str, list[str]] = {
    "important": ["significant", "key", "notable"],
    "significant": ["considerable", "notable", "substantial"],
    "crucial": ["critical", "essential", "vital"],
    "threaten": ["endanger", "put at risk", "jeopardise"],
    "threatens": ["endangers", "puts at risk", "jeopardises"],
    "around the world": ["across the globe", "worldwide", "globally"],
    "the world": ["the globe", "the planet"],
    "people": ["individuals", "communities", "populations"],
    "displace": ["uproot", "force out", "relocate"],
    "displaces": ["uproots", "forces out", "relocates"],
    "millions of": ["countless", "vast numbers of"],
    "rising": ["increasing", "climbing", "growing"],
    "increase": ["rise", "grow", "climb"],
    "increases": ["rises", "grows", "climbs"],
    "decrease": ["decline", "drop", "fall"],
    "decreases": ["declines", "drops", "falls"],
    "cause": ["bring about", "trigger", "lead to"],
    "causes": ["brings about", "triggers", "leads to"],
    "result in": ["lead to", "bring about"],
    "due to": ["because of", "owing to"],
    "because of": ["due to", "owing to"],
    "in order to": ["to", "so as to"],
    "however": ["nevertheless", "that said", "still"],
    "therefore": ["as a result", "consequently", "thus"],
    "moreover": ["furthermore", "in addition", "what's more"],
    "additionally": ["furthermore", "also", "in addition"],
    "furthermore": ["moreover", "additionally", "in addition"],
    "show": ["reveal", "indicate", "demonstrate"],
    "shows": ["reveals", "indicates", "demonstrates"],
    "demonstrate": ["show", "illustrate", "reveal"],
    "demonstrates": ["shows", "illustrates", "reveals"],
    "indicate": ["suggest", "show", "point to"],
    "indicates": ["suggests", "shows", "points to"],
    "suggest": ["indicate", "imply", "point to"],
    "suggests": ["indicates", "implies", "points to"],
    "provide": ["offer", "supply", "deliver"],
    "provides": ["offers", "supplies", "delivers"],
    "develop": ["build", "construct", "devise"],
    "develops": ["builds", "constructs", "devises"],
    "create": ["produce", "generate", "build"],
    "creates": ["produces", "generates", "builds"],
    "establish": ["set up", "build", "form"],
    "ensure": ["guarantee", "make certain", "secure"],
    "ensures": ["guarantees", "makes certain", "secures"],
    "maintain": ["sustain", "uphold", "preserve"],
    "maintains": ["sustains", "upholds", "preserves"],
    "require": ["need", "call for", "demand"],
    "requires": ["needs", "calls for", "demands"],
    "allow": ["permit", "let", "enable"],
    "allows": ["permits", "lets", "enables"],
    "enable": ["allow", "permit", "make possible"],
    "enables": ["allows", "permits", "makes possible"],
    "achieve": ["attain", "accomplish", "reach"],
    "achieves": ["attains", "accomplishes", "reaches"],
    "obtain": ["acquire", "secure", "gain"],
    "reduce": ["lower", "cut", "diminish"],
    "reduces": ["lowers", "cuts", "diminishes"],
    "improve": ["enhance", "strengthen", "boost"],
    "improves": ["enhances", "strengthens", "boosts"],
    "approach": ["method", "strategy", "technique"],
    "method": ["approach", "technique", "procedure"],
    "process": ["procedure", "mechanism", "workflow"],
    "factor": ["element", "consideration", "variable"],
    "factors": ["elements", "considerations", "variables"],
    "issue": ["concern", "matter", "problem"],
    "issues": ["concerns", "matters", "problems"],
    "challenge": ["difficulty", "obstacle", "hurdle"],
    "challenges": ["difficulties", "obstacles", "hurdles"],
    "solution": ["remedy", "fix", "answer"],
    "solutions": ["remedies", "fixes", "answers"],
    "strategy": ["approach", "plan", "tactic"],
    "strategies": ["approaches", "plans", "tactics"],
    "focus on": ["concentrate on", "centre on", "home in on"],
    "consider": ["examine", "weigh", "look at"],
    "examine": ["investigate", "study", "scrutinise"],
    "explore": ["investigate", "examine", "delve into"],
    "investigate": ["examine", "explore", "look into"],
    "assess": ["evaluate", "appraise", "gauge"],
    "evaluate": ["assess", "appraise", "judge"],
    "determine": ["establish", "ascertain", "work out"],
    "identify": ["pinpoint", "recognise", "single out"],
    "propose": ["put forward", "suggest", "recommend"],
    "recommend": ["advise", "suggest", "propose"],
    "conclude": ["determine", "deduce", "infer"],
    "discuss": ["examine", "address", "cover"],
    "describe": ["depict", "outline", "characterise"],
    "explain": ["clarify", "account for", "elucidate"],
    "highlight": ["underscore", "spotlight", "draw attention to"],
    "reveal": ["show", "expose", "uncover"],
    "confirm": ["verify", "corroborate", "validate"],
    "support": ["back", "bolster", "reinforce"],
    "contribute to": ["add to", "play a part in", "feed into"],
    "particularly": ["notably", "especially", "in particular"],
    "especially": ["particularly", "notably", "above all"],
    "specifically": ["in particular", "precisely", "namely"],
    "generally": ["broadly", "typically", "on the whole"],
    "typically": ["generally", "usually", "as a rule"],
    "often": ["frequently", "regularly", "commonly"],
    "frequently": ["often", "regularly", "repeatedly"],
    "rarely": ["seldom", "infrequently", "hardly ever"],
    "sometimes": ["occasionally", "at times", "now and then"],
    "usually": ["typically", "generally", "normally"],
    "mainly": ["primarily", "largely", "chiefly"],
    "primarily": ["mainly", "principally", "largely"],
    "ultimately": ["in the end", "eventually", "finally"],
    "eventually": ["ultimately", "in time", "over time"],
    "consequently": ["as a result", "therefore", "hence"],
    "despite": ["notwithstanding", "in spite of", "even with"],
    "although": ["though", "even though", "while"],
    "community": ["neighbourhood", "locality"],
    "communities": ["neighbourhoods", "localities"],
    "society": ["the public", "the population"],
    "population": ["inhabitants", "residents"],
    "government": ["the authorities", "policymakers", "administration"],
}


def _lexical_substitute(text: str, rng: random.Random) -> str:
    for phrase in sorted(_PARAPHRASE_SYNONYMS, key=len, reverse=True):
        options = _PARAPHRASE_SYNONYMS[phrase]
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        if not pattern.search(text):
            continue
        text = pattern.sub(lambda m: _match_case(m.group(), rng.choice(options)), text, count=1)
    return text


_CLAUSE_SWAP_RE = re.compile(r"^(.*?),\s+(and|but)\s+(.+?)([.!?])$", re.IGNORECASE)


def _swap_clauses(sentence: str, rng: random.Random) -> str:
    """Swap 'A, and/but B.' -> 'B, and/but A.' to disturb word order wholesale."""
    m = _CLAUSE_SWAP_RE.match(sentence.strip())
    if not m or rng.random() >= 0.7:
        return sentence
    first, conj, second, end = m.group(1), m.group(2), m.group(3), m.group(4)
    new_first = second[0].upper() + second[1:]
    new_second = first[0].lower() + first[1:]
    return f"{new_first}, {conj} {new_second}{end}"


# Vocabulary-independent fallback. Lexical substitution needs a word the
# synonym map knows, and clause swap needs an "A, and/but B" shape — neither
# fires on a plain sentence built from everyday words ("Sarah walked to the
# corner shop..."). Shingles are about word *content*, not position, so the
# only way to guarantee disruption with no vocabulary hook to grab is to break
# the original word sequence into pieces no longer than SHINGLE_SIZE - 1 words
# by inserting short parenthetical asides between them — that leaves zero
# original 8-word windows intact, regardless of what the words actually are.
_TRANSITION_FILLERS = [
    "in fact", "notably", "indeed", "as it happens", "to be specific",
    "for context", "interestingly", "as a result", "in this instance",
    "it should be noted", "incidentally", "more precisely", "in turn",
]

# Avoid snapping a split point right after one of these — splitting a
# determiner/preposition/conjunction off from what follows it reads as broken
# rather than just informal.
_NO_BREAK_AFTER = {
    "the", "a", "an", "this", "that", "these", "those", "his", "her", "its",
    "their", "our", "your", "my", "of", "to", "in", "on", "at", "for", "with",
    "by", "from", "into", "onto", "upon", "over", "under", "about", "between",
    "among", "through", "during", "before", "after", "since", "until", "while",
    "and", "but", "or", "nor", "so", "yet", "as", "if", "than", "because", "not",
}

_END_PUNCT_RE = re.compile(r"^(.*?)([.!?]+)$")


def _snap_split_point(words: list[str], target: int) -> int | None:
    n = len(words)
    for radius in range(n):
        for cand in (target + radius, target - radius):
            if 1 <= cand <= n - 1:
                prior = words[cand - 1].lower().strip(".,;:!?\"')(")
                if prior not in _NO_BREAK_AFTER:
                    return cand
    return None


def _insert_transitions(sentence: str, rng: random.Random) -> str:
    """Break a long, vocabulary-resistant sentence into <=7-word pieces joined
    by short parenthetical asides, so no original 8-word shingle survives."""
    stripped = sentence.strip()
    m = _END_PUNCT_RE.match(stripped)
    body, end = (m.group(1), m.group(2)) if m else (stripped, "")
    words = body.split()
    if len(words) <= SHINGLE_SIZE:
        return sentence

    split_points: list[int] = []
    target = SHINGLE_SIZE - 1
    while target < len(words):
        snapped = _snap_split_point(words, target)
        if snapped is not None and (not split_points or snapped > split_points[-1]):
            split_points.append(snapped)
        target += SHINGLE_SIZE - 1

    if not split_points:
        return sentence

    fillers = _TRANSITION_FILLERS.copy()
    rng.shuffle(fillers)
    pieces = []
    cursor = 0
    for sp in split_points:
        pieces.append(" ".join(words[cursor:sp]))
        cursor = sp
    pieces.append(" ".join(words[cursor:]))

    out = pieces[0]
    for i, piece in enumerate(pieces[1:]):
        filler = fillers[i % len(fillers)]
        out += f", {filler}, {piece}"
    return out + end


def paraphrase_sentence(sentence: str, rng: random.Random) -> str:
    text = _lexical_substitute(sentence, rng)
    text = _swap_clauses(text, rng)
    if text.strip() == sentence.strip():
        text = _insert_transitions(text, rng)
    return text


def reduce_similarity(text: str, flagged_texts: set[str], seed: str | None = None) -> str:
    """
    Rewrite only the sentences in `flagged_texts` (matched/similar passages
    found by check_plagiarism), leaving everything else untouched.
    """
    if not flagged_texts:
        return text
    rng = random.Random(seed or text[:200])
    paragraphs = text.split("\n")
    out_paragraphs = []
    for para in paragraphs:
        if not para.strip():
            out_paragraphs.append(para)
            continue
        sentences = _SENT_RE.split(para.strip())
        rebuilt = [
            paraphrase_sentence(s, rng) if s.strip() in flagged_texts else s
            for s in sentences
        ]
        out_paragraphs.append(" ".join(rebuilt))
    return "\n".join(out_paragraphs)
