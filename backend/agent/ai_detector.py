"""
AI content detection engine, plus a StealthWriter-style rule-based humaniser.

Detection approximates Turnitin's two-signal methodology:
  1. Perplexity  – how predictable is each sentence to a language model?
     (approximated via known-phrase + structural signals)
  2. Burstiness  – humans vary their sentence complexity; LLMs stay uniform.
     B = σ(per-sentence score) / μ(per-sentence score)

Overall document AI probability combines the mean sentence score (55 %)
with a burstiness inversion signal (45 %).

The humaniser mirrors the three signals a detector scores, attacking each
directly instead of doing a single phrase find/replace pass:
  Pass 1  — cliché/fingerprint phrase substitution (lowers phrase score)
  Pass 1b — clause/nominalisation restructuring: a curated set of templates
            that convert specific noun-phrase and passive-voice shapes
            common in AI academic prose (e.g. "the impact of X on Y" ->
            "how X affects Y", "Data were collected through X" -> fronted
            source/instrument, connector swaps) into a differently-shaped
            but grammatically equivalent phrase or clause. No dependency
            parser is available in this environment, so each rule targets
            one well-defined pattern rather than a generic transform.
  Pass 2  — lexical substitution of high-predictability words with lower-
            frequency synonyms (raises perplexity / lowers predictability)
  Pass 3  — sentence-length restructuring: split long uniform sentences and
            merge short uniform ones so length variance increases (raises
            burstiness, since LLM output is unusually uniform in length)
"""
from __future__ import annotations

import random
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
    r"\brobust (?:framework|methodology|approach|solution|system)s?\b",
    r"\bemphasise?s? the (?:importance|need|significance)\b",
    r"\bintricacies? of\b",
    r"\bever-?(?:evolving|changing|growing|increasing)\b",
    r"\bmyriad(?:\s+of)? (?:ways?|factors?|reasons?|benefits?|challenges?)\b",
    r"\bshed(?:s|ding)? light on\b",
    r"\bnavigate(?:s|d)? the complexit(?:y|ies) of\b",
    r"\bstands? as a testament to\b",
    r"\bserves? as a\b",
    r"\bgarner(?:s|ed|ing)? (?:attention|interest|support)\b",
    r"\bembark(?:s|ed|ing)? on a journey\b(?:\s+of)?",
    r"\bin (?:today's|this) (?:fast-paced|rapidly changing) world\b",
    r"\btapestry of\b",
    r"\bunprecedented (?:levels?|growth|challenges?)\b",
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


# ── Rule-based humaniser ───────────────────────────────────────────────────
# Applied when no LLM key is available, or as a pre-pass before LLM polish.
# Each entry: (compiled_regex, replacement_string)

_HUMANISE_RULES: list[tuple[re.Pattern[str], str]] = [
    # Hedge openers
    (re.compile(r"\bIt is important to note that\b", re.I), "Note that"),
    (re.compile(r"\bIt is worth noting that\b", re.I), "Worth noting:"),
    (re.compile(r"\bIt is (?:crucial|essential|vital) to\b", re.I), lambda m: _match_case(m.group(), "to")),
    (re.compile(r"\bIt is evident that\b", re.I), "Clearly,"),
    (re.compile(r"\bIt (?:is|can be) (?:argued|said|noted|observed) that\b", re.I), "I would argue that"),
    (re.compile(r"\bNeedless to say,?\s*", re.I), ""),
    (re.compile(r"\bIt goes without saying that\b", re.I), "Obviously,"),
    (re.compile(r"\bHaving said that,?\s*", re.I), "That said, "),
    (re.compile(r"\bWithout (?:a )?(?:doubt|question|reservation)\b", re.I), "Clearly"),
    (re.compile(r"\bBy and large\b", re.I), "Overall"),
    (re.compile(r"\bBy no means\b", re.I), "Not"),
    # Time/world clichés
    (re.compile(r"\bIn today['’]?s (?:world|society|age|era|landscape)\b", re.I), "Today"),
    (re.compile(r"\bin (?:today['’]?s|this) (?:fast-paced|rapidly changing) world\b", re.I), "now"),
    (re.compile(r"\btapestry of\b", re.I), "mix of"),
    (re.compile(r"\bunprecedented (?:levels?|growth|challenges?)\b", re.I), lambda m: f"sharp {m.group().split()[-1]}"),
    (re.compile(r"\bshed(?:s|ding)? light on\b", re.I), "clarify"),
    (re.compile(r"\bnavigate(?:s|d)? the complexit(?:y|ies) of\b", re.I), "work through"),
    (re.compile(r"\bstands? as a testament to\b", re.I), "reflects"),
    (re.compile(r"\bserves? as a\b", re.I), "acts as a"),
    (re.compile(r"\bgarner(?:s|ed|ing)? (?:attention|interest|support)\b", re.I), lambda m: f"attract {m.group().split()[-1]}"),
    (re.compile(r"\bembark(?:s|ed|ing)? on a journey\b(?:\s+of)?", re.I), "begin"),
    (re.compile(r"\bit cannot be denied that\b", re.I), "clearly,"),
    (re.compile(r"\bin (?:conclusion|summary)\b,?", re.I), "overall,"),
    (re.compile(r"\bon the other hand\b", re.I), "by contrast"),
    (re.compile(r"\bThroughout history\b", re.I), "Historically"),
    (re.compile(r"\bThroughout the ages\b", re.I), "Over the years"),
    # Meta-essay phrases
    (re.compile(r"\bThis (?:essay|paper|study|section|chapter|report) (?:will|aims to|seeks to|endeavours? to)\b", re.I), "This study"),
    (re.compile(r"\bThis (?:essay|paper|study|section|report) (?:will|aims to) (?:provide|offer|present) a (?:comprehensive|detailed)\b", re.I), "This paper presents"),
    (re.compile(r"\bThis (?:highlights?|underscores?|demonstrates?|showcases?|illuminates?) the\b", re.I), "This shows the"),
    (re.compile(r"\bIn (?:conclusion|summary),? (?:it is|we can|this study|this paper)\b", re.I), "In short,"),
    # Buzzwords
    (re.compile(r"\bdelve(?:s|d)?\s+into\b", re.I), "explore"),
    (re.compile(r"\bcomprehensive (?:overview|analysis|examination|understanding)\b", re.I), "overview"),
    (re.compile(r"\bholistic approach\b", re.I), "broad approach"),
    (re.compile(r"\bparadigm shift\b", re.I), "major change"),
    (re.compile(r"\binextricably linked\b", re.I), "closely connected"),
    (re.compile(r"\bseamlessly integrat\b", re.I), "integrat"),
    (re.compile(r"\brobust (?:framework|methodology|approach|solution|system)s?\b", re.I), lambda m: m.group().split()[-1]),
    (re.compile(r"\bthe (?:realm|landscape|domain) of\b", re.I), "the area of"),
    (re.compile(r"\bthe aforementioned\b", re.I), "these"),
    (re.compile(r"\baforementioned\b", re.I), "above-mentioned"),
    (re.compile(r"\bever-?(?:evolving|changing|growing)\b", re.I), "changing"),
    (re.compile(r"\bmyriad(?:\s+of)?\b", re.I), "many"),
    (re.compile(r"\bintricacies? of\b", re.I), "details of"),
    (re.compile(r"\bfoster(?:s|ed|ing)? (?:a )?(?:deeper|better|greater|more nuanced) understanding\b", re.I), "build understanding"),
    (re.compile(r"\bunderscores? the (?:importance|significance|need|necessity)\b", re.I), "highlights the importance"),
    (re.compile(r"\bemphasise?s? the (?:importance|need|significance)\b", re.I), "highlights the importance"),
    (re.compile(r"\b(plays?) (?:a )?(?:crucial|pivotal|significant|vital|key|important) role\b", re.I),
     lambda m: "matters" if m.group(1).lower() == "plays" else "matter"),
    (re.compile(r"\bsignificantly impacts?\b", re.I), "affects"),
    (re.compile(r"\bpotential (?:benefits?|implications?|challenges?)\b", re.I), lambda m: m.group().split()[-1]),
    # Connective filler
    (re.compile(r"\bFurthermore,?\s+", re.I), "Also, "),
    (re.compile(r"\bMoreover,?\s+", re.I), "And "),
    (re.compile(r"\bConsequently,?\s+", re.I), "So "),
    (re.compile(r"\bAdditionally,?\s+", re.I), "Also, "),
    (re.compile(r"\bNotably,?\s+", re.I), ""),
    (re.compile(r"\bImportantly,?\s+", re.I), ""),
    (re.compile(r"\bEssentially,?\s+", re.I), ""),
    (re.compile(r"\bFundamentally,?\s+", re.I), ""),
    (re.compile(r"\bUltimately,?\s+", re.I), "In the end, "),
    (re.compile(r"\bthe importance of (.{3,40}) cannot be (?:overstated|understated|emphasised|emphasized)\b", re.I), lambda m: f"{m.group(1)} matters a great deal"),
    # Passive constructions (common AI pattern)
    (re.compile(r"\bIt (?:has|have) been (?:noted|observed|suggested|argued) that\b", re.I), "Research suggests that"),
    (re.compile(r"\bit is (?:widely )?(?:recognised|recognized|acknowledged|accepted) that\b", re.I), "Most researchers agree that"),
]


# ── Pass 1b: clause / nominalisation restructuring ─────────────────────────
# Curated, hand-written templates for specific noun-phrase and passive-voice
# shapes that recur in AI-generated academic prose. Every rule here converts
# the matched span into a *grammatically equivalent* replacement — NP -> NP,
# full clause -> full clause, word -> word — so it stays safe to drop into
# whatever sentence position the original phrase occupied. Deliberately
# narrower than a generic syntactic transform: with no dependency parser
# available, each template targets one well-defined pattern rather than
# guessing at sentence structure.

def _restructure_clauses(text: str, rng: random.Random) -> str:
    # "The findings/results revealed/showed/indicated that X" -> "It was found that X"
    text = re.sub(
        r"\bThe (?:findings|results) (?:revealed|showed|indicated) that\b",
        lambda m: _match_case(m.group(), "it was found that"),
        text, flags=re.IGNORECASE,
    )

    # "the impact/effect/influence of X on Y" -> "how X affects/influences/shapes Y"
    # Safe after the governing verbs this phrase typically follows in academic
    # writing (determine/investigate/examine/assess/study/establish), which
    # accept either a noun-phrase or a wh-clause complement interchangeably.
    def _impact_to_clause(m: re.Match) -> str:
        verb = rng.choice(["affects", "influences", "shapes"])
        return f"how {m.group(1).strip()} {verb} {m.group(2).strip()}"
    text = re.sub(
        r"\bthe (?:impact|effect|influence) of ([^,.;:]+?) on ([^,.;:]+)",
        _impact_to_clause, text, flags=re.IGNORECASE,
    )

    # "customer/revenue/job/market share loss" -> "the loss of customers/..."  (NP -> NP)
    def _loss_phrase(m: re.Match) -> str:
        noun = m.group(1).lower()
        plural = {"customer": "customers", "job": "jobs"}.get(noun, noun)
        return f"the loss of {plural}"
    text = re.sub(
        r"\b(customer|revenue|job|market share) loss\b",
        _loss_phrase, text, flags=re.IGNORECASE,
    )

    # "technology/knowledge/skills transfer" -> "transfer of technology/..."  (NP -> NP)
    def _transfer_phrase(m: re.Match) -> str:
        result = f"transfer of {m.group(1).lower()}"
        return result[:1].upper() + result[1:] if m.group(0)[:1].isupper() else result
    text = re.sub(
        r"\b(technology|knowledge|skills?) transfer\b",
        _transfer_phrase, text, flags=re.IGNORECASE,
    )

    # "creates/create opportunities for X to Y" -> "gives/give X the chance to Y"
    # Keeps the same subject + verb-phrase slot, just swaps the verb-phrase shape.
    text = re.sub(
        r"\bcreates opportunities for ([^,.;:]+?) to ([^,.;:]+)",
        lambda m: f"gives {m.group(1).strip()} the chance to {m.group(2).strip()}",
        text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bcreate opportunities for ([^,.;:]+?) to ([^,.;:]+)",
        lambda m: f"give {m.group(1).strip()} the chance to {m.group(2).strip()}",
        text, flags=re.IGNORECASE,
    )

    # "(Quantitative/Qualitative) data were/was collected through/using X (from Y)"
    # -> front the source as subject, keeping any leading qualifier attached to
    # "data" rather than letting it dangle in front of the new subject.
    def _data_collected(m: re.Match) -> str:
        starts_upper = m.group(0)[:1].isupper()
        qualifier = (m.group(1) or "").strip()
        instrument = m.group(2).strip()
        source = m.group(3)
        qualifier_text = f"{qualifier.lower()} " if qualifier else ""
        if source:
            result = f"{source.strip()} were used to gather {qualifier_text}data through {instrument}"
        else:
            # Crude plurality heuristic ("interviews"/"surveys" vs "a questionnaire")
            # since no parser is available to check real number agreement.
            last_word = instrument.rstrip(".").split()[-1].lower() if instrument else ""
            verb = "were" if last_word.endswith("s") and not last_word.endswith("ss") else "was"
            result = f"{instrument} {verb} used to gather {qualifier_text}data"
        return result[:1].upper() + result[1:] if starts_upper else result
    text = re.sub(
        r"\b(?:(Quantitative|Qualitative)\s+)?[Dd]ata (?:were|was) collected (?:through|using) "
        r"([^,.;:]+?)(?:\s+from\s+([^,.;:]+))?(?=[.,;:]|$)",
        _data_collected, text, flags=re.IGNORECASE,
    )

    # "limited access/funding/resources/capacity" -> "a lack of access/..."  (NP -> NP)
    text = re.sub(
        r"\blimited (access|funding|resources|capacity)\b",
        lambda m: f"a lack of {m.group(1)}",
        text, flags=re.IGNORECASE,
    )

    # "increased competition" -> "more intense/heightened competition"  (NP -> NP)
    text = re.sub(
        r"\bincreased competition\b",
        lambda m: _match_case(m.group(), rng.choice(["more intense competition", "heightened competition"])),
        text, flags=re.IGNORECASE,
    )

    # Connector substitution — preserves clause structure, swaps the linking word.
    # "X, however, Y" -> "X, but Y" (mid-clause insertion, drop the trailing comma "but" doesn't take)
    text = re.sub(r",\s+however,\s+", ", but ", text, flags=re.IGNORECASE)
    # Sentence-initial "However, X" -> "But X"
    text = re.sub(
        r"\bHowever,\s+",
        lambda m: _match_case("However", "But") + " ",
        text, flags=re.IGNORECASE,
    )
    # Any remaining bare "however" -> "but"
    text = re.sub(r"\bHowever\b", lambda m: _match_case(m.group(), "but"), text, flags=re.IGNORECASE)

    # ", while Y verbed" -> ", and Y verbed" — only when "while" is followed by
    # what looks like a fresh clause subject (not "still"/an -ing participle),
    # since "while trying to..."/"while still..." are reduced clauses with no
    # subject of their own and "and" would leave a dangling fragment.
    def _while_to_and(m: re.Match) -> str:
        return m.group(0) if rng.random() >= 0.5 else ", and "
    text = re.sub(
        r",\s+while\s+(?!still\b|\w+ing\b)",
        _while_to_and, text, flags=re.IGNORECASE,
    )

    return text


# ── Pass 2: lexical substitution ───────────────────────────────────────────
# Common high-frequency / high-predictability words that LLMs over-use,
# mapped to lower-frequency synonyms. Word-boundary, case-preserving swap.
_SYNONYM_MAP: dict[str, list[str]] = {
    "utilize": ["use"], "utilizes": ["uses"], "utilizing": ["using"],
    "leverage": ["use", "draw on"], "leverages": ["uses", "draws on"], "leveraging": ["using", "drawing on"],
    "numerous": ["many", "several"], "facilitate": ["help", "support"],
    "facilitates": ["helps", "supports"], "demonstrate": ["show"], "demonstrates": ["shows"],
    "demonstrated": ["showed"], "subsequently": ["later", "then"], "additionally": ["also"],
    "approximately": ["about", "roughly"], "in order to": ["to"],
    "due to the fact that": ["because"], "a number of": ["several", "a few"],
    "is able to": ["can"], "are able to": ["can"], "prior to": ["before"],
    "with regard to": ["regarding", "about"], "in the event that": ["if"],
    "a majority of": ["most"], "in close proximity to": ["near"],
    "indicate": ["show", "suggest"], "indicates": ["shows", "suggests"],
    "significant": ["notable", "considerable"], "significantly": ["notably", "considerably"],
    "obtain": ["get"], "obtained": ["got", "gathered"], "endeavor": ["try"],
    "commence": ["begin", "start"], "commenced": ["began", "started"],
    "terminate": ["end"], "ascertain": ["determine", "find out"],
    "various": ["several", "a range of"],
    "in conclusion": ["overall", "to sum up"], "on the other hand": ["conversely", "by contrast"],
    "it cannot be denied that": ["clearly"], "shed light on": ["clarify", "explain"],
    "in the realm of": ["in"], "cutting-edge": ["recent", "advanced"],
    "garner": ["attract", "gain"], "garnered": ["attracted", "gained"],
    "embark on": ["begin", "start"], "navigate the complexities of": ["address", "work through"],
    "stands as a testament to": ["reflects", "confirms"],
    "serves as a": ["acts as a", "functions as a"],
    # Academic/dissertation-prose vocabulary — same part-of-speech and same
    # verb form/tense as the source word in every entry, so chained swaps
    # (a value that is itself a key elsewhere) never change grammatical role.
    "examine": ["discuss", "investigate", "look at"], "examines": ["discusses", "investigates", "looks at"],
    "examined": ["discussed", "investigated", "looked at"], "examining": ["discussing", "investigating", "looking at"],
    "analyse": ["examine", "investigate"], "analyses": ["examines", "investigates"],
    "analysed": ["examined", "investigated"], "analysing": ["examining", "investigating"],
    "analyze": ["examine", "investigate"], "analyzes": ["examines", "investigates"],
    "analyzed": ["examined", "investigated"], "analyzing": ["examining", "investigating"],
    "motivate": ["inspire", "drive"], "motivates": ["inspires", "drives"], "motivated": ["inspired", "driven"],
    "understand": ["learn about", "grasp"], "understands": ["learns about", "grasps"],
    "understood": ["learned about", "grasped"],
    "seek": ["aim", "set out"], "seeks": ["aims", "sets out"], "sought": ["aimed", "set out"],
    "adopt": ["use", "apply"], "adopts": ["uses", "applies"],
    "adopted": ["used", "applied"], "adopting": ["using", "applying"],
    "collect": ["gather", "compile"], "collects": ["gathers", "compiles"],
    "collected": ["gathered", "compiled"], "collecting": ["gathering", "compiling"],
    "reveal": ["find", "show"], "reveals": ["finds", "shows"],
    "revealed": ["found", "showed"], "revealing": ["finding", "showing"],
    "establish": ["determine", "find"], "establishes": ["determines", "finds"],
    "established": ["determined", "found"],
    "promote": ["facilitate", "support", "encourage"], "promotes": ["facilitates", "supports", "encourages"],
    "promoted": ["facilitated", "supported", "encouraged"],
    "identify": ["find", "note", "flag"], "identifies": ["finds", "notes", "flags"],
    "identified": ["found", "noted", "flagged"],
    "contributes to": ["plays a role in", "helps drive"], "contribute to": ["play a role in", "help drive"],
    "impacts": ["affects", "influences"], "affects": ["influences", "shapes"],
    "need for": ["necessity for", "requirement for"],
    "enterprise": ["business", "firm"], "enterprises": ["businesses", "firms"],
    "investment": ["capital", "funding"], "investments": ["capital", "funds"],
    "inflow": ["influx"], "inflows": ["influxes"],
    "extent": ["degree"],
    "technique": ["method", "approach"], "techniques": ["methods", "approaches"],
    "increasing": ["growing", "rising"],
    "competition": ["rivalry"],
}
_SYNONYM_RES = sorted(_SYNONYM_MAP.keys(), key=len, reverse=True)


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _lexical_substitute(text: str, rng: random.Random) -> str:
    """Swap predictable high-frequency words for lower-frequency synonyms."""
    for key in _SYNONYM_RES:
        options = _SYNONYM_MAP[key]
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)

        def _sub(m: re.Match, options: list[str] = options) -> str:
            choice = rng.choice(options)
            return _match_case(m.group(), choice)

        text = pattern.sub(_sub, text)
    return text


# ── Pass 3: sentence-length restructuring (burstiness) ─────────────────────
_CLAUSE_SPLIT_RE = re.compile(r",\s+(and|but|which|so|because)\s+", re.IGNORECASE)


def _restructure_for_burstiness(text: str, rng: random.Random) -> str:
    """
    Split unusually long sentences and merge unusually short adjacent ones so
    sentence-length variance rises — flat, uniform length is itself an AI
    fingerprint (low burstiness), independent of word choice.
    """
    paragraphs = text.split("\n")
    out_paragraphs = []
    for para in paragraphs:
        if not para.strip():
            out_paragraphs.append(para)
            continue
        sentences = _SENT_RE.split(para.strip())
        rebuilt: list[str] = []
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            words = sent.split()
            # Split long/medium sentences (> 20 words) at a clause boundary,
            # falling back to the first comma if no conjunction is present.
            if len(words) > 20:
                m = _CLAUSE_SPLIT_RE.search(sent)
                cut = m.start() if m else None
                if cut is None:
                    comma = sent.find(",", 15)
                    if comma != -1:
                        cut = comma
                if cut is not None and rng.random() < 0.85:
                    first = sent[:cut].strip().rstrip(",") + "."
                    rest = sent[cut:].strip()
                    rest = re.sub(r"^,\s*(?:(?:and|but|so|which|while)\s+)?", "", rest, flags=re.IGNORECASE)
                    rest = rest[:1].upper() + rest[1:]
                    rebuilt.append(first)
                    rebuilt.append(rest)
                    i += 1
                    continue
            # Merge two consecutive short sentences (< 11 words) into one.
            if (
                len(words) < 11
                and i + 1 < len(sentences)
                and len(sentences[i + 1].split()) < 16
                and rng.random() < 0.6
            ):
                nxt = sentences[i + 1]
                joiner = rng.choice([", and", ";", ", while"])
                merged = sent.rstrip(".") + joiner + " " + nxt[:1].lower() + nxt[1:]
                rebuilt.append(merged)
                i += 2
                continue
            rebuilt.append(sent)
            i += 1

        # Final pass: if every sentence in the paragraph clusters in a narrow
        # length band (flat = AI-like), force-split the longest one at its
        # first comma to inject a short, punchy fragment.
        lens = [len(s.split()) for s in rebuilt]
        if len(rebuilt) >= 2 and lens and (max(lens) - min(lens)) < 7:
            longest_i = max(range(len(rebuilt)), key=lambda idx: lens[idx])
            longest = rebuilt[longest_i]
            comma = longest.find(",", 10)
            if comma != -1:
                first = longest[:comma].strip().rstrip(",") + "."
                rest = longest[comma:].strip()
                rest = re.sub(r"^,\s*(?:(?:and|but|so|which|while)\s+)?", "", rest, flags=re.IGNORECASE)
                rest = rest[:1].upper() + rest[1:]
                rebuilt[longest_i:longest_i + 1] = [first, rest]

        out_paragraphs.append(" ".join(rebuilt))
    return "\n".join(out_paragraphs)


def rule_based_humanise(text: str, seed: str | None = None) -> str:
    """
    StealthWriter-style multi-pass rewrite, applied without any LLM:
      1. Replace known AI cliché phrases with plainer alternatives.
      1b. Apply curated clause/nominalisation restructuring templates.
      2. Swap predictable high-frequency words for lower-frequency synonyms
         (raises perplexity).
      3. Restructure sentence lengths — split long uniform sentences, merge
         short uniform ones (raises burstiness).
    `seed` makes synonym/restructuring choices deterministic per-call while
    still varying between different input texts.
    """
    rng = random.Random(seed or text[:200])

    result = text
    for pattern, repl in _HUMANISE_RULES:
        result = pattern.sub(repl, result)
    # Collapse accidental double-spaces
    result = re.sub(r"  +", " ", result)
    # Collapse duplicate connectors left by stacked substitutions
    # (e.g. "Furthermore, it is evident that" -> "Also, " + "Clearly," -> "Also, Clearly,")
    result = re.sub(
        r"\b(?:Also|And|So|Overall),\s+(?=(?:Also|And|So|Clearly|Obviously|Overall),)",
        "",
        result,
    )
    # Remove stray leading commas after blank substitutions
    result = re.sub(r"\.\s+,\s+", ". ", result)
    result = re.sub(r"^\s*,\s*", "", result, flags=re.MULTILINE)

    result = _restructure_clauses(result, rng)
    result = _lexical_substitute(result, rng)
    result = _restructure_for_burstiness(result, rng)

    return result.strip()


# ── Academic writing quality checker ──────────────────────────────────────────

_WEAK_WORDS = frozenset([
    "very", "quite", "rather", "somewhat", "stuff", "things", "basically",
    "pretty", "really", "nice", "fine", "big", "small", "good", "bad",
    "lots", "a lot", "got", "get", "kind of", "sort of",
])

_EVIDENCE_CUE_RE = re.compile(
    r"\b(?:according to|as (?:noted|argued|shown|stated|demonstrated|found) by|"
    r"studies (?:show|suggest|indicate|demonstrate)|research (?:shows?|suggests?|finds?|indicates?)|"
    r"\(\d{4}\)|\[\d+\]|et al\b|cited in|evidence (?:shows?|suggests?)|"
    r"(?:scholars?|researchers?|authors?) (?:argue|suggest|note|claim|find|report))\b",
    re.IGNORECASE,
)

_INFORMAL_FP_RE = re.compile(
    r"\bI (?:think|feel|believe|know|want|am\b|don't|didn't|cannot|can't)\b",
    re.IGNORECASE,
)


def academic_quality_check(text: str) -> dict[str, Any]:
    """
    Rule-based academic writing quality analysis — no LLM required.
    Returns a score (0–100), verdict, and a list of actionable issues.
    """
    from collections import Counter

    if not (text or "").strip():
        return {"quality_score": 0, "verdict": "insufficient_text", "issues": [], "word_count": 0}

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return {"quality_score": 0, "verdict": "insufficient_text", "issues": [], "word_count": len(text.split())}

    words_lower = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words_lower)
    issues: list[dict[str, Any]] = []

    # 1. Weak / vague language
    weak_hits = [w for w in words_lower if w in _WEAK_WORDS]
    if len(weak_hits) > 3:
        sample = ", ".join(f'"{w}"' for w in dict.fromkeys(weak_hits[:5]))
        issues.append({
            "type": "weak_language",
            "severity": "medium",
            "message": (
                f"Found {len(weak_hits)} vague or informal words ({sample}). "
                "Replace with precise academic vocabulary (e.g. 'very large' → 'substantial')."
            ),
        })

    # 2. Passive voice overuse
    passive_count = sum(1 for s in sentences if _PASSIVE_RE.search(s))
    passive_ratio = passive_count / len(sentences)
    if passive_ratio > 0.5:
        issues.append({
            "type": "passive_overuse",
            "severity": "medium",
            "message": (
                f"{int(passive_ratio * 100)}% of sentences use passive voice. "
                "Prefer active constructions for clarity ('The study found…' not 'It was found that…')."
            ),
        })

    # 3. Underdeveloped paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 10]
    if paragraphs:
        thin = sum(1 for p in paragraphs if len(p.split()) < 40)
        if thin > max(1, len(paragraphs) * 0.4):
            issues.append({
                "type": "thin_paragraphs",
                "severity": "medium",
                "message": (
                    f"{thin} paragraph(s) appear underdeveloped (< 40 words). "
                    "Build full PEEL paragraphs: Point → Evidence → Explanation → Link."
                ),
            })

    # 4. Repetitive sentence starters
    starters = [s.split()[0].lower().rstrip(",.;:") for s in sentences if s.split()]
    starter_counts = Counter(starters)
    repeated = [(w, c) for w, c in starter_counts.items() if c >= 3 and w not in {"the", "a", "an", "this", "these"}]
    if repeated:
        reps = ", ".join(f'"{w}" (×{c})' for w, c in sorted(repeated, key=lambda x: -x[1])[:3])
        issues.append({
            "type": "repetitive_starters",
            "severity": "low",
            "message": f"Repeated sentence openers detected: {reps}. Vary structure to improve flow and readability.",
        })

    # 5. Missing evidence / citation cues
    evidence_count = len(_EVIDENCE_CUE_RE.findall(text))
    if word_count > 150 and evidence_count < 2:
        issues.append({
            "type": "missing_evidence",
            "severity": "high",
            "message": (
                "No citation or evidence cues detected. Academic writing requires referenced support — "
                "add citations (Author, Year) or attribution phrases ('Research by X indicates…')."
            ),
        })

    # 6. Hedge word overuse
    hedge_count = sum(1 for w in words_lower if w in _HEDGE_WORDS)
    hedge_ratio = hedge_count / max(word_count, 1)
    if hedge_ratio > 0.06:
        issues.append({
            "type": "hedge_overuse",
            "severity": "low",
            "message": (
                f"High density of filler connectors ({int(hedge_ratio * 100)}% of words). "
                "Words like 'furthermore', 'additionally', 'essentially' inflate length without adding meaning."
            ),
        })

    # 7. Informal first-person
    fp_matches = _INFORMAL_FP_RE.findall(text)
    if len(fp_matches) > 2:
        issues.append({
            "type": "informal_register",
            "severity": "low",
            "message": (
                f"Found {len(fp_matches)} informal first-person phrases (e.g. 'I think', 'I feel'). "
                "In formal academic writing prefer 'This study argues…' or 'The evidence suggests…'."
            ),
        })

    # 8. Sentence length monotony (all sentences similar length)
    sent_lengths = [len(s.split()) for s in sentences]
    if len(sent_lengths) >= 5:
        mean_len = statistics.mean(sent_lengths)
        std_len = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0
        cv = std_len / mean_len if mean_len > 0 else 0
        if cv < 0.25:
            issues.append({
                "type": "monotonous_rhythm",
                "severity": "low",
                "message": (
                    "Sentence lengths are very uniform — the text may feel monotonous. "
                    "Mix short, punchy sentences with longer analytical ones for better rhythm."
                ),
            })

    # Compute overall quality score
    severity_weights = {"high": 25, "medium": 15, "low": 7}
    penalty = sum(severity_weights.get(i["severity"], 10) for i in issues)
    score = max(0, 100 - penalty)

    verdict = (
        "strong" if score >= 80
        else "adequate" if score >= 60
        else "needs_improvement" if score >= 40
        else "poor"
    )

    return {
        "quality_score": score,
        "verdict": verdict,
        "word_count": word_count,
        "sentence_count": len(sentences),
        "passive_ratio": round(passive_ratio, 2),
        "evidence_cues": evidence_count,
        "issues": issues,
    }
