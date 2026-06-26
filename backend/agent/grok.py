"""Grok API wrappers for the autonomous document agent.
Reads GROK_API_KEY from Django settings (loaded from .env).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests
from django.conf import settings

logger = logging.getLogger(__name__)
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = getattr(settings, "GROK_MODEL", "llama-3.3-70b-versatile")


def get_model_label() -> str:
    return "Grok"


def _strip_fences(text: str) -> str:
    text = text.strip()
    # Extract the first JSON object or array from inside a code fence
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if m:
        return m.group(1).strip()
    # No fence - return as-is so json.loads can try it directly
    return text


def _parse_json(text: str) -> Any:
    return json.loads(_strip_fences(text))


def _request_text(prompt: str) -> str:
    api_key = getattr(settings, "GROK_API_KEY", "")
    if not api_key:
        raise ValueError("GROK_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Empty response from Grok API")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def generate_text(prompt: str) -> str:
    return _request_text(prompt)


def _json_response(prompt: str) -> Any:
    text = _request_text(
        prompt
        + "\n\nIMPORTANT: Respond with valid JSON ONLY. No markdown, no extra text."
    )
    return _parse_json(text)


def classify_intent(message: str, doc_context: str) -> dict[str, Any]:
    prompt = f"""You are an AI document assistant. Classify the user's intent.

Document excerpt:
{doc_context[:4000]}

User message: \"{message}\"

Choose ONE intent:
- enhance_document
- enhance_section
- humanise_ai_sections
- reduce_plagiarism_similarity
- write_section
- write_dissertation
- write_document
- create_outline
- add_chart
- add_image
- search_image
- generate_ai_image
- check_academic_quality
- export_bibtex
- chat

Guidance:
- If user asks to find/search for a REAL photo/image/picture ("real photo of X", "find an image of X", "search the web/internet for a photo of X", a Wikimedia/Commons/stock photo) -> search_image.
- If user asks to GENERATE/CREATE an AI/photorealistic/DALL-E image (not a diagram/chart) -> generate_ai_image.
- If user says "humanise", "humanize", "make it sound human", "remove AI", "bypass AI detection", "make less AI", "sound more natural", "rewrite AI passages", "human-like" -> humanise_ai_sections.
- If user says "reduce similarity", "reduce plagiarism", "fix plagiarism", "remove plagiarism", "lower the plagiarism", "make this original", "rewrite the plagiarised content", "de-plagiarise" -> reduce_plagiarism_similarity. Note this is different from just asking to "check" or "scan for" plagiarism, which is a read-only request and should be classified as chat.
- If user says "correct", "fix", "improve" for a specific part -> enhance_section.
- If user says "improve 2.7" or "fix 3.4" (subsection number) -> enhance_section with that exact number as target_section.
- If user says "redo chapter X" or "rewrite chapter X" -> write_section with target_section.
- CRITICAL: "improve 2.7", "fix section 2.7", "enhance 3.4" mean improve ONLY that subsection — set intent=enhance_section and target_section="2.7". Do NOT set intent=write_dissertation or write_section.
- If user says "write full dissertation", "write thesis", "write project on <topic>", "full dissertation", "complete thesis" -> write_dissertation.
- If user asks to write ANY kind of document (article, report, assignment, essay, paper, presentation, proposal, case study, etc.) -> write_document. The AI will decide the structure.
- If user asks for a full/complete/entire project with chapters -> write_dissertation.
- If user asks to generate substantial new document content, do NOT return chat; use write_document.
- If user asks about "academic quality", "writing quality", "check writing", "writing check" -> check_academic_quality.
- If user asks for "the bibtex", "bib file", "references file", "export references/bibliography/citations", "what sources did you use/cite" -> export_bibtex.
- IMPORTANT: "explain X", "what is X", "what are X", "describe X", "how does X work", "tell me about X", "define X" are ALL chat — do NOT classify these as any write intent.
- IMPORTANT: Any message that ends with "?" is a question and should be classified as chat.
- IMPORTANT: Only classify as write_* if the user is explicitly asking to ADD or CHANGE content IN the document.

Return JSON exactly:
{{"intent": "<intent>", "target_section": "<section name or null>", "topic": "<main topic or null>"}}"""
    try:
        result = _json_response(prompt)
        if isinstance(result, dict) and "intent" in result:
            return {
                "intent": result.get("intent", "chat"),
                "target_section": result.get("target_section"),
                "topic": result.get("topic"),
            }
    except Exception as exc:
        logger.warning("grok classify_intent failed: %s", exc)
    return {"intent": "chat", "target_section": None, "topic": None}


def chat_with_document(message: str, doc_context: str) -> str:
    prompt = (
        "You are an expert academic writing assistant embedded in a word processor.\n"
        f"Document:\n{doc_context[:15000]}\n\n"
        f"User: {message}\n\n"
        "Give a helpful, direct response. Be concise and professional."
    )
    return generate_text(prompt)


def humanise_text(text: str, topic: str, ai_phrases: list[str] | None = None) -> str:
    """Rewrite text to reduce AI-detection signals: remove stock phrases, add burstiness."""
    phrase_list = ""
    if ai_phrases:
        phrase_list = (
            "\n\nAvoid or replace these detected AI phrases:\n"
            + "\n".join(f"  • {p}" for p in ai_phrases[:12])
        )
    prompt = (
        f"Rewrite the following academic text on '{topic}' so it defeats AI-detection tools "
        "(GPTZero, Turnitin, Originality.ai) the way a professional humaniser like StealthWriter "
        "does — by attacking the two signals those detectors score: low perplexity (predictable "
        "word choice) and low burstiness (uniform sentence length/structure).\n\n"
        "Do this as three internal passes, then output only the final result:\n"
        "PASS 1 — Paraphrase: restructure clauses and sentence order so the wording isn't a "
        "near-copy of the source, while preserving every fact, figure, and citation exactly.\n"
        "PASS 2 — Raise perplexity: replace common, predictable words/phrases with precise but "
        "less-expected synonyms a human writer would reach for; avoid the most statistically "
        "obvious next word at each point.\n"
        "PASS 3 — Raise burstiness: deliberately vary sentence length — alternate short, blunt "
        "sentences with longer, multi-clause ones. LLM output is unnaturally uniform in length; "
        "uneven rhythm is the strongest human signal.\n\n"
        "Additional rules:\n"
        "- Use concrete details and first-person voice where natural ('I found', 'The results showed').\n"
        "- REMOVE these AI clichés: 'it is important to note', 'in today's world', "
        "'holistic approach', 'paradigm shift', 'delve into', 'comprehensive overview', "
        "'it is crucial', 'furthermore', 'moreover', 'it is evident that', 'the realm of', "
        "'seamlessly', 'robust framework', 'plays a crucial role', 'underscores the importance'.\n"
        "- Prefer active voice over passive; vary sentence openers (don't start consecutive "
        "sentences with the same word or transition).\n"
        "- Show genuine perspective — hedges like 'I think', 'It seemed', 'Surprisingly' feel human.\n"
        "- Keep ALL facts, data, citations, and the core argument intact — do not invent content.\n"
        f"{phrase_list}\n\n"
        f"TEXT:\n{text[:4000]}\n\n"
        "Return ONLY the rewritten text. No explanations, no markdown, no pass labels."
    )
    return generate_text(prompt)


def _classify_section_type(hint: str) -> str:
    h = hint.lower()
    if any(k in h for k in ["abstract", "executive summary"]):
        return "abstract"
    if any(k in h for k in ["introduction", "background", "problem statement", "objective", "scope", "significance"]):
        return "introduction"
    if any(k in h for k in ["literature", "theoretical", "conceptual", "empirical", "related work", "review"]):
        return "literature_review"
    if any(k in h for k in ["methodology", "research design", "sampling", "data collection", "validity", "reliability"]):
        return "methodology"
    if any(k in h for k in ["result", "finding", "analysis", "data present"]):
        return "results"
    if any(k in h for k in ["discussion", "interpretation"]):
        return "discussion"
    if any(k in h for k in ["conclusion", "recommendation", "limitation", "future research"]):
        return "conclusion"
    return "general"


_AI_CLICHE_PHRASES = (
    "'in today's world', 'in this day and age', 'since time immemorial', 'it is important/crucial to note', "
    "'it should be noted that', 'needless to say', 'delve into', 'navigate the complexities of', "
    "'in the realm of', 'a testament to', 'this section will discuss/explore', 'as previously mentioned'"
)

_SECTION_GUIDES: dict[str, str] = {
    "abstract": (
        "Section type: ABSTRACT\n"
        "• Write as a SINGLE dense paragraph — no sub-headings, no bullet points, no citations.\n"
        "• Cover, in order: context/problem, purpose/objectives, method, key finding(s), and the study's contribution.\n"
        "• Every sentence must carry information — no throat-clearing or background padding.\n"
        "• Use past tense for what was done and found; present tense only for the study's significance.\n"
        "• Keep it self-contained — a reader should understand the whole study from this paragraph alone."
    ),
    "introduction": (
        "Section type: INTRODUCTION\n"
        "• Open with a precise, contextualised problem statement — not a generic claim.\n"
        "• Define the research gap clearly: what is unknown or contested?\n"
        "• State objectives/research questions in specific, measurable language.\n"
        "• Signal scope: what is included and what is excluded.\n"
        "• Use signpost phrases: 'This study investigates…', 'The aim is to…'.\n"
        "• Avoid clichés: 'In today's world', 'Since time immemorial', 'It is a fact that'."
    ),
    "literature_review": (
        "Section type: LITERATURE REVIEW\n"
        "• Synthesise sources thematically — do NOT merely summarise each source in sequence.\n"
        "• Use attribution: 'According to Smith (2020)…', 'As argued by Jones (2019)…'.\n"
        "• Compare and contrast positions: highlight agreements, contradictions, and gaps.\n"
        "• Conclude with a clear statement of the gap this study fills.\n"
        "• Use hedging appropriately: 'suggests', 'indicates', 'argues' — not 'proves'.\n"
        "• Vary sentence openings — avoid starting every sentence with an author name."
    ),
    "methodology": (
        "Section type: METHODOLOGY\n"
        "• Justify every methodological choice with explicit reasoning (WHY, not just WHAT).\n"
        "• Follow logical order: design → population → sampling → instruments → procedures → analysis.\n"
        "• Be precise: sample sizes, instrument names, timeframes, software used.\n"
        "• Address validity and reliability explicitly.\n"
        "• Cite established frameworks where appropriate (e.g. Creswell, 2014; Bryman, 2016).\n"
        "• Use past tense for completed procedures."
    ),
    "results": (
        "Section type: RESULTS / FINDINGS\n"
        "• Report findings objectively — describe data, do not interpret here.\n"
        "• Reference tables and figures precisely ('As shown in Table 3…').\n"
        "• Use precise quantitative language appropriate to the data type — survey/perception data "
        "('78% of respondents…', 'mean = 4.2, SD = 0.8'), or technical/experimental data "
        "('94.2% accuracy', 'latency = 12ms', 'throughput improved by 18%'). Match the language to "
        "what this specific study actually measured — do not assume a human survey if there isn't one.\n"
        "• Organise by research question or theme, not chronologically.\n"
        "• Do NOT introduce new literature or sweeping interpretation here.\n"
        "• Use past tense for findings; present tense for tables and figures."
    ),
    "discussion": (
        "Section type: DISCUSSION\n"
        "• Open with the most significant finding, then explain its importance.\n"
        "• Link every major finding to the literature — agree, contrast, or extend prior work.\n"
        "• Use hedged interpretation: 'This suggests…', 'These results may indicate…'.\n"
        "• Address unexpected or contradictory findings candidly.\n"
        "• Show how findings advance the field — avoid over-claiming.\n"
        "• Do not introduce new data."
    ),
    "conclusion": (
        "Section type: CONCLUSION\n"
        "• Synthesise key findings — do NOT simply restate them.\n"
        "• Show how the study met its research objectives.\n"
        "• Acknowledge limitations with professional candour.\n"
        "• Offer specific, actionable recommendations (practice, policy, future research).\n"
        "• End with a strong closing statement on the study's contribution.\n"
        "• Do NOT introduce new information."
    ),
    "general": (
        "Academic writing standards:\n"
        "• Use precise, formal vocabulary — replace vague words (very, quite, things, basically) "
        "with specific academic terms.\n"
        "• Build PEEL paragraphs: Point → Evidence → Explanation → Link.\n"
        "• Vary sentence length — mix short analytical punches with longer elaborations.\n"
        "• Prefer active voice; use passive only when the agent is unknown or unimportant.\n"
        "• Each paragraph needs a clear topic sentence and logical internal progression.\n"
        "• Use precise transitions: 'However', 'In contrast', 'Building on this', 'Consequently'.\n"
        "• Do not open consecutive paragraphs with the same word or phrase "
        "(e.g. two paragraphs both starting with 'Moreover' or 'Furthermore').\n"
        "• Ground every analytical claim in a concrete example, statistic, or citation — "
        "never leave a claim as an unsupported generalisation.\n"
        f"• Remove filler and AI clichés: {_AI_CLICHE_PHRASES}."
    ),
}


def enhance_text(text: str, topic: str, instruction: str = "", section_title: str = "") -> str:
    section_type = _classify_section_type(section_title or instruction or "")
    guide = _SECTION_GUIDES.get(section_type, _SECTION_GUIDES["general"])
    instr_block = ("USER INSTRUCTION: " + instruction.strip() + "\n\n") if instruction.strip() else ""
    prompt = (
        f"You are an expert academic editor improving a dissertation on '{topic}'.\n\n"
        f"{guide}\n\n"
        f"{instr_block}"
        "EDITING TASKS — apply ALL of the following:\n"
        "1. Strengthen argument — every claim must be supported or clearly signposted.\n"
        "2. Improve sentence variety — break monotonous rhythms, vary length and structure.\n"
        "3. Sharpen vocabulary — replace vague words with precise academic terms.\n"
        "4. Fix transitions — each sentence must flow logically from the previous.\n"
        "5. Remove redundancy — cut phrases that add no meaning.\n"
        f"6. Remove AI clichés: {_AI_CLICHE_PHRASES}.\n"
        "7. Preserve ALL factual content, data, citations, and specific details.\n\n"
        f"TEXT TO IMPROVE:\n{text[:3500]}\n\n"
        "Return ONLY the improved text. No explanations, no markdown, no headings."
    )
    return generate_text(prompt)


def generate_section_content(
    title: str,
    topic: str,
    context: str = "",
    word_count: int = 220,
) -> str:
    section_type = _classify_section_type(title)
    guide = _SECTION_GUIDES.get(section_type, _SECTION_GUIDES["general"])
    ctx_block = ("Context and instructions:\n" + context[:3000] + "\n\n") if context else ""
    prompt = (
        f"You are writing a formal academic dissertation section.\n"
        f"Section: '{title}'\n"
        f"Study topic: '{topic}'\n\n"
        f"{guide}\n\n"
        f"{ctx_block}"
        f"Write ~{word_count} words of scholarly content for this section.\n"
        "Requirements:\n"
        "• Write the actual content this heading promises — never describe, summarize, or "
        "refer to the section/chapter itself. Banned: 'this section discusses/covers/examines', "
        "'this chapter discusses', 'as discussed/mentioned/noted above', 'the above discussion', "
        "'the foregoing analysis', or any sentence that recaps a heading instead of delivering "
        "its content. A reader must never catch you talking ABOUT the writing instead of writing.\n"
        "• Write SPECIFICALLY about this topic — not generic filler that fits any study.\n"
        "• Ground claims in the context provided (objectives, methodology, findings).\n"
        "• Use varied sentence rhythm — short punchy sentences alongside longer analyses.\n"
        "• Include precise claims, concrete examples, and academic evidence language.\n"
        "• Do NOT include the section heading in your response.\n"
        "• Do NOT use HTML tags — plain text with blank lines between paragraphs.\n"
        f"• Avoid AI clichés: {_AI_CLICHE_PHRASES}.\n"
        "• Do not open consecutive paragraphs with the same word or phrase.\n"
        f"• Aim for ~{word_count} words — do not stop early.\n\n"
        "Begin writing the substantive content now — no preamble, no meta-commentary:"
    )
    return generate_text(prompt)


def generate_outline_sections(topic: str) -> list[dict[str, Any]]:
    prompt = f"""Generate a complete dissertation chapter outline for: '{topic}'.

Return a JSON array with 8 items. Each item must have:
- title: string
- subsections: array of strings

Example item:
{{"title": "Chapter 1: Introduction", "subsections": ["1.1 Background", "1.2 Problem Statement"]}}

Include these major chapters:
Introduction, Literature Review, Methodology, Results and Analysis,
Discussion, Conclusion, References, Appendices."""
    try:
        result = _json_response(prompt)
        if isinstance(result, list) and result:
            normalized: list[dict[str, Any]] = []
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    normalized.append(
                        {
                            "title": str(item.get("title") or f"Chapter {i + 1}"),
                            "subsections": item.get("subsections", []),
                        }
                    )
                elif isinstance(item, str):
                    normalized.append({"title": item, "subsections": []})
            if normalized:
                return normalized[:8]
    except Exception as exc:
        logger.warning("grok generate_outline_sections failed: %s", exc)

    defaults = [
        "Chapter 1: Introduction",
        "Chapter 2: Literature Review",
        "Chapter 3: Methodology",
        "Chapter 4: Results and Analysis",
        "Chapter 5: Discussion",
        "Chapter 6: Conclusion",
        "References",
        "Appendices",
    ]
    return [{"title": title, "subsections": []} for title in defaults]
