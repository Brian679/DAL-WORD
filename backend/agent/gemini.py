"""Gemini API wrappers for the autonomous document agent.
Reads GEMINI_API_KEY from Django settings (loaded from .env).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import google.generativeai as genai
from django.conf import settings

logger = logging.getLogger(__name__)
MODEL_NAME = "gemini-1.5-flash"


def _model() -> genai.GenerativeModel:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def get_model_label() -> str:
    return "Gemini 1.5 Flash"


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


def generate_text(prompt: str) -> str:
    response = _model().generate_content(prompt)
    return (response.text or "").strip()


def _json_response(prompt: str) -> Any:
    response = _model().generate_content(
        prompt
        + "\n\nIMPORTANT: Respond with valid JSON ONLY. No markdown, no extra text."
    )
    return _parse_json(response.text or "")


def classify_intent(message: str, doc_context: str) -> dict[str, Any]:
    """Classify a user message into one autonomous action intent."""
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
- check_academic_quality
- export_bibtex
- chat

Guidance:
- If user says "humanise", "humanize", "make it sound human", "remove AI", "bypass AI detection", "make less AI", "sound more natural", "rewrite AI passages", "human-like" -> humanise_ai_sections.
- If user says "reduce similarity", "reduce plagiarism", "fix plagiarism", "remove plagiarism", "lower the plagiarism", "make this original", "rewrite the plagiarised content", "de-plagiarise" -> reduce_plagiarism_similarity. Note this is different from just asking to "check" or "scan for" plagiarism, which is a read-only request and should be classified as chat.
- If user says "correct", "fix", "improve" for a specific part -> enhance_section.
- If user says "improve 2.7" or "fix 3.4" (subsection number) -> enhance_section with that exact number as target_section.
- If user says "redo chapter X" or "rewrite chapter X" -> write_section with target_section.
- CRITICAL: "improve 2.7", "fix section 2.7", "enhance 3.4" mean improve ONLY that subsection — set intent=enhance_section and target_section="2.7" (the number). Do NOT set intent=write_dissertation or write_section.
- If user says "write full dissertation", "write thesis", "write project on <topic>", "full dissertation", "complete thesis" -> write_dissertation.
- If user asks to write ANY kind of document (article, report, assignment, essay, paper, presentation, proposal, case study, brief, plan, etc.) -> write_document. The AI planner will decide the structure, length, and to-do list automatically.
- If user asks for a full/complete/entire project with multiple chapters -> write_dissertation.
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
        logger.warning("classify_intent failed: %s", exc)
    return {"intent": "chat", "target_section": None, "topic": None}


def create_execution_plan(intent: str) -> list[str]:
    plans = {
        "humanise_ai_sections": [
            "Running AI content detection across all sections",
            "Identifying AI-detected sentences (high perplexity / low burstiness)",
            "Rewriting flagged passages with authentic human voice",
            "Varying sentence length and structure for natural burstiness",
            "Saving humanised document",
        ],
        "reduce_plagiarism_similarity": [
            "Scanning document for matched/similar passages",
            "Comparing against other documents in the workspace",
            "Rewriting flagged passages to reduce textual overlap",
            "Re-checking similarity after rewrite",
            "Saving updated document",
        ],
        "enhance_document": [
            "Reading and analysing document structure",
            "Identifying weak sections",
            "Enhancing each section for clarity and academic tone",
            "Checking language flow and consistency",
            "Saving updated document",
        ],
        "enhance_section": [
            "Locating target section",
            "Analysing existing content",
            "Rewriting with improved clarity and academic tone",
            "Saving updated section",
        ],
        "write_section": [
            "Locating or creating target section",
            "Generating academic content",
            "Inserting content into document",
            "Saving changes",
        ],
        "write_dissertation": [
            "Generating dissertation outline",
            "Writing Introduction",
            "Writing Literature Review",
            "Writing Methodology",
            "Writing Results and Analysis",
            "Writing Discussion",
            "Writing Conclusion",
            "Saving complete dissertation",
        ],
        "create_outline": [
            "Generating chapter structure for topic",
            "Creating section entries",
            "Saving outline to document",
        ],
        "write_document": [
            "Reading your request and planning the document",
            "AI designing section structure and word targets",
            "Writing each section in sequence",
            "Saving complete document",
        ],
        # Legacy aliases — route through write_document at runtime
        "write_report": ["Planning and writing report based on your request"],
        "write_assignment": ["Planning and writing assignment based on your request"],
        "write_presentation": ["Planning and writing presentation based on your request"],
        "write_spreadsheet": ["Planning and writing spreadsheet based on your request"],
        "write_article": ["Planning and writing article based on your request"],
        "check_academic_quality": [
            "Scanning document for academic writing quality",
            "Checking vocabulary, evidence, and argument strength",
            "Generating section-by-section quality report",
        ],
        "export_bibtex": [
            "Reading the document's saved bibliography",
            "Rendering BibTeX entries",
        ],
        "add_chart": [
            "Locating target section",
            "Generating chart from data",
            "Inserting chart into section",
            "Saving document",
        ],
        "add_image": [
            "Locating target section",
            "Generating image",
            "Inserting image into section",
            "Saving document",
        ],
        "address_comments": [
            "Scanning document for reviewer comments",
            "Reading each annotated section",
            "Generating revisions to address comments",
            "Applying changes to document",
            "Saving updated document",
        ],
        "chat": [
            "Reading document context",
            "Understanding user intent",
            "Reviewing relevant document sections",
            "Drafting response strategy",
            "Generating response",
            "Final quality pass",
            "Returning chat summary",
        ],
    }
    return plans.get(intent, ["Analysing request", "Executing", "Saving"])


def chat_with_document(message: str, doc_context: str) -> str:
    prompt = (
        "You are an expert academic writing assistant embedded in a word processor.\n"
        f"Document:\n{doc_context[:15000]}\n\n"
        f"Request: {message}\n\n"
        "Instructions:\n"
        "- Give a direct, concise response (no more than 10 sentences).\n"
        "- Do NOT ask for more information or say 'please provide'.\n"
        "- Do NOT repeat or echo the request text.\n"
        "- Do NOT include labels like 'User:' or 'Assistant:'.\n"
        "- Do NOT output prompt templates unless the user explicitly asked for one.\n"
        "- Write the answer immediately, starting with the actual content."
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
        f"You are an expert academic editor helping a student humanise their writing on '{topic}' "
        "so it passes AI-detection tools (GPTZero, Turnitin, Originality.ai), using the same "
        "approach as professional humanisers like StealthWriter: attack perplexity and burstiness, "
        "the two signals those detectors score, not just surface phrasing.\n\n"
        "Work through three internal passes, then output only the final rewritten text:\n"
        "PASS 1 — Paraphrase: re-order clauses and restructure sentences so the result isn't a "
        "near-copy of the source, while keeping every fact, figure, and citation exact.\n"
        "PASS 2 — Raise perplexity: swap common, highly predictable words for precise but less "
        "obvious synonyms a human writer would choose; avoid the statistically expected word.\n"
        "PASS 3 — Raise burstiness: vary sentence length deliberately. "
        "   Abrupt stops. Then a longer, more reflective sentence that develops the idea further. "
        "LLM text is unnaturally uniform in length — uneven rhythm is the strongest human signal.\n\n"
        "Additional rules:\n"
        "- Use concrete, specific details and first-person perspective where natural "
        "  (e.g. 'I found', 'The data showed', 'Three participants said').\n"
        "- Avoid these AI clichés entirely: 'it is important to note', 'in today's world', "
        "  'holistic approach', 'paradigm shift', 'delve into', 'comprehensive overview', "
        "  'it is crucial', 'furthermore', 'moreover', 'it is evident that', "
        "  'the realm of', 'seamlessly', 'robust framework', 'plays a crucial role'.\n"
        "- Passive voice sparingly — prefer active constructions; vary sentence openers so "
        "  consecutive sentences don't start with the same word or transition.\n"
        "- Let opinions or observations show. Real writers hedge naturally ('I think', 'It seemed').\n"
        "- Preserve ALL factual claims, citations, and core argument — only the phrasing changes.\n"
        "- Do NOT add new claims or fabricate data.\n"
        f"{phrase_list}\n\n"
        f"TEXT TO REWRITE:\n{text[:4000]}\n\n"
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
    ctx_block = ("Context and instructions:\n" + context[:4000] + "\n\n") if context else ""
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
    """Generate structured dissertation chapters with optional subsections."""
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
        logger.warning("generate_outline_sections failed: %s", exc)

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


def extract_formal_objectives(full_document_text: str, topic: str) -> list[str]:
    """Analyze the full document context to extract or synthesize clear, professional research objectives."""
    prompt = f"""Read the full research document text provided below and determine the core research objectives.
Topic: {topic}

Task:
1. Analyze the whole document text.
2. Understand the core requirements and research trajectory.
3. Write the key research objectives in clear point form.
4. Use standard, highly professional academic phrasing. Do not write rubbish or vague filler; rely on precise academic verbs (e.g., 'To examine...', 'To evaluate...', 'To determine...').

Document Text Snippet (first 40000 chars for context):
{full_document_text[:40000]}

Return exactly a JSON array of strings, where each string is a single professional objective.
Example output format:
["To evaluate the impact of X on Y.", "To assess the factors influencing Z.", "To determine the relationship between A and B."]"""
    try:
        result = _json_response(prompt)
        if isinstance(result, list) and all(isinstance(x, str) for x in result) and result:
            return result
    except Exception as exc:
        logger.warning(f"extract_formal_objectives failed: {exc}")

    # Fallback if prediction fails
    short_topic = (topic or "the study topic").strip()
    return [
        f"To determine the current state of {short_topic}",
        f"To evaluate key drivers and constraints affecting {short_topic}",
        f"To propose evidence-based recommendations for improving outcomes in {short_topic}",
    ]
