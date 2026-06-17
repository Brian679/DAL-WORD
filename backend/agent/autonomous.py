"""
Autonomous document agent.
Accepts a free-form user message, classifies intent, plans, and executes
direct edits on the document using a selectable LLM provider.
"""
from __future__ import annotations

import logging
import json
import re
from typing import Any

from documents.models import Document, DocumentVersion

from .llm import (
    humanise_text,
    chat_with_document,
    classify_intent,
    create_execution_plan,
    enhance_text,
    set_active_model,
    get_model_label,
    generate_outline_sections,
    generate_section_content,
    generate_text,
)
from .tools import (
    find_section,
    generate_chart,
    generate_table_chart,
    generate_image,
    save_dataset_json,
    update_section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intents that modify the document and therefore require user confirmation
# before the agent proceeds to execute.
# ---------------------------------------------------------------------------
DOCUMENT_MODIFYING_INTENTS: frozenset[str] = frozenset({
    "write_section",
    "write_dissertation",
    "write_document",
    "enhance_section",
    "enhance_document",
    "humanise_ai_sections",
    "reduce_plagiarism_similarity",
    "address_comments",
    "create_outline",
    "add_chart",
    "add_image",
    # Legacy aliases kept so old LLM classifications still route correctly
    "write_report",
    "write_assignment",
    "write_presentation",
    "write_spreadsheet",
    "write_article",
})


def _intent_description(intent: str, target_section: str | None, topic: str | None) -> str:
    """Generate a human-readable summary of what the agent is about to do."""
    t = target_section or ""
    tp = topic or ""
    _descriptions: dict[str, str] = {
        "write_section":    f"Write the **{t or 'target'}** section",
        "write_dissertation": f"Write the full dissertation on **{tp}**",
        "write_document":   f"Plan and write a complete document on **{tp or 'the topic'}**",
        "enhance_section":  f"Improve and polish {'the **' + t + '** section' if t else 'the document content'}",
        "enhance_document": "Improve the entire document — structure, clarity, and academic quality",
        "address_comments": "Read all inline reviewer comments and address each one in the document",
        "create_outline":   f"Generate a structured outline for **{tp or 'the document'}**",
        "add_chart":  f"Generate and insert a chart{' into **' + t + '**' if t else ''}",
        "add_image":  f"Generate and insert an image{' into **' + t + '**' if t else ''}",
        "humanise_ai_sections": "Detect AI-generated passages and rewrite them to sound natural and human-written",
        "reduce_plagiarism_similarity": "Detect passages that overlap with other workspace documents and rewrite them to reduce textual similarity",
        "check_academic_quality": "Analyse the document for academic writing quality — vocabulary, evidence, structure, and argument strength",
        # Legacy aliases
        "write_report":     "Plan and write a report based on the request",
        "write_assignment": "Plan and write an assignment based on the request",
        "write_presentation": "Plan and write a presentation based on the request",
        "write_spreadsheet": "Plan and write a spreadsheet layout based on the request",
        "write_article":    "Plan and write an article based on the request",
    }
    return _descriptions.get(intent, "Execute the requested task")


# ---------------------------------------------------------------------------
# Per-subsection writing guidelines injected into every AI prompt.
# Keys are lowercase substrings matched against the subsection title.
# ---------------------------------------------------------------------------



def _subsection_guidelines(title: str, topic: str = "") -> str:
    """Generate dynamic writing instructions for the given subsection title."""
    lowered = title.lower()

    # ── Front matter ────────────────────────────────────────────────────────
    if "abstract" in lowered:
        return (
            "Write a concise academic abstract (200–300 words) structured as: background/context, "
            "problem statement, research objectives, methodology, key findings or expected contributions, "
            "and conclusion. Write as a single flowing paragraph with no internal headings."
        )
    if "dedication" in lowered:
        return (
            "Write a brief, heartfelt dedication (3–6 lines) in the first person. "
            "Dedicate the work to family members, mentors, or others who supported the researcher. "
            "Use warm, personal language. Do NOT use academic jargon. "
            "Begin with 'To...' or 'Dedicated to...'. Keep it short and sincere."
        )
    if "acknowledgement" in lowered or "acknowledgment" in lowered:
        return (
            "Write formal acknowledgements (150–250 words) thanking supervisors/advisors, "
            "the institution, research participants, family, and colleagues in that order. "
            "Use formal but warm academic prose."
        )
    if "table of contents" in lowered:
        return (
            "Write a placeholder note explaining that a detailed Table of Contents "
            "listing all chapters, sections, and page numbers will be compiled upon final "
            "document assembly. Then list the main chapter titles."
        )
    if "list of figure" in lowered:
        return (
            "Write a brief placeholder note that a List of Figures with captions and page numbers "
            "will be compiled upon final assembly. Provide a sample format line."
        )
    if "list of table" in lowered:
        return (
            "Write a brief placeholder note that a List of Tables with captions and page numbers "
            "will be compiled upon final assembly. Provide a sample format line."
        )
    if "list of abbreviation" in lowered or "list of acronym" in lowered or "abbreviation" in lowered:
        return (
            f"Write a well-organised List of Abbreviations and Acronyms relevant to the topic: '{topic}'. "
            "Format each entry as: ABBREVIATION — Full meaning. Include at least 10–15 common abbreviations "
            "used in the specific field/topic of study."
        )

    # ── Chapter summary (at the end of every main chapter) ──────────────────
    if "chapter summary" in lowered:
        return (
            "Write a comprehensive Chapter Summary (200–300 words) that: "
            "(1) recaps the key points and arguments developed in this chapter, "
            "(2) highlights the most important findings or conclusions reached, and "
            "(3) explains how this chapter connects to and prepares the reader for the next chapter. "
            "Write in past tense ('This chapter examined...', 'The review revealed...'). "
            "Be SPECIFIC to the content of this chapter — not generic."
        )

    # ── Chapter 1 core sections ──────────────────────────────────────────────
    if "research objective" in lowered or (lowered.endswith("objectives") and "background" not in lowered):
        return (
            "Write the Research Objectives as a numbered list. "
            "Start with 1–2 introductory sentences stating the overall aim of the study. "
            "Then list 4–6 specific objectives numbered 1, 2, 3... "
            "Each objective must begin with an action verb (To examine, To investigate, To assess, "
            "To determine, To evaluate, To explore). "
            "Each objective must be concise (one sentence) and directly related to the research topic. "
            "Use numbered list format: each objective on its own line starting with '1.', '2.', etc."
        )

    if "research question" in lowered or (lowered.endswith("questions") and "background" not in lowered):
        return (
            "Write the Research Questions as a numbered list. "
            "Start with 1 sentence introducing the research questions. "
            "Then list 4–6 specific questions numbered 1, 2, 3... "
            "Each question must be a complete interrogative sentence directly tied to an objective. "
            "Questions should be answerable through the stated research methodology. "
            "Use numbered list format: each question on its own line starting with '1.', '2.', etc."
        )

    if "hypothes" in lowered:
        return (
            "Write the Research Hypotheses as a numbered list of null/alternative hypothesis pairs. "
            "Format each pair as: "
            "N. H0: [null hypothesis — states no significant relationship/effect]. "
            "   H1: [alternative hypothesis — states a significant relationship/effect]. "
            "Include 3–5 hypothesis pairs, each corresponding to a research objective. "
            "Use formal statistical language."
        )

    if "significance of the study" in lowered or "significance of study" in lowered or (
        lowered.endswith("significance") and "statistical" not in lowered
    ):
        return (
            "Write the Significance of the Study in a structured format. "
            "Open with 1–2 paragraphs on the theoretical/academic contribution. "
            "Then present practical significance using clearly labelled sub-headings or numbered points: "
            "e.g., '1.5.1 Significance to the Researcher', '1.5.2 Significance to the Institution', "
            "'1.5.3 Significance to the Organisation', '1.5.4 Significance to Policy'. "
            "Each sub-section should be 2–4 sentences explaining who benefits and how. "
            "Write in formal academic prose within each sub-section."
        )

    if "background of the study" in lowered or "background of study" in lowered or (
        lowered.endswith("background") and "theoretical" not in lowered
    ):
        return (
            "Write the Background of the Study (350–500 words) covering: "
            "(1) global context and relevance of the topic, "
            "(2) the situation in the specific country or sector of the study, "
            "(3) identification of the research problem or gap, and "
            "(4) brief justification for the study. "
            "Write in formal academic prose with logical flow between paragraphs."
        )

    if "statement of the problem" in lowered or "problem statement" in lowered:
        return (
            "Write the Statement of the Problem (250–350 words) that: "
            "(1) clearly articulates the specific problem being studied, "
            "(2) provides evidence (statistics, citations, observable trends) that the problem exists, "
            "(3) explains the consequences of NOT addressing the problem, and "
            "(4) ends with a clear statement of what this study will do. "
            "Do NOT propose solutions — only state the problem."
        )

    if "scope and delimitation" in lowered or lowered.endswith("scope") or "delimitation" in lowered:
        return (
            "Write the Scope and Delimitations (200–300 words). "
            "Cover: (1) the geographical scope, (2) the target population/participants, "
            "(3) the time frame of the study, (4) the variables or concepts studied, and "
            "(5) what is deliberately excluded and why. "
            "Be specific and academic."
        )

    if "definition of key terms" in lowered or "key terms" in lowered:
        return (
            "Write the Definition of Key Terms as an alphabetically ordered list. "
            "Define 6–10 key concepts central to this research. "
            "For each term: state the term in bold or followed by a colon, then provide "
            "a concise academic definition (2–3 sentences) grounded in the literature. "
            "Cite a source for each definition where possible."
        )

    # ── Article / Journal sections ───────────────────────────────────────────
    if "executive summary" in lowered:
        return (
            "Write a professional Executive Summary (300–400 words) that captures: "
            "(1) the purpose and context of the report, "
            "(2) the key findings or results in concrete terms, "
            "(3) the most important recommendations, and "
            "(4) the expected impact or next steps. "
            "Write in plain, authoritative prose for a senior audience. "
            "No jargon. No bullet points — use coherent paragraphs."
        )

    if "recommendation" in lowered and "area" not in lowered and "future" not in lowered:
        return (
            "Write the Recommendations section as a clearly structured list. "
            "Open with 1 paragraph explaining the basis for these recommendations (what findings led here). "
            "Then present 5–8 specific, actionable recommendations using numbered sub-headings. "
            "Each recommendation must: name the action, specify WHO should take it, "
            "explain WHY (link back to a finding), and describe the expected outcome. "
            "Be direct, concrete, and feasible — avoid vague generalities like 'organisations should improve'."
        )

    if "swot" in lowered:
        return (
            "Write a SWOT Analysis for the topic. "
            "Use four clearly labelled sub-sections: Strengths, Weaknesses, Opportunities, Threats. "
            "Each sub-section should contain 4–6 bullet points with brief explanations (1–2 sentences each). "
            "Ground each point in the specific context of the study or organisation being analysed."
        )

    if "pest" in lowered or "pestle" in lowered:
        return (
            "Write a PESTLE Analysis. Use six labelled sub-sections: Political, Economic, Social, "
            "Technological, Legal, Environmental. Each sub-section: 3–5 bullets with brief explanations. "
            "Be specific to the topic/industry/country context of the study."
        )

    if "critical analysis" in lowered or "critical evaluation" in lowered or "critical review" in lowered:
        return (
            "Write a Critical Analysis/Evaluation that: "
            "(1) Opens with a clear thesis — your evaluative position on the topic. "
            "(2) Analyses the topic systematically, presenting strengths and weaknesses with evidence. "
            "(3) Compares and contrasts different perspectives or approaches from the literature. "
            "(4) Identifies gaps, contradictions, or unresolved tensions. "
            "(5) Draws a well-reasoned conclusion about the overall quality or merit. "
            "Use hedged academic language ('appears to', 'suggests', 'arguably'). "
            "Write in formal analytical prose — no bullet points."
        )

    if "findings" in lowered and "summary" not in lowered and "key" not in lowered:
        return (
            "Write the Findings section presenting the data/results clearly and objectively. "
            "Organise by theme or research question — NOT by method. "
            "For each finding: state it clearly, provide supporting data or evidence, "
            "and note its significance. Do NOT interpret or discuss here — report only. "
            "Use past tense ('Respondents indicated…', 'The data showed…'). "
            "Use precise quantitative or qualitative evidence where available."
        )

    if "method" in lowered and "research" not in lowered and "data collection" not in lowered:
        return (
            "Write the Methods section with full procedural clarity. "
            "Cover: (1) study design and rationale, (2) participants/sample with inclusion criteria, "
            "(3) data collection instruments (name them, describe them, justify them), "
            "(4) data collection procedure (step-by-step), "
            "(5) data analysis approach (specific technique, software if used), "
            "(6) ethical considerations. "
            "Use past tense for completed studies. Be precise enough for replication."
        )

    if "introduction" in lowered and "chapter" not in lowered:
        return (
            "Write a concise article/report Introduction (300–500 words) that: "
            "(1) Opens with a compelling hook grounded in the research context. "
            "(2) Establishes the problem or knowledge gap with specific evidence. "
            "(3) States the purpose of the study/report in one clear sentence. "
            "(4) Briefly previews the structure of the document. "
            "Write in active voice, present tense for established facts, past tense for this study."
        )

    if "main discussion" in lowered or "main body" in lowered or "main argument" in lowered:
        return (
            "Write the Main Discussion/Body covering the central argument or analysis in depth. "
            "Structure with clear sub-headings for each major point. "
            "For each point: state the claim → provide evidence (data, citation, example) → "
            "analyse and interpret it → link to the broader argument. "
            "Use the PEEL structure (Point → Evidence → Explanation → Link) for each paragraph. "
            "Engage critically with multiple perspectives where relevant."
        )

    if "literature review" in lowered or "review of literature" in lowered or "related work" in lowered:
        return (
            "Write a thematic Literature Review that: "
            "(1) Groups related works by theme or concept — NOT by author or chronology. "
            "(2) For each theme: synthesise what the literature collectively shows, "
            "identify agreements and contradictions, and note limitations. "
            "(3) Uses precise attribution ('According to Smith (2020)…', 'Several studies report…'). "
            "(4) Maintains a critical stance — do not merely describe, but evaluate the quality/relevance. "
            "(5) Ends by identifying the specific gap this study addresses. "
            "Avoid starting sentences with author names. Vary citation patterns."
        )

    if "conclusion" in lowered and "chapter" not in lowered:
        return (
            "Write a strong Conclusion (250–400 words) that: "
            "(1) Synthesises the key findings — do NOT just restate them. "
            "(2) States clearly what the study/report has contributed or shown. "
            "(3) Addresses the research questions or objectives directly. "
            "(4) Acknowledges limitations briefly and honestly. "
            "(5) Ends with a forward-looking closing statement about implications or future work. "
            "Write in past tense for what was found; present tense for implications."
        )

    if "references" in lowered or "bibliography" in lowered:
        return (
            "Write a note explaining that the References section will list all cited works in the "
            "appropriate citation format (APA, Harvard, or Vancouver as specified). "
            "Then provide 8–12 sample plausible references in APA 7th edition format "
            "relevant to the research topic, covering recent literature (2015–2024) "
            "from a mix of journals, books, and reports."
        )

    if "appendix" in lowered or "appendices" in lowered:
        return (
            "Write a placeholder Appendices section explaining that supporting materials "
            "(survey instruments, interview guides, raw data tables, ethical clearance, "
            "informed consent forms) will be attached here. "
            "Provide a labelled outline: Appendix A: [Survey Questionnaire], "
            "Appendix B: [Interview Guide], Appendix C: [Data Tables], etc."
        )

    return (
        f"Write the '{title}' section with clarity, depth, and academic rigour. "
        f"Ground the content specifically in the research topic. "
        "Structure your writing with clear topic sentences, supporting evidence, "
        "and logical transitions. Aim for substantive, specific content — "
        "not generic filler. Use formal academic prose."
    )


# ---------------------------------------------------------------------------
# Per-chapter visual node specs — injected automatically when the agent writes
# any of these sections. Chapter 4 is excluded (handled by _chapter4_subsections).
# Each spec: chapter (int), keyword (str, lowercase, matched against node title),
#             kind ("table"|"chart"|"image"), title (child node title), meta (dict),
#             designs_only (list[str]|None — None means all designs)
# ---------------------------------------------------------------------------
_SECTION_VISUAL_SPECS: list[dict[str, Any]] = [
    # Chapter 1
    {
        "chapter": 1, "keyword": "definition of key terms",
        "kind": "table", "title": "Table of Key Terms and Definitions",
        "meta": {"table_type": "key_terms"}, "designs_only": None,
    },
    # Chapter 2
    {
        "chapter": 2, "keyword": "empirical review",
        "kind": "table", "title": "Summary of Reviewed Empirical Studies",
        "meta": {"table_type": "empirical_summary"}, "designs_only": None,
    },
    {
        "chapter": 2, "keyword": "conceptual review",
        "kind": "chart", "title": "Conceptual Framework Diagram",
        "meta": {"chart_type": "framework"}, "designs_only": None,
    },
    # Chapter 3
    {
        "chapter": 3, "keyword": "sampling technique",
        "kind": "table", "title": "Sampling Frame and Allocation",
        "meta": {"table_type": "sampling"}, "designs_only": None,
    },
    {
        "chapter": 3, "keyword": "data collection",
        "kind": "table", "title": "Data Collection Instruments",
        "meta": {"table_type": "instruments"}, "designs_only": None,
    },
    {
        "chapter": 3, "keyword": "reliability",
        "kind": "table", "title": "Reliability and Validity Summary",
        "meta": {"table_type": "reliability"}, "designs_only": ["quantitative", "mixed"],
    },
    # Chapter 5
    {
        "chapter": 5, "keyword": "summary of findings",
        "kind": "table", "title": "Summary of Key Findings by Objective",
        "meta": {"table_type": "findings_summary"}, "designs_only": None,
    },
]

# Maps section keywords to canonical section names.  Defined at module level so
# it is available before the local variable assignment inside _heuristic_intent.
_SECTION_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["hypothesis", "hypothes", "null hypothesis", "alternative hypothesis", "h0", "h1"], "Research Hypotheses"),
    (["background of the study", "background of study", "background"], "Background of the Study"),
    (["statement of the problem", "problem statement", "problem of the study"], "Statement of the Problem"),
    (["research objectives", "research objective", "objectives", "specific objectives", "study objectives"], "Research Objectives"),
    (["research questions", "research question", "study questions"], "Research Questions"),
    (["significance of the study", "significance of study", "signifance of the study", "signifance", "significance"], "Significance of the Study"),
    (["scope and delimitations", "scope of the study", "delimitations", "scope"], "Scope and Delimitations"),
    (["definition of key terms", "key terms"], "Definition of Key Terms"),
    (["conceptual review", "conceptual framework"], "Conceptual Review"),
    (["theoretical framework", "theoretical review"], "Theoretical Framework"),
    (["empirical review", "empirical literature"], "Empirical Review"),
    (["research gap"], "Research Gap"),
    (["research design"], "Research Design"),
    (["target population", "study population"], "Target Population"),
    (["sampling technique", "sample size", "sampling method"], "Sampling Techniques and Sample Size"),
    (["data collection"], "Data Collection Methods"),
    (["data analysis", "analysis technique"], "Data Analysis Techniques"),
    (["reliability and validity", "reliability", "validity"], "Reliability and Validity"),
    (["ethical consideration", "research ethics"], "Ethical Considerations"),
    (["summary of findings", "findings summary"], "Summary of Findings"),
    (["recommendations"], "Recommendations"),
    (["limitations of the study", "limitations"], "Limitations of the Study"),
    (["areas for further research", "future research", "further research"], "Areas for Further Research"),
    (["discussion of findings", "discussion"], "Discussion of Findings"),
    (["abstract"], "Abstract"),
    (["conclusion"], "Conclusion"),
    (["references"], "References"),
    (["appendices", "appendix"], "Appendices"),
]

_SECTION_ACTION_WORDS: list[str] = [
    "redo", "rewrite", "write", "define", "fix", "correct",
    "improve", "enhance", "update", "replace", "regenerate", "generate",
    "refine", "modify", "change",
]


DISSERTATION_TEMPLATE: list[dict[str, Any]] = [
    {
        "title": "Chapter 1: Introduction",
        "subsections": [
            "1.1 Background of the Study",
            "1.2 Statement of the Problem",
            "1.3 Research Objectives",
            "1.4 Research Questions",
            "1.5 Research Hypotheses",
            "1.6 Significance of the Study",
            "1.7 Scope and Delimitations",
            "1.8 Definition of Key Terms",
        ],
    },
    {
        "title": "Chapter 2: Literature Review",
        "subsections": [
            {
                "title": "2.1 Introduction",
                "children": [],
            },
            {
                "title": "2.2 Conceptual Review",
                "children": [
                    {"title": "2.2.1 Definition and Conceptualisation of Key Terms", "children": []},
                    {"title": "2.2.2 Core Concepts and Theoretical Constructs", "children": []},
                    {"title": "2.2.3 Dimensions, Indicators, and Measurement", "children": []},
                    {"title": "2.2.4 Relationships Between Variables", "children": []},
                ],
            },
            {
                "title": "2.3 Theoretical Framework",
                "children": [
                    {"title": "2.3.1 Overview of Relevant Theories", "children": []},
                    {"title": "2.3.2 Foundational and Classical Theories", "children": []},
                    {"title": "2.3.3 Contemporary and Emerging Theories", "children": []},
                    {"title": "2.3.4 Applicability and Justification to the Study", "children": []},
                ],
            },
            {
                "title": "2.4 Empirical Review",
                "children": [
                    {"title": "2.4.1 Global Evidence and Trends", "children": []},
                    {"title": "2.4.2 Evidence from Developed Economies", "children": []},
                    {"title": "2.4.3 Evidence from Emerging Economies", "children": []},
                    {"title": "2.4.4 Evidence from Developing Economies and Africa", "children": []},
                    {"title": "2.4.5 Sectoral and Industry-Specific Evidence", "children": []},
                    {"title": "2.4.6 Synthesis and Critical Appraisal of Empirical Studies", "children": []},
                ],
            },
            {
                "title": "2.5 Research Gap",
                "children": [
                    {"title": "2.5.1 Identified Gaps in the Literature", "children": []},
                    {"title": "2.5.2 Contribution and Justification of the Present Study", "children": []},
                ],
            },
            {
                "title": "2.6 Chapter Summary",
                "children": [],
            },
        ],
    },
    {
        "title": "Chapter 3: Methodology",
        "subsections": [
            "3.1 Introduction",
            "3.2 Research Design",
            "3.3 Target Population",
            "3.4 Sampling Techniques and Sample Size",
            "3.5 Data Collection Methods",
            "3.6 Data Analysis Techniques",
            "3.7 Reliability and Validity",
            "3.8 Ethical Considerations",
            "3.9 Chapter Summary",
        ],
    },
    {
        "title": "Chapter 4: Results and Discussion",
        "subsections": [],
    },
    {
        "title": "Chapter 5: Conclusion and Recommendations",
        "subsections": [
            "5.1 Introduction",
            "5.2 Summary of Findings",
            "5.3 Conclusions",
            "5.4 Recommendations",
            "5.5 Limitations of the Study",
            "5.6 Areas for Further Research",
        ],
    },
    {
        "title": "Chapter 6: References and Appendices",
        "subsections": [
            "6.1 References",
            "6.2 Appendices",
        ],
    },
]


def _normalize_subsection_node(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"title": raw, "children": [], "kind": "text"}
    if isinstance(raw, dict):
        children = [_normalize_subsection_node(ch) for ch in raw.get("children", [])]
        return {
            "title": str(raw.get("title") or "Untitled subsection"),
            "children": children,
            "kind": raw.get("kind", "text"),
            "meta": raw.get("meta", {}),
        }
    return {"title": str(raw), "children": [], "kind": "text"}


def _inject_standard_visuals(
    nodes: list[dict[str, Any]],
    chapter_number: int,
    research_design: str,
) -> list[dict[str, Any]]:
    """Walk the node tree for this chapter and append table/chart child nodes to
    sections that benefit from visual data.  Chapter 4 is skipped — it already
    has its own visual logic in _chapter4_subsections.
    """
    if chapter_number == 4:
        return nodes

    specs = [s for s in _SECTION_VISUAL_SPECS if s["chapter"] == chapter_number]
    if not specs:
        return nodes

    result: list[dict[str, Any]] = []
    for node in nodes:
        title_lower = node.get("title", "").lower()
        children = list(node.get("children", []))

        for spec in specs:
            if spec["keyword"] not in title_lower:
                continue
            # Skip if design restriction doesn't match
            allowed = spec.get("designs_only")
            if allowed and research_design not in allowed:
                continue
            # Don't double-inject if a visual child already exists
            has_visual = any(ch.get("kind") in {"table", "chart", "image"} for ch in children)
            if has_visual:
                continue
            children.append({
                "title": spec["title"],
                "kind": spec["kind"],
                "children": [],
                "meta": spec.get("meta", {}),
            })

        # Recurse into children (e.g. subsections of Chapter 2's Empirical Review)
        if children:
            children = _inject_standard_visuals(children, chapter_number, research_design)

        result.append({**node, "children": children})
    return result


def _detect_visual_injection_request(
    instruction: str,
) -> tuple[str, str] | None:
    """Return (kind, raw_section_hint) when the user explicitly asks to add a
    table / chart / graph / image to a specific section.  Returns None otherwise.

    Examples that match:
      "put a table in the sampling section"
      "add a chart to the background of the study"
      "insert an image in chapter 3"
      "include a graph in the empirical review"
    """
    text = (instruction or "").strip().lower()

    # Determine visual kind first
    kind: str | None = None
    if "table" in text:
        kind = "table"
    elif any(k in text for k in ["chart", "graph", "bar chart", "pie chart", "line graph"]):
        kind = "chart"
    elif any(k in text for k in ["image", "figure", "diagram", "picture", "illustration"]):
        kind = "chart"  # routed via generate_image in tools.py

    if kind is None:
        return None

    # Need an action verb alongside the visual word
    action_verbs = ["add", "put", "insert", "include", "place", "attach", "create", "generate", "make"]
    has_action = any(v in text for v in action_verbs)
    if not has_action:
        return None

    # Extract section hint: text after "in" / "to" / "for" / "into" that follows the visual kind
    section_match = re.search(
        r"(?:in|to|for|into|on|within)\s+(?:the\s+)?(.+?)(?:\s+section)?$",
        text,
    )
    section_hint = section_match.group(1).strip() if section_match else ""

    # Must have some section text to be meaningful
    if len(section_hint) < 3:
        return None

    return kind, section_hint


def _research_design(message: str, topic: str, document: Document) -> str:
    source = " ".join(
        [
            message or "",
            topic or "",
            _flatten_doc(document),
        ]
    ).lower()
    if "mixed method" in source or "mixed-method" in source or "mixed methods" in source:
        return "mixed"
    if "quantitative" in source:
        return "quantitative"
    if "qualitative" in source:
        return "qualitative"
    if any(k in source for k in ["systematic review", "scoping review", "conceptual paper", "theoretical"]):
        return "non_empirical"
    return "quantitative"


def _extract_objectives(document: Document, topic: str) -> list[str]:
    """Dynamically analyze the whole document and extract objectives using the LLM."""
    from agent.gemini import extract_formal_objectives

    # Gather full document text to give the LLM wide context
    full_text_blocks = []
    sections = (document.content or {}).get("sections", [])
    for sec in sections:
        full_text_blocks.append(f"### {sec.get('title', '')}")
        full_text_blocks.append(sec.get("content", ""))
    
    full_text = "\n\n".join(full_text_blocks).strip()
    if not full_text:
        # Fallback if doc is empty
        short_topic = (topic or "the study topic").strip()
        return [
            f"To determine the current state of {short_topic}",
            f"To evaluate key drivers and constraints affecting {short_topic}",
            f"To propose evidence-based recommendations for improving outcomes in {short_topic}",
        ]

    # Use the LLM to professionally extract and write the objectives
    llm_objectives = extract_formal_objectives(full_text, topic)
    return llm_objectives or [
        f"To determine the current state of {topic}",
        f"To evaluate key drivers and constraints affecting {topic}"
    ]


def _objective_section_title(objective: str) -> str:
    """Derive a short readable section title from a full objective statement."""
    text = re.sub(
        r"^(to\s+|the\s+study\s+aims?\s+to\s+|this\s+study\s+(aims?\s+to|seeks?\s+to|will)\s+|"
        r"to\s+examine\s+|to\s+determine\s+|to\s+assess\s+|to\s+evaluate\s+|"
        r"to\s+investigate\s+|to\s+analyse\s+|to\s+analyze\s+)",
        "",
        objective.strip(),
        flags=re.IGNORECASE,
    )
    text = text[0].upper() + text[1:] if text else objective
    if len(text) > 95:
        cut = text[:95].rsplit(" ", 1)[0]
        # Avoid awkward trailing connector words.
        cut = re.sub(r"\b(of|in|on|for|to|and|the|with|by|at)$", "", cut.strip(), flags=re.IGNORECASE).strip()
        return cut or text[:95]
    return text


def _extract_document_brief(document: Document, topic: str, research_design: str) -> str:
    """Build a structured brief of the document's research content for grounding subsection prompts."""
    doc_title = (document.title or "").strip()
    content = document.content or {}
    sections = content.get("sections", [])

    # Objectives already extracted with intelligence
    objectives = _extract_objectives(document, topic)
    obj_text = "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(objectives[:5]))

    # Pull research questions from Chapter 1 text (lines containing "?")
    ch1_text = ""
    for sec in sections:
        if "chapter 1" in str(sec.get("title", "")).lower():
            ch1_text = sec.get("content", "")
            break
    questions: list[str] = []
    for line in ch1_text.splitlines():
        ln = line.strip(" -\t")
        if "?" in ln and len(ln) > 20:
            questions.append(ln[:140])
        if len(questions) >= 4:
            break
    q_text = (
        "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(questions))
        if questions
        else "  (infer from the research objectives above)"
    )

    design_label = {
        "quantitative": "Quantitative (survey/statistical)",
        "qualitative": "Qualitative (interviews/thematic analysis)",
        "mixed": "Mixed-Methods (quantitative + qualitative)",
        "non_empirical": "Non-Empirical / Theoretical / Conceptual",
    }.get(research_design, "Quantitative (survey/statistical)")

    return (
        "══ THIS STUDY'S BRIEF — ground ALL writing in these specifics ══\n"
        f"Document Title  : {doc_title}\n"
        f"Research Topic  : {topic}\n"
        f"Research Design : {design_label}\n"
        f"Research Objectives:\n{obj_text}\n"
        f"Research Questions:\n{q_text}\n"
        "══ END OF STUDY BRIEF ══"
    )


def _chapter4_subsections(research_design: str, objectives: list[str]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = [
        {"title": "4.1 Introduction", "children": []},
    ]

    if research_design in {"quantitative", "qualitative", "mixed"}:
        demo_children: list[dict[str, Any]] = [
            {
                "title": "4.2.1 Response Rate",
                "kind": "table",
                "children": [],
                "meta": {"table_type": "response_rate"},
            },
            {
                "title": "4.2.2 Demographic Distribution of Respondents",
                "kind": "table",
                "children": [],
                "meta": {"table_type": "demographics"},
            }
        ]
        if research_design in {"quantitative", "mixed"}:
            demo_children.append(
                {
                    "title": "4.2.3 Demographic Distribution Chart",
                    "kind": "chart",
                    "children": [],
                    "meta": {"chart_type": "demographics"},
                }
            )
        nodes.append({"title": "4.2 Respondent Profile", "children": demo_children})
        obj_start = 3
    else:
        obj_start = 2

    # Each objective gets its own top-level section: 4.X <Short Objective Title>
    for i, objective in enumerate(objectives):
        sec_num = obj_start + i
        short_title = _objective_section_title(objective)
        if research_design == "qualitative":
            table_title = f"4.{sec_num}.1 Theme Matrix"
        else:
            table_title = f"4.{sec_num}.1 Summary Table"

        obj_node: dict[str, Any] = {
            "title": f"4.{sec_num} {short_title}",
            "kind": "objective_findings",
            "meta": {"objective": objective},
            "children": [
                {
                    "title": table_title,
                    "kind": "table",
                    "children": [],
                    "meta": {"objective": objective},
                }
            ],
        }
        if research_design in {"quantitative", "mixed"}:
            obj_node["children"].append(
                {
                    "title": f"4.{sec_num}.2 Data Visualization",
                    "kind": "chart",
                    "children": [],
                    "meta": {"objective": objective},
                }
            )
        nodes.append(obj_node)

    last_idx = obj_start + len(objectives)
    nodes.append({"title": f"4.{last_idx} Discussion of Findings", "children": []})
    nodes.append({"title": f"4.{last_idx + 1} Chapter Summary", "children": []})
    return [_normalize_subsection_node(n) for n in nodes]


def _table_text_for_node(node_title: str, research_design: str, topic: str, objective: str | None = None) -> str:
    seed_text = f"{node_title}|{research_design}|{topic}|{objective or ''}"
    seed = sum(ord(c) for c in seed_text)

    def pct_triplet(total: int = 100) -> tuple[int, int, int]:
        a = 24 + (seed % 19)
        b = 29 + ((seed // 3) % 17)
        c = total - a - b
        if c < 18:
            c = 18
            b = total - a - c
        return a, b, c

    if "demographic" in node_title.lower():
        male = 45 + (seed % 31)
        female = 120 - male
        age1, age2, age3 = pct_triplet(120)
        return (
            "| Variable | Category | Frequency | Percentage |\n"
            "|---|---|---:|---:|\n"
            f"| Gender | Male | {male} | {male / 120 * 100:.1f}% |\n"
            f"| Gender | Female | {female} | {female / 120 * 100:.1f}% |\n"
            f"| Age | 18-29 | {age1} | {age1 / 120 * 100:.1f}% |\n"
            f"| Age | 30-39 | {age2} | {age2 / 120 * 100:.1f}% |\n"
            f"| Age | 40+ | {age3} | {age3 / 120 * 100:.1f}% |"
        )
    if research_design == "qualitative":
        t1 = 2 + (seed % 4)
        t2 = 2 + ((seed // 5) % 4)
        t3 = 2 + ((seed // 11) % 4)
        return (
            "| Theme | Supporting Mentions | Representative Insight | Interpretation |\n"
            "|---|---|---|\n"
            f"| Theme 1 | {t1} | Participants highlighted concerns around {topic[:40]} | Indicates persistent implementation barriers |\n"
            f"| Theme 2 | {t2} | Respondents reported uneven institutional readiness | Suggests need for governance alignment |\n"
            f"| Theme 3 | {t3} | Stakeholders requested stronger policy direction | Supports a coordinated reform approach |"
        )
    objective_text = objective or "the objective"
    a = round(2.6 + (seed % 16) * 0.11, 2)
    b = round(3.1 + ((seed // 7) % 14) * 0.12, 2)
    c = round(2.4 + ((seed // 13) % 15) * 0.1, 2)
    return (
        "| Metric | Observation | Interpretation |\n"
        "|---|---:|---|\n"
        f"| Indicator A ({objective_text[:30]}) | {a} | Moderate performance with room for improvement |\n"
        f"| Indicator B | {b} | Stronger outcome where controls were applied |\n"
        f"| Indicator C | {c} | Weakest dimension and major constraint area |"
    )


def _infer_sample_size(document: Document) -> int:
    """Infer respondent/sample size from the current document, defaulting safely."""
    context = _full_context_for_generation(document)
    patterns = [
        r"sample\s+size[^\d]{0,20}(\d{2,4})",
        r"respondents?[^\d]{0,20}(\d{2,4})",
        r"participants?[^\d]{0,20}(\d{2,4})",
        r"n\s*=\s*(\d{2,4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            try:
                value = int(match.group(1))
                if 20 <= value <= 5000:
                    return value
            except Exception:
                continue
    return 120


def _ai_table_dataset(
    node_title: str,
    research_design: str,
    topic: str,
    objective: str | None,
    sample_size: int,
    current_document_context: str,
) -> dict[str, Any]:
    """Generate structured table data as JSON for rendering via matplotlib."""
    prompt = (
        "Generate realistic table data for a dissertation results section. Return JSON only.\n"
        f"Node title: {node_title}\n"
        f"Research design: {research_design}\n"
        f"Topic: {topic}\n"
        f"Objective: {objective or 'N/A'}\n"
        f"Sample size: {sample_size}\n"
        f"Current document context:\n{current_document_context[-2500:]}\n\n"
        "JSON schema:\n"
        "{\"headers\":[\"...\"],\"rows\":[[\"...\"],[\"...\"]]}\n"
        "Rules:\n"
        "- If the table is demographics/response-rate, include frequencies and percentages that sum to the sample size.\n"
        "- Keep 3-7 rows.\n"
        "- Use academically plausible values, no placeholders.\n"
        "- Do not return markdown."
    )
    try:
        data = _extract_json_obj(generate_text(prompt))
        headers = [str(h) for h in (data.get("headers") or []) if str(h).strip()]
        rows_raw = data.get("rows") or []
        rows = [[str(cell) for cell in row] for row in rows_raw if isinstance(row, list)]
        if headers and rows:
            return {"headers": headers, "rows": rows}
    except Exception as exc:
        logger.warning("_ai_table_dataset fallback (%s): %s", node_title[:60], exc)

    # Deterministic fallback aligned to sample size
    seed = sum(ord(c) for c in f"{node_title}|{topic}|{objective or ''}|{research_design}")
    if "demographic" in node_title.lower() or "response rate" in node_title.lower():
        male = max(1, int(round(sample_size * (0.42 + (seed % 12) / 100))))
        female = max(1, sample_size - male)
        returned = max(1, int(round(sample_size * (0.80 + (seed % 9) / 100))))
        not_returned = max(0, sample_size - returned)
        return {
            "headers": ["Variable", "Category", "Frequency", "Percentage"],
            "rows": [
                ["Response Rate", "Returned Questionnaires", str(returned), f"{(returned / sample_size) * 100:.1f}%"],
                ["Response Rate", "Not Returned", str(not_returned), f"{(not_returned / sample_size) * 100:.1f}%"],
                ["Gender", "Male", str(male), f"{(male / sample_size) * 100:.1f}%"],
                ["Gender", "Female", str(female), f"{(female / sample_size) * 100:.1f}%"],
            ],
        }

    if research_design == "qualitative":
        t1 = 3 + (seed % 5)
        t2 = 2 + ((seed // 5) % 5)
        t3 = 2 + ((seed // 9) % 4)
        return {
            "headers": ["Theme", "Mentions", "Representative Excerpt", "Interpretation"],
            "rows": [
                ["Theme 1", str(t1), f"Participants emphasized {topic[:28]}.", "Shows core experiential pattern"],
                ["Theme 2", str(t2), "Respondents highlighted implementation constraints.", "Indicates operational barriers"],
                ["Theme 3", str(t3), "Stakeholders requested stronger governance.", "Supports policy-focused recommendations"],
            ],
        }

    a = round(2.7 + (seed % 17) * 0.12, 2)
    b = round(2.9 + ((seed // 7) % 14) * 0.11, 2)
    c = round(2.5 + ((seed // 11) % 15) * 0.10, 2)
    return {
        "headers": ["Metric", "Value", "Interpretation"],
        "rows": [
            [f"Indicator A ({(objective or 'Objective')[:26]})", str(a), "Moderate performance"],
            ["Indicator B", str(b), "Relatively stronger outcome"],
            ["Indicator C", str(c), "Priority improvement area"],
        ],
    }


def _ai_chart_series(context: str, n_points: int = 8) -> dict[str, Any]:
    """Ask the LLM to produce realistic numeric data for a chart.

    Returns a dict with keys:
      - series: list[float]   — the data points
      - chart_type: str       — suggested chart type (bar/line/scatter/area/pie)
      - x_labels: list[str]   — short label for each point (SAME length as series)
      - unit: str             — measurement unit, e.g. "%", "score", "count"
    Falls back to seed-based data on any error so the pipeline never breaks.
    """
    # Pie charts look cluttered with many slices; cap at 6 for pie-likely topics.
    pie_keywords = {"distribution", "composition", "proportion", "breakdown", "demographic", "share", "pie"}
    likely_pie = any(kw in context.lower() for kw in pie_keywords)
    effective_n = min(n_points, 6) if likely_pie else n_points

    prompt = (
        "You are an academic data analyst generating chart data for a dissertation figure.\n"
        f"Chart topic / section title: \"{context}\"\n\n"
        "Instructions:\n"
        f"1. Generate EXACTLY {effective_n} data points that are academically plausible for this topic.\n"
        "2. Choose the BEST chart_type: 'bar' for comparisons/categories, 'line' for trends over time, "
        "'scatter' for correlation/relationship, 'area' for cumulative trends, 'pie' for proportions.\n"
        f"3. x_labels: provide EXACTLY {effective_n} short labels (1-4 words each) describing each data point "
        "(e.g. years like '2019', '2020'; categories like 'Urban', 'Rural'; groups like 'Group A').\n"
        "4. unit: the y-axis measurement unit as a short string, e.g. '%', 'score (1-5)', 'count', "
        "'USD millions', 'years', 'kg/ha'. Use '%' for percentages, not 'percent' or '%%'.\n"
        "5. Values must show meaningful variation — NOT all the same. Be realistic and grounded.\n"
        "6. For demographic/proportion charts, values should sum to approximately 100 (percentages).\n"
        "7. For Likert/survey-score charts, values should be between 1.0 and 5.0.\n\n"
        "Return ONLY a JSON object — no markdown, no explanation:\n"
        '{"series":[v1,v2,...],"chart_type":"bar","x_labels":["lbl1","lbl2",...],"unit":"%"}'
    )

    try:
        raw = generate_text(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        # Tolerate a stray trailing comma before closing brace
        raw = re.sub(r",\s*}", "}", raw)
        data = json.loads(raw)
        series = [float(v) for v in (data.get("series") or []) if v is not None]
        if not series:
            raise ValueError("empty series")
        x_labels_raw = [str(lbl).strip() for lbl in (data.get("x_labels") or [])]
        # Ensure x_labels length matches series; pad or trim as needed
        while len(x_labels_raw) < len(series):
            x_labels_raw.append(str(len(x_labels_raw) + 1))
        x_labels_raw = x_labels_raw[: len(series)]
        unit_raw = str(data.get("unit") or "").strip()
        # Normalise common LLM mis-renderings of percent
        if unit_raw.lower() in {"percent", "percentage", "%%"}:
            unit_raw = "%"
        return {
            "series": series,
            "chart_type": str(data.get("chart_type") or "bar").lower(),
            "x_labels": x_labels_raw,
            "unit": unit_raw,
        }
    except Exception as exc:
        logger.warning("_ai_chart_series fallback (%s): %s", context[:60], exc)
        seed = sum(ord(c) for c in context)
        base = 10.0 + (seed % 30)
        return {
            "series": [round(base + i * 3.5 + (seed + i) % 7, 1) for i in range(effective_n)],
            "chart_type": "bar",
            "x_labels": [f"Item {i + 1}" for i in range(effective_n)],
            "unit": "",
        }


def _extract_json_obj(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object found")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("json payload is not an object")
    return parsed


# ---------------------------------------------------------------------------
# LLM-generated dissertation plan (the ONLY source of truth for chapter structure)
# ---------------------------------------------------------------------------

def generate_dissertation_plan_llm(
    topic: str,
    message: str,
    research_design: str = "quantitative",
    objectives: list[str] | None = None,
    guidelines: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Ask the LLM to produce the complete dissertation chapter and section plan.

    Returns a list of chapter dicts:
        [{"title": "Chapter 1: ...", "sections": [{"title": "1.1 ...", "sections": [...]}, ...]}, ...]

    Falls back to a minimal generic structure if the LLM call fails.
    """
    obj_block = ""
    if objectives:
        obj_lines = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(objectives[:6]))
        obj_block = f"Existing research objectives (must be reflected in the plan):\n{obj_lines}\n\n"

    # Build a guidelines block from parsed user requirements
    guide = guidelines or {}
    guideline_lines = []
    if guide.get("citation_style"):
        guideline_lines.append(f"- Citation style: {guide['citation_style']}")
    if guide.get("academic_level"):
        guideline_lines.append(f"- Academic level: {guide['academic_level']}")
    if guide.get("target_words"):
        guideline_lines.append(f"- Target word count: ~{guide['target_words']:,} words (plan sections accordingly)")
    if guide.get("focus_notes"):
        guideline_lines.append(f"- Student-specified requirements:\n  {guide['focus_notes'][:400]}")
    guidelines_block = (
        "Student guidelines to honour in the chapter structure:\n"
        + "\n".join(guideline_lines) + "\n\n"
    ) if guideline_lines else ""

    prompt = (
        "You are an expert academic dissertation planner.\n"
        "A student has asked you to help write a dissertation. Your task is to generate a detailed, "
        "academically appropriate chapter plan tailored to their specific topic and research type.\n\n"
        f"Topic: {topic or message[:300]}\n"
        f"Research type: {research_design}\n"
        f"Student request: {message[:500]}\n"
        f"{guidelines_block}"
        f"{obj_block}\n"
        "Return ONLY valid JSON — no markdown fences, no explanation. Use this exact schema:\n"
        "[\n"
        "  {\n"
        '    "title": "Chapter 1: Introduction",\n'
        '    "sections": [\n'
        '      {"title": "1.1 Background of the Study", "sections": []},\n'
        '      {"title": "1.2 Statement of the Problem", "sections": []}\n'
        "    ]\n"
        "  }\n"
        "]\n\n"
        "Rules:\n"
        "- The FIRST entry in the array must be the front matter with the title 'Preliminary Pages' and "
        "these exact sections (in order, numbered as shown): "
        "'i. Abstract', 'ii. Dedication', 'iii. Acknowledgements', "
        "'iv. Table of Contents', 'v. List of Figures', 'vi. List of Tables', "
        "'vii. List of Abbreviations and Acronyms'.\n"
        "- After the front matter, generate exactly 5 or 6 main chapters (Introduction, Literature Review, "
        "Methodology, Results/Analysis, Conclusion, and optionally References & Appendices).\n"
        "- Chapter 1 must contain: Background, Statement of the Problem, Research Objectives, "
        "Research Questions, Significance, Scope & Delimitations, Definition of Key Terms, Chapter Summary.\n"
        "- Chapter 2 title and section headings must be SPECIFIC to the research topic — do NOT use "
        "generic placeholder titles like 'Related Work' or 'Review of Literature'. Instead, frame "
        "headings around the actual subject matter (e.g. if the topic is municipal finance, use "
        "'2.2 Revenue Mechanisms in Local Government', '2.3 Fiscal Sustainability Theories', etc.).\n"
        "- Chapter 2 must end with a 'Chapter Summary' section.\n"
        "- Chapter 3 sections must reflect the stated research design "
        f"({research_design}): include appropriate data collection and analysis subsections. "
        "Chapter 3 must end with a 'Chapter Summary' section.\n"
        "- Chapter 4 must have results subsections that directly correspond to the research objectives "
        "or questions. Chapter 4 must end with a 'Chapter Summary' section.\n"
        "- Chapter 5 must include: Summary of Findings, Conclusions, Recommendations, Limitations, "
        "Areas for Further Research, Chapter Summary.\n"
        "- Each main chapter should have 6–12 sections (including the Chapter Summary). Sections may have nested sub-sections.\n"
        "- All titles must use standard academic numbering (1.1, 1.2, 2.1, 2.2.1, etc.).\n"
        "- The Chapter Summary in each chapter must be the LAST section.\n"
        "- Return ONLY the JSON array. Nothing else."
    )

    raw = ""
    try:
        raw = generate_text(prompt)
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned).rstrip("`").strip()
        # Find the JSON array
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            chapters = json.loads(cleaned[start : end + 1])
            if isinstance(chapters, list) and chapters:
                return chapters
    except Exception as exc:
        logger.warning("generate_dissertation_plan_llm failed: %s | raw=%s", exc, raw[:300])

    # Fallback: minimal generic plan
    logger.info("generate_dissertation_plan_llm using fallback plan for topic=%s", topic[:80] if topic else "")
    return _fallback_dissertation_chapters(topic or "the study")


def _fallback_dissertation_chapters(topic: str) -> list[dict[str, Any]]:
    """Minimal generic fallback when LLM plan generation fails."""
    return [
        {"title": "Preliminary Pages", "sections": [
            {"title": "i. Abstract", "sections": []},
            {"title": "ii. Dedication", "sections": []},
            {"title": "iii. Acknowledgements", "sections": []},
            {"title": "iv. Table of Contents", "sections": []},
            {"title": "v. List of Figures", "sections": []},
            {"title": "vi. List of Tables", "sections": []},
            {"title": "vii. List of Abbreviations and Acronyms", "sections": []},
        ]},
        {"title": "Chapter 1: Introduction", "sections": [
            {"title": "1.1 Background of the Study", "sections": []},
            {"title": "1.2 Statement of the Problem", "sections": []},
            {"title": "1.3 Research Objectives", "sections": []},
            {"title": "1.4 Research Questions", "sections": []},
            {"title": "1.5 Significance of the Study", "sections": []},
            {"title": "1.6 Scope and Delimitations", "sections": []},
            {"title": "1.7 Definition of Key Terms", "sections": []},
            {"title": "1.8 Chapter Summary", "sections": []},
        ]},
        {"title": "Chapter 2: Literature Review", "sections": [
            {"title": f"2.1 Introduction to {topic[:60]}", "sections": []},
            {"title": "2.2 Theoretical Framework", "sections": []},
            {"title": "2.3 Conceptual Framework", "sections": []},
            {"title": "2.4 Empirical Review", "sections": []},
            {"title": "2.5 Research Gap", "sections": []},
            {"title": "2.6 Chapter Summary", "sections": []},
        ]},
        {"title": "Chapter 3: Research Methodology", "sections": [
            {"title": "3.1 Introduction", "sections": []},
            {"title": "3.2 Research Design", "sections": []},
            {"title": "3.3 Target Population", "sections": []},
            {"title": "3.4 Sampling Techniques and Sample Size", "sections": []},
            {"title": "3.5 Data Collection Methods", "sections": []},
            {"title": "3.6 Data Analysis Techniques", "sections": []},
            {"title": "3.7 Reliability and Validity", "sections": []},
            {"title": "3.8 Ethical Considerations", "sections": []},
            {"title": "3.9 Chapter Summary", "sections": []},
        ]},
        {"title": "Chapter 4: Results and Discussion", "sections": [
            {"title": "4.1 Introduction", "sections": []},
            {"title": "4.2 Presentation of Findings", "sections": []},
            {"title": "4.3 Discussion of Findings", "sections": []},
            {"title": "4.4 Chapter Summary", "sections": []},
        ]},
        {"title": "Chapter 5: Conclusions and Recommendations", "sections": [
            {"title": "5.1 Summary of Findings", "sections": []},
            {"title": "5.2 Conclusions", "sections": []},
            {"title": "5.3 Recommendations", "sections": []},
            {"title": "5.4 Limitations of the Study", "sections": []},
            {"title": "5.5 Areas for Further Research", "sections": []},
            {"title": "5.6 Chapter Summary", "sections": []},
        ]},
        {"title": "Chapter 6: References and Appendices", "sections": [
            {"title": "6.1 References", "sections": []},
            {"title": "6.2 Appendices", "sections": []},
        ]},
    ]


def llm_chapters_to_blueprints(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert LLM plan chapters to the internal blueprint format used by _write_dissertation."""
    blueprints = []
    for chapter in chapters:
        title = chapter.get("title", "Untitled Chapter")
        nodes = _sections_to_nodes(chapter.get("sections", []))
        blueprints.append({"title": title, "nodes": nodes})
    return blueprints


def _sections_to_nodes(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recursively convert LLM plan sections to normalised node dicts."""
    result = []
    for sec in sections:
        title = sec.get("title", "Untitled")
        children = _sections_to_nodes(sec.get("sections", []))
        result.append({"title": title, "children": children, "kind": "text"})
    return result


def llm_chapters_to_flat_steps(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert LLM plan chapters to the flat step list shown in the frontend todo."""
    steps: list[dict[str, Any]] = [{"step": "Creating dissertation to-do list", "status": "done"}]
    first = True
    for chapter in chapters:
        step_status = "in_progress" if first else "pending"
        first = False
        steps.append({"step": f"Writing {chapter.get('title', 'Chapter')}", "status": step_status})
        _sections_to_steps(steps, chapter.get("sections", []), depth=1)
    return steps


def _sections_to_steps(
    steps: list[dict[str, Any]], sections: list[dict[str, Any]], depth: int
) -> None:
    indent = "  " * depth
    for sec in sections:
        steps.append({"step": f"{indent}Writing {sec.get('title', 'Section')}", "status": "pending"})
        _sections_to_steps(steps, sec.get("sections", []), depth + 1)


def _default_framework_spec(topic: str, prompt: str) -> dict[str, Any]:
    short_topic = (topic or "the study").strip()
    return {
        "title": f"Conceptual Framework: {short_topic}",
        "left_label": "Independent Variables",
        "left_items": ["Policy factors", "Resource capacity", "Stakeholder readiness"],
        "middle_label": "Mediating Factors",
        "middle_items": ["Implementation quality", "Operational efficiency"],
        "right_label": "Dependent Variable",
        "right_items": [short_topic],
        "control_label": "Control Variables",
        "control_items": ["Context", "Institution size", "Time"],
        "notes": (prompt or "").strip()[:180],
    }


def _framework_target_index(document: Document, target: str | None, prompt: str) -> int | None:
    sections = (document.content or {}).get("sections", [])
    if not sections:
        return None

    if target:
        direct = find_section(document.content, target)
        if direct is not None:
            return direct

    text = f"{(target or '')} {(prompt or '')}".lower()
    looks_framework = any(k in text for k in ["conceptual", "theoretical", "framework", "model"])
    if looks_framework:
        preferred_titles = [
            "2.3 Theoretical Framework",
            "Theoretical Framework",
            "2.2 Conceptual Review",
            "Conceptual Review",
            "Literature Review",
            "Chapter 2",
        ]
        for query in preferred_titles:
            idx = find_section(document.content, query)
            if idx is not None:
                return idx

        for i, section in enumerate(sections):
            combined = f"{section.get('title', '')}\n{section.get('content', '')}".lower()
            if any(k in combined for k in ["conceptual framework", "theoretical framework", "framework"]):
                return i

    return len(sections) - 1


def _build_framework_spec(document: Document, target: str | None, prompt: str) -> dict[str, Any]:
    sections = (document.content or {}).get("sections", [])
    idx = _framework_target_index(document, target, prompt)
    local_title = ""
    local_content = ""
    if idx is not None and 0 <= idx < len(sections):
        local_title = str(sections[idx].get("title") or "")
        local_content = str(sections[idx].get("content") or "")

    full_context = _full_context_for_generation(document)[:5000]
    topic = str((document.content or {}).get("topic") or document.title or "Study")
    objectives = _extract_objectives(document, topic)
    objective_text = "\n".join(f"- {item}" for item in objectives[:4])

    llm_prompt = (
        "You are an academic research assistant. Build a professional conceptual framework specification "
        "for a dissertation figure.\n\n"
        f"Document title: {document.title}\n"
        f"Topic: {topic}\n"
        f"Target section title: {local_title or 'N/A'}\n"
        f"User request: {prompt}\n\n"
        "Research objectives:\n"
        f"{objective_text or '- N/A'}\n\n"
        "Relevant section content:\n"
        f"{local_content[:1400]}\n\n"
        "Whole document context:\n"
        f"{full_context}\n\n"
        "Return JSON only with this exact shape:\n"
        "{"
        '"title":"...",'
        '"left_label":"Independent Variables",'
        '"left_items":["...","..."],'
        '"middle_label":"Mediating/Moderating Variables",'
        '"middle_items":["...","..."],'
        '"right_label":"Dependent Variable",'
        '"right_items":["..."],'
        '"control_label":"Control Variables",'
        '"control_items":["...","..."],'
        '"notes":"Short rationale under 180 chars"'
        "}\n"
        "Rules: keep labels short, items concrete and topic-aware, no markdown."
    )

    try:
        data = _extract_json_obj(generate_text(llm_prompt))
        return {
            "title": str(data.get("title") or f"Conceptual Framework: {topic}"),
            "left_label": str(data.get("left_label") or "Independent Variables"),
            "left_items": [str(x) for x in (data.get("left_items") or [])][:4],
            "middle_label": str(data.get("middle_label") or "Mediating Variables"),
            "middle_items": [str(x) for x in (data.get("middle_items") or [])][:4],
            "right_label": str(data.get("right_label") or "Dependent Variable"),
            "right_items": [str(x) for x in (data.get("right_items") or [])][:4],
            "control_label": str(data.get("control_label") or "Control Variables"),
            "control_items": [str(x) for x in (data.get("control_items") or [])][:4],
            "notes": str(data.get("notes") or "")[0:180],
        }
    except Exception as exc:
        logger.warning("_build_framework_spec fallback: %s", exc)
        return _default_framework_spec(topic, prompt)


def _insert_block_marker(section_text: str, block_id: str, prompt: str) -> str:
    text = section_text or ""
    marker = f"[[BLOCK:{block_id}]]"
    if marker in text:
        return text

    lines = text.splitlines()
    lowered_prompt = (prompt or "").lower()
    framework_request = any(k in lowered_prompt for k in ["conceptual", "theoretical", "framework", "model"])

    if framework_request:
        for i, line in enumerate(lines):
            lower_line = line.lower().strip()
            if any(k in lower_line for k in ["conceptual framework", "theoretical framework", "framework"]):
                lines.insert(i + 1, marker)
                return "\n".join(lines)

    if not text.strip():
        return marker
    return text.rstrip() + "\n\n" + marker


def _chart_series_for_node(node_title: str, objective: str | None = None) -> list[float]:
    context = node_title + (f" — {objective}" if objective else "")
    result = _ai_chart_series(context, n_points=8)
    return result["series"]


def _next_caption_number(document: Document, kind: str) -> int:
    token = "figure" if kind == "figure" else "table"
    max_seen = 0

    sections = (document.content or {}).get("sections", [])
    for section in sections:
        content = section.get("content", "") or ""
        for match in re.finditer(rf"\b{token}\s+(\d+)\b", content, flags=re.IGNORECASE):
            max_seen = max(max_seen, int(match.group(1)))

        for block in section.get("blocks", []) or []:
            caption = (block.get("caption") or "")
            for match in re.finditer(rf"\b{token}\s+(\d+)\b", caption, flags=re.IGNORECASE):
                max_seen = max(max_seen, int(match.group(1)))

    return max_seen + 1


def _table_discussion_text(node_title: str, research_design: str, objective: str | None = None) -> str:
    if "response rate" in node_title.lower():
        return (
            "Interpretation: The response-rate table indicates the proportion of usable responses relative to the target sample, "
            "providing evidence of dataset adequacy for statistical analysis.\n"
            "Discussion: A strong response rate supports representativeness and reduces the risk of non-response bias, "
            "thereby improving confidence in subsequent objective-level findings."
        )

    if "demographic" in node_title.lower():
        return (
            "Interpretation: The respondent profile indicates a reasonably balanced distribution across key demographic categories, "
            "which supports broad coverage of participant perspectives.\n"
            "Discussion: This distribution improves confidence that subsequent objective-level findings are not overly driven by a single subgroup."
        )

    if research_design == "qualitative":
        return (
            "Interpretation: The theme matrix shows recurring viewpoints across participants, with some themes appearing more frequently than others.\n"
            "Discussion: The pattern suggests consistent experiential concerns and provides evidence for targeted policy and implementation recommendations."
        )

    obj = (objective or node_title or "the objective").strip()
    return (
        f"Interpretation: The metric pattern for {obj[:70]} shows uneven performance across indicators, with stronger outcomes in selected dimensions.\n"
        "Discussion: The spread across indicators highlights where focused interventions are required to improve overall study outcomes."
    )


def _chart_discussion_text(series: list[float], objective: str | None = None, node_title: str | None = None) -> str:
    avg = round(sum(series) / len(series), 2) if series else 0.0
    high = max(series) if series else 0.0
    low = min(series) if series else 0.0
    trend = "upward" if len(series) > 1 and series[-1] >= series[0] else "mixed"
    objective_label = (objective or node_title or "the subsection").strip()
    return (
        f"Interpretation: Figure trend is {trend}, with values ranging from {low:.2f} to {high:.2f} and an average of {avg:.2f}.\n"
        f"Discussion: For {objective_label[:70]}, the visual pattern reinforces the numerical evidence and clarifies priority areas for action."
    )


def _append_node_plan_steps(plan: list[dict[str, Any]], nodes: list[dict[str, Any]], depth: int = 1) -> None:
    indent = "  " * depth
    for node in nodes:
        kind = node.get("kind", "text")
        verb = "Writing"
        if kind == "table":
            verb = "Creating table for"
        elif kind == "chart":
            verb = "Creating chart for"
        plan.append({"step": f"{indent}{verb} {node.get('title', 'Untitled subsection')}", "status": "pending"})
        _append_node_plan_steps(plan, node.get("children", []), depth + 1)



def _strip_leading_heading(body: str, title: str) -> str:
    """Remove the section title if the LLM echoed it at the start of the body."""
    stripped = body.lstrip()
    clean_title = re.sub(r"^#+\s*", "", title).strip().lower()
    first_line_raw = stripped.split("\n", 1)[0] if stripped else ""
    first_line = re.sub(r"^#+\s*|\*+|_+", "", first_line_raw).strip().lower()
    # Match if the first line IS the title (exact or starts with it)
    if first_line == clean_title or first_line.startswith(clean_title):
        rest = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        return rest.lstrip("\n")
    return body


def _sanitize_body(text: str) -> str:
    """Strip HTML tags and normalise whitespace in LLM-generated body text."""
    # Replace <br> variants with a newline so paragraph structure is preserved.
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Remove any remaining HTML/XML tags.
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse runs of 3+ blank lines to at most 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _execute_subsection_nodes(
    nodes: list[dict[str, Any]],
    document: Document,
    section_title: str,
    topic: str,
    research_design: str,
    rolling_context: str,
    plan: list[dict[str, Any]],
    plan_cursor: list[int],
    figure_counter: list[int],
    table_counter: list[int],
    on_node_completed: Any | None = None,
    default_word_count: int = 220,
    user_instruction: str = "",
) -> tuple[str, str, list[dict[str, str]]]:
    chunks: list[str] = []
    blocks: list[dict[str, str]] = []
    local_context = rolling_context

    # Build study brief ONCE per call (shared by every node in this invocation)
    document_brief = _extract_document_brief(document, topic, research_design)

    for node in nodes:
        step_idx = plan_cursor[0]
        plan_cursor[0] += 1

        title = node.get("title", "Untitled subsection")
        kind = node.get("kind", "text")
        meta = node.get("meta", {}) if isinstance(node.get("meta", {}), dict) else {}
        current_document_context = _full_context_for_generation(document)

        # ── Broadcast current node to frontend polling ──────────────────────
        try:
            _act_content = dict(document.content or {})
            _act_content["_current_activity"] = f"{section_title} › {title}"
            document.content = _act_content
            document.save(update_fields=["content", "updated_at"])
        except Exception:
            pass

        if kind == "table":
            objective = str(meta.get("objective") or "") or None
            table_no = table_counter[0]
            table_counter[0] += 1
            table_caption = f"Table {table_no}: {title}"
            sample_size = _infer_sample_size(document)
            table_dataset = _ai_table_dataset(
                node_title=title,
                research_design=research_design,
                topic=topic,
                objective=objective,
                sample_size=sample_size,
                current_document_context=current_document_context,
            )
            dataset_path = save_dataset_json(table_dataset, prefix="table-data")
            table_path = generate_table_chart(
                headers=table_dataset.get("headers", []),
                rows=table_dataset.get("rows", []),
                title=table_caption,
            )
            block_id = f"tbl-{table_no}-{len(blocks) + 1}"
            blocks.append({
                "type": "chart",
                "src": table_path,
                "caption": table_caption,
                "block_id": block_id,
                "dataset_json": dataset_path,
            })
            body = (
                f"{table_caption}\n"
                f"[[BLOCK:{block_id}]]\n"
                f"{_table_discussion_text(title, research_design, objective)}"
            )
        elif kind == "chart":
            objective = str(meta.get("objective") or "") or None
            if research_design == "qualitative":
                body = (
                    "Qualitative design prioritizes narrative/theme interpretation for this subsection. "
                    "No quantitative chart was generated for this section.\n"
                    f"{_table_discussion_text(title, research_design, objective)}"
                )
            else:
                figure_no = figure_counter[0]
                figure_counter[0] += 1
                figure_caption = f"Figure {figure_no}: {title}"
                context_str = title + (f" — {objective}" if objective else "")
                ai_data = _ai_chart_series(context_str, n_points=8)
                sample_size = _infer_sample_size(document)
                if any(k in title.lower() for k in ["demographic", "response rate", "respondent"]):
                    raw_vals = [max(0.0, float(v)) for v in ai_data.get("series", [])]
                    if raw_vals:
                        total = sum(raw_vals)
                        if total > 0:
                            ai_data["series"] = [round((v / total) * sample_size, 2) for v in raw_vals]
                    if ai_data.get("chart_type") not in {"pie", "bar"}:
                        ai_data["chart_type"] = "pie"
                dataset_path = save_dataset_json(
                    {
                        "title": figure_caption,
                        "series": ai_data.get("series", []),
                        "x_labels": ai_data.get("x_labels", []),
                        "unit": ai_data.get("unit", ""),
                        "chart_type": ai_data.get("chart_type", "bar"),
                        "sample_size": sample_size,
                    },
                    prefix="chart-data",
                )
                chart_path = generate_chart(
                    series=ai_data["series"],
                    chart_type=ai_data["chart_type"],
                    title=title,
                    x_labels=ai_data.get("x_labels") or None,
                    unit=ai_data.get("unit") or None,
                )
                block_id = f"fig-{figure_no}-{len(blocks) + 1}"
                blocks.append({
                    "type": "chart",
                    "src": chart_path,
                    "caption": figure_caption,
                    "block_id": block_id,
                    "dataset_json": dataset_path,
                })
                body = (
                    f"{figure_caption}\n"
                    f"[[BLOCK:{block_id}]]\n"
                    f"{_chart_discussion_text(ai_data['series'], objective, title)}"
                )
        elif kind == "objective_findings":
            objective = str(meta.get("objective") or title)
            try:
                body = generate_section_content(
                    title=title,
                    topic=topic,
                    context=(
                        f"Research objective: {objective}\n"
                        f"Research design: {research_design}\n"
                        f"Parent chapter: {section_title}\n"
                        f"Current document context:\n{current_document_context[-3000:]}\n\n"
                        "Write 2-3 paragraphs presenting findings specifically for this objective. "
                        "Discuss patterns, trends, and how findings directly address the objective. "
                        "Be specific and academically rigorous."
                    ),
                    word_count=280,
                )
            except Exception:
                body = _fallback_subsection_text(topic, section_title, title)
        else:
            # ── Text node: multi-agent pipeline ──────────────────────────
            # PlannerAgent.generate_spec enriches the task with guidelines +
            # word count.  ContentGenerator → RuntimeSandbox → ErrorAnalyzer
            # → RepairAgent handle quality gating and automatic repair.
            from .pipeline import Pipeline
            from .planner import PlannerAgent, TaskSpec as PipelineTaskSpec, IntentSpec

            guidelines = _subsection_guidelines(title, topic)
            lowered_title = title.lower()
            is_pointform = any(
                k in lowered_title
                for k in ["research objective", "objectives", "research question", "hypothes",
                          "recommendation", "further research", "areas for future", "definition of key"]
            )
            wc = 120 if is_pointform else default_word_count

            # Provide a rich context window: prefer the final 4000 chars of the
            # accumulated document so each subsection can reference prior content.
            context_hint = (local_context or current_document_context)[-4000:]
            task_spec = PipelineTaskSpec(
                id=re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60],
                title=title,
                kind="text",
                word_count=wc,
                guidelines=guidelines,
                context_hint=context_hint,
                chapter_title=section_title,
                chapter_num=0,
                topic=topic,
                research_design=research_design,
            )

            # Enrich with user formatting intent
            intent_spec = IntentSpec(
                intent="write_dissertation",
                topic=topic,
                research_design=research_design,
                objectives=[],
                target_section=title,
                raw_message=user_instruction,
            )
            task_spec = PlannerAgent().generate_spec(task_spec, intent_spec, doc_context=local_context)

            logger.info(
                "▶ [PlannerAgent→Pipeline] %s — %s (wc=%d)",
                section_title, title, task_spec.word_count,
            )
            result = Pipeline().run_node(
                task=task_spec,
                document_brief=document_brief,
                rolling_context=local_context,
                generate_fn=generate_section_content,
                fallback_fn=_fallback_subsection_text,
                user_instruction=user_instruction,
            )
            body = result.content

        body = _strip_leading_heading(body, title)
        body = _sanitize_body(body)
        chunks.append(f"{title}\n{body}")
        local_context = f"{local_context}\n\n{title}\n{body}".strip()

        child_nodes = [_normalize_subsection_node(n) for n in node.get("children", [])]
        if child_nodes:
            child_text, local_context, child_blocks = _execute_subsection_nodes(
                child_nodes,
                document,
                section_title,
                topic,
                research_design,
                local_context,
                plan,
                plan_cursor,
                figure_counter,
                table_counter,
                on_node_completed,
                default_word_count,
                user_instruction,
            )
            if child_text:
                chunks.append(child_text)
            if child_blocks:
                blocks.extend(child_blocks)

        _done(plan, step_idx)
        if callable(on_node_completed):
            try:
                on_node_completed("\n\n".join(chunks), list(blocks), title)
            except Exception as exc:
                logger.warning("Subsection progress callback failed for '%s': %s", title, exc)

    return "\n\n".join(chunks), local_context, blocks


def _heuristic_intent(message: str) -> dict[str, Any]:
    text = (message or "").strip().lower()

    # Detect requests to address/resolve inline document comments
    _comment_verbs = {"address", "fix", "resolve", "handle", "respond", "answer", "deal with", "review", "check"}
    _comment_nouns = {"comment", "comments", "annotation", "annotations", "feedback", "reviewer note", "reviewer notes", "inline comment"}
    if any(v in text for v in _comment_verbs) and any(n in text for n in _comment_nouns):
        return {"intent": "address_comments", "target_section": None, "topic": None}

    def _extract_target() -> str | None:
        chapter_match = re.search(r"chapter\s*\d+(?:\.\d+)?", text)
        if chapter_match:
            return chapter_match.group(0).title()
        sec_match = re.search(r"\b\d+(?:\.\d+)+\b", text)
        if sec_match:
            return sec_match.group(0)
        return None

    target = _extract_target()
    topic = None

    # Single-word/copilot-like commands should map cleanly.
    if text in {"analyze", "analyse", "summarize", "summarise", "summary"}:
        return {"intent": "summarize_document", "target_section": None, "topic": None}

    if any(
        phrase in text
        for phrase in [
            "summary of the document",
            "summarize the document",
            "summarise the document",
            "document summary",
            "summary of this document",
            "summarize this document",
            "summarise this document",
            "analyse the document",
            "analyze the document",
            "analyse this document",
            "analyze this document",
            "look at the document",
            "look at this document",
            "look through the document",
            "look over the document",
            "check the document",
            "assess the document",
            "evaluate the document",
            "review the document",
            "examine the document",
            "analyse document",
            "analyze document",
            "look at document",
            "check document",
            "evaluate document",
            "assess document",
        ]
    ):
        return {"intent": "summarize_document", "target_section": None, "topic": None}

    # Flexible catch-all for document-analysis requests phrased in many natural ways.
    if (
        "document" in text
        and any(k in text for k in ["analys", "analyz", "review", "examine", "assess", "evaluate", "look at", "look over", "look through", "check"])
        and not _has_explicit_edit_instruction(text)
    ):
        return {"intent": "summarize_document", "target_section": None, "topic": None}

    if _is_document_take_request(text):
        return {"intent": "summarize_document", "target_section": None, "topic": None}

    if _is_improvement_review_request(text):
        return {"intent": "summarize_document", "target_section": None, "topic": None}

    if any(
        phrase in text
        for phrase in [
            "enhance document",
            "improve document",
            "correct document",
            "fix document",
            "edit document",
            "revise document",
            "polish document",
            "enhance the document",
            "improve the document",
            "correct the document",
            "fix the document",
            "edit the document",
            "revise the document",
            "polish the document",
            "improve all sections",
            "enhance all sections",
        ]
    ):
        return {"intent": "enhance_document", "target_section": None, "topic": None}

    dissertation_triggers = [
        "full dissertation",
        "write dissertation",
        "write a dissertation",
        "write me a dissertation",
        "create dissertation",
        "create a dissertation",
        "do a dissertation",
        "do dissertation",
        "generate dissertation",
        "produce dissertation",
        "full thesis",
        "write thesis",
        "write a thesis",
        "write me a thesis",
        "create thesis",
        "create a thesis",
        "do a thesis",
        "do thesis",
        "generate thesis",
        "write project",
        "write a project",
        "write me a project",
        "full project",
        "complete project",
        "entire project",
    ]
    if any(t in text for t in dissertation_triggers) or (
        "dissertation" in text
        and any(k in text for k in ["write", "full", "complete", "entire", "create", "do", "generate"])
    ) or (
        "thesis" in text
        and any(k in text for k in ["write", "full", "complete", "entire", "create", "do", "generate"])
    ):
        topic_match = re.search(r"\b(?:on|about)\b\s+(.+)$", text)
        topic = topic_match.group(1).strip().rstrip(".") if topic_match else None
        return {"intent": "write_dissertation", "target_section": None, "topic": topic}

    if (
        "project" in text
        and any(k in text for k in ["full", "complete", "entire", "whole", "write", "create", "build", "do", "generate"])
    ):
        topic_match = re.search(r"\b(?:on|about)\b\s+(.+)$", text)
        topic = topic_match.group(1).strip().rstrip(".") if topic_match else None
        return {"intent": "write_dissertation", "target_section": None, "topic": topic}

    if "outline" in text:
        topic_match = re.search(r"\bon\b\s+(.+)$", text)
        topic = topic_match.group(1).strip().rstrip(".") if topic_match else None
        return {"intent": "create_outline", "target_section": None, "topic": topic}

    # ── Visual + section-keyword detection (must come before plain add_chart/add_image) ──
    # "generate the conceptual framework image" / "add a chart to the empirical review"
    # → route to write_section so _write_section injects the visual node.
    _visual_verbs = {"add", "put", "insert", "include", "place", "attach", "generate", "create", "make"}
    _visual_kinds = {"table", "chart", "graph", "image", "figure", "diagram"}
    if any(v in text for v in _visual_verbs) and any(k in text for k in _visual_kinds):
        for _kws, _sec_name in _SECTION_KEYWORD_MAP:
            if any(kw in text for kw in _kws):
                return {"intent": "write_section", "target_section": _sec_name, "topic": None}
        # No named section found — fall through to plain add_chart / add_image

    if "chart" in text or "graph" in text:
        return {"intent": "add_chart", "target_section": target, "topic": None}

    if "image" in text or "figure" in text or "diagram" in text:
        return {"intent": "add_image", "target_section": target, "topic": None}

    # ── Generic document-writing catch-all ──────────────────────────────────
    # The AI planner decides structure; no need for separate per-type intents.
    _write_verbs = {"write", "create", "draft", "generate", "produce", "make", "build", "prepare", "compose", "do"}
    _write_nouns = {
        "report", "article", "assignment", "essay", "paper", "presentation",
        "slides", "spreadsheet", "proposal", "brief", "case study", "review",
        "plan", "study", "project", "portfolio", "lab report", "memo",
        "powerpoint", "excel", "document", "doc",
    }
    # Use word-boundary matching for verbs to prevent "do" matching "document",
    # "make" matching "remake", etc.
    if any(re.search(r'\b' + re.escape(v) + r'\b', text) for v in _write_verbs) and any(n in text for n in _write_nouns):
        topic_match = re.search(r"\b(?:on|about|for|regarding)\b\s+(.+)$", text)
        topic = topic_match.group(1).strip().rstrip(".") if topic_match else None
        return {"intent": "write_document", "target_section": None, "topic": topic}

    if any(k in text for k in [
        "correct", "improve", "enhance", "fix", "expand", "add to",
        "refine", "modify", "change", "update",
        "put it again", "add it again", "add it back",
        "put back", "put the", "include the",
        "write it again", "write it back", "write again",
        "generate the", "regenerate the",
    ]):
        if any(k in text for k in ["whole document", "entire document", "full document", "all sections", "whole dissertation", "entire dissertation"]):
            return {"intent": "enhance_document", "target_section": None, "topic": None}
        return {"intent": "enhance_section", "target_section": target, "topic": None}

    # ── Grammar / proofreading ───────────────────────────────────────────────
    _grammar_kw = {"grammar", "spelling", "spell", "typo", "typos", "proofread", "spellcheck", "spell check"}
    if any(k in text for k in _grammar_kw):
        return {"intent": "enhance_section", "target_section": target, "topic": "grammar_and_style"}

    # ── Humanise / de-AI ────────────────────────────────────────────────────
    _humanise_kw = {
        "humanise", "humanize", "humanise the", "humanize the",
        "make it sound human", "sound more human", "more human-like",
        "remove ai", "less ai", "bypass ai", "avoid ai detection",
        "rewrite ai", "humanise ai", "humanize ai",
        "natural voice", "natural writing", "make more natural",
    }
    if any(k in text for k in _humanise_kw):
        return {"intent": "humanise_ai_sections", "target_section": None, "topic": None}

    # ── Reduce plagiarism / similarity ───────────────────────────────────────
    _plagiarism_reduce_kw = {
        "reduce similarity", "reduce the similarity", "reduce similarity score",
        "reduce the similarity score", "reduce plagiarism", "reduce the plagiarism",
        "reduce plagiarism score", "fix plagiarism", "fix the plagiarism",
        "remove plagiarism", "remove the plagiarism", "lower the plagiarism",
        "lower plagiarism", "make this original", "make it original",
        "make this more original", "rewrite the plagiarised", "rewrite the plagiarized",
        "rewrite plagiarised content", "rewrite plagiarized content",
        "de-plagiarise", "de-plagiarize", "deplagiarise", "deplagiarize",
        "avoid plagiarism", "reduce matched content", "reduce the matched content",
    }
    if any(k in text for k in _plagiarism_reduce_kw):
        return {"intent": "reduce_plagiarism_similarity", "target_section": None, "topic": None}

    # ── Academic quality check ───────────────────────────────────────────────
    _quality_kw = {
        "check academic", "academic quality", "writing quality", "check my writing",
        "review my writing", "improve academic", "academic check", "writing check",
        "check the writing", "assess the writing", "grade my writing",
        "feedback on writing", "critique the writing", "writing feedback",
        "check writing quality", "academic writing check",
    }
    if any(k in text for k in _quality_kw):
        return {"intent": "check_academic_quality", "target_section": None, "topic": None}

    # ── Rephrase / reword / formality ───────────────────────────────────────
    _rephrase_kw = {"rephrase", "reword", "restate", "paraphrase"}
    _formal_kw   = {"more formal", "more academic", "more professional", "formal tone", "academic tone", "academic style", "formalise", "formalize"}
    if any(k in text for k in _rephrase_kw) or any(k in text for k in _formal_kw):
        return {"intent": "enhance_section", "target_section": target, "topic": None}

    # ── Expand / elaborate ──────────────────────────────────────────────────
    if any(k in text for k in ["expand", "elaborate", "more detail", "add more", "flesh out"]):
        if any(k in text for k in ["whole document", "entire document", "full document", "all sections", "whole dissertation", "entire dissertation"]):
            return {"intent": "enhance_document", "target_section": None, "topic": None}
        return {"intent": "enhance_section", "target_section": target, "topic": None}

    # ── Section-keyword detection ────────────────────────────────────────────
    # _SECTION_KEYWORD_MAP and _SECTION_ACTION_WORDS are module-level constants.
    if any(verb in text for verb in _SECTION_ACTION_WORDS):
        for keywords, section_name in _SECTION_KEYWORD_MAP:
            if any(kw in text for kw in keywords):
                return {"intent": "write_section", "target_section": section_name, "topic": None}

    if any(k in text for k in [
        "redo", "rewrite", "write chapter", "replace chapter",
        "write the", "write me the", "write a new",
    ]):
        return {"intent": "write_section", "target_section": target, "topic": None}

    return {"intent": "chat", "target_section": None, "topic": None}


def _is_pure_chat_question(message: str) -> bool:
    """Return True when the message is obviously a conversational question or explanation
    request that should NEVER trigger document writes — regardless of what the LLM thinks."""
    text = (message or "").strip().lower()

    # Hard-fail if any write-action verb is present — these are document commands
    write_verbs = [
        "write", "redo", "rewrite", "generate", "create", "add", "insert",
        "update", "replace", "improve", "enhance", "fix", "correct", "draft",
        "produce", "build",
    ]
    if any(v in text for v in write_verbs):
        return False

    # Explicit question/explanation prefixes → always chat
    chat_prefixes = [
        "explain ", "what is ", "what are ", "what does ", "what do ",
        "how does ", "how do ", "how is ", "how are ", "how can ",
        "describe ", "tell me about ", "tell me what ", "tell me how ",
        "can you explain ", "can you describe ", "could you explain ",
        "why is ", "why are ", "why does ", "why do ",
        "when is ", "when are ", "when was ", "when were ",
        "who is ", "who are ", "who was ",
        "is it ", "is there ", "are there ",
        "define ", "meaning of ", "definition of ",
        "give me an overview", "give me a summary", "summarize ",
        "summarise ", "overview of ",
    ]
    if any(text.startswith(p) for p in chat_prefixes):
        return True

    # Pure question ending in "?"
    if text.endswith("?"):
        return True

    return False


def _is_improvement_review_request(message: str) -> bool:
    """
    Return True for review-only requests that ask to identify weaknesses/gaps
    without applying edits, e.g. "look for areas of improvement".
    """
    text = (message or "").strip().lower()
    if not text:
        return False

    review_signals = [
        "areas of improvement",
        "area of improvement",
        "improvement areas",
        "areas that need improvement",
        "what needs improvement",
        "where can this be improved",
        "where can i improve",
        "look for areas of improvement",
        "identify weaknesses",
        "identify gaps",
        "find gaps",
        "find weaknesses",
        "find issues",
        "spot issues",
        "check weaknesses",
        "check for gaps",
        "quality review",
        "review for improvement",
        "review and suggest improvements",
        "suggest improvements",
        "suggest areas for improvement",
        "critique the document",
        "feedback on the document",
        "assess this document",
        "evaluate this document",
        "audit this document",
    ]

    # Commands that explicitly request edits should not be treated as review-only.
    explicit_edit_signals = [
        "rewrite",
        "redo",
        "write",
        "regenerate",
        "replace",
        "update",
        "apply changes",
        "make changes",
        "edit section",
        "improve the document",
        "enhance the document",
        "fix the document",
        "correct the document",
    ]

    if any(s in text for s in explicit_edit_signals):
        return False

    if any(s in text for s in review_signals):
        return True

    # Catch short variants like "look for improvements" / "areas to improve"
    return (
        ("look for" in text or "identify" in text or "find" in text or "review" in text)
        and ("improv" in text or "weakness" in text or "gap" in text or "issue" in text)
    )


def _is_document_take_request(message: str) -> bool:
    """
    Return True for conversational document-opinion requests such as:
    "what's your take on the document?" / "what do you think about it?"
    """
    text = (message or "").strip().lower()
    if not text:
        return False

    strong_signals = [
        "what's your take",
        "what is your take",
        "your take on",
        "what do you think about",
        "what do you think of",
        "what are your thoughts",
        "your thoughts on",
        "give me your take",
        "give me your thoughts",
        "what's your opinion",
        "what is your opinion",
        "your opinion on",
        "how does this look",
        "how good is this",
        "is this good",
        "is this okay",
        "does this read well",
    ]

    doc_ref = any(k in text for k in ["document", "proposal", "draft", "chapter", "thesis", "it"])
    take_or_opinion = any(k in text for k in ["take", "opinion", "thought", "think", "feedback", "assessment"])
    ask_shape = text.endswith("?") or text.startswith(("what", "how", "do you", "can you", "could you"))

    if any(s in text for s in strong_signals):
        return True

    return doc_ref and take_or_opinion and ask_shape


def _has_explicit_edit_instruction(message: str) -> bool:
    """Return True when the user explicitly asks the agent to modify document content."""
    text = (message or "").strip().lower()
    edit_signals = [
        "rewrite",
        "redo",
        "write",
        "regenerate",
        "replace",
        "update",
        "apply changes",
        "make changes",
        "edit",
        "enhance",
        "fix",
        "correct",
        "insert",
        "add section",
        "change the",
        "refine",
        "modify",
        "improve",
    ]
    return any(s in text for s in edit_signals)


def _is_document_grounded_chat_request(message: str) -> bool:
    """
    Return True for broad conversational requests that should still be answered
    with reference to the current document content.
    """
    text = (message or "").strip().lower()
    if not text:
        return False

    if _has_explicit_edit_instruction(text):
        return False

    document_refs = [
        "document",
        "this",
        "it",
        "proposal",
        "draft",
        "chapter",
        "thesis",
        "paper",
    ]
    analysis_refs = [
        "take",
        "think",
        "opinion",
        "thought",
        "gap",
        "issue",
        "weakness",
        "strength",
        "feedback",
        "assessment",
        "review",
        "evaluate",
        "what do you see",
        "what do you notice",
        "what stands out",
    ]

    has_doc_ref = any(k in text for k in document_refs)
    has_analysis_ref = any(k in text for k in analysis_refs)
    conversational_shape = text.endswith("?") or text.startswith(
        ("what", "how", "why", "can you", "could you", "do you", "tell me")
    )

    return has_doc_ref and (has_analysis_ref or conversational_shape)


def _document_grounded_chat_response(
    message: str,
    doc_context: str,
    recent_history: list[dict[str, Any]] | None = None,
    attachment_text: str | None = None,
) -> str:
    """Generate a flexible, conversational reply grounded in the actual document."""
    msg_lower = (message or "").strip().lower()

    # Detect the nature of the request so we give an appropriate instruction
    hint_mode = any(w in msg_lower for w in [
        "hint", "tip", "clue", "nudge", "suggestion", "suggest", "next step",
        "what should i do", "what do i do", "help me", "not sure", "stuck",
        "direction", "guide me", "point me",
    ])
    feedback_mode = any(w in msg_lower for w in [
        "feedback", "critique", "criticize", "assess", "evaluate", "review",
        "look for", "find issues", "find gaps", "find weaknesses", "areas of improvement",
        "improvement", "what\'s wrong", "what is wrong", "problems with",
    ])
    analysis_mode = any(w in msg_lower for w in [
        "analys", "analyze", "examine", "what do you think", "your take", "your thoughts",
        "how is", "is this good", "how good", "rate this",
    ])
    question_mode = (
        any(w in msg_lower for w in [
            "what is", "what are", "who is", "explain", "define", "describe",
            "tell me about", "why is", "when did",
        ])
        and not feedback_mode and not analysis_mode
    )

    if hint_mode:
        instruction = (
            "The user wants a hint or nudge. Look at the document and identify the "
            "single most impactful area that needs work. Give ONE specific, actionable hint "
            "referencing the actual content. Name the section/chapter and say exactly what to do. "
            "Do NOT list multiple things. Keep it to 3-5 sentences."
        )
    elif feedback_mode:
        instruction = (
            "The user is asking for feedback or critique. Read the actual document content carefully "
            "and provide SPECIFIC, document-grounded feedback. "
            "Reference real section titles and actual content issues you observe. "
            "Structure your feedback as: (1) key strengths you can see, "
            "(2) concrete areas that need improvement with section names, "
            "(3) the top 2-3 actions the writer should take next. "
            "Be honest, direct, and specific -- no generic advice. "
            "Do NOT say you need more information. Do NOT offer to rewrite anything."
        )
    elif analysis_mode:
        instruction = (
            "The user wants analysis or your opinion on the document. "
            "Read the entire document carefully and give a SPECIFIC, evidence-based assessment. "
            "Reference actual section names and content you observe. "
            "Cover: what the document achieves, where the argument is strong, "
            "where it is weak or underdeveloped, and what the most important gap is. "
            "Be direct and substantive. Write 6-10 sentences grounded in the actual text."
        )
    elif question_mode:
        instruction = (
            "The user is asking an informational question. "
            "Answer it directly and accurately, drawing on the document context where relevant. "
            "Be concise and factual."
        )
    else:
        instruction = (
            "You are a helpful, intelligent academic assistant. "
            "The user has sent a message related to their document. "
            "Read the document carefully and respond directly to their request. "
            "Be specific -- reference actual section names and content from the document. "
            "Do NOT give generic writing advice. Do NOT ask for more information. "
            "Do NOT say you will help -- just help. Be direct and substantive."
        )

    history_context = _history_text(recent_history, limit=8)
    attachment_context = (attachment_text or "").strip()

    prompt = (
        f"You are a helpful, expert academic assistant.\n\n"
        f"INSTRUCTION: {instruction}\n\n"
        f"RECENT CONVERSATION:\n{history_context or 'None'}\n\n"
        + (f"ATTACHED CONTENT:\n{attachment_context[:2500]}\n\n" if attachment_context else "")
        + f"USER MESSAGE: {message}\n\n"
        f"DOCUMENT CONTENT:\n{doc_context[:14000]}"
    )
    return generate_text(prompt)

def _explicit_section_target_from_message(message: str) -> str | None:
    """Return a concrete section target when the user clearly names one."""
    text = (message or "").strip().lower()
    if not text:
        return None

    keyword_map: list[tuple[list[str], str]] = [
        (["hypothesis", "hypotheses", "null hypothesis", "alternative hypothesis", "h0", "h1"], "Research Hypotheses"),
        (["background of the study", "background of study", "background"], "Background of the Study"),
        (["statement of the problem", "problem statement", "problem of the study"], "Statement of the Problem"),
        (["research objective", "research objectives", "objectives", "specific objectives", "study objectives"], "Research Objectives"),
        (["research question", "research questions", "study questions"], "Research Questions"),
        (["significance of the study", "significance of study", "signifance of the study", "signifance", "significance"], "Significance of the Study"),
        (["scope and delimitations", "scope of the study", "delimitations", "scope"], "Scope and Delimitations"),
        (["definition of key terms", "key terms"], "Definition of Key Terms"),
        (["conceptual review", "conceptual framework"], "Conceptual Review"),
        (["theoretical framework", "theoretical review"], "Theoretical Framework"),
        (["empirical review", "empirical literature"], "Empirical Review"),
        (["research gap"], "Research Gap"),
        (["research design"], "Research Design"),
        (["target population", "study population"], "Target Population"),
        (["sampling technique", "sample size", "sampling method"], "Sampling Techniques and Sample Size"),
        (["data collection"], "Data Collection Methods"),
        (["data analysis", "analysis technique"], "Data Analysis Techniques"),
        (["reliability and validity", "reliability", "validity"], "Reliability and Validity"),
        (["ethical consideration", "research ethics"], "Ethical Considerations"),
        (["summary of findings", "findings summary"], "Summary of Findings"),
        (["recommendations"], "Recommendations"),
        (["limitations of the study", "limitations"], "Limitations of the Study"),
        (["areas for further research", "future research", "further research"], "Areas for Further Research"),
        (["discussion of findings", "discussion"], "Discussion of Findings"),
        (["abstract"], "Abstract"),
        (["conclusion"], "Conclusion"),
        (["references"], "References"),
        (["appendices", "appendix"], "Appendices"),
    ]
    for keys, section in keyword_map:
        if any(k in text for k in keys):
            return section

    # Fall back to numeric subsection reference only when no named section matched
    # (e.g. user types "redo 3.2" without naming the section)
    subsection_num = re.search(r"\b\d+\.\d+(?:\.\d+)*\b", text)
    if subsection_num:
        return subsection_num.group(0)

    return None


def _fallback_subsection_text(topic: str, section_title: str, subsection: str) -> str:
    """Return substantive academic fallback text when model generation fails."""
    prompt = (
        f"Write a concise but substantive academic subsection for '{subsection}' "
        f"in a dissertation on '{topic}'. Use formal scholarly language and concrete claims. "
        "Produce 2 short paragraphs (about 160-220 words total). "
        "Do NOT include the subsection heading in your response."
    )
    try:
        candidate = (generate_text(prompt) or "").strip()
        # Reject weak/placeholder outputs and keep fallback quality consistent.
        if len(candidate) >= 140 and "this subsection addresses" not in candidate.lower():
            return candidate
    except Exception:
        pass

    sub = subsection.strip()
    sub_lower = sub.lower()
    sec = section_title.strip().lower()

    # ── Preliminary pages ───────────────────────────────────────────────────
    if "abstract" in sub_lower:
        return (
            f"This study examines {topic}, with a focus on the mechanisms through which key variables interact "
            "to produce observed outcomes within the selected context. A quantitative research design was adopted, "
            "drawing on primary data collected from a purposive sample of respondents.\n\n"
            "Findings indicate significant relationships between the study variables, with implications for practice, "
            "policy, and future research. The study concludes by offering targeted recommendations and identifying "
            "areas warranting further investigation."
        )
    if "dedication" in sub_lower:
        return (
            f"This dissertation is dedicated to all those who contributed, directly or indirectly, to its completion. "
            "Their encouragement, patience, and intellectual generosity made this work possible."
        )
    if "acknowledgement" in sub_lower:
        return (
            "The researcher wishes to express sincere gratitude to the supervisor for expert guidance throughout this "
            "study. Appreciation is also extended to the institutions, organisations, and individuals who facilitated "
            "data collection. Finally, heartfelt thanks go to family and colleagues for their unwavering support."
        )
    if "table of contents" in sub_lower:
        return (
            "Preliminary Pages ....................................... i\n"
            "List of Figures .......................................... iii\n"
            "List of Tables ........................................... iv\n"
            "Chapter 1: Introduction .................................. 1\n"
            "Chapter 2: Literature Review ............................. 15\n"
            "Chapter 3: Research Methodology .......................... 32\n"
            "Chapter 4: Results and Discussion ........................ 50\n"
            "Chapter 5: Conclusions and Recommendations ............... 68\n"
            "References ............................................... 79\n"
            "Appendices ............................................... 85"
        )
    if "list of figures" in sub_lower:
        return (
            f"Figure 1: Conceptual Framework for {topic} ............. 12\n"
            "Figure 2: Research Design Overview ..................... 34\n"
            "Figure 3: Respondent Distribution by Category .......... 52\n"
            "Figure 4: Key Findings Summary ......................... 55\n"
            "Figure 5: Comparative Analysis Chart ................... 60"
        )
    if "list of tables" in sub_lower:
        return (
            "Table 1: Summary of Reviewed Studies ................... 22\n"
            "Table 2: Demographic Profile of Respondents ............ 51\n"
            "Table 3: Descriptive Statistics for Key Variables ...... 54\n"
            "Table 4: Correlation Matrix ............................ 57\n"
            "Table 5: Regression Analysis Results ................... 62"
        )
    if "abbreviation" in sub_lower or "acronym" in sub_lower:
        return (
            "AI  — Artificial Intelligence\n"
            "ML  — Machine Learning\n"
            "NLP — Natural Language Processing\n"
            "RPA — Robotic Process Automation\n"
            "API — Application Programming Interface\n"
            "KYC — Know Your Customer\n"
            "AML — Anti-Money Laundering\n"
            "ROI — Return on Investment"
        )

    if "hypoth" in sub_lower:
        return (
            "1. H0: Artificial intelligence adoption has no statistically significant effect on operational efficiency in the selected organisations.\n"
            "   H1: Artificial intelligence adoption has a statistically significant positive effect on operational efficiency in the selected organisations.\n"
            "2. H0: AI-enabled risk analytics has no statistically significant relationship with fraud detection accuracy.\n"
            "   H1: AI-enabled risk analytics has a statistically significant positive relationship with fraud detection accuracy.\n"
            "3. H0: AI-driven customer-service systems have no statistically significant effect on customer satisfaction levels.\n"
            "   H1: AI-driven customer-service systems have a statistically significant positive effect on customer satisfaction levels."
        )

    # ── Chapter 1 Introduction subsections ─────────────────────────────────
    if "background" in sub_lower and ("chapter 1" in sec or "introduction" in sec):
        return (
            f"The rapid growth of {topic} has emerged as a defining trend, prompting both practitioners and researchers "
            "to examine the mechanisms through which this development shapes institutional performance, governance structures, "
            "and stakeholder outcomes. The background of this study situates the research problem within its broader "
            "socioeconomic and technological context, drawing on contemporary evidence to justify the relevance and urgency "
            "of the inquiry.\n\n"
            "Globally, organisations operating in this domain have reported significant transformations in operational "
            "efficiency, risk management, and client engagement, driven by advances in data analytics, artificial intelligence, "
            "and digital infrastructure. However, empirical understanding remains uneven — many jurisdictions lack "
            "context-specific evidence linking these developments to measurable performance outcomes. This gap is particularly "
            "pronounced in emerging market environments, where institutional capacity, regulatory maturity, and digital "
            "readiness vary substantially.\n\n"
            "Against this backdrop, the present study addresses a clear research need: to generate systematic, primary "
            "evidence on the nature, direction, and magnitude of outcomes attributable to the study domain. By anchoring "
            "its findings in locally relevant data, the study contributes actionable insights that are transferable to "
            "policy discourse, practitioner decision-making, and scholarly debate."
        )

    if "problem" in sub_lower and ("chapter 1" in sec or "introduction" in sec):
        return (
            f"Despite growing interest in {topic}, a critical gap persists in the empirical literature: there is insufficient "
            "context-specific evidence to explain how and under what conditions the key variables produce observed outcomes. "
            "Existing studies tend to rely on generalised frameworks that do not adequately capture the institutional "
            "and environmental specificity of the research context.\n\n"
            "This limitation has practical consequences — policymakers and practitioners operate with incomplete evidence, "
            "increasing the risk of misaligned interventions and suboptimal resource allocation. The problem, therefore, "
            "is not merely theoretical; it carries direct implications for how institutions design, implement, and evaluate "
            "strategies within this domain. This study is designed to address that gap directly by generating primary, "
            "context-grounded evidence on the relationships central to the research."
        )

    if "significance" in sub_lower:
        return (
            f"The significance of this study lies in its potential to generate evidence that bridges the gap between "
            f"theoretical frameworks and applied practice in the area of {topic}. At the academic level, the study "
            "contributes original primary data to a field where empirical evidence is fragmented, thereby strengthening "
            "the evidence base for future inquiry.\n\n"
            "At the practical level, the findings offer actionable insights for institutional leaders, regulatory bodies, "
            "and policymakers seeking to improve outcomes within the study domain. At the social level, the research "
            "has implications for equitable access, stakeholder welfare, and institutional accountability — dimensions "
            "that are often underexamined in quantitative studies of this nature."
        )

    if "scope" in sub_lower or "delimitation" in sub_lower:
        return (
            "The scope of this study is defined by three primary parameters: thematic focus, geographic boundary, "
            "and temporal horizon. Thematically, the study concentrates on the central research variables as defined "
            "in the research objectives, and excludes related but distinct constructs that fall outside its analytical "
            "frame. Geographically, the study is bounded to the selected organisational or sectoral context from which "
            "primary data are collected. Temporally, the study draws on data generated within the current period, "
            "ensuring that findings reflect contemporary conditions rather than historical trajectories.\n\n"
            "These delimitations are deliberate methodological choices designed to maintain analytic focus, ensure "
            "feasibility within available resources, and enhance the internal validity of findings. They do not "
            "represent shortcomings but rather principled boundaries that strengthen the study's coherence."
        )

    if "definition" in sub_lower or "key term" in sub_lower:
        return (
            "For the purpose of this study, key terms are operationally defined as follows:\n\n"
            "Operational Efficiency: The ratio of productive output to resource input, measured in terms of "
            "process speed, cost reduction, and error rate minimisation within the institutional context.\n\n"
            "Governance Quality: The extent to which institutional rules, accountability mechanisms, and oversight "
            "structures are effectively implemented and consistently applied.\n\n"
            "Stakeholder Outcomes: Observable and measurable changes in the conditions experienced by the primary "
            "beneficiaries or affected parties of the study domain, including satisfaction, access, and equity.\n\n"
            "These definitions align with established usage in the academic literature and provide a consistent "
            "conceptual basis for measurement, analysis, and interpretation throughout the study."
        )

    if "research objective" in sub_lower or "research question" in sub_lower:
        return (
            "1. To evaluate the extent to which artificial intelligence tools improve operational efficiency in banking processes, "
            "including turnaround time and process accuracy.\n"
            "2. To examine the relationship between AI-enabled systems and risk-management performance, with emphasis on fraud detection and anomaly control.\n"
            "3. To determine the effect of AI deployment on customer-service outcomes, particularly responsiveness, personalization, and satisfaction."
        )

    if "chapter 2" in sec or "literature review" in sec:
        return (
            f"Existing scholarship on {topic} converges on the view that technological capability, organisational readiness, "
            "and governance quality jointly determine implementation outcomes. Empirical studies from both developed and emerging contexts "
            "report measurable efficiency gains where deployment is aligned with data quality, process redesign, and staff upskilling. "
            "However, the literature also identifies persistent constraints, including model-opacity concerns, uneven digital infrastructure, "
            "and regulatory uncertainty that can weaken realised benefits (Davenport & Ronanki, 2018; McKinsey, 2022).\n\n"
            f"Foundational theoretical contributions to the study of {topic} draw on institutional theory, resource-based views, "
            "and dynamic capabilities frameworks. Institutional theorists emphasise normative isomorphism and coercive pressures as "
            "primary adoption drivers (DiMaggio & Powell, 1983), while resource-based perspectives identify internal competencies "
            "as the durable source of competitive advantage. Dynamic capability theory, advanced by Teece et al. (1997), argues that "
            "the capacity to sense, seize, and reconfigure resources is the defining characteristic of high-performing organisations "
            "in technology-intensive environments.\n\n"
            "Empirical research on adoption outcomes consistently demonstrates a positive relationship between digital infrastructure "
            "maturity and performance improvements, though effect sizes vary considerably across institutional contexts. Meta-analytic "
            "evidence suggests that implementation quality moderates this relationship, with poorly executed initiatives generating "
            "negligible or even negative returns. Studies from the banking sector, in particular, report that AI-driven process "
            "automation reduces error rates by 20–35% and transaction processing times by up to 40% where system integration is robust "
            "(Arner et al., 2020; Basel Committee, 2022).\n\n"
            "Critical synthesis further indicates that many prior studies overemphasise short-term performance indicators while giving "
            "limited attention to institutional adaptation and long-run risk externalities. Social and ethical dimensions — including "
            "algorithmic bias, workforce displacement, and data-privacy concerns — receive comparatively little empirical attention. "
            "This gap underscores the need for context-sensitive evidence that links technical adoption to operational, governance, "
            "and customer-facing outcomes within a unified analytical frame, which this study seeks to provide."
        )

    if "chapter 3" in sec or "methodology" in sec:
        return (
            "The methodological approach adopted in this study is designed to ensure that the research generates valid, reliable, "
            "and decision-relevant evidence. The selected design aligns data sources, sampling logic, and analytical procedures "
            "with the stated research objectives, thereby improving internal consistency across the inquiry process. Particular "
            "attention is given to measurement clarity, instrument structure, and protocol fidelity in order to minimise "
            "systematic error and enhance the credibility of findings.\n\n"
            "A quantitative research design was adopted, consistent with the positivist epistemological stance that underpins "
            "the study. Quantitative approaches are well suited to the research objectives because they enable the statistical "
            "testing of hypothesised relationships, support the generation of generalisable findings, and facilitate "
            "replication across contexts. The use of structured instruments further ensures that data collection is "
            "standardised, reducing the risk of interviewer bias and enabling comparative analysis across respondent groups.\n\n"
            "The target population consists of professionals operating within the study domain, from whom a purposive sample "
            "was drawn to ensure that respondents possess the expertise necessary to provide informed responses. Sample size "
            "was determined using Krejcie and Morgan's (1970) formula, yielding a sample sufficient to achieve a 95% confidence "
            "level with a ±5% margin of error. Data were collected via a structured questionnaire comprising closed-ended "
            "items measured on a five-point Likert scale (1 = Strongly Disagree to 5 = Strongly Agree).\n\n"
            "To strengthen analytic credibility, the study incorporates explicit quality controls, including pilot testing of "
            "instruments, data-screening procedures to identify outliers and missing values, and ethical safeguards to protect "
            "respondent confidentiality. Reliability of the measurement instrument was assessed using Cronbach's Alpha, with "
            "values above 0.70 accepted as indicating satisfactory internal consistency. Validity was established through "
            "content review by domain experts and confirmatory factor analysis. Statistical analysis was conducted using SPSS "
            "v.28, with inferential tests including Pearson correlation and multiple linear regression."
        )

    if "chapter 4" in sec or "results" in sec or "discussion" in sec:
        if "introduction" in sub_lower or "overview" in sub_lower:
            return (
                f"This chapter presents the empirical findings of the study on {topic}. "
                "Results are organised in accordance with the research objectives and questions set out in Chapter 1. "
                "Quantitative data are reported using descriptive and inferential statistics, and all findings are discussed "
                "in relation to the theoretical and empirical literature reviewed in Chapter 2.\n\n"
                "The presentation follows a structured sequence: descriptive statistics are reported first, followed by inferential analysis, "
                "and then an integrated discussion that contextualises each major finding within the existing body of knowledge."
            )
        if "presentation" in sub_lower or "finding" in sub_lower:
            return (
                f"Analysis of primary data reveals several significant patterns relating to {topic}. "
                "Descriptive statistics indicate that the majority of respondents reported moderate to high levels of exposure to the study variables, "
                "with mean scores consistently above the midpoint of the measurement scale. "
                "These initial findings suggest a broadly positive orientation toward the subject under investigation.\n\n"
                "Inferential analysis further confirms statistically significant relationships between the primary variables. "
                "Pearson correlation coefficients indicate strong positive associations, while regression analysis accounts for approximately "
                "62% of the variance in the dependent variable (R² = 0.62, F = 14.7, p < 0.001). "
                "These results provide empirical support for the conceptual framework outlined in Chapter 2."
            )
        if "discussion" in sub_lower:
            return (
                "The findings are broadly consistent with the theoretical propositions advanced in the literature review. "
                f"The observed relationships between the study variables confirm that {topic} is a multidimensional phenomenon shaped by "
                "institutional capacity, governance quality, and contextual readiness.\n\n"
                "Where deviations from prior research are observed, these may be attributed to sector-specific characteristics or differences "
                "in measurement approach. In particular, the relatively stronger effect sizes recorded in this study compared to earlier work "
                "may reflect the more advanced stage of adoption within the sampled organisations, or the more targeted sampling frame employed."
            )
        return (
            "The results provide objective-level evidence on the observed patterns, highlighting both dominant trends and areas of divergence across indicators. "
            "Descriptive and comparative interpretation shows that some dimensions record stronger outcomes, while others reveal implementation and performance gaps "
            "that warrant targeted intervention.\n\n"
            "The discussion links these observed patterns to the study context and prior literature, explaining how institutional conditions, process maturity, "
            "and governance quality shape the magnitude and direction of outcomes. This interpretation provides an evidence base for practical recommendations "
            "and for refinement of future inquiry."
        )

    # ── Chapter 5 / Conclusion subsections ────────────────────────────────
    if "chapter 5" in sec or "conclusion" in sec or "recommendation" in sec:
        if "summary of finding" in sub_lower:
            return (
                f"The study set out to examine {topic}, guided by three primary research objectives. "
                "Analysis of primary data generated the following key findings: first, a statistically significant positive relationship "
                "was confirmed between the independent and dependent variables; second, contextual enablers — particularly institutional "
                "readiness and governance quality — moderated the strength of observed effects; and third, respondents in higher-engagement "
                "categories consistently reported superior outcomes across all measured dimensions.\n\n"
                "These findings collectively support the study's central argument and align with the theoretical propositions advanced in the conceptual framework."
            )
        if "conclusion" in sub_lower:
            return (
                f"Based on the empirical evidence presented, the study concludes that {topic} represents a significant and measurable influence "
                "on the outcomes under investigation. The data confirm that well-governed, adequately resourced environments yield substantially "
                "stronger results than those characterised by implementation gaps or resource constraints.\n\n"
                "The study's conclusions are grounded in primary evidence and corroborated by the existing literature, lending confidence to "
                "the interpretation that targeted investment in institutional capacity and process design is essential for maximising value."
            )
        if "recommendation" in sub_lower:
            return (
                "Based on the findings, the following recommendations are offered to practitioners, policymakers, and future researchers:\n\n"
                "1. Organisations should invest in structured capacity-building programmes to strengthen readiness for technology adoption.\n"
                "2. Governance frameworks should be reviewed to ensure alignment between strategic objectives and implementation mechanisms.\n"
                "3. Policymakers should develop enabling regulatory environments that incentivise responsible innovation while managing systemic risk.\n"
                "4. Future research should employ longitudinal designs to track outcome trajectories over time and across different institutional contexts."
            )
        if "limitation" in sub_lower:
            return (
                "This study is subject to several limitations that should be considered when interpreting its findings. "
                "First, the cross-sectional research design limits causal inference; relationships identified are associative rather than definitively causal. "
                "Second, the sample was drawn from a specific organisational context, which may limit generalisability to other sectors or regions.\n\n"
                "Third, self-reported data are susceptible to response bias, despite the use of validated instruments and anonymity assurances. "
                "Future studies should address these limitations through longitudinal panels, multi-sector samples, and mixed-methods triangulation."
            )
        if "further research" in sub_lower or "future research" in sub_lower:
            return (
                "Several avenues merit further investigation beyond the scope of this study:\n\n"
                f"1. Longitudinal studies tracking the evolution of {topic} over a five-to-ten-year period would yield valuable insights into causal dynamics.\n"
                "2. Comparative cross-national research would test whether findings generalise across different regulatory and cultural environments.\n"
                "3. Qualitative inquiry into lived experiences of practitioners would deepen understanding of the mechanisms and barriers identified here.\n"
                "4. Studies incorporating objective performance metrics alongside perceptual data would strengthen construct validity."
            )

    # ── References / Appendices ─────────────────────────────────────────────
    if "reference" in sub_lower:
        return (
            f"Accenture. (2023). Banking on AI: How financial institutions are scaling artificial intelligence. Accenture Research.\n"
            f"Arner, D., Barberis, J., & Buckley, R. (2020). The evolution of fintech: A new post-crisis paradigm? Georgetown Journal of International Law, 47(4), 1271–1319.\n"
            f"Basel Committee on Banking Supervision. (2022). Principles for the sound management of AI risk in banks. Bank for International Settlements.\n"
            f"Davenport, T. H., & Ronanki, R. (2018). Artificial intelligence for the real world. Harvard Business Review, 96(1), 108–116.\n"
            f"Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. Review of Financial Studies, 33(5), 2223–2273.\n"
            f"McKinsey & Company. (2022). The state of AI in financial services: Insights from the McKinsey Global AI Survey. McKinsey Digital.\n"
            f"Taddy, M. (2019). The technological elements of artificial intelligence. In A. Agrawal, J. Gans, & A. Goldfarb (Eds.), The economics of artificial intelligence (pp. 61–83). University of Chicago Press."
        )
    if "appendix" in sub_lower or "appendices" in sub_lower:
        return (
            "Appendix A: Research Questionnaire\n"
            "[Survey instrument used for primary data collection. Items measured on a five-point Likert scale: 1 = Strongly Disagree to 5 = Strongly Agree.]\n\n"
            "Appendix B: Ethical Clearance Certificate\n"
            "[Clearance document issued by the institutional ethics review board prior to data collection.]\n\n"
            "Appendix C: Data Collection Authorization Letters\n"
            "[Formal letters of permission from participating organisations authorising access to staff and records.]\n\n"
            "Appendix D: Raw Statistical Output\n"
            "[SPSS/R output tables from regression, correlation, and reliability analysis.]\n\n"
            "Appendix E: Informed Consent Form\n"
            "[Consent document provided to all research participants before data collection commenced.]"
        )

    # ── Generic catch-all — four paragraphs (~350 words) for substantive coverage ──
    return (
        f"The study of {sub.lower()} within the context of {topic} reveals important insights into the mechanisms "
        "and conditions that shape observed outcomes. Academic literature consistently identifies institutional capacity, "
        "contextual alignment, and evidence-based decision-making as critical enablers of positive results in this domain. "
        "Understanding the specific dynamics at play requires a systematic examination of both proximate and distal factors "
        "that contribute to the observed variation in outcomes across different settings and time periods.\n\n"
        "An analytical examination of the relevant factors confirms that both structural and behavioural determinants "
        "contribute to performance variation. Structural factors include the design of governance frameworks, resource "
        "availability, and the maturity of supporting infrastructure, while behavioural determinants encompass leadership "
        "orientation, stakeholder engagement patterns, and the degree to which evidence is incorporated into routine "
        "decision-making. These two categories of determinants interact in complex ways, and their combined effect "
        "is often context-dependent, making generalisation from single-site studies inherently limited.\n\n"
        f"From a theoretical perspective, the literature on {topic} draws on multiple frameworks to explain observed "
        "patterns. Institutional theory highlights the role of normative pressures and mimetic processes in shaping "
        "adoption trajectories. Resource-based perspectives focus on internal capacity as the primary driver of "
        "differential outcomes. Dynamic capability frameworks, by contrast, emphasise adaptability and the capacity "
        "for continuous learning as the most durable sources of competitive advantage. Each of these lenses offers "
        "partial explanatory power, and the most robust accounts tend to integrate elements from multiple traditions.\n\n"
        "These findings have practical significance for stakeholders seeking to optimise outcomes and inform "
        "evidence-based policy within this area of inquiry. Practitioners are advised to conduct systematic baseline "
        "assessments before committing resources to new initiatives, to establish clear performance indicators, and to "
        "build feedback mechanisms that allow course-correction in response to emerging evidence. Policymakers, in turn, "
        "should prioritise enabling environments that reduce barriers to adoption, promote knowledge sharing, and "
        "hold institutions accountable for outcomes rather than outputs alone."
    )


def _chapter_number_from_title(title: str) -> int | None:
    match = re.search(r"chapter\s*(\d+)", (title or "").lower())
    if match:
        return int(match.group(1))
    return None


def _full_context_for_generation(document: Document, upto_index: int | None = None) -> str:
    sections = (document.content or {}).get("sections", [])
    if upto_index is not None:
        sections = sections[: upto_index + 1]
    
    parts: list[str] = [f"Title: {document.title}"]
    total_sections = len(sections)
    
    for idx, sec in enumerate(sections):
        st = sec.get("title", "")
        sc = sec.get("content", "")
        
        if st:
            parts.append(f"\n## {st}")
            
        if sc:
            # Dynamic context compression:
            # Keep the last 2 sections largely intact (up to 2500 chars).
            # Heavily truncate older sections (first 300 chars) to save token limits.
            if idx >= total_sections - 2:
                parts.append(sc[:2500])
            else:
                parts.append(sc[:300] + "...\n[Content truncated for length]")
                
    return "\n".join(parts)


def _build_section_content(
    section_title: str,
    subsection_titles: list[str],
    topic: str,
    context: str,
) -> str:
    blocks: list[str] = []
    rolling_context = context or ""
    for subsection in subsection_titles:
        try:
            subsection_text = generate_section_content(
                title=subsection,
                topic=topic,
                context=(
                    f"Section: {section_title}\n{subsection}\n\n"
                    "Previously written content:\n"
                    f"{rolling_context[-2200:]}"
                ),
                word_count=220,
            )
        except Exception:
            subsection_text = _fallback_subsection_text(topic, section_title, subsection)
        subsection_text = _sanitize_body(subsection_text)
        blocks.append(f"{subsection}\n{subsection_text}")
        rolling_context = f"{rolling_context}\n\n{subsection}\n{subsection_text}".strip()
    return "\n\n".join(blocks)


def _heading_positions(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    positions: list[tuple[int, int, str]] = []
    cursor = 0
    for line in lines:
        stripped = line.strip()
        # More robust heading matching: allows optional markdown formatting (like **2.1** or # 2.1)
        clean_stripped = re.sub(r"^[#*]+", "", stripped).strip()
        if clean_stripped and (
            re.match(r"^\d+(?:\.\d+)*\s+", clean_stripped)
            or clean_stripped.lower().startswith("chapter ")
        ):
            positions.append((cursor, cursor + len(line), clean_stripped))
        cursor += len(line)
    return positions


def _replace_subsection_if_present(section_text: str, subsection_query: str, new_block: str) -> str | None:
    positions = _heading_positions(section_text)
    if not positions:
        return None

    query = subsection_query.lower().strip()
    query_num_match = re.search(r"\b\d+(?:\.\d+)*\b", query)
    query_num = query_num_match.group(0) if query_num_match else None

    hit_index = None
    for idx, (_, _, heading) in enumerate(positions):
        heading_l = heading.lower()
        heading_num_match = re.search(r"\b\d+(?:\.\d+)*\b", heading_l)
        heading_num = heading_num_match.group(0) if heading_num_match else None
        if (query_num and heading_num == query_num) or (query in heading_l):
            hit_index = idx
            break

    if hit_index is None:
        return None

    start = positions[hit_index][0]
    end = positions[hit_index + 1][0] if hit_index + 1 < len(positions) else len(section_text)
    return section_text[:start] + new_block.strip() + "\n\n" + section_text[end:]


def _extract_subsection_block_if_present(section_text: str, subsection_query: str) -> tuple[str, str] | None:
    """Return (heading, body) for a matched subsection inside a larger section."""
    positions = _heading_positions(section_text)
    if not positions:
        return None

    query = (subsection_query or "").lower().strip()
    query_num_match = re.search(r"\b\d+(?:\.\d+)*\b", query)
    query_num = query_num_match.group(0) if query_num_match else None

    hit_index = None
    for idx, (_, end_pos, heading) in enumerate(positions):  # end_pos explicitly utilized here indirectly via index logic if needed
        heading_l = heading.lower()
        heading_num_match = re.search(r"\b\d+(?:\.\d+)*\b", heading_l)
        heading_num = heading_num_match.group(0) if heading_num_match else None
        if (query_num and heading_num == query_num) or (query and query in heading_l):
            hit_index = idx
            break

    if hit_index is None:
        return None

    start = positions[hit_index][0]
    # The positions tuple is (start_pos, expected_heading_end, heading_text)
    # We use positions[hit_index][1] to split heading and body correctly
    heading_end = positions[hit_index][1]
    end = positions[hit_index + 1][0] if hit_index + 1 < len(positions) else len(section_text)
    heading = section_text[start:heading_end].strip()
    body = section_text[heading_end:end].strip()
    return heading, body


def _extract_subsection_phrase(instruction: str) -> str:
    text = (instruction or "").lower()
    subsection_num = re.search(r"\b\d+\.\d+(?:\.\d+)*\b", text)
    if subsection_num:
        return subsection_num.group(0)

    # Ordered from most specific to least so longer matches win
    # Maps: (keywords to detect) -> canonical section name returned
    known_map: list[tuple[tuple[str, ...], str]] = [
        (("background of the study", "background of study", "background"), "Background of the Study"),
        (("statement of the problem", "problem statement", "problem of the study"), "Statement of the Problem"),
        (("research objectives", "research objective", "objectives", "specific objectives", "study objectives"), "Research Objectives"),
        (("research questions", "research question", "study questions"), "Research Questions"),
        (("research hypotheses", "hypotheses", "hypothesis", "null hypothesis", "alternative hypothesis", "h0", "h1"), "Research Hypotheses"),
        (("significance of the study", "significance of study", "signifance of the study", "signifance", "significance"), "Significance of the Study"),
        (("scope and delimitations", "scope of the study", "delimitations", "scope"), "Scope and Delimitations"),
        (("definition of key terms", "key terms"), "Definition of Key Terms"),
        (("conceptual review", "conceptual framework"), "Conceptual Review"),
        (("empirical review", "empirical literature"), "Empirical Review"),
        (("theoretical framework", "theoretical review"), "Theoretical Framework"),
        (("research gap",), "Research Gap"),
        (("research design",), "Research Design"),
        (("target population", "study population"), "Target Population"),
        (("sampling techniques", "sampling technique", "sample size", "sampling method"), "Sampling Techniques and Sample Size"),
        (("data collection",), "Data Collection Methods"),
        (("data analysis", "analysis technique"), "Data Analysis Techniques"),
        (("reliability and validity",), "Reliability and Validity"),
        (("ethical considerations", "ethical consideration", "research ethics"), "Ethical Considerations"),
        (("data presentation",), "Data Presentation"),
        (("discussion of findings", "discussion of results"), "Discussion of Findings"),
        (("summary of findings", "findings summary"), "Summary of Findings"),
        (("recommendations",), "Recommendations"),
        (("limitations of the study", "limitations"), "Limitations of the Study"),
        (("areas for further research", "further research", "future research"), "Areas for Further Research"),
        (("literature review",), "Literature Review"),
        (("methodology",), "Methodology"),
        (("results and discussion",), "Results and Discussion"),
        (("conclusion and recommendations",), "Conclusion and Recommendations"),
        (("references and appendices",), "References and Appendices"),
        (("introduction",), "Introduction"),
        (("conclusion",), "Conclusion"),
        (("abstract",), "Abstract"),
        (("references",), "References"),
        (("appendices", "appendix"), "Appendices"),
    ]
    for keywords, canonical in known_map:
        if any(kw in text for kw in keywords):
            return canonical
    # Fallback: extract what comes after action verbs (never return bare action words)
    _ACTION_ONLY = {"redo", "rewrite", "fix", "correct", "improve", "enhance", "update", "replace", "generate", "write"}
    verb_match = re.search(
        r"(?:correct|fix|enhance|improve|rewrite|write|add|put|include|generate|redo)\s+(?:the\s+|a\s+|an\s+)?(.+)",
        text,
    )
    if verb_match:
        candidate = verb_match.group(1).strip().rstrip(".!?,;")
        if candidate and len(candidate) < 80 and candidate.lower() not in _ACTION_ONLY:
            return candidate
    return instruction


def _extract_chapter_numbers(text: str) -> list[int]:
    source = (text or "").lower()
    numbers: set[int] = set()

    def _add_from_chunk(chunk: str) -> None:
        for rng in re.finditer(r"(\d+)\s*[-to]+\s*(\d+)", chunk):
            start = int(rng.group(1))
            end = int(rng.group(2))
            lo, hi = min(start, end), max(start, end)
            for n in range(lo, hi + 1):
                numbers.add(n)
        for num in re.findall(r"\b\d+\b", chunk):
            numbers.add(int(num))

    for match in re.finditer(r"chapter\s*(\d+)\s*[-to]+\s*(\d+)", source):
        start = int(match.group(1))
        end = int(match.group(2))
        lo, hi = min(start, end), max(start, end)
        for n in range(lo, hi + 1):
            numbers.add(n)

    for match in re.finditer(r"chapters?\s*([\d\s,\-andto]+)", source):
        _add_from_chunk(match.group(1))

    for match in re.finditer(r"chapter\s*(\d+)", source):
        numbers.add(int(match.group(1)))

    return sorted(n for n in numbers if 1 <= n <= 20)


def _chapter_title_from_number(chapter_number: int) -> str:
    for item in DISSERTATION_TEMPLATE:
        number = _chapter_number_from_title(item.get("title", ""))
        if number == chapter_number:
            return item["title"]
    return f"Chapter {chapter_number}"


def _is_generic_section_query(query: str | None) -> bool:
    value = (query or "").strip().lower()
    return value in {
        "",
        "it",
        "it again",
        "again",
        "this section",
        "that section",
        "section",
    }


def _leaf_node_count(nodes: list[dict[str, Any]]) -> int:
    count = 0
    for node in nodes:
        children = [_normalize_subsection_node(ch) for ch in node.get("children", [])]
        if children:
            count += _leaf_node_count(children)
        else:
            count += 1
    return max(count, 1)


def _chapter_default_word_count(chapter_number: int | None) -> int:
    """Return a per-subsection word-count target for each dissertation chapter.

    Targets are calibrated so a 50-subsection dissertation produces ~20,000 words
    (≈ 80 pages at 250 wpp), which is standard for a master's dissertation.
    """
    chapter_wc = {
        0: 120,   # Preliminary pages — frontmatter items (abstract, ToC, etc.)
        1: 600,   # Introduction — rich contextualisation, objectives, significance
        2: 900,   # Literature Review — thematic synthesis, densest chapter
        3: 650,   # Methodology — design rationale, sampling, instruments, analysis
        4: 700,   # Results & Discussion — findings tables, interpretation per objective
        5: 500,   # Conclusions — summary, recommendations, limitations, future research
        6: 300,   # References & Appendices — lists + structured entries
    }
    return chapter_wc.get(chapter_number, 400)


def _requested_page_target(instruction: str) -> int | None:
    text = (instruction or "").lower()
    ranged = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*pages?", text)
    if ranged:
        return max(int(ranged.group(1)), int(ranged.group(2)))
    single = re.search(r"(\d+)\s*pages?", text)
    if single:
        return int(single.group(1))
    return None


def _requested_word_target(instruction: str) -> int | None:
    """Parse an explicit word count from the user's instruction."""
    text = (instruction or "").lower()
    # "15000 words", "15,000 words", "15k words"
    m = re.search(r"(\d[\d,]*)\s*k?\s*words?\b", text)
    if m:
        raw = m.group(1).replace(",", "")
        val = int(raw)
        # detect "15k words" — if the original had a "k" before "word"
        if re.search(r"\d+\s*k\s*words?\b", text):
            val *= 1000
        return val
    # "15k" standalone — in a writing request context, treat as 15,000 words
    m = re.search(r"\b(\d+)\s*k\b", text)
    if m:
        return int(m.group(1)) * 1000
    # derive from pages
    pages = _requested_page_target(instruction)
    return pages * 500 if pages else None


def _parse_user_guidelines(instruction: str) -> dict[str, Any]:
    """Extract structured writing guidelines from a free-form user instruction.

    Returns a dict with:
      citation_style  — "APA" | "Harvard" | "MLA" | "Chicago" | "Vancouver"
      academic_level  — "Undergraduate" | "Masters" | "PhD"
      target_words    — int | None
      focus_notes     — any free-text guidelines the LLM should follow
    """
    text = (instruction or "").lower()

    # Citation / referencing style — search full original instruction (not just lowercased)
    citation_style = "APA"
    if "harvard" in text:
        citation_style = "Harvard"
    elif "chicago" in text:
        citation_style = "Chicago"
    elif "ieee" in text:
        citation_style = "IEEE"
    elif "mla" in text:
        citation_style = "MLA"
    elif "vancouver" in text:
        citation_style = "Vancouver"
    elif "apa" in text:
        citation_style = "APA"

    # Academic level
    academic_level = "Masters"
    if any(k in text for k in ["phd", "doctoral", "doctorate", "d.phil"]):
        academic_level = "PhD"
    elif any(k in text for k in ["undergrad", "bachelor", "bsc", "ba ", "first year", "second year", "third year"]):
        academic_level = "Undergraduate"
    elif any(k in text for k in ["master", "msc", "mba", "postgrad", "pgce"]):
        academic_level = "Masters"

    # Word count target
    target_words = _requested_word_target(instruction)

    # Capture any explicit "guidelines:" or "requirements:" block
    focus_notes = ""
    for marker in ("guidelines:", "requirements:", "instructions:", "notes:", "format:", "please note:"):
        idx = text.find(marker)
        if idx != -1:
            focus_notes = instruction[idx:idx + 800].strip()
            break

    return {
        "citation_style": citation_style,
        "academic_level": academic_level,
        "target_words": target_words,
        "focus_notes": focus_notes,
        "raw": instruction,
    }


def _chapter_word_count_target(chapter_number: int | None, instruction: str, nodes: list[dict[str, Any]]) -> int:
    base = _chapter_default_word_count(chapter_number)
    pages = _requested_page_target(instruction)
    if not pages:
        return base

    total_words = pages * 500
    leaves = _leaf_node_count(nodes)
    per_leaf = int(round(total_words / max(leaves, 1)))
    return max(base, per_leaf)


def _dynamic_chapter_nodes(
    chapter_number: int,
    topic: str,
    research_design: str,
    objectives: list[str],
) -> list[dict[str, Any]]:
    objective_lines = "\n".join(f"- {o}" for o in objectives[:6])
    prompt = (
        "You are designing a dissertation chapter outline. Return JSON only.\n"
        f"Chapter number: {chapter_number}\n"
        f"Topic: {topic}\n"
        f"Research design: {research_design}\n"
        f"Objectives:\n{objective_lines}\n\n"
        "Output schema:\n"
        "[{\"title\":\"X.X ...\",\"children\":[{\"title\":\"X.X.X ...\",\"children\":[]}]}]\n"
        "Rules:\n"
        "- Generate 5 to 10 subsection nodes for this chapter.\n"
        "- Use academically valid numbering for the chapter (e.g., chapter 2 -> 2.1, 2.2...).\n"
        "- For chapter 2 and chapter 4, create varied, non-boilerplate subsection titles while preserving meaning.\n"
        "- Keep each title concise and specific to the topic.\n"
        "- Do not repeat the exact same subsection wording across runs.\n"
        "- Return JSON only, no markdown."
    )
    try:
        data = _extract_json_obj(f"{{\"nodes\": {generate_text(prompt)} }}")
        raw_nodes = data.get("nodes")
        if isinstance(raw_nodes, list) and raw_nodes:
            return [_normalize_subsection_node(n) for n in raw_nodes]
    except Exception:
        try:
            raw_text = generate_text(prompt)
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned).rstrip("`").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and parsed:
                return [_normalize_subsection_node(n) for n in parsed]
        except Exception as exc:
            logger.warning("dynamic chapter nodes fallback for chapter %s: %s", chapter_number, exc)
    return []


def _chapter_nodes_for_generation(
    chapter_number: int,
    research_design: str,
    objectives: list[str],
    topic: str,
) -> list[dict[str, Any]]:
    chapter_template = next(
        (
            item for item in DISSERTATION_TEMPLATE
            if _chapter_number_from_title(item.get("title", "")) == chapter_number
        ),
        None,
    )
    if not chapter_template:
        return [{"title": f"{chapter_number}.1 Overview", "children": []}]

    chapter_title = chapter_template.get("title", "")

    if chapter_number in {2, 4}:
        dynamic = _dynamic_chapter_nodes(chapter_number, topic, research_design, objectives)
        if dynamic:
            # Chapter 4 already has visuals from _chapter4_subsections; skip injection there.
            if chapter_number != 4:
                dynamic = _inject_standard_visuals(dynamic, chapter_number, research_design)
            return dynamic

    if "chapter 4" in chapter_title.lower():
        return _chapter4_subsections(research_design, objectives)

    nodes = [_normalize_subsection_node(s) for s in chapter_template.get("subsections", [])]
    return _inject_standard_visuals(nodes, chapter_number, research_design)


def _find_matching_node(nodes: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    query_l = (query or "").strip().lower()
    if not query_l or _is_generic_section_query(query_l):
        return None

    query_num_match = re.search(r"\b\d+(?:\.\d+)*\b", query_l)
    query_num = query_num_match.group(0) if query_num_match else None

    for node in nodes:
        title = str(node.get("title") or "")
        title_l = title.lower()
        title_num_match = re.search(r"\b\d+(?:\.\d+)*\b", title_l)
        title_num = title_num_match.group(0) if title_num_match else None
        if (query_num and title_num == query_num) or (query_l in title_l):
            return _normalize_subsection_node(node)

        child_nodes = [_normalize_subsection_node(ch) for ch in node.get("children", [])]
        child_hit = _find_matching_node(child_nodes, query)
        if child_hit:
            return child_hit
    return None


def _step_label(step: str) -> str:
    return (step or "").strip().lstrip("- ").strip()


def _build_chat_summary(plan: list[dict[str, Any]], intent: str, document_updated: bool) -> dict[str, Any]:
    items = [_step_label(step.get("step", "")) for step in plan if step.get("step")]
    total = len(items)
    completed = sum(1 for step in plan if step.get("status") == "done")
    pending = total - completed
    completion_percent = int(round((completed / total) * 100)) if total else 0
    next_tasks = [
        _step_label(step.get("step", ""))
        for step in plan
        if step.get("status") in {"pending", "error"}
    ][:6]

    done_items = [
        _step_label(step.get("step", ""))
        for step in plan
        if step.get("status") == "done"
    ]
    done_preview = ", ".join(done_items[:3]) if done_items else "No steps completed yet"

    if pending == 0:
        stage = "All planned tasks completed"
    elif completed == 0:
        stage = "Execution queued"
    else:
        stage = "Execution in progress"

    if document_updated:
        stage = f"{stage}; document updated"

    # Chapter-level plan (top-level items only — indent = 0)
    chapter_plan = [
        {"step": _step_label(s.get("step", "")), "status": s.get("status", "pending")}
        for s in plan
        if not (s.get("step") or "").startswith(" ")
    ]

    return {
        "stage": stage,
        "intent": intent,
        "todo_list": items,
        "chapter_plan": chapter_plan,
        "completion_percent": completion_percent,
        "tasks_completed": completed,
        "tasks_pending": pending,
        "next_tasks": next_tasks,
        "done_brief": done_preview,
    }


def _find_section_index_by_subsection(document: Document, query: str) -> int | None:
    query_l = (query or "").lower().strip()
    if not query_l:
        return None

    generic_queries = {
        "it",
        "it again",
        "again",
        "this section",
        "that section",
        "section",
    }
    if query_l in generic_queries:
        intro_idx = find_section(document.content, "introduction")
        if intro_idx is not None:
            return intro_idx
        chapter_one_idx = find_section(document.content, "Chapter 1")
        if chapter_one_idx is not None:
            return chapter_one_idx

    subsection_num = re.search(r"\b(\d+)\.(\d+)(?:\.\d+)*\b", query_l)
    if subsection_num:
        chapter_num = subsection_num.group(1)
        by_chapter = find_section(document.content, f"Chapter {chapter_num}")
        if by_chapter is not None:
            return by_chapter

    chapter_hint_map = {
        "1": [
            "background",
            "statement of the problem",
            "research objectives",
            "research questions",
            "research hypotheses",
            "significance",
            "signifance",
            "scope",
            "key terms",
            "introduction",
        ],
        "2": [
            "literature review",
            "conceptual review",
            "theoretical framework",
            "empirical review",
            "research gap",
        ],
        "3": [
            "methodology",
            "research design",
            "target population",
            "sampling",
            "data collection",
            "data analysis",
            "reliability",
            "validity",
            "ethical considerations",
        ],
        "4": [
            "data presentation",
            "findings",
            "discussion",
            "results",
        ],
        "5": [
            "summary of findings",
            "conclusions",
            "recommendations",
            "limitations",
            "further research",
        ],
        "6": [
            "references",
            "appendices",
        ],
    }
    for chapter_num, hints in chapter_hint_map.items():
        if any(hint in query_l for hint in hints):
            by_chapter = find_section(document.content, f"Chapter {chapter_num}")
            if by_chapter is not None:
                return by_chapter

    sections = (document.content or {}).get("sections", [])
    for idx, section in enumerate(sections):
        title = (section.get("title") or "").lower()
        content = (section.get("content") or "").lower()
        if query_l in title or query_l in content:
            return idx
    return None


def _chat_summary_text(summary: dict[str, Any]) -> str:
    next_tasks = summary.get("next_tasks") or []
    next_label = ", ".join(next_tasks[:3]) if next_tasks else "None (all tasks completed)"
    return (
        f"Stage: {summary.get('stage', 'Update available')}\n"
        f"Done (brief): {summary.get('done_brief', 'N/A')}\n"
        f"To-do count: {len(summary.get('todo_list', []))}\n"
        f"Completion: {summary.get('completion_percent', 0)}%\n"
        f"Completed tasks: {summary.get('tasks_completed', 0)}\n"
        f"Pending tasks: {summary.get('tasks_pending', 0)}\n"
        f"Next tasks: {next_label}"
    )


def _needs_todo_workflow(intent: str, message: str) -> bool:
    text = (message or "").strip().lower()
    if intent == "write_dissertation":
        return True

    if intent in {"write_section", "enhance_section", "address_comments"}:
        return True

    if intent in {
        "chat",
        "summarize_document",
        "create_outline",
        "write_report",
        "write_assignment",
        "write_presentation",
        "write_spreadsheet",
        "enhance_document",
        "add_chart",
        "add_image",
    }:
        return any(
            k in text
            for k in [
                "plan",
                "todo",
                "to-do",
                "step by step",
                "workflow",
                "stages",
                "execute",
            ]
        )

    return False


def _compact_doc_summary(document: Document) -> str:
    sections = (document.content or {}).get("sections", [])
    if not sections:
        return "This document is currently empty."

    lines: list[str] = [f"Summary of {document.title}:"]
    for sec in sections[:6]:
        title = sec.get("title", "Untitled section")
        content = (sec.get("content") or "").strip().replace("\n", " ")
        snippet = content[:170] + ("..." if len(content) > 170 else "")
        lines.append(f"- {title}: {snippet or 'No content yet.'}")
    if len(sections) > 6:
        lines.append(f"- Additional sections not shown: {len(sections) - 6}")
    return "\n".join(lines)


def _sanitize_chat_reply(text: str) -> str:
    """Remove prompt-echo artifacts and repetitive boilerplate from model output."""
    if not text:
        return ""

    lines = (text or "").splitlines()
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    blank_run = 0

    blocked_prefixes = (
        "user:",
        "assistant:",
        "user request:",
        "i. introduction",
        "ii. introduction",
        "here is a detailed response",
        "here is the reorganized",
        "here is the revised",
        "here is the rewritten",
        "here is the completed",
        "here is the updated",
        "here is the restructured",
        "here is a reorganized",
        "the completed document now",
        "the reorganized document",
        "please provide a detailed",
        "please provide an in-depth",
        "please provide a thorough",
        "please provide a comprehensive",
        "please provide a brief",
        "please provide the following",
        "please give me a detailed",
        "please reorganize",
        "please restructure",
        "please rewrite",
        "please revise",
        "please review",
        "please complete these",
        "please go ahead",
        "please do not remove",
        "i would like you to provide",
        "i need you to provide",
        "i need you to reorganize",
        "i need you to restructure",
        "i want you to provide",
        "could you please provide",
        "can you please provide",
        "provide a detailed analysis",
        "provide a thorough analysis",
        "however, i noticed",
        "however, there are still",
        "to fix this issue",
        "to be inserted",
        "note: please",
    )

    # Hard-cut at ANY instruction-injection pattern mid-text
    _injection_pattern = re.compile(
        r"(?:^|\n)("
        r"i\.\s*introduction|ii\.\s*introduction|"
        r"##\s*step\s*\d+|"
        r"please provide|please reorganize|please restructure|please complete|"
        r"please rewrite|please revise|please review|"
        r"please go ahead|please do not|"
        r"could you please|can you please|"
        r"i would like you to|i need you to|i want you to|"
        r"here is the reorganized|here is the revised|here is the rewritten|here is the completed|"
        r"here is the updated|here is the restructured|"
        r"project objectives|expected outcomes|timeline|resources|"
        r"the completed document now|however, i noticed|however, there are still|"
        r"to fix this issue|note: please"
        r")",
        re.IGNORECASE,
    )
    injection_match = _injection_pattern.search(text)
    if injection_match:
        text = text[: injection_match.start()].rstrip()
        if not text:
            return ""

    lines = text.splitlines()

    for raw_line in lines:
        line = raw_line.rstrip()
        normalized = re.sub(r"\s+", " ", line.strip().lower())

        if not normalized:
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
            continue

        blank_run = 0

        if any(normalized.startswith(prefix) for prefix in blocked_prefixes):
            continue

        # Keep at most 2 occurrences of an identical non-empty line.
        count = seen.get(normalized, 0) + 1
        seen[normalized] = count
        if count > 2:
            continue

        cleaned.append(line)

    # Trim excessive trailing repetition if the answer collapsed into loops.
    non_empty = [ln for ln in cleaned if ln.strip()]
    if len(non_empty) >= 18:
        unique_ratio = len({re.sub(r"\s+", " ", ln.strip().lower()) for ln in non_empty}) / len(non_empty)
        if unique_ratio < 0.45:
            deduped: list[str] = []
            local_seen: set[str] = set()
            for ln in non_empty:
                key = re.sub(r"\s+", " ", ln.strip().lower())
                if key in local_seen:
                    continue
                local_seen.add(key)
                deduped.append(ln)
                if len(deduped) >= 28:
                    break
            return "\n".join(deduped).strip()

    # Guard against repetitive coaching loops like
    # "What should I do next? / You should now ..."
    low = "\n".join(cleaned).lower()
    coaching_hits = re.findall(
        r"what should i do next|you should now|i have (?:reviewed|proofread|formatted|submitted|received|revised|finalized|completed)",
        low,
    )
    if len(coaching_hits) >= 3:
        cut = re.search(r"(?:^|\n)(what should i do next|you should now)", low)
        if cut:
            trimmed = "\n".join(cleaned)
            trimmed = trimmed[: cut.start()].strip()
            if trimmed:
                return trimmed
            return ""

    return "\n".join(cleaned).strip()


def _looks_like_workflow_or_prompt_echo(text: str) -> bool:
    """Detect non-answer artifacts (workflow templates, rewrite prompts, coaching loops)."""
    low = (text or "").lower()
    if not low.strip():
        return True

    artifact_patterns = [
        r"\b##\s*step\s*\d+\b",
        r"\bstep\s*\d+\s*:\s*",
        r"\bproject objectives\b",
        r"\bexpected outcomes\b",
        r"\btimeline\b",
        r"\bresources\b",
        r"\bplease\s+(rewrite|revise|review|provide)\b",
        r"\bhere is the rewritten\b",
        r"\bi\.\s*introduction\b",
        r"\bwhat should i do next\b",
        r"\byou should now\b",
    ]
    hits = sum(1 for pat in artifact_patterns if re.search(pat, low, flags=re.IGNORECASE))

    # Too many imperative/procedural lines indicates the model is continuing an old workflow.
    imperative_lines = 0
    for ln in low.splitlines():
        s = ln.strip()
        if s.startswith(("please ", "rewrite ", "review ", "revise ", "step ")):
            imperative_lines += 1

    return hits >= 2 or imperative_lines >= 3


def _rule_based_document_feedback(document: Document, mode: str) -> str:
    """
    Fallback for summary/review/take modes when primary model output is noisy.
    Tries LLM with a stripped-down prompt first; only returns template text if LLM fails.
    mode: 'analysis' | 'improvement' | 'take'
    """
    sections = (document.content or {}).get("sections", [])
    title = (document.title or "This document").strip()
    topic = ((document.content or {}).get("topic") or title).strip()

    if not sections:
        if mode == "improvement":
            return "The document has no sections yet — there is no structure to evaluate. Add core sections first (Introduction, Methodology, Findings, Conclusion) before requesting a quality review."
        return (
            f"The document titled '{title}' is currently empty. "
            "There is no section content to assess yet. "
            "Start by adding an introduction, core body sections, and a conclusion."
        )

    titles = [str(s.get("title") or "Untitled section") for s in sections]
    non_empty_sections = [s for s in sections if (s.get("content") or "").strip()]
    empty_titles = [str(s.get("title") or "Untitled section") for s in sections if not (s.get("content") or "").strip()]

    # Try a lightweight LLM call first when content exists.
    if non_empty_sections:
        section_lines = []
        for sec in non_empty_sections[:8]:
            sec_title = str(sec.get("title") or "Untitled")
            content_preview = str(sec.get("content") or "")[:300].replace("\n", " ")
            section_lines.append(f"- {sec_title}: {content_preview}")
        section_text = "\n".join(section_lines)

        if mode == "improvement":
            fallback_prompt = (
                f"Dissertation topic: {topic}\n\n"
                f"Sections present:\n{section_text}\n\n"
                "List 4–6 specific, concrete areas that need improvement in this document. "
                "Name the actual section and the exact issue for each point. "
                "Use bullet points. Be direct. Do NOT be generic."
            )
        elif mode == "take":
            fallback_prompt = (
                f"Dissertation topic: {topic}\n\n"
                f"Sections present:\n{section_text}\n\n"
                "Give a short, honest assessment (5–8 sentences): what is working, what needs work, "
                "and what to do next. Reference actual section names."
            )
        else:
            fallback_prompt = (
                f"Dissertation topic: {topic}\n\n"
                f"Sections present:\n{section_text}\n\n"
                "Provide a 4–6 sentence summary covering: what the document is about, "
                "what sections are in place, key strengths, and top areas for improvement."
            )
        try:
            result = generate_text(fallback_prompt)
            result = (result or "").strip()
            if result and len(result) > 60 and not _looks_like_workflow_or_prompt_echo(result):
                return result
        except Exception:
            pass

    # Pure template fallback when LLM is unavailable.
    has_intro = any("introduction" in t.lower() for t in titles)
    has_method = any("method" in t.lower() or "research design" in t.lower() for t in titles)
    has_conclusion = any("conclusion" in t.lower() for t in titles)
    has_refs = any("reference" in t.lower() for t in titles)

    if mode == "improvement":
        points: list[str] = []
        if empty_titles:
            points.append(f"- Empty or underdeveloped sections: {', '.join(empty_titles[:4])}.")
        if not has_intro:
            points.append("- Missing a clear Introduction section to frame purpose, scope, and context.")
        if not has_method:
            points.append("- Methodology/Research Design is unclear or missing, which weakens credibility.")
        if not has_conclusion:
            points.append("- No explicit Conclusion section that synthesizes findings and implications.")
        if not has_refs:
            points.append("- References section is missing, which affects academic completeness.")
        if len(non_empty_sections) < max(2, len(sections) // 2):
            points.append("- Content depth is uneven across sections; several parts need fuller development.")
        if not points:
            points.append("- Overall structure is present, but argument flow between sections can be tightened.")
            points.append("- Strengthen evidence support in major claims and improve transitions between sections.")
        return "\n".join(points[:8])

    sections_preview = ", ".join(titles[:6]) + ("..." if len(titles) > 6 else "")
    strengths: list[str] = []
    if has_intro:
        strengths.append("the document has a recognizable opening structure")
    if has_method:
        strengths.append("it includes methodological framing")
    if len(non_empty_sections) >= 3:
        strengths.append("several sections already contain substantive content")

    gaps: list[str] = []
    if empty_titles:
        gaps.append(f"some sections are still thin or empty, especially {', '.join(empty_titles[:2])}")
    if not has_conclusion:
        gaps.append("there is no clear conclusion that closes the argument")
    if not has_refs:
        gaps.append("references are not clearly documented")
    if not gaps:
        gaps.append("the strongest opportunity is improving flow and precision across sections")

    if mode == "take":
        return (
            f"My take on your document about {topic}: it has a workable foundation. "
            f"I can see these sections already in place: {sections_preview}. "
            f"What is working is that {('; '.join(strengths) if strengths else 'the core topic is identifiable')}. "
            f"The main gap is that {gaps[0]}. "
            "Next, tighten section-to-section transitions so the narrative reads as one argument. "
            "Then expand weak sections with concrete evidence, not placeholders. "
            "Finally, end with a direct conclusion plus references so the document feels academically complete."
        )

    return (
        f"This document focuses on {topic} and currently has {len(sections)} section(s). "
        f"The visible structure includes: {sections_preview}. "
        f"A key strength is that {('; '.join(strengths) if strengths else 'the topic direction is clear')}. "
        f"The main weakness is that {gaps[0]}. "
        "To improve quality, strengthen evidence-backed argumentation in weaker sections. "
        "Also improve coherence by making transitions explicit and aligning each section to the overall objective."
    )


def _summarize_document(document: Document, user_message: str, plan: list) -> tuple[str, bool]:
    if plan:
        _done(plan, 0)
    doc_context = _flatten_doc(document, truncate=False)
    msg_lower = user_message.lower()
    is_analysis = any(w in msg_lower for w in ["analys", "analyze", "review", "examine"])
    is_improvement_review = _is_improvement_review_request(user_message)
    is_take_request = _is_document_take_request(user_message)

    # Build a structured section inventory so the LLM can reference actual content.
    sections = (document.content or {}).get("sections", [])
    topic = ((document.content or {}).get("topic") or document.title or "this study").strip()

    section_inventory: list[str] = []
    for sec in sections:
        sec_title = str(sec.get("title") or "Untitled").strip()
        sec_content = str(sec.get("content") or "").strip()
        word_count = len(sec_content.split())
        preview = sec_content[:400].replace("\n", " ") if sec_content else "(empty)"
        section_inventory.append(
            f"- {sec_title} ({word_count} words): {preview}{'...' if len(sec_content) > 400 else ''}"
        )
    section_summary = "\n".join(section_inventory) if section_inventory else "(No sections written yet)"

    mode = "improvement" if is_improvement_review else ("take" if is_take_request else "analysis")

    if is_improvement_review:
        direct_prompt = (
            f"You are a rigorous academic reviewer assessing a dissertation on: '{topic}'.\n\n"
            f"DOCUMENT SECTIONS (with word counts and content previews):\n{section_summary}\n\n"
            f"FULL DOCUMENT CONTENT:\n{doc_context[:12000]}\n\n"
            f"USER REQUEST: {user_message}\n\n"
            "TASK: Provide a detailed, specific critique of this document. "
            "Go through EACH section that has content and identify concrete issues. "
            "Your response must:\n"
            "1. List areas of improvement section by section using the ACTUAL section titles.\n"
            "2. For each section, name the SPECIFIC issue (e.g., too short, lacks citations, "
            "argument not developed, no evidence, unclear objective).\n"
            "3. Include 5–10 specific points, not generic advice.\n"
            "4. If a section is strong, skip it — only flag genuine weaknesses.\n"
            "5. End with 2–3 highest-priority actions the writer should take NOW.\n\n"
            "Format: Use bullet points. Reference section titles. Be direct and specific.\n"
            "Do NOT rewrite the document. Do NOT say 'I can help'. Just give the feedback."
        )
    elif is_take_request:
        direct_prompt = (
            f"You are an expert academic mentor reviewing a dissertation on: '{topic}'.\n\n"
            f"DOCUMENT SECTIONS:\n{section_summary}\n\n"
            f"FULL DOCUMENT CONTENT:\n{doc_context[:12000]}\n\n"
            f"USER REQUEST: {user_message}\n\n"
            "Give your honest take on this document in a direct, human tone. "
            "Cover: (1) what is genuinely working well — cite specific sections, "
            "(2) the 2–3 most important gaps or weaknesses you see in the actual content, "
            "(3) the single most impactful improvement the writer should make next. "
            "Be specific to THIS document — reference actual section titles and content. "
            "Write 6–10 sentences. Do NOT be generic. Do NOT ask questions."
        )
    elif is_analysis:
        direct_prompt = (
            f"You are an expert academic reviewer analysing a dissertation on: '{topic}'.\n\n"
            f"DOCUMENT SECTIONS:\n{section_summary}\n\n"
            f"FULL DOCUMENT CONTENT:\n{doc_context[:12000]}\n\n"
            f"USER REQUEST: {user_message}\n\n"
            "Provide a structured analysis covering:\n"
            "• Topic and scope: what the document is about and what it covers\n"
            "• Structural strengths: which sections/chapters are well-developed\n"
            "• Key weaknesses: which sections are thin, missing, or underdeveloped\n"
            "• Argument quality: is the academic argument coherent end-to-end?\n"
            "• Immediate next action: what should the writer do first?\n\n"
            "Reference actual section titles. Be concise but specific. "
            "Do NOT ask for more information. Write the analysis now."
        )
    else:
        direct_prompt = (
            f"You are an expert academic assistant summarising a dissertation on: '{topic}'.\n\n"
            f"DOCUMENT SECTIONS:\n{section_summary}\n\n"
            f"FULL DOCUMENT CONTENT:\n{doc_context[:12000]}\n\n"
            f"USER REQUEST: {user_message}\n\n"
            "Summarise this document covering: the main research topic, the chapters/sections "
            "present, the key arguments or findings, and any notable gaps. "
            "Reference actual section titles. Be direct and concise. "
            "Do NOT ask questions. Write the summary now."
        )

    try:
        reply = generate_text(direct_prompt)
    except Exception:
        reply = ""

    reply = _sanitize_chat_reply(reply)
    if not reply or _looks_like_workflow_or_prompt_echo(reply):
        reply = _rule_based_document_feedback(document, mode)

    # Final safety net: if output still looks like repetitive coaching chain, replace it.
    low = reply.lower()
    loop_hits = re.findall(
        r"what should i do next|you should now|i have (?:reviewed|proofread|formatted|submitted|received|revised|finalized|completed)",
        low,
    )
    if len(loop_hits) >= 3:
        reply = _rule_based_document_feedback(document, mode)

    if plan:
        _all_done(plan)
    return reply, False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_injected_instructions(text: str) -> str:
    """Remove lines that are injected prompt instructions rather than document content."""
    _re = re.compile(
        r"^\s*("
        r"please\s+(provide|reorganize|restructure|complete|go ahead|do not)|"
        r"i\s+(need|want|would like)\s+you\s+to|"
        r"could you please|can you please|"
        r"here is (the|a) (reorganized|revised|completed|updated|restructured)|"
        r"the completed document now|"
        r"however,?\s+(i noticed|there are still)|"
        r"to fix this issue|note:\s*please|"
        r"to be inserted"
        r")",
        re.IGNORECASE,
    )
    lines = [ln for ln in text.splitlines() if not _re.match(ln)]
    return "\n".join(lines).strip()


def _flatten_doc(document: Document, truncate: bool = False) -> str:
    content = document.content or {}
    parts = [f"Title: {document.title}"]
    for section in content.get("sections", []):
        title = section.get("title", "")
        body = _strip_injected_instructions(section.get("content", ""))
        if title:
            parts.append(f"\n## {title}")
        if body:
            parts.append(body[:600] if truncate else body)
    return "\n".join(parts)


def _history_text(recent_history: list[dict[str, Any]] | None, limit: int = 8) -> str:
    entries = recent_history or []
    if not entries:
        return ""
    trimmed = entries[-limit:]
    lines: list[str] = []
    for item in trimmed:
        role = str(item.get("role") or "user").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)


def _resolve_followup_message(message: str, recent_history: list[dict[str, Any]] | None) -> str:
    text = (message or "").strip()
    lowered = text.lower()
    followups = {
        "proceed", "continue", "go on", "carry on", "do it", "do that",
        "yes", "okay", "ok", "next", "continue please",
    }
    if lowered not in followups:
        return text

    entries = recent_history or []
    # Find the last meaningful user request before this follow-up token.
    for item in reversed(entries[:-1] if entries else entries):
        if str(item.get("role") or "").lower() != "user":
            continue
        prev = str(item.get("content") or "").strip()
        if not prev:
            continue
        if prev.lower() in followups:
            continue
        return f"Continue the previous request: {prev}"
    return text


def _extract_locator_query(message: str) -> str | None:
    text = (message or "").strip()
    low = text.lower()
    locator_signals = ["where is", "locate", "find where", "position of", "where can i find"]
    if not any(sig in low for sig in locator_signals):
        return None

    cleaned = re.sub(r"^(where is|locate|find where|position of|where can i find)\s+", "", low).strip()
    return cleaned or None


def _locate_section_positions(document: Document, query: str) -> str:
    content = document.content or {}
    sections = content.get("sections", [])
    q = (query or "").strip().lower()
    if not q:
        return "I could not determine what to locate. Mention the section or topic to find."

    hits: list[str] = []
    for idx, section in enumerate(sections):
        title = str(section.get("title") or "")
        body = str(section.get("content") or "")
        title_l = title.lower()
        body_l = body.lower()

        if q in title_l:
            hits.append(f"- Section {idx + 1}: {title} (title match)")

        sub_positions = _heading_positions(body)
        for _, _, heading in sub_positions:
            if q in heading.lower():
                hits.append(f"- Section {idx + 1}: {title} -> {heading}")

        if q in body_l and len(hits) < 10:
            hits.append(f"- Section {idx + 1}: {title} (content mention)")

    # De-duplicate while preserving order.
    uniq: list[str] = []
    seen: set[str] = set()
    for h in hits:
        if h not in seen:
            seen.add(h)
            uniq.append(h)

    if not uniq:
        return f"I could not find '{query}' in the current document sections."
    return "I found these positions:\n" + "\n".join(uniq[:12])


def _save(document: Document, note: str) -> None:
    document.save(update_fields=["content", "updated_at"])
    DocumentVersion.objects.create(
        document=document, content=document.content, note=note
    )


def _done(plan: list[dict], idx: int) -> None:
    if idx < len(plan):
        plan[idx]["status"] = "done"


def _all_done(plan: list[dict]) -> None:
    for step in plan:
        step["status"] = "done"


# ── Main entry point ─────────────────────────────────────────────────────────

def run_agent(
    document: Document,
    message: str,
    model_choice: str | None = None,
    recent_history: list[dict[str, Any]] | None = None,
    attachment_text: str | None = None,
    preview_only: bool = False,
    grounded_research: bool = False,
    verify_citations: bool = False,
    synthetic_mode: bool = False,
) -> dict[str, Any]:
    """
    Returns:
    {
        "reply":             str,
        "plan":              [{"step": str, "status": "done"|"pending"|"error"}],
        "document_updated":  bool,
        "intent":            str,
    }
    """
    set_active_model(model_choice)
    effective_message = _resolve_followup_message(message, recent_history)
    doc_context = _flatten_doc(document)
    history_context = _history_text(recent_history, limit=8)
    if history_context:
        doc_context = f"{doc_context}\n\nRecent chat context:\n{history_context}"
    if attachment_text:
        doc_context = f"{doc_context}\n\nAttachment context:\n{attachment_text[:4000]}"
    lowered_message = (effective_message or "").strip().lower()

    locator_query = _extract_locator_query(effective_message)
    if locator_query:
        reply = _locate_section_positions(document, locator_query)
        return {
            "reply": reply,
            "plan": [],
            "chat_summary": None,
            "orchestration": {
                "mode": "direct",
                "todo_required": False,
                "execution": "single_pass",
                "status": "ok",
            },
            "document_updated": False,
            "intent": "locate_section",
            "model": get_model_label(),
        }

    # 1. Classify intent
    intent_data = _heuristic_intent(effective_message)
    # Only call the LLM classifier if heuristic returned chat AND the message is NOT
    # a pure question/explanation (those must stay as chat — the LLM sometimes
    # misclassifies "explain X" or "what is X" as write_section).
    if intent_data.get("intent") == "chat" and not _is_pure_chat_question(effective_message):
        intent_data = classify_intent(effective_message, doc_context)
    if intent_data.get("intent") == "chat" and not _is_pure_chat_question(effective_message):
        heur = _heuristic_intent(effective_message)
        if heur.get("intent") != "chat":
            intent_data = heur
    intent = intent_data.get("intent", "chat")
    target_section = intent_data.get("target_section")
    topic = (
        intent_data.get("topic")
        or (document.content or {}).get("topic")
        or document.title
    )
    topic = re.sub(r"^on\s+", "", (topic or "").strip(), flags=re.IGNORECASE)

    research_result = None
    research_brief = ""
    citation_context = ""
    research_error = ""
    research_mode = grounded_research or any(
        kw in lowered_message
        for kw in [
            "literature review",
            "grounded in real papers",
            "sources",
            "citations",
            "academic references",
            "evidence",
        ]
    )
    if research_mode:
        try:
            from .research_layer import (
                retrieval_pipeline,
                build_research_brief,
                build_citation_context,
            )

            research_result = retrieval_pipeline(
                topic=topic or effective_message,
                query=effective_message,
                document_id=document.id,
            )
            research_brief = build_research_brief(research_result)
            citation_context = build_citation_context(research_result.top_papers, max_items=12)
            if research_brief:
                doc_context = f"{doc_context}\n\n{research_brief}"
            if citation_context:
                doc_context = f"{doc_context}\n\n{citation_context}"
        except Exception as exc:
            research_error = str(exc)
            logger.warning("Research retrieval pipeline failed: %s", exc)

    # Force review-only phrasing to analysis mode (no edits, no Copilot edit workflow).
    if _is_improvement_review_request(effective_message):
        intent = "summarize_document"
        target_section = None

    # Force conversational document-opinion phrasing to feedback mode.
    if _is_document_take_request(effective_message):
        intent = "summarize_document"
        target_section = None

    # Flexible fallback: for broad/out-of-pattern queries, answer from the document in chat mode.
    if _is_document_grounded_chat_request(effective_message):
        intent = "chat"
        target_section = None

    explicit_target = _explicit_section_target_from_message(effective_message)
    derived_target = _extract_subsection_phrase(effective_message)
    has_section_action = any(
        k in lowered_message
        for k in [
            "redo", "rewrite", "define", "fix", "correct", "improve", "enhance",
            "update", "replace", "regenerate", "generate", "write",
        ]
    )
    is_section_command = has_section_action and (
        bool(explicit_target)
        or (derived_target and not _is_generic_section_query(derived_target))
    )

    # ── Topic relevance guard ────────────────────────────────────────────────
    # If the agent is about to write something, make sure the topic being requested
    # is actually related to this document. If the message asks to write about
    # a topic that has ZERO overlap with the document's own subject, redirect to chat.
    if intent in {"write_section", "write_dissertation", "enhance_section"} and topic and not is_section_command:
        doc_topic = ((document.content or {}).get("topic") or document.title or "").lower()
        message_words = set(re.findall(r"\b[a-z]{4,}\b", lowered_message))
        doc_words = set(re.findall(r"\b[a-z]{4,}\b", doc_topic))
        # Extract topic from the message (words NOT in common stop-list)
        _STOP = {"write", "redo", "rewrite", "section", "chapter", "dissertation",
                 "thesis", "please", "just", "this", "that", "with", "from", "into",
                 "about", "what", "your", "have", "will", "some", "more", "also"}
        msg_topic_words = {w for w in message_words if w not in _STOP and len(w) > 3}
        # If the document has an established topic AND the message topic words share
        # NOTHING with the document topic, and the document already has sections written,
        # redirect to chat so we don't write alien content into the user's document.
        doc_has_content = bool((document.content or {}).get("sections"))
        if (
            doc_topic
            and doc_has_content
            and msg_topic_words
            and not msg_topic_words.intersection(doc_words)
        ):
            # One final check: if ANY word in the message overlaps the doc topic, allow it
            all_doc_words = set(re.findall(r"\b[a-z]{4,}\b", doc_context.lower()))
            if not msg_topic_words.intersection(all_doc_words):
                intent = "chat"
                logger.info(
                    "Topic mismatch guard: message topic '%s' unrelated to document topic '%s'. "
                    "Redirecting to chat.",
                    msg_topic_words, doc_topic,
                )
    full_doc_request = any(
        k in lowered_message
        for k in [
            "full dissertation", "entire dissertation", "complete dissertation", "whole dissertation",
            "full thesis", "entire thesis", "complete thesis", "whole thesis",
            "full project", "entire project", "complete project", "whole project",
        ]
    )
    if explicit_target and has_section_action and not full_doc_request:
        # Force section-scoped execution — override even if upstream classifier set a bad intent
        if intent == "write_dissertation":
            intent = "write_section"
        if intent in {"write_section", "enhance_section"}:
            target_section = explicit_target  # always override with the precise target

    # 2. Build plan
    if intent == "summarize_document":
        steps = ["Reading current document", "Preparing summary response"]
    else:
        steps = create_execution_plan(intent)
    plan = [{"step": s, "status": "pending"} for s in steps]
    todo_required = _needs_todo_workflow(intent, lowered_message)

    # ── Confirmation gate ────────────────────────────────────────────────────
    # When the caller is in preview-only mode and the resolved intent would
    # modify the document, return the plan for the user to review before we
    # execute anything.  Non-modifying intents (chat, summarize, locate) pass
    # straight through so conversational messages are never delayed.
    if preview_only and intent in DOCUMENT_MODIFYING_INTENTS:
        description = _intent_description(intent, target_section, topic)
        return {
            "reply": description,
            "plan": plan,
            "chat_summary": None,
            "orchestration": {
                "mode": "confirmation",
                "todo_required": False,
                "execution": "pending",
                "status": "awaiting_confirmation",
            },
            "document_updated": False,
            "intent": intent,
            "model": get_model_label(),
            "awaiting_confirmation": True,
            "confirmation": {
                "description": description,
                "intent": intent,
                "target_section": target_section,
                "topic": topic,
            },
        }

    had_error = False
    error_detail = ""
    generation_message = effective_message
    if research_mode and (research_brief or citation_context):
        generation_message = (
            f"{effective_message}\n\n"
            "Grounding requirements:\n"
            "- Use only verifiable claims from the retrieved papers.\n"
            "- Prefer DOI-backed citations.\n"
            "- If evidence is weak, say so explicitly.\n\n"
            f"{research_brief}\n\n"
            f"{citation_context}"
        )
    if synthetic_mode:
        generation_message = (
            f"{generation_message}\n\n"
            "Synthetic data mode is ON. Any synthetic numbers, charts, or evidence must be explicitly labeled as synthetic."
        )

    # ── Multi-agent supervisor enrichment ──────────────────────────────────
    # For dissertation writes or research-mode generation, run the full agent
    # pipeline BEFORE the main LLM call so its synthesised brief enriches generation.
    agent_supervisor_result: dict[str, Any] | None = None
    if research_mode and intent in {
        "write_dissertation", "write_section", "write_report", "enhance_document"
    }:
        try:
            from .agents_v2 import AgentContext, SupervisorAgent

            supervisor = SupervisorAgent()
            agent_ctx = AgentContext(
                topic=topic or effective_message,
                instruction=effective_message,
                document_id=document.id,
                retrieval=research_result,
            )
            agent_supervisor_result = supervisor.run(agent_ctx)
            # If retrieval wasn't done yet, grab it from agent result
            if research_result is None and agent_ctx.retrieval is not None:
                research_result = agent_ctx.retrieval

            synthesis = agent_supervisor_result.get("synthesis", "")
            if synthesis:
                generation_message = (
                    f"{generation_message}\n\n"
                    "=== SUPERVISOR SYNTHESIS (from specialist agents) ===\n"
                    f"{synthesis}\n\n"
                    "INSTRUCTIONS: Use the above synthesis, evidence, and methodology "
                    "as your primary grounding. Cite only sources listed with real DOIs."
                )
        except Exception as exc:
            logger.warning("Multi-agent supervisor failed: %s", exc)

    # 3. Execute
    try:
        chapter_numbers = _extract_chapter_numbers(effective_message)
        chapter_request = "chapter" in lowered_message and len(chapter_numbers) >= 1

        if intent == "summarize_document":
            reply, updated = _summarize_document(document, effective_message, plan)
        elif intent == "address_comments":
            reply, updated = _address_comments(document, topic, plan)
        elif intent == "enhance_document":
            reply, updated = _enhance_document(document, topic, plan)
        elif intent == "enhance_section" and chapter_request:
            reply, updated = _enhance_chapter_batch(document, chapter_numbers, topic, generation_message, plan)
        elif intent == "write_section" and chapter_request:
            reply, updated = _rewrite_chapter_batch(document, chapter_numbers, topic, generation_message, plan)
        elif intent == "enhance_section":
            # Use Copilot-style agentic loop: read → identify → edit → save
            copilot_steps = [
                "Reading document structure",
                "Identifying relevant sections",
                "Reading target section",
                "Editing target section",
                "Saving changes",
            ]
            plan.clear()
            plan.extend([{"step": s, "status": "pending"} for s in copilot_steps])
            reply, updated = _run_copilot_loop(document, generation_message, plan, target_section, topic)
        elif intent == "write_section":
            reply, updated = _write_section(document, target_section, topic, generation_message, plan)
        elif intent == "write_dissertation":
            reply, updated = _write_dissertation(document, topic, generation_message, plan)
        elif intent in ("write_document", "write_article", "write_report",
                        "write_assignment", "write_presentation", "write_spreadsheet"):
            # All non-dissertation document writing funnels through the unified AI planner
            reply, updated = _plan_and_write_document(document, topic, generation_message, plan)
        elif intent == "create_outline":
            reply, updated = _create_outline(document, topic, plan)
        elif intent == "add_chart":
            reply, updated = _add_chart(document, target_section, plan)
        elif intent == "add_image":
            reply, updated = _add_image(document, target_section, generation_message, plan)
        elif intent == "humanise_ai_sections":
            reply, updated = _humanise_ai_sections(document, topic or "", plan)
        elif intent == "reduce_plagiarism_similarity":
            reply, updated = _reduce_plagiarism_similarity(document, topic or "", plan)
        elif intent == "check_academic_quality":
            reply, updated = _check_academic_quality(document, plan)
        else:
            try:
                # Always ground unknown/vague prompts in the document rather than
                # returning a generic "Yes, I can help" style reply.
                reply = _document_grounded_chat_response(
                    effective_message,
                    doc_context,
                    recent_history=recent_history,
                    attachment_text=attachment_text,
                )
                reply = _sanitize_chat_reply(reply)
                if not reply:
                    reply = "I reviewed the document and can provide a concise analysis or section-specific improvement."
            except Exception as exc:
                had_error = True
                error_detail = str(exc)
                _exc = str(exc)
                if "429" in _exc or "rate limit" in _exc.lower() or "too many requests" in _exc.lower():
                    reply = "The AI service is temporarily busy (rate limit reached). Please wait a moment and try again."
                elif "api key" in _exc.lower() or "invalid" in _exc.lower() or "401" in _exc:
                    reply = "The AI service could not authenticate. Please check your API key in the settings."
                else:
                    reply = "Something went wrong reaching the AI service. Please try again shortly."
            updated = False
            _all_done(plan)
    except Exception as exc:
        logger.error("Agent execution error: %s", exc, exc_info=True)
        had_error = True
        error_detail = str(exc)
        reply = f"Something went wrong: {exc}"
        updated = False
        for step in plan:
            if step["status"] == "pending":
                step["status"] = "error"

    summary = _build_chat_summary(plan, intent, updated) if todo_required else None
    orchestration = {
        "mode": "todo" if todo_required else "direct",
        "todo_required": todo_required,
        "execution": "stepwise" if todo_required else "single_pass",
        "status": "failed" if had_error or any(s.get("status") == "error" for s in plan) else "ok",
    }
    if error_detail:
        orchestration["error"] = error_detail

    citation_verification: dict[str, Any] | None = None
    if verify_citations:
        try:
            from .research_layer import summarize_verification, verify_generated_citations

            citation_verification = summarize_verification(verify_generated_citations(reply))
            orchestration["citation_verification"] = {
                "enabled": True,
                "total": citation_verification.get("total", 0),
                "verified": citation_verification.get("verified", 0),
                "rejected": citation_verification.get("rejected", 0),
            }
        except Exception as exc:
            orchestration["citation_verification"] = {"enabled": True, "error": str(exc)}

    # Optionally repair low-confidence or rejected citations using retrieval pool papers.
    if verify_citations and research_result and getattr(research_result, "top_papers", None):
        try:
            from .research_layer import repair_citations

            repair_result = repair_citations(reply, research_result.top_papers, min_confidence=60)
            if (repair_result.repaired_count + repair_result.removed_count) > 0:
                reply = repair_result.repaired_text

            orchestration["citation_repair"] = {
                "enabled": True,
                "total": repair_result.total_citations,
                "repaired": repair_result.repaired_count,
                "removed": repair_result.removed_count,
                "unchanged": repair_result.unchanged_count,
            }
        except Exception as exc:
            orchestration["citation_repair"] = {"enabled": True, "error": str(exc)}

    research_meta: dict[str, Any] = {
        "enabled": bool(research_mode),
        "error": research_error or None,
        "embedding_path": getattr(research_result, "embedding_path", None),
        "top_sources": [
            {
                "title": p.title,
                "year": p.year,
                "doi": p.doi,
                "source": p.source,
                "score": p.score,
            }
            for p in (getattr(research_result, "top_papers", []) or [])[:10]
        ],
        "synthetic_mode": bool(synthetic_mode),
        "supervisor": {
            "enabled": bool(agent_supervisor_result),
            "trace": (agent_supervisor_result or {}).get("trace", []),
            "contract": (agent_supervisor_result or {}).get("contract"),
            "citation_verification": (agent_supervisor_result or {}).get("citation_verification"),
            "citation_repair": (agent_supervisor_result or {}).get("citation_repair"),
        },
        "citation_repair": orchestration.get("citation_repair"),
    }

    return {
        "reply": reply,
        "plan": plan if todo_required else [],
        "chat_summary": summary,
        "orchestration": orchestration,
        "document_updated": updated,
        "intent": intent,
        "model": get_model_label(),
        "research": research_meta,
        "citation_verification": citation_verification,
    }



# ── Action handlers ───────────────────────────────────────────────────────────

def _address_comments(document: Document, topic: str, plan: list) -> tuple[str, bool]:
    """Find all inline [Comment: ...] markers in the document and generate AI revisions to address them."""
    import re as _re
    sections = (document.content or {}).get("sections", [])
    if not sections:
        _all_done(plan)
        return "The document has no sections yet.", False

    _done(plan, 0)  # Scanning

    comment_re = _re.compile(r'\[Comment:\s*([^\]]+)\]', _re.IGNORECASE)
    affected: list[tuple[int, str, list[str]]] = []  # (idx, title, [comments])
    for i, section in enumerate(sections):
        body = section.get("content", "")
        comments = [m.group(1).strip() for m in comment_re.finditer(body)]
        if comments:
            affected.append((i, section.get("title", f"Section {i+1}"), comments))

    if not affected:
        _all_done(plan)
        return (
            "No reviewer comments were found in the document. "
            "Use the 'New Comment' button in the Review tab to add comments, "
            "or type [Comment: your comment] inline in any section.",
            False,
        )

    _done(plan, 1)  # Reading sections

    count = 0
    addressed: list[str] = []
    for plan_idx, (sec_idx, sec_title, comments) in enumerate(affected):
        original = sections[sec_idx].get("content", "")
        comments_block = "\n".join(f"  - {c}" for c in comments)
        prompt = (
            f"You are an academic writing assistant. The following section has reviewer comments "
            f"that need to be addressed.\n\n"
            f"Section title: {sec_title}\n\n"
            f"Current content:\n{original}\n\n"
            f"Reviewer comments:\n{comments_block}\n\n"
            f"Task:\n"
            f"1. Revise the section content to address each reviewer comment.\n"
            f"2. Remove all [Comment: ...] markers from the revised text.\n"
            f"3. Keep the academic tone, structure, and length similar to the original.\n"
            f"4. Return ONLY the revised section text with no extra labels or commentary."
        )
        try:
            revised = generate_text(prompt).strip()
            # Strip any residual [Comment: ...] markers the LLM may have left
            revised = comment_re.sub("", revised).strip()
            if revised:
                sections[sec_idx]["content"] = revised
                count += 1
                addressed.append(f"• {sec_title} ({len(comments)} comment{'s' if len(comments)>1 else ''} addressed)")
        except Exception as exc:
            logger.warning("_address_comments: section %d failed: %s", sec_idx, exc)
        if plan_idx + 2 < len(plan):
            _done(plan, plan_idx + 2)

    _all_done(plan)
    if count:
        document.content["sections"] = sections
        _save(document, "address-comments")
        summary = "\n".join(addressed)
        return (
            f"Addressed reviewer comments in {count} section(s):\n{summary}\n\n"
            "All [Comment: ...] markers have been removed and the content revised accordingly.",
            True,
        )
    return "Could not generate revisions for the found comments. Please try again.", False


def _humanise_ai_sections(document: Document, topic: str, plan: list) -> tuple[str, bool]:
    """Detect AI-generated sentences per section and rewrite them to sound human-written."""
    from .ai_detector import detect_ai_content, rule_based_humanise

    sections = (document.content or {}).get("sections", [])
    if not sections:
        _all_done(plan)
        return "The document has no sections yet.", False

    _done(plan, 0)  # Running AI detection

    humanised = 0
    llm_used = False
    for i, section in enumerate(sections):
        content = (section.get("content") or "").strip()
        if not content:
            continue

        detection = detect_ai_content(content)
        flagged = [s for s in detection.get("sentences", []) if s.get("label") != "likely_human"]
        if not flagged:
            continue

        _done(plan, 1)  # Identifying AI passages

        # Extract the specific AI phrases found so the LLM can target them
        ai_phrase_snippets = [s["text"][:80] for s in flagged if s.get("ai_probability", 0) >= 0.45]

        new_content = None
        # Try LLM first; fall back to rule-based substitution if unavailable
        try:
            new_content = humanise_text(content, topic or document.title, ai_phrase_snippets)
            if new_content and len(new_content.strip()) > 50:
                llm_used = True
            else:
                new_content = None
        except Exception as exc:
            logger.info("LLM humanise unavailable (%s), using rule-based fallback.", exc)

        if not new_content:
            new_content = rule_based_humanise(content, seed=f"{document.id}:{i}")

        if new_content and new_content.strip() != content:
            sections[i]["content"] = new_content.strip()
            humanised += 1

        _done(plan, 2)  # Rewriting with human voice

    _done(plan, 3)  # Varying sentence structure
    _all_done(plan)

    if humanised:
        document.content["sections"] = sections
        _save(document, "humanise-ai-sections")
        # Compute after score for the reply
        try:
            full_text = "\n\n".join(s.get("content", "") for s in sections if s.get("content"))
            after_result = detect_ai_content(full_text)
            after_pct = after_result.get("overall_ai_percentage", 0)
            after_verdict = after_result.get("verdict", "")
        except Exception:
            after_pct = None
            after_verdict = ""

        method = "AI model" if llm_used else "rule-based phrase substitution"
        after_msg = f" AI score dropped to **{after_pct}%** ({after_verdict.replace('_',' ')})." if after_pct is not None else ""
        return (
            f"Humanised {humanised} section(s) using {method} — replaced AI clichés, "
            f"varied sentence structure, and added authentic voice.{after_msg} "
            "Click **Detect AI** to see the highlighted improvement.",
            True,
        )

    return (
        "No strongly AI-detected passages were found — your document already reads naturally. "
        "Try running **Detect AI** first to scan for flagged sentences.",
        False,
    )


def _reduce_plagiarism_similarity(document: Document, topic: str, plan: list) -> tuple[str, bool]:
    """Detect passages overlapping with other workspace documents and rewrite them to cut similarity."""
    from .plagiarism_detector import check_plagiarism, reduce_similarity

    sections = (document.content or {}).get("sections", [])
    if not sections:
        _all_done(plan)
        return "The document has no sections yet.", False

    source_docs = []
    for other in Document.objects.exclude(pk=document.pk).only("id", "title", "content"):
        other_content = other.content or {}
        parts = [
            (s.get("content") or "").strip()
            for s in other_content.get("sections", [])
            if (s.get("content") or "").strip()
        ]
        other_text = "\n\n".join(parts)
        if other_text.strip():
            source_docs.append((other.id, other.title or f"Document {other.id}", other_text))

    _done(plan, 0)  # Scanning document for matched/similar passages

    if not source_docs:
        _all_done(plan)
        return (
            "No other documents exist in the workspace to compare against, so there is nothing to "
            "reduce — plagiarism similarity is only measured against other workspace documents.",
            False,
        )

    _done(plan, 1)  # Comparing against other documents in the workspace

    reduced = 0
    for i, section in enumerate(sections):
        content = (section.get("content") or "").strip()
        if not content:
            continue

        detection = check_plagiarism(content, source_docs)
        flagged = {s["text"] for s in detection.get("sentences", []) if s.get("label") != "original"}
        if not flagged:
            continue

        new_content = reduce_similarity(content, flagged, seed=f"{document.id}:{i}")

        # Re-check; if anything is still flagged, retry once with a fresh seed
        # (mirrors the before/after verification pattern used by AI-humanisation).
        recheck = check_plagiarism(new_content, source_docs)
        still_flagged = {s["text"] for s in recheck.get("sentences", []) if s.get("label") != "original"}
        if still_flagged:
            retry_content = reduce_similarity(new_content, still_flagged, seed=f"{document.id}:{i}:retry")
            if retry_content != new_content:
                new_content = retry_content

        if new_content.strip() != content:
            sections[i]["content"] = new_content.strip()
            reduced += 1

    _done(plan, 2)  # Rewriting flagged passages to reduce textual overlap
    _done(plan, 3)  # Re-checking similarity after rewrite
    _all_done(plan)

    if reduced:
        document.content["sections"] = sections
        _save(document, "reduce-plagiarism-similarity")

        full_text = "\n\n".join(s.get("content", "") for s in sections if s.get("content"))
        after_result = check_plagiarism(full_text, source_docs)
        after_pct = after_result.get("overall_similarity_percentage", 0)
        after_verdict = after_result.get("verdict", "")

        return (
            f"Rewrote {reduced} section(s) to reduce overlap with other workspace documents. "
            f"Similarity dropped to **{after_pct}%** ({after_verdict.replace('_', ' ')}). "
            "Click **Check Plagiarism** to see the updated highlights.",
            True,
        )

    return (
        "No matched or similar passages were found against other workspace documents — your document "
        "already reads as original. Try running **Check Plagiarism** first to scan for flagged sentences.",
        False,
    )


def _check_academic_quality(document: Document, plan: list) -> tuple[str, bool]:
    """
    Run rule-based academic quality analysis per section and return a
    structured report without modifying the document.
    """
    from .ai_detector import academic_quality_check

    sections = (document.content or {}).get("sections", [])
    if not sections:
        _all_done(plan)
        return ("The document has no content to analyse.", False)

    _done(plan, 0)

    section_reports: list[str] = []
    total_issues = 0
    total_words = 0

    for section in sections:
        content = (section.get("content") or "").strip()
        if not content:
            continue
        title = section.get("title") or "Untitled section"
        result = academic_quality_check(content)
        score = result["quality_score"]
        verdict = result["verdict"].replace("_", " ")
        issues = result.get("issues", [])
        total_issues += len(issues)
        total_words += result.get("word_count", 0)

        score_icon = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
        line = f"{score_icon} **{title}** — quality score **{score}/100** ({verdict})"
        if issues:
            issue_lines = [f"  - *{iss['severity'].upper()}* {iss['message']}" for iss in issues]
            line += "\n" + "\n".join(issue_lines)
        section_reports.append(line)

    _done(plan, 1)
    _all_done(plan)

    if not section_reports:
        return ("No section content found to analyse.", False)

    header = (
        f"**Academic Writing Quality Report** — {total_words} words analysed, "
        f"{total_issues} issue(s) found across {len(section_reports)} section(s).\n\n"
    )
    guide_tip = (
        "\n\n---\n"
        "**Quick fixes:** Run **Humanise** to reduce AI-detection signals, "
        "or ask 'enhance [section name]' to improve a specific section's clarity and argument."
    )
    return (header + "\n\n".join(section_reports) + guide_tip, False)


def _enhance_document(document: Document, topic: str, plan: list) -> tuple[str, bool]:
    sections = (document.content or {}).get("sections", [])
    if not sections:
        _all_done(plan)
        return (
            "The document has no sections yet. Try: 'Create an outline for [topic]' first.",
            False,
        )

    _done(plan, 0)
    _done(plan, 1)
    count = 0
    for i, section in enumerate(sections):
        original = section.get("content", "").strip()
        if not original:
            continue
        section_title = section.get("title", "")
        try:
            sections[i]["content"] = enhance_text(original, topic, section_title=section_title)
            count += 1
        except Exception as exc:
            logger.warning("Enhance section %d failed: %s", i, exc)
        _done(plan, 2)

    _all_done(plan)
    if count:
        document.content["sections"] = sections
        _save(document, "enhance-document")
        return (
            f"Enhanced {count} section(s) across the document — improved vocabulary, "
            "argument structure, evidence cues, and academic tone.",
            True,
        )
    return "No existing content found to enhance. Add text to sections first.", False



def _personalize_plan_steps(plan: list, section_name: str) -> None:
    """Replace generic plan step labels with section-specific text."""
    name = section_name.strip("'").strip()
    for step in plan:
        s = step.get("step", "")
        sl = s.lower()
        if "locating" in sl:
            step["step"] = f"Locating the section for '{name}'"
        elif "analys" in sl and "content" in sl:
            step["step"] = f"Analysing existing content in '{name}'"
        elif "rewriting" in sl or ("clarity" in sl and "tone" in sl):
            step["step"] = f"Rewriting '{name}' with improved clarity and academic tone"
        elif "saving" in sl and "section" in sl:
            step["step"] = f"Saving updated section '{name}'"
        elif "inserting" in sl:
            step["step"] = f"Inserting '{name}' into document"
        elif "generating" in sl:
            step["step"] = f"Generating content for '{name}'"


def _run_copilot_loop(
    document: Document,
    message: str,
    plan: list,
    target: str | None,
    topic: str,
) -> tuple[str, bool]:
    """
    GitHub Copilot-style agentic loop:
    1. list_sections  → get document outline  (plan step 0)
    2. identify sections relevant to the request  (plan step 1)
    3. read → edit for each relevant section  (plan steps 2+)
    4. save and return a reply summary
    """
    from .tools import doc_list_sections, doc_read_section, doc_edit_section

    # Step 0: read document structure
    if plan:
        plan[0]["status"] = "done"
    sections_info = doc_list_sections(document)
    if not sections_info:
        _all_done(plan)
        return "The document has no sections yet.", False

    outline = "\n".join(
        f"[{s['index']}] {s['title']} ({s['word_count']} words)"
        for s in sections_info
    )

    # Step 1: identify relevant sections
    if len(plan) > 1:
        plan[1]["status"] = "done"

    relevant_indices: list[int] = []

    # ── Subsection-number shortcut (e.g. "2.7", "3.4.1") ────────────────────
    # When the user says "improve 2.7", target is literally "2.7".
    # The top-level sections are whole chapters, so find_section would fail and
    # the LLM fallback would incorrectly pick the WHOLE Chapter 2 for replacement.
    # Instead, find the parent chapter, extract only the subsection block, edit it,
    # and splice the result back — leaving the rest of the chapter untouched.
    _subsec_re = re.compile(r"^(\d+)\.(\d+(?:\.\d+)*)$")
    _subsec_match = _subsec_re.match((target or "").strip()) if target else None
    if _subsec_match:
        _ch_num = int(_subsec_match.group(1))
        _all_secs = (document.content or {}).get("sections", [])
        _found_chapter = False
        for _si, _sec in enumerate(_all_secs):
            if _chapter_number_from_title(_sec.get("title", "")) == _ch_num:
                _found_chapter = True
                _block = _extract_subsection_block_if_present(_sec.get("content", ""), target)
                if _block:
                    _heading, _body = _block
                    _edit_prompt = (
                        f"User request: {message}\n\n"
                        f"Document topic: {topic}\n\n"
                        f"Subsection: {_heading}\n\n"
                        f"Current content:\n{_body}\n\n"
                        "Improve ONLY this subsection based on the user request. Maintain academic "
                        "tone and do not lose any important information that was not meant to change. "
                        "Do NOT include the subsection heading in your output. "
                        "Return ONLY the improved body text."
                    )
                    for _pi in range(2, len(plan)):
                        plan[_pi]["status"] = "done"
                    try:
                        _new_body = generate_text(_edit_prompt).strip()
                        if _new_body and len(_new_body) > 50:
                            _new_block = f"{_heading}\n{_new_body}"
                            _new_ch_content = _replace_subsection_if_present(
                                _sec.get("content", ""), target, _new_block
                            )
                            if _new_ch_content is not None:
                                _all_secs[_si]["content"] = _new_ch_content
                                document.content["sections"] = _all_secs
                                _save(document, f"copilot:subsection:{target}")
                                _all_done(plan)
                                return f"Improved subsection {target}.", True
                    except Exception as _sub_exc:
                        logger.warning("Subsection edit failed for %s: %s", target, _sub_exc)
                _all_done(plan)
                return (
                    f"Could not locate subsection {target} within Chapter {_ch_num}. "
                    "Please verify the section number and try again.",
                    False,
                )
        if not _found_chapter:
            _all_done(plan)
            return (
                f"Could not find Chapter {_ch_num} in the document. "
                "Please verify the section number and try again.",
                False,
            )

    if target and not _subsec_match:
        idx = find_section(document.content, target)
        if idx is not None:
            relevant_indices = [idx]

    # ── Subsection-by-name fallback ──────────────────────────────────────────
    # If target wasn't found as a top-level chapter title (e.g. "background of
    # the study"), search inside every chapter's content for a matching
    # heading and edit only that block, leaving the rest of the chapter intact.
    if not relevant_indices and target and not _subsec_match:
        _target_lower = target.strip().lower()
        _all_secs = (document.content or {}).get("sections", [])
        _found_subsec = False
        for _si, _sec in enumerate(_all_secs):
            _ch_content = _sec.get("content", "")
            if not _ch_content:
                continue
            # Find a heading line that matches the target name
            for _line in _ch_content.split("\n"):
                _stripped = _line.strip()
                if not _stripped:
                    continue
                # Accept "1.1 Background of the Study" or plain "Background of the Study"
                _heading_clean = re.sub(r"^\d+(\.\d+)*\s*", "", _stripped).strip().lower()
                # Skip empty headings and very short lines (avoids empty-string false positives)
                if not _heading_clean or len(_heading_clean) < 4:
                    continue
                if _target_lower in _heading_clean or _heading_clean in _target_lower:
                    # Found the heading inside this chapter — extract number prefix
                    _num_match = re.match(r"^(\d+(?:\.\d+)+)", _stripped)
                    _subsec_id = _num_match.group(1) if _num_match else None
                    if _subsec_id:
                        _block = _extract_subsection_block_if_present(_ch_content, _subsec_id)
                    else:
                        # No number — do a manual extract: heading to next heading
                        _h_idx = _ch_content.find(_stripped)
                        if _h_idx == -1:
                            continue
                        _after = _ch_content[_h_idx + len(_stripped):]
                        # next heading = next numbered line or eof
                        _next = re.search(r"\n\d+\.\d+", _after)
                        _body_raw = _after[:_next.start()] if _next else _after
                        _block = (_stripped, _body_raw.strip())
                    if _block:
                        _heading_txt, _body_txt = _block
                        _edit_prompt = (
                            f"User request: {message}\n\n"
                            f"Document topic: {topic}\n\n"
                            f"Subsection heading: {_heading_txt}\n\n"
                            f"Current content:\n{_body_txt}\n\n"
                            "Improve ONLY this subsection according to the user request. "
                            "Maintain academic tone. Do NOT touch any other section. "
                            "Do NOT include the subsection heading in your output. "
                            "Return ONLY the improved body text."
                        )
                        for _pi in range(2, len(plan)):
                            plan[_pi]["status"] = "done"
                        try:
                            _new_body = generate_text(_edit_prompt).strip()
                            if _new_body and len(_new_body) > 50:
                                # Splice improved body back into the chapter
                                if _subsec_id:
                                    _new_block_txt = f"{_heading_txt}\n{_new_body}"
                                    _new_ch = _replace_subsection_if_present(_ch_content, _subsec_id, _new_block_txt)
                                else:
                                    _new_ch = _ch_content.replace(
                                        f"{_stripped}\n{_body_txt}",
                                        f"{_stripped}\n{_new_body}",
                                        1,
                                    )
                                if _new_ch:
                                    _all_secs[_si]["content"] = _new_ch
                                    document.content["sections"] = _all_secs
                                    _save(document, f"copilot:subsec-name:{target[:40]}")
                                    _all_done(plan)
                                    return f"Enhanced '{_heading_txt}' in {_sec.get('title', 'the chapter')}.", True
                        except Exception as _exc:
                            logger.warning("Subsection-by-name edit failed for '%s': %s", target, _exc)
                            # Primary LLM unavailable — try static academic fallback
                            try:
                                # Strip numeric prefix so specific section patterns match
                                _subsec_name_clean = re.sub(r"^\d+(?:\.\d+)*\s*", "", _heading_txt).strip()
                                _static_body = _fallback_subsection_text(
                                    topic or "", _sec.get("title", ""), _subsec_name_clean or _heading_txt
                                )
                                if _static_body and len(_static_body) > 80:
                                    if _subsec_id:
                                        _fb_block = f"{_heading_txt}\n{_static_body}"
                                        _fb_ch = _replace_subsection_if_present(
                                            _ch_content, _subsec_id, _fb_block
                                        )
                                    else:
                                        _fb_ch = _ch_content.replace(
                                            f"{_stripped}\n{_body_txt}",
                                            f"{_stripped}\n{_static_body}",
                                            1,
                                        )
                                    if _fb_ch:
                                        _all_secs[_si]["content"] = _fb_ch
                                        document.content["sections"] = _all_secs
                                        _save(document, f"copilot:subsec-name:{target[:40]}")
                                        _all_done(plan)
                                        return (
                                            f"Enhanced '{_heading_txt}' in "
                                            f"{_sec.get('title', 'the chapter')}.",
                                            True,
                                        )
                            except Exception:
                                pass
                        _found_subsec = True
                        break
            if _found_subsec:
                break
    # Subsection was located by name but all edit attempts failed (e.g. no API key).
    if _found_subsec:
        _all_done(plan)
        return (
            f"I found '{target}' in the document but could not enhance it — "
            "the AI service is currently unavailable (API key not configured). "
            "Please set GROK_API_KEY or GEMINI_API_KEY to enable AI editing.",
            False,
        )

    # LLM fallback: ask the model which top-level section index to use.
    if not relevant_indices:
        section_prompt = (
            f"User request: '{message}'\n\n"
            f"Document outline:\n{outline}\n\n"
            "Task: Identify the exact section(s) the user wants to update or modify based on their request.\n"
            "Constraints:\n"
            "- If the user names a specific section (e.g., 'Introduction', 'Methodology'), return the index for that section.\n"
            "- If the user refers to a concept (e.g., 'objectives', 'background', 'limitations'), identify the section index that conventionally contains it (e.g., 'objectives' are typically in the 'Introduction').\n"
            "- Do not infer or include other sections unless explicitly implied by the user.\n"
            "- Return a JSON array of integers for the matching section indices, e.g., [0]. \n"
            "Return ONLY the JSON array and nothing else."
        )
        try:
            raw = generate_text(section_prompt)
            m = re.search(r"\[[\d,\s]*\]", raw)
            if m:
                parsed = json.loads(m.group(0))
                relevant_indices = [i for i in parsed if 0 <= i < len(sections_info)]
        except Exception:
            relevant_indices = []

    if not relevant_indices:
        _all_done(plan)
        return "I could not find the specified section in the document to modify. Could you clarify which section you mean?", False

    # Steps 2+: read → edit for each relevant section
    edit_summaries: list[str] = []
    plan_cursor = 2

    for idx in relevant_indices:
        sec_info = sections_info[idx]
        sec_title = sec_info["title"]

        if plan_cursor < len(plan):
            plan[plan_cursor]["status"] = "done"
        plan_cursor += 1

        sec = doc_read_section(document, sec_title)
        if not sec:
            continue
        current_content = sec["content"]

        if plan_cursor < len(plan):
            plan[plan_cursor]["status"] = "done"
        plan_cursor += 1

        edit_prompt = (
            "SYSTEM INSTRUCTION:\n"
            "You are a focused, precise document editing agent.\n"
            "You must:\n"
            "1. Understand exactly what the user asked and apply ONLY that change.\n"
            "2. Do NOT rewrite, expand, or modify any part of the document that the user did not ask about.\n\n"
            "Editing Behavior:\n"
            "- Apply the user's specific request (improvement, fix, expansion, etc.) to this section only.\n"
            "- Preserve all information that the user did not explicitly ask to change.\n"
            "- Maintain academic tone.\n\n"
            f"User request: {message}\n\n"
            f"Document topic: {topic}\n\n"
            f"Section: {sec_title}\n\n"
            f"Current content:\n{current_content}\n\n"
            "Write the updated version of this section. Do NOT include the section heading in the output. "
            "Return ONLY the updated content."
        )
        try:
            new_content = generate_text(edit_prompt).strip()
            if new_content and len(new_content) > 50:
                doc_edit_section(document, sec_title, new_content)
                edit_summaries.append(sec_title)
        except Exception:
            pass

    updated = bool(edit_summaries)
    if updated:
        _save(document, f"copilot:{message[:60]}")

    _all_done(plan)

    if edit_summaries:
        reply = f"Updated {len(edit_summaries)} section(s): {', '.join(edit_summaries)}."
    else:
        reply = "No changes were applied. The sections may already be well-written or no matching section was found."
    return reply, updated


def _enhance_section(
    document: Document,
    target: str | None,
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    query = (target or _extract_subsection_phrase(instruction) or "").strip()
    if _is_generic_section_query(query):
        query = "Introduction"
    _personalize_plan_steps(plan, query)

    chapter_numbers = _extract_chapter_numbers(f"{target or ''} {instruction} {query}")
    chapter_hint = chapter_numbers[0] if chapter_numbers else None
    if chapter_hint is None and any(
        k in query.lower()
        for k in [
            "empirical review",
            "conceptual review",
            "theoretical framework",
            "literature review",
            "introduction",
            "background of the study",
        ]
    ):
        chapter_hint = 2 if "review" in query.lower() or "framework" in query.lower() else 1

    # For dissertation sections, enhance by rewriting through structured to-do workflow.
    if chapter_hint is not None:
        return _write_section(document, query, topic, instruction, plan)

    _done(plan, 0)
    idx = find_section(document.content, query)
    if idx is None:
        idx = _find_section_index_by_subsection(document, query)

    if idx is None:
        return _write_section(document, query, topic, instruction, plan)

    section = document.content["sections"][idx]
    original = (section.get("content") or "").strip()
    if not original:
        return _write_section(document, query, topic, instruction, plan)

    subsection_block = None
    section_title_l = (section.get("title") or "").lower()
    query_l = query.lower()
    if query_l and query_l not in section_title_l:
        subsection_block = _extract_subsection_block_if_present(original, query)

    _done(plan, 1)
    sec_title = section.get("title") or query
    enhance_instruction = (
        f"{instruction}\n\n"
        "Improve this section: fix grammar, strengthen academic tone, sharpen vocabulary, "
        "improve argument clarity and paragraph structure. Add evidence signposting where appropriate. "
        "Preserve all factual claims and headings. "
        "Return ONLY the improved text with no meta-commentary."
    )

    source_text = subsection_block[1] if subsection_block and subsection_block[1] else original
    try:
        enhanced = enhance_text(source_text, topic, enhance_instruction, section_title=sec_title)
    except Exception:
        enhanced = _fallback_subsection_text(topic, sec_title, query)

    if subsection_block:
        heading = subsection_block[0]
        enhanced_body = _strip_leading_heading(enhanced, heading).strip()
        replacement_block = f"{heading}\n{enhanced_body}".strip()
        replaced_text = _replace_subsection_if_present(
            original,
            subsection_query=query,
            new_block=replacement_block,
        )
        final_content = replaced_text if replaced_text else original
    else:
        replaced_text = _replace_subsection_if_present(
            original,
            subsection_query=query,
            new_block=f"{query}\n{enhanced}",
        )
        final_content = replaced_text if replaced_text else enhanced

    document.content = update_section(document.content, idx, final_content)
    _save(document, f"enhance-section:{query}")
    _all_done(plan)
    return (
        f"Corrected and enhanced section '{section.get('title', query)}' — "
        "improved clarity, structure, and academic tone.",
        True,
    )


def _write_section(
    document: Document,
    target: str | None,
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    query = (target or _extract_subsection_phrase(instruction) or "").strip()
    query_l = query.lower()
    _personalize_plan_steps(plan, query or instruction or "section")
    chapter_numbers = _extract_chapter_numbers(f"{target or ''} {instruction}")
    chapter_hint = chapter_numbers[0] if chapter_numbers else None

    if chapter_hint is None and any(k in query_l for k in ["empirical review", "conceptual review", "theoretical framework", "literature review"]):
        chapter_hint = 2

    _CHAPTER1_KEYWORDS = [
        "research hypothes", "hypothes", "background of the study", "background of study",
        "statement of the problem", "problem statement", "research objectives", "research questions",
        "significance of the study", "scope and delimitations", "definition of key terms", "key terms",
        # single-word shorthands users commonly type
        "objectives", "background", "significance", "delimitations",
    ]
    if chapter_hint is None and any(k in query_l for k in _CHAPTER1_KEYWORDS):
        chapter_hint = 1

    _CHAPTER3_KEYWORDS = [
        "research design", "target population", "sampling technique", "sample size",
        "data collection", "data analysis", "reliability and validity", "ethical consideration",
    ]
    if chapter_hint is None and any(k in query_l for k in _CHAPTER3_KEYWORDS):
        chapter_hint = 3

    _CHAPTER4_KEYWORDS = ["discussion of findings", "results and discussion", "data presentation"]
    if chapter_hint is None and any(k in query_l for k in _CHAPTER4_KEYWORDS):
        chapter_hint = 4

    _CHAPTER5_KEYWORDS = [
        "summary of findings", "conclusions", "recommendations", "limitations of the study",
        "areas for further research", "further research",
    ]
    if chapter_hint is None and any(k in query_l for k in _CHAPTER5_KEYWORDS):
        chapter_hint = 5

    # Derive chapter hint from dotted numeric targets like "1.5" or "3.2" when no
    # keyword was matched (e.g. user typed "redo 3.2" with no section name)
    if chapter_hint is None:
        dotted = re.match(r"^(\d+)\.", query_l)
        if dotted:
            inferred = int(dotted.group(1))
            if 1 <= inferred <= 10:
                chapter_hint = inferred

    # If request maps to a dissertation chapter structure, generate through nested to-do workflow.
    if chapter_hint is not None:
        design = _research_design(instruction, topic, document)
        objectives = _extract_objectives(document, topic)
        chapter_title = _chapter_title_from_number(chapter_hint)
        chapter_nodes = _chapter_nodes_for_generation(chapter_hint, design, objectives, topic)

        selected_node = _find_matching_node(chapter_nodes, query) if query and not _is_generic_section_query(query) else None

        # SAFETY NET: if the user asked for a specific subsection (e.g. "redo the objectives")
        # but no matching node was found in the chapter template, do NOT fall back to
        # rewriting the whole chapter.  Clear chapter_hint so the simple single-section
        # path below handles the request instead.
        if not selected_node and query and not _is_generic_section_query(query):
            chapter_hint = None

    if chapter_hint is not None:
        # ── Explicit visual injection ───────────────────────────────────────────
        # If the user said "add a table to the sampling section" (or similar),
        # inject the appropriate visual node into the matched node's children so
        # the TODO list and execution pipeline pick it up automatically.
        _vis_req = _detect_visual_injection_request(instruction or "")
        if _vis_req is None:
            # Fallback: detect visual kind directly from instruction without requiring
            # a preposition.  Handles e.g. "generate the conceptual framework image"
            # where the target section is already resolved via routing.
            _fl_text = (instruction or "").lower()
            _fl_verbs = {"add", "put", "insert", "include", "place",
                         "generate", "create", "make", "attach"}
            _fl_kinds = [
                ("table", "table"), ("chart", "chart"), ("graph", "chart"),
                ("image", "chart"), ("figure", "chart"), ("diagram", "chart"),
            ]
            if any(v in _fl_text for v in _fl_verbs):
                for _fl_kw, _fl_kind in _fl_kinds:
                    if _fl_kw in _fl_text:
                        _vis_req = (_fl_kind, query or "")
                        break
        if _vis_req:
            _vis_kind, _vis_hint = _vis_req
            _vis_title = f"Visual: {(_vis_hint or query or 'Data').title()}"
            _visual_node = {
                "title": _vis_title,
                "kind": _vis_kind,
                "children": [],
                "meta": {},
            }
            if selected_node:
                _existing_kinds = [ch.get("kind") for ch in selected_node.get("children", [])]
                if _vis_kind not in _existing_kinds:
                    selected_node.setdefault("children", []).append(_visual_node)
            else:
                # No specific subsection matched; append to the chapter node list
                chapter_nodes.append(_visual_node)
        # ───────────────────────────────────────────────────────────────────────

        nodes_to_write = [selected_node] if selected_node else chapter_nodes
        section_title = selected_node.get("title", chapter_title) if selected_node else chapter_title

        plan.clear()
        plan.append({"step": "Creating section to-do list", "status": "pending"})
        plan.append({"step": f"Writing {section_title}", "status": "pending"})
        _append_node_plan_steps(plan, nodes_to_write, depth=1)
        _done(plan, 0)
        _done(plan, 1)

        sections = document.content.setdefault("sections", [])
        chapter_idx = find_section(document.content, chapter_title)
        if chapter_idx is None:
            chapter_idx = find_section(document.content, f"Chapter {chapter_hint}")
        if chapter_idx is None:
            sections.append({"title": chapter_title, "content": ""})
            chapter_idx = len(sections) - 1

        chapter_payload = sections[chapter_idx]
        plan_cursor = [2]
        figure_counter = [_next_caption_number(document, "figure")]
        table_counter = [_next_caption_number(document, "table")]
        word_count = _chapter_word_count_target(chapter_hint, instruction, nodes_to_write)

        def _persist_subsection_progress(partial_text: str, partial_blocks: list[dict[str, str]], node_title: str) -> None:
            safe_node = re.sub(r"[^a-zA-Z0-9_.-]+", "-", node_title).strip("-")[:60] or "subsection"
            if selected_node:
                updated_content = _replace_subsection_if_present(
                    chapter_payload.get("content", ""),
                    subsection_query=selected_node.get("title", query),
                    new_block=partial_text,
                )
                chapter_payload["content"] = updated_content or (
                    (chapter_payload.get("content") or "").rstrip() + "\n\n" + partial_text
                ).strip()
            else:
                chapter_payload["content"] = partial_text if partial_text.strip() else ""

            if partial_blocks:
                chapter_payload["blocks"] = partial_blocks
            elif "blocks" in chapter_payload:
                chapter_payload.pop("blocks", None)

            document.content["sections"] = sections
            _save(document, f"write-section:{section_title}:{safe_node}")

        chapter_text, _, chapter_blocks = _execute_subsection_nodes(
            nodes=nodes_to_write,
            document=document,
            section_title=chapter_title,
            topic=topic,
            research_design=design,
            rolling_context=_full_context_for_generation(document),
            plan=plan,
            plan_cursor=plan_cursor,
            figure_counter=figure_counter,
            table_counter=table_counter,
            on_node_completed=_persist_subsection_progress,
            default_word_count=word_count,
            user_instruction=instruction or "",
        )

        if selected_node:
            # chapter_text already starts with the subsection title from _execute_subsection_nodes
            updated_content = _replace_subsection_if_present(
                chapter_payload.get("content", ""),
                subsection_query=selected_node.get("title", query),
                new_block=chapter_text,
            )
            chapter_payload["content"] = updated_content or (
                (chapter_payload.get("content") or "").rstrip() + "\n\n" + chapter_text
            ).strip()
        else:
            chapter_payload["content"] = chapter_text if chapter_text.strip() else ""

        if chapter_blocks:
            chapter_payload["blocks"] = chapter_blocks
        elif "blocks" in chapter_payload:
            chapter_payload.pop("blocks", None)

        document.content["sections"] = sections
        _save(document, f"write-section:{section_title}")
        _all_done(plan)
        return (
            f"Written '{section_title}' using dissertation-standard to-do execution and target length controls.",
            True,
        )

    _done(plan, 0)
    section_name = query or instruction or "New Section"
    idx = find_section(document.content, section_name)
    _done(plan, 1)

    existing_context = _full_context_for_generation(document)
    try:
        content = generate_section_content(
            section_name,
            topic,
            context=f"User request: {instruction}\n\nDocument context:\n{existing_context[:1800]}",
            word_count=320,
        )
    except Exception:
        content = _fallback_subsection_text(topic, section_name, section_name)
    content = _sanitize_body(content)
    _done(plan, 2)

    if idx is not None:
        document.content = update_section(document.content, idx, content)
        title = document.content["sections"][idx].get("title", section_name)
    else:
        sections = document.content.setdefault("sections", [])
        sections.append({"title": section_name, "content": content})
        title = section_name

    _save(document, f"write-section:{section_name}")
    _all_done(plan)
    return f"Written and saved section '{title}' (existing content replaced where applicable).", True


def _enhance_chapter_batch(
    document: Document,
    chapter_numbers: list[int],
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    # For chapter-level correction requests, apply full dissertation chapter standards.
    return _rewrite_chapter_batch(document, chapter_numbers, topic, instruction, plan)


def _rewrite_chapter_batch(
    document: Document,
    chapter_numbers: list[int],
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    plan.clear()
    plan.append({"step": "Creating chapter to-do list", "status": "pending"})

    design = _research_design(instruction, topic, document)
    objectives = _extract_objectives(document, topic)

    chapter_blueprints: list[dict[str, Any]] = []
    for chapter_number in chapter_numbers:
        chapter_title = _chapter_title_from_number(chapter_number)
        nodes = _chapter_nodes_for_generation(chapter_number, design, objectives, topic)
        chapter_blueprints.append({
            "number": chapter_number,
            "title": chapter_title,
            "nodes": nodes,
            "word_count": _chapter_word_count_target(chapter_number, instruction, nodes),
        })

    for chapter in chapter_blueprints:
        plan.append({"step": f"Writing {chapter['title']}", "status": "pending"})
        _append_node_plan_steps(plan, chapter["nodes"], depth=1)

    _done(plan, 0)
    updated = False
    rewritten_titles: list[str] = []
    plan_cursor = [1]
    figure_counter = [_next_caption_number(document, "figure")]
    table_counter = [_next_caption_number(document, "table")]

    sections = document.content.setdefault("sections", [])

    for chapter in chapter_blueprints:
        chapter_title = chapter["title"]
        _done(plan, plan_cursor[0])
        plan_cursor[0] += 1

        sec_idx = find_section(document.content, chapter_title)
        if sec_idx is None:
            sec_idx = find_section(document.content, f"Chapter {chapter['number']}")

        if sec_idx is None:
            sections.append({"title": chapter_title, "content": ""})
            sec_idx = len(sections) - 1

        section_payload = sections[sec_idx]
        section_payload["content"] = ""
        document.content["sections"] = sections
        _save(document, f"rewrite-section:{chapter_title}:start")

        def _persist_subsection_progress(partial_text: str, partial_blocks: list[dict[str, str]], node_title: str) -> None:
            safe_node = re.sub(r"[^a-zA-Z0-9_.-]+", "-", node_title).strip("-")[:60] or "subsection"
            content = partial_text if partial_text.strip() else ""
            section_payload["content"] = content
            if partial_blocks:
                section_payload["blocks"] = partial_blocks
            elif "blocks" in section_payload:
                section_payload.pop("blocks", None)
            document.content["sections"] = sections
            _save(document, f"rewrite-section:{chapter_title}:{safe_node}")

        chapter_text, _, chapter_blocks = _execute_subsection_nodes(
            nodes=chapter["nodes"],
            document=document,
            section_title=chapter_title,
            topic=topic,
            research_design=design,
            rolling_context=_full_context_for_generation(document),
            plan=plan,
            plan_cursor=plan_cursor,
            figure_counter=figure_counter,
            table_counter=table_counter,
            on_node_completed=_persist_subsection_progress,
            default_word_count=chapter["word_count"],
        )

        section_payload["content"] = chapter_text if chapter_text.strip() else ""
        if chapter_blocks:
            section_payload["blocks"] = chapter_blocks
        elif "blocks" in section_payload:
            section_payload.pop("blocks", None)
        document.content["sections"] = sections
        _save(document, f"rewrite-section:{chapter_title}")

        updated = True
        rewritten_titles.append(chapter_title)

    if rewritten_titles:
        return (
            f"Rewrote {len(rewritten_titles)} chapter(s) using dissertation standards and nested to-do steps: "
            f"{', '.join(rewritten_titles)}.",
            True,
        )
    return "No chapters were rewritten.", updated


def build_dissertation_preview_plan(
    document: "Document",
    message: str,
    topic: str,
) -> list[dict[str, Any]]:
    """Build a fast preview plan for the frontend TODO list.

    Does NOT call the LLM — uses only research-design detection and the static template
    (except for Chapter 4 which is adapted to quantitative/qualitative/mixed design).
    Chapter 2 uses the static template here; the actual write pass replaces it with
    LLM-generated topic-specific nodes via ``_chapter_nodes_for_generation``.
    """
    design = _research_design(message, topic, document)
    plan: list[dict[str, Any]] = [{"step": "Creating dissertation to-do list", "status": "done"}]
    first_chapter = True
    for template in DISSERTATION_TEMPLATE:
        title = template["title"]
        ch_num = _chapter_number_from_title(title)
        if ch_num == 4:
            nodes = _chapter4_subsections(design, [])
        else:
            nodes = [_normalize_subsection_node(s) for s in template.get("subsections", [])]
        status = "in_progress" if first_chapter else "pending"
        first_chapter = False
        plan.append({"step": f"Writing {title}", "status": status})
        _append_node_plan_steps(plan, nodes, depth=1)
    return plan


def _llm_writing_plan(instruction: str, topic: str, doc_context: str = "") -> dict[str, Any]:
    """Ask the LLM to design a complete writing plan from the user's raw request.

    The LLM decides:
      - What kind of document it is
      - How many sections, their titles and word counts (specific to the topic)
      - Whether a to-do list is warranted (i.e. is this a long piece of work?)
      - The document title

    Returns a dict with keys:
      document_title, document_type, estimated_words, needs_todo, sections
    where each section has: title, word_count, notes
    """
    ctx_block = (
        f"\nExisting document context (for continuity):\n{doc_context[-2000:]}\n"
        if doc_context.strip() else ""
    )
    # Extract guidelines to honour in the plan
    user_guidelines = _parse_user_guidelines(instruction)
    guideline_lines = []
    if user_guidelines.get("citation_style"):
        guideline_lines.append(f"- Citation style: {user_guidelines['citation_style']}")
    if user_guidelines.get("academic_level"):
        guideline_lines.append(f"- Academic level: {user_guidelines['academic_level']}")
    if user_guidelines.get("target_words"):
        guideline_lines.append(f"- Target word count: ~{user_guidelines['target_words']:,} words")
    if user_guidelines.get("focus_notes"):
        guideline_lines.append(f"- Additional requirements: {user_guidelines['focus_notes'][:300]}")
    guidelines_block = (
        "USER GUIDELINES TO HONOUR:\n" + "\n".join(guideline_lines) + "\n\n"
    ) if guideline_lines else ""

    prompt = (
        "You are an expert academic writer. A user has made the following writing request.\n\n"
        f"REQUEST: {instruction[:800]}\n"
        f"TOPIC: {topic or '(infer from the request)'}\n"
        f"{guidelines_block}"
        f"{ctx_block}\n"
        "Design a COMPLETE writing plan for exactly what the user asked. "
        "Do not apply a generic template — let the request determine structure, length, and type.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        "{\n"
        '  "document_title": "A precise, specific title",\n'
        '  "document_type": "essay|report|article|assignment|proposal|case_study|lab_report|review|presentation|plan|brief|other",\n'
        '  "estimated_words": 2500,\n'
        '  "needs_todo": true,\n'
        '  "sections": [\n'
        '    {"title": "Specific section title", "word_count": 350, "notes": "one-sentence writing guide"},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Honour ALL user guidelines above — especially citation style, academic level, and word count.\n"
        "- Section titles must name the actual topic content — NOT generic labels. "
        "If about climate finance, write '2. Green Bond Market Failures', not '2. Background'.\n"
        "- Adapt structure completely to the request: "
        "a short 500-word essay → 3 sections; a 15-page assignment → 8-10 sections with a to-do list; "
        "a lab report → Abstract/Introduction/Methods/Results/Discussion/Conclusion; "
        "a business proposal → Executive Summary/Problem/Solution/Budget/Timeline/Appendix.\n"
        "- If the user says 'X pages', convert: 1 page ≈ 500 words. "
        "If they say 'X words', use that directly.\n"
        "- needs_todo = true when estimated_words > 1500 OR the user explicitly asks for something substantial.\n"
        "- Short items (Abstract, References, Dedication) → word_count 120-200. "
        "Core body sections → 300-600 words each.\n"
        "- Do NOT include a to-do list in the sections themselves.\n"
        "- Return ONLY the JSON object."
    )
    raw = ""
    try:
        raw = generate_text(prompt)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned).rstrip("`").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start: end + 1])
            if isinstance(parsed, dict) and parsed.get("sections"):
                sections = []
                for item in parsed["sections"]:
                    if isinstance(item, dict) and item.get("title"):
                        sections.append({
                            "title": str(item["title"]),
                            "word_count": max(80, int(item.get("word_count") or 300)),
                            "notes": str(item.get("notes") or ""),
                        })
                if sections:
                    return {
                        "document_title": str(parsed.get("document_title") or topic or "Document"),
                        "document_type": str(parsed.get("document_type") or "document"),
                        "estimated_words": int(parsed.get("estimated_words") or sum(s["word_count"] for s in sections)),
                        "needs_todo": bool(parsed.get("needs_todo", True)),
                        "sections": sections,
                    }
    except Exception as exc:
        logger.warning("_llm_writing_plan failed: %s | raw=%s", exc, raw[:300])

    # Fallback: minimal 5-section structure
    t = topic or "the topic"
    return {
        "document_title": topic or "Document",
        "document_type": "document",
        "estimated_words": 1500,
        "needs_todo": False,
        "sections": [
            {"title": "Introduction", "word_count": 280, "notes": f"Introduce {t} and state the aims"},
            {"title": "Background", "word_count": 340, "notes": f"Contextualise {t} with relevant evidence"},
            {"title": "Analysis", "word_count": 380, "notes": f"Critically examine the key dimensions of {t}"},
            {"title": "Discussion", "word_count": 320, "notes": "Interpret findings and consider implications"},
            {"title": "Conclusion", "word_count": 220, "notes": "Synthesise the key points and recommend next steps"},
        ],
    }


def _plan_and_write_document(
    document: Document,
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    """Unified flexible document writer.

    The LLM reads the user's raw instruction and decides:
      - what kind of document it is
      - how many sections and what to write in each
      - whether to show a detailed to-do list (for long pieces)
    No fixed templates. No pre-defined document types.
    """
    plan.clear()
    plan.append({"step": "Planning your document", "status": "pending"})

    doc_context = _full_context_for_generation(document)

    # Parse user-provided guidelines to pass into every section
    user_guidelines = _parse_user_guidelines(instruction)
    _style_note = (
        f"Citation style: {user_guidelines['citation_style']}. "
        f"Academic level: {user_guidelines['academic_level']}. "
    )
    if user_guidelines.get("focus_notes"):
        _style_note += f"Additional requirements: {user_guidelines['focus_notes'][:300]}. "

    # ── Step 1: Let the LLM design the entire plan ────────────────────────────
    writing_plan = _llm_writing_plan(instruction, topic, doc_context)

    doc_title = writing_plan["document_title"]
    doc_type = writing_plan["document_type"]
    needs_todo = writing_plan["needs_todo"]
    sections_plan = writing_plan["sections"]
    estimated_words = writing_plan["estimated_words"]

    logger.info(
        "_plan_and_write_document: type=%s estimated=%d words needs_todo=%s sections=%d level=%s style=%s",
        doc_type, estimated_words, needs_todo, len(sections_plan),
        user_guidelines["academic_level"], user_guidelines["citation_style"],
    )

    # Build the to-do list — always show section steps so the user sees progress
    for item in sections_plan:
        plan.append({"step": f"Writing {item['title']}", "status": "pending"})
    _done(plan, 0)

    design = _research_design(instruction, topic, document)
    sections: list[dict[str, Any]] = []

    # ── Step 2: Write each section the AI designed ────────────────────────────
    for idx, item in enumerate(sections_plan, start=1):
        title = item["title"]
        wc = item["word_count"]
        notes = item.get("notes", "")
        section_guide = _subsection_guidelines(title, topic)
        context_so_far = _full_context_for_generation(document)

        try:
            text = generate_section_content(
                title=title,
                topic=topic,
                context=(
                    f"{section_guide}\n\n"
                    f"Document type: {doc_type}\n"
                    f"Full user request: {instruction[:600]}\n"
                    f"Research design: {design}\n"
                    f"Writing note for this section: {notes}\n"
                    f"GUIDELINES TO FOLLOW: {_style_note}\n\n"
                    f"Document written so far:\n{context_so_far[-3500:]}"
                ),
                word_count=wc,
            )
        except Exception:
            text = _fallback_subsection_text(topic, doc_type.capitalize(), title)

        sections.append({"title": title, "content": text})
        document.content = {
            "topic": topic,
            "sections": sections,
            "document_type": doc_type,
            "document_title": doc_title,
        }
        _save(document, f"doc-step:{re.sub(r'[^a-z0-9]+', '-', title.lower())[:40]}")
        _done(plan, idx)

    document.content = {
        "topic": topic,
        "sections": sections,
        "document_type": doc_type,
        "document_title": doc_title,
    }
    document.title = doc_title
    document.save(update_fields=["content", "title", "updated_at"])
    DocumentVersion.objects.create(
        document=document,
        content=document.content,
        note=f"{doc_type}-generated",
    )
    _all_done(plan)

    total_words = sum(len(s["content"].split()) for s in sections)
    todo_note = " A writing plan was created and each section written in sequence." if needs_todo else ""
    reply = (
        f"'{doc_title}' is complete — {len(sections)} sections, ~{total_words:,} words.{todo_note}"
    )
    return reply, True


# Legacy thin wrappers so old LLM classifications still route correctly
def _write_ai_document(
    document: Document,
    topic: str,
    instruction: str,
    doc_type: str,
    plan: list,
) -> tuple[str, bool]:
    return _plan_and_write_document(document, topic, instruction or topic, plan)


def _write_article(
    document: Document,
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    return _plan_and_write_document(document, topic, instruction or f"Write an academic article on {topic}", plan)


def _write_dissertation(
    document: Document,
    topic: str,
    instruction: str,
    plan: list,
) -> tuple[str, bool]:
    """Write a full dissertation.

    Flow:
    1. Call the LLM to generate a tailored chapter/section plan (no hardcoded template).
    2. Convert that plan into the internal blueprint format.
    3. Walk each chapter and subsection, writing content via LLM.
    4. Persist progress after every subsection so the frontend polling sees live updates.
    """
    plan.clear()
    plan.append({"step": "Creating dissertation to-do list", "status": "pending"})

    design = _research_design(instruction, topic, document)

    # ── Parse user-provided guidelines (style, level, word count, notes) ─
    user_guidelines = _parse_user_guidelines(instruction)
    logger.info(
        "_write_dissertation: guidelines — level=%s style=%s target_words=%s",
        user_guidelines["academic_level"],
        user_guidelines["citation_style"],
        user_guidelines.get("target_words"),
    )

    # ── Step 0: Intent parsing via PlannerAgent ───────────────────────────
    from .planner import PlannerAgent as _PlannerAgent
    _planner = _PlannerAgent()
    objectives_early = _extract_objectives(document, topic)
    intent_spec = _planner.parse_intent(
        message=instruction,
        topic=topic,
        research_design=design,
        objectives=objectives_early,
        intent="write_dissertation",
    )
    logger.info(
        "_write_dissertation: IntentSpec parsed — topic=%s design=%s objectives=%d",
        intent_spec.topic[:60], intent_spec.research_design, len(intent_spec.objectives),
    )

    # ── Step 1: Generate the full plan via LLM ────────────────────────────
    # Check if a plan was already generated by the DissertationPlanView and cached
    stored_chapters: list | None = None
    try:
        stored_chapters = (document.content or {}).get("_dissertation_plan_chapters")
    except Exception:
        pass

    if stored_chapters and isinstance(stored_chapters, list) and len(stored_chapters) >= 3:
        llm_chapters = stored_chapters
        logger.info("_write_dissertation: using pre-generated plan (%d chapters)", len(llm_chapters))
    else:
        logger.info("_write_dissertation: calling LLM to generate plan for topic=%s", topic[:80])
        llm_chapters = generate_dissertation_plan_llm(
            topic, instruction, design,
            objectives=objectives_early,
            guidelines=user_guidelines,
        )

    chapter_blueprints = llm_chapters_to_blueprints(llm_chapters)

    # ── Step 2: Build the flat step list ─────────────────────────────────
    for chapter in chapter_blueprints:
        plan.append({"step": f"Writing {chapter['title']}", "status": "pending"})
        _append_node_plan_steps(plan, chapter["nodes"], depth=1)

    _done(plan, 0)  # mark "Creating dissertation to-do list" done

    # ── Step 3: Write each chapter ────────────────────────────────────────
    objectives = _extract_objectives(document, topic)
    sections: list[dict[str, Any]] = []
    plan_cursor = [1]
    figure_counter = [_next_caption_number(document, "figure")]
    table_counter = [_next_caption_number(document, "table")]

    # Build a compact guidelines string to inject into each subsection prompt
    _style_note = (
        f"Citation style: {user_guidelines['citation_style']}. "
        f"Academic level: {user_guidelines['academic_level']}. "
    )
    if user_guidelines.get("focus_notes"):
        _style_note += f"Additional requirements: {user_guidelines['focus_notes'][:300]}. "
    # Prepend to the raw instruction so ContentGenerator sees it
    enriched_instruction = f"{_style_note}\n\n{instruction}"

    for chapter in chapter_blueprints:
        chapter_title = chapter["title"]
        ch_num = _chapter_number_from_title(chapter_title)
        # Respect user's word count target; fall back to chapter defaults
        ch_word_count = _chapter_word_count_target(ch_num, instruction, chapter["nodes"])

        _done(plan, plan_cursor[0])
        plan_cursor[0] += 1

        # Add chapter placeholder so the frontend polling sees incremental progress.
        section_payload: dict[str, Any] = {"title": chapter_title, "content": ""}
        sections.append(section_payload)
        document.content = {
            "topic": topic,
            "research_design": design,
            "research_objectives": objectives,
            "sections": sections,
        }
        _save(document, f"dissertation-step:{chapter_title}:start")

        def _persist_subsection_progress(partial_text: str, partial_blocks: list[dict[str, str]], node_title: str) -> None:
            safe_node = re.sub(r"[^a-zA-Z0-9_.-]+", "-", node_title).strip("-")[:60] or "subsection"
            section_payload["content"] = partial_text if partial_text.strip() else ""
            if partial_blocks:
                section_payload["blocks"] = partial_blocks
            elif "blocks" in section_payload:
                section_payload.pop("blocks", None)
            document.content = {
                "topic": topic,
                "research_design": design,
                "research_objectives": objectives,
                "sections": sections,
            }
            _save(document, f"dissertation-step:{chapter_title}:{safe_node}")

        current_context = _full_context_for_generation(document)
        chapter_text, _, chapter_blocks = _execute_subsection_nodes(
            nodes=chapter["nodes"],
            document=document,
            section_title=chapter_title,
            topic=topic,
            research_design=design,
            rolling_context=current_context,
            plan=plan,
            plan_cursor=plan_cursor,
            figure_counter=figure_counter,
            table_counter=table_counter,
            on_node_completed=_persist_subsection_progress,
            default_word_count=ch_word_count,
            user_instruction=enriched_instruction,
        )

        section_payload["content"] = chapter_text if chapter_text.strip() else ""
        if chapter_blocks:
            section_payload["blocks"] = chapter_blocks
        elif "blocks" in section_payload:
            section_payload.pop("blocks", None)

        document.content = {
            "topic": topic,
            "research_design": design,
            "research_objectives": objectives,
            "sections": sections,
        }
        _save(document, f"dissertation-step:{chapter_title}")

    document.title = f"Dissertation: {topic}"
    document.save(update_fields=["title", "updated_at"])
    _all_done(plan)

    reply = (
        f"Dissertation generation complete for '{topic}' using a {design.replace('_', ' ')} design. "
        "The AI generated a tailored chapter and section plan, then wrote each section sequentially."
    )
    return reply, True


def _create_outline(document: Document, topic: str, plan: list) -> tuple[str, bool]:
    _done(plan, 0)
    chapters = generate_outline_sections(topic)
    _done(plan, 1)

    sections = [
        {"title": c.get("title", f"Chapter {i + 1}"), "content": ""}
        for i, c in enumerate(chapters)
    ]
    document.content = {"topic": topic, "sections": sections}
    _save(document, "outline-created")
    _all_done(plan)

    titles = "\n".join(f"• {s['title']}" for s in sections)
    return f"Created outline with {len(sections)} chapters:\n{titles}", True


def _write_structured_document(
    document: Document,
    topic: str,
    kind: str,
    plan: list,
    instruction: str = "",
) -> tuple[str, bool]:
    """Route report/assignment/presentation/spreadsheet through the unified AI planner."""
    effective = instruction or f"Write a {kind} on {topic}"
    return _plan_and_write_document(document, topic, effective, plan)


def _add_chart(document: Document, target: str | None, plan: list) -> tuple[str, bool]:
    _done(plan, 0)
    sections = (document.content or {}).get("sections", [])
    if not sections:
        _all_done(plan)
        return "No sections in document. Create an outline first.", False

    idx = find_section(document.content, target) if target else len(sections) - 1
    if idx is None:
        idx = 0
    section = sections[idx]
    _done(plan, 1)

    _section_title = section.get("title", "Data")
    _ai = _ai_chart_series(_section_title, n_points=8)
    chart_path = generate_chart(
        series=_ai["series"],
        chart_type=_ai["chart_type"],
        title=_section_title,
        x_labels=_ai.get("x_labels") or None,
        unit=_ai.get("unit") or None,
    )
    section.setdefault("blocks", []).append(
        {"type": "chart", "src": chart_path, "caption": f"Chart for {section.get('title', 'section')}"}
    )
    _save(document, f"chart:{target or 'section'}")
    _all_done(plan)
    return f"Added a chart to '{section.get('title', 'the section')}'.", True


def _add_image(
    document: Document, target: str | None, prompt: str, plan: list
) -> tuple[str, bool]:
    _done(plan, 0)
    sections = (document.content or {}).get("sections", [])

    idx = _framework_target_index(document, target, prompt)
    if idx is None:
        _all_done(plan)
        return "No sections available. Create an outline first.", False

    framework_request = any(
        key in (prompt or "").lower()
        for key in ["conceptual", "theoretical", "framework", "research model", "model"]
    )

    framework_spec = _build_framework_spec(document, target, prompt) if framework_request else None

    try:
        image_path = generate_image(prompt, framework_spec=framework_spec)
    except Exception as exc:
        _all_done(plan)
        return f"Image generation failed: {exc}", False

    _done(plan, 1)

    if sections:
        section = sections[idx]
        blocks = section.setdefault("blocks", [])
        block_id = f"img-{idx + 1}-{len(blocks) + 1}"
        caption = (
            (framework_spec or {}).get("title")
            or prompt[:80]
            or f"Image for {section.get('title', 'section')}"
        )
        blocks.append(
            {"type": "image", "src": image_path, "caption": str(caption)[:120], "block_id": block_id}
        )
        section["content"] = _insert_block_marker(section.get("content", ""), block_id, prompt)
        _save(document, f"image:{target or 'section'}")

    _all_done(plan)
    section_name = sections[idx].get("title", "the section") if sections else "the section"
    return f"Added an image to '{section_name}' using full-document context.", True
