"""
Autonomous document agent.
Accepts a free-form user message, classifies intent, plans, and executes
direct edits on the document using a selectable LLM provider.
"""
from __future__ import annotations

import logging
import json
import math
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
            "Write the Research Hypotheses. If this study is quantitative or mixed-methods and its "
            "objectives genuinely test a relationship, effect, or difference between measurable "
            "variables, write a numbered list of null/alternative hypothesis pairs, one pair per "
            "such objective (typically 3–5 pairs). Format each pair as: "
            "N. H0: [null hypothesis — states no significant relationship/effect]. "
            "   H1: [alternative hypothesis — states a significant relationship/effect]. "
            "Use formal statistical language. "
            "If the study is qualitative, purely descriptive, or exploratory — with no measurable "
            "variables to correlate — do NOT force statistical H0/H1 pairs onto it. Instead write "
            "numbered working propositions or guiding assumptions the study explores, phrased as "
            "plain declarative statements tied to the actual objectives."
        )

    if "significance of the study" in lowered or "significance of study" in lowered or (
        lowered.endswith("significance") and "statistical" not in lowered
    ):
        return (
            "Write the Significance of the Study in a structured format. "
            "Open with 1–2 paragraphs on the theoretical/academic contribution. "
            "Then present practical significance using clearly labelled sub-headings or numbered "
            "points — but choose the beneficiaries that genuinely apply to THIS study rather than "
            "reusing the same four labels regardless of fit. Common options include 'Significance to "
            "the Researcher', 'Significance to the Institution', 'Significance to the Organisation' "
            "(only if the study is set within or studies a specific organisation/sector), "
            "'Significance to Policy/Policymakers' (only if the findings plausibly inform policy), "
            "'Significance to Practitioners', or 'Significance to Future Researchers'. Pick 3–4 "
            "beneficiaries that make sense for this topic and drop any that would be a stretch. "
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
            "Write the Scope and Delimitations (200–300 words). Cover whichever of these genuinely "
            "apply to this study — skip any that don't fit rather than forcing them: "
            "(1) the geographical scope, (2) the target population/participants (or unit of analysis "
            "— e.g. documents, systems, datasets — for studies with no human subjects), "
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
            "Then present 5–8 specific, actionable recommendations, each named and numbered. "
            "Each recommendation must: name the action, specify WHO should take it, "
            "explain WHY (link back to a finding), and describe the expected outcome. "
            "Be direct, concrete, and feasible — avoid vague generalities like 'organisations should improve'. "
            "If — and only if — this study's topic is applied/organizational/policy-oriented enough that a "
            "rollout timeline genuinely makes sense, group the recommendations under three labelled "
            "sub-headings by implementation horizon: 'Short-term (0–6 months)', 'Medium-term (6–18 months)', "
            "and 'Long-term (18 months and beyond)'. For a purely theoretical, lab-based, or exploratory "
            "topic where no real-world rollout is implied, skip that grouping and just number the list."
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
            "Write the Methods section with full procedural clarity. Cover whichever of the "
            "following genuinely apply — for a study with no human subjects (e.g. a build, "
            "simulation, document/data analysis), replace 'participants' with the actual unit of "
            "analysis (datasets, artefacts, systems, documents) rather than inventing participants: "
            "(1) study design and rationale, (2) participants/sample with inclusion criteria, or the "
            "materials/data sources used, "
            "(3) data collection instruments or tools (name them, describe them, justify them), "
            "(4) procedure (step-by-step), "
            "(5) data analysis approach (specific technique, software if used), "
            "(6) ethical considerations, where relevant. "
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

    if "contribution" in lowered:
        return (
            "Write the Contributions of the Study using three clearly labelled sub-headings: "
            "'Theoretical Contributions' (how this study extends, refines, or challenges existing theory "
            "or frameworks), 'Practical Contributions' (concrete value for practitioners, organizations, "
            "or policy), and 'Methodological Contributions' (anything notable about the approach, "
            "instruments, or analysis technique used). Each sub-section: 2–4 sentences, specific to this "
            "study's actual findings and design — not generic claims any study could make."
        )

    if "references" in lowered or "bibliography" in lowered:
        return (
            "No verified external sources were retrievable for this document (the literature search "
            "returned nothing, e.g. due to no network access). Write a short note stating plainly that "
            "automatic source retrieval failed and that the entries below are ILLUSTRATIVE PLACEHOLDERS, "
            "not verified citations — they must be replaced with real literature before submission. "
            "Then provide 8–12 example references in APA 7th edition format, each prefixed with "
            "'[Placeholder]' so they cannot be mistaken for verified sources."
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
# any of these sections. Chapter 4 is excluded (handled by _plan_chapter4_structure).
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
        "chapter": 2, "keyword": ("conceptual review", "conceptual framework"),
        "kind": "chart", "title": "Conceptual Framework Diagram",
        "meta": {"chart_type": "framework"}, "designs_only": None,
    },
    {
        "chapter": 2, "keyword": "theoretical framework",
        "kind": "chart", "title": "Theoretical Framework Diagram",
        "meta": {"chart_type": "theory_model"}, "designs_only": None,
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
    has its own visual logic in _plan_chapter4_structure.
    """
    if chapter_number == 4:
        return nodes

    specs = [s for s in _SECTION_VISUAL_SPECS if s["chapter"] == chapter_number]
    if not specs:
        return nodes

    result: list[dict[str, Any]] = []
    for node in nodes:
        title_lower = node.get("title", "").lower()

        # Recurse into the node's OWN (pre-injection) children first. Visual nodes
        # appended below are never fed back into this recursion, so an injected
        # node whose own title happens to contain its trigger keyword (e.g. "Conceptual
        # Framework Diagram" containing "conceptual framework") can't self-match and
        # recurse forever.
        original_children = list(node.get("children", []))
        children = (
            _inject_standard_visuals(original_children, chapter_number, research_design)
            if original_children else []
        )

        for spec in specs:
            spec_keywords = spec["keyword"]
            if isinstance(spec_keywords, str):
                spec_keywords = (spec_keywords,)
            if not any(kw in title_lower for kw in spec_keywords):
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

        result.append({**node, "children": children})
    return result


def _has_chart_type(nodes: list[dict[str, Any]], chart_type: str) -> bool:
    for n in nodes:
        if n.get("kind") == "chart" and (n.get("meta") or {}).get("chart_type") == chart_type:
            return True
        if _has_chart_type(n.get("children", []) or [], chart_type):
            return True
    return False


def _ensure_chart_visual(
    nodes: list[dict[str, Any]],
    chart_type: str,
    title: str,
    keywords: tuple[str, ...],
    required: bool = True,
    exclude_keywords: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Guarantee a Chapter 2 diagram of the given chart_type exists even if the LLM-tailored
    section titles don't literally contain one of the _SECTION_VISUAL_SPECS keywords (the plan
    prompt deliberately encourages topic-specific, non-generic headings).

    When required is False, this only attaches the diagram to a section whose title already
    suggests it belongs there — it never invents a home for it on an unrelated node. Used for
    diagrams (e.g. a named-theory model) that don't fit every study, so the planner's own
    decision to omit the matching heading is respected rather than overridden.

    exclude_keywords skips candidate nodes that belong to a different, more specific diagram
    (e.g. a "Theoretical Framework" heading also contains the substring "framework", which would
    otherwise wrongly steal the Conceptual Framework diagram meant for its own heading).
    """
    if not nodes or _has_chart_type(nodes, chart_type):
        return nodes

    target_idx = next(
        (
            i for i, n in enumerate(nodes)
            if any(kw in (n.get("title") or "").lower() for kw in keywords)
            and not any(kw in (n.get("title") or "").lower() for kw in exclude_keywords)
        ),
        None,
    )
    if target_idx is None:
        if not required:
            return nodes
        # No obvious match — place it on the second node (after the chapter intro),
        # or the first if the chapter has only one section.
        target_idx = 1 if len(nodes) > 1 else 0

    result = list(nodes)
    target = dict(result[target_idx])
    target["children"] = list(target.get("children", [])) + [{
        "title": title,
        "kind": "chart",
        "children": [],
        "meta": {"chart_type": chart_type},
    }]
    result[target_idx] = target
    return result


def _ensure_framework_visual(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Guarantee Chapter 2 gets a conceptual-framework diagram (near-universal across designs),
    and a theoretical-framework (named theory) diagram only when a section already suggests one
    is intended. Theory is checked first so it claims a "theoretical"-titled node before the
    conceptual fallback's broader keyword match would otherwise grab it; the conceptual pass also
    explicitly excludes "theoretical"-titled nodes so it can't steal that heading even when no
    theory diagram ends up attached to it (e.g. when required=False skipped it).
    """
    nodes = _ensure_chart_visual(
        nodes, "theory_model", "Theoretical Framework Diagram", ("theoretical", "theory"), required=False,
    )
    nodes = _ensure_chart_visual(
        nodes, "framework", "Conceptual Framework Diagram", ("conceptual", "framework", "model"),
        required=True, exclude_keywords=("theoretical", "theory"),
    )
    return nodes


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


# Specific named methodologies, and the broader family (quantitative/qualitative/mixed) each
# implies. Checked when the source text names one of these without ever saying the literal
# word "qualitative"/"quantitative"/"mixed" — without this, an unrecognised design used to fall
# through to the hardcoded "quantitative" default, producing survey/Likert-scale fallback text
# for studies (e.g. grounded theory, ethnography, an experiment) that never used a survey at all.
_SPECIFIC_METHODOLOGY_HINTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("case_study", ("case study", "case-study", "multiple case study", "single case study"), "qualitative"),
    ("action_research", ("action research",), "qualitative"),
    ("grounded_theory", ("grounded theory",), "qualitative"),
    ("ethnography", ("ethnograph",), "qualitative"),
    ("phenomenology", ("phenomenolog",), "qualitative"),
    ("narrative_inquiry", ("narrative inquiry", "narrative research"), "qualitative"),
    ("content_analysis", ("content analysis",), "qualitative"),
    ("design_science", ("design science", "design and development research", "design-based research"), "mixed"),
    ("quasi_experimental", ("quasi-experimental", "quasi experimental"), "quantitative"),
    ("experimental", ("true experimental", "experimental design", "randomized control", "randomised control", " rct ", "pretest-posttest", "pre-test post-test"), "quantitative"),
    ("correlational", ("correlational",), "quantitative"),
    ("cross_sectional", ("cross-sectional survey", "cross sectional survey"), "quantitative"),
)

# Human-readable label and one-line framing used to correctly name a specific design in
# generated text (e.g. "A case study design was adopted..." instead of the generic
# "A qualitative research design was adopted...").
_SPECIFIC_METHODOLOGY_LABELS: dict[str, str] = {
    "case_study": "case study",
    "action_research": "action research",
    "grounded_theory": "grounded theory",
    "ethnography": "ethnographic",
    "phenomenology": "phenomenological",
    "narrative_inquiry": "narrative inquiry",
    "content_analysis": "content analysis",
    "design_science": "design science",
    "quasi_experimental": "quasi-experimental",
    "experimental": "experimental",
    "correlational": "correlational",
    "cross_sectional": "cross-sectional survey",
}

# Per-design overrides for the Chapter 3 "Research Design", "Sampling", "Data Collection", and
# "Analysis" subsections, used in place of the generic qualitative/quantitative family text
# whenever a specific named methodology (see _SPECIFIC_METHODOLOGY_HINTS) is detected. A design
# is only given an override here where the generic family text would otherwise describe a
# materially different procedure (e.g. grounded theory's open/axial/selective coding, or an
# experiment's random assignment to control/treatment groups) — designs left out of a given
# category (e.g. "correlational" has no sampling/collection override) simply fall through to
# the existing, already-applicable quantitative/qualitative text for that category.
_SPECIFIC_DESIGN_TEXT: dict[str, dict[str, str]] = {
    "case_study": {
        "design": (
            "A case study research design was adopted, enabling an in-depth, holistic investigation of {topic} "
            "within its real-life context. This design was selected because the phenomenon under investigation is "
            "best understood through detailed examination of a bounded case (or a small number of cases) rather "
            "than through statistical generalisation across a large sample.\n\n"
            "Multiple sources of evidence were drawn upon and triangulated to strengthen the validity of the "
            "findings, consistent with established case study methodology, with the boundaries of the case "
            "defined clearly in relation to the research objectives set out in Chapter 1."
        ),
        "sampling": (
            "The case (or cases) was selected purposively on the basis of its relevance to the research "
            "objectives and the richness of information it could be expected to yield, rather than through "
            "probability sampling. Within the selected case, key individuals were identified and invited to "
            "participate based on their direct knowledge of, or involvement with, the phenomenon under study.\n\n"
            "This purposive approach prioritises depth and contextual understanding of the bounded case over "
            "statistical representativeness, consistent with the explanatory aims of case study research."
        ),
        "collection": (
            "Data were collected from multiple sources of evidence, including semi-structured interviews, "
            "relevant documents, and, where applicable, direct observation, in order to triangulate findings and "
            "build a comprehensive picture of the case. Interviews were audio-recorded with consent and "
            "transcribed verbatim.\n\n"
            "Drawing on more than one source of evidence is a defining feature of rigorous case study research, "
            "reducing reliance on any single account and strengthening confidence in the patterns identified."
        ),
        "analysis": (
            "Data were analysed using within-case analysis to identify patterns specific to the case, followed by "
            "pattern-matching against the theoretical propositions guiding the study. Where more than one case was "
            "examined, cross-case analysis was used to identify similarities and differences between cases.\n\n"
            "This analytic approach allows findings to remain grounded in the specific context of the case while "
            "still supporting broader analytic generalisation in relation to the study's research objectives."
        ),
    },
    "action_research": {
        "design": (
            "An action research design was adopted, structured around iterative cycles of planning, action, "
            "observation, and reflection. This design was selected because the research objectives call for "
            "practical change within a specific setting, generated collaboratively with those directly affected "
            "by {topic}, rather than the detached observation typical of more conventional designs.\n\n"
            "Each cycle's findings directly informed the design of the subsequent cycle, allowing the intervention "
            "and the understanding of its effects to develop together over the course of the study."
        ),
        "sampling": (
            "Participants were drawn from within the specific organisational or community setting where the "
            "action research was conducted, selected because of their direct involvement in the practices being "
            "examined and improved. All relevant stakeholders within the setting were invited to participate in "
            "one or more cycles of the study.\n\n"
            "This site-based approach reflects the collaborative, change-oriented nature of action research, "
            "where participants are co-investigators rather than passive subjects of the research."
        ),
        "collection": (
            "Data were collected throughout each action research cycle using a combination of methods, including "
            "field notes, reflective journals, semi-structured interviews, and, where relevant, documentary "
            "evidence of the changes implemented. Data collection was repeated across cycles to capture the "
            "evolving effects of each intervention.\n\n"
            "This continuous, cyclical approach to data collection allowed the research team to adjust the "
            "intervention in near real time in response to emerging evidence."
        ),
        "analysis": (
            "Data from each cycle were analysed reflectively, in collaboration with participating stakeholders, "
            "to evaluate the effects of the action taken and to inform the plan for the subsequent cycle. Across "
            "cycles, recurring patterns were identified using thematic analysis of the qualitative data generated.\n\n"
            "This iterative analytic process is central to action research, ensuring that interpretation of the "
            "data directly and continuously shapes practical action within the study setting."
        ),
    },
    "grounded_theory": {
        "design": (
            "A grounded theory research design was adopted, with the aim of generating a theory grounded in, and "
            "explanatory of, the data collected on {topic}, rather than testing a theory specified in advance. "
            "This design was selected because existing theory does not yet adequately account for the processes "
            "under investigation.\n\n"
            "Data collection and analysis proceeded iteratively and concurrently, with each round of data "
            "collection informed by categories emerging from the analysis of previous data, consistent with "
            "established grounded theory procedure."
        ),
        "sampling": (
            "Theoretical sampling was used, whereby initial participants were selected purposively for their "
            "relevant experience, and subsequent participants were then selected specifically to refine, test, "
            "and saturate the categories emerging from ongoing analysis. Sampling and data collection continued "
            "until theoretical saturation was reached.\n\n"
            "This approach differs from conventional purposive sampling in that the sample itself evolves "
            "iteratively in direct response to the developing theoretical categories."
        ),
        "collection": (
            "Data were collected primarily through semi-structured interviews, conducted iteratively alongside "
            "ongoing analysis so that emerging categories could be probed and refined in subsequent interviews. "
            "Interviews were audio-recorded with consent and transcribed verbatim, with memo-writing used "
            "throughout to capture analytic insights as data collection proceeded.\n\n"
            "This concurrent collection-and-analysis process is a defining feature of the grounded theory "
            "approach, distinguishing it from designs in which all data are collected before analysis begins."
        ),
        "analysis": (
            "Data were analysed using the constant comparative method, proceeding through open coding, axial "
            "coding, and selective coding to identify a core category capable of explaining the patterns observed "
            "across the data. Memos were used throughout to document the development of categories and their "
            "interrelationships.\n\n"
            "This iterative coding process continued until theoretical saturation was reached, culminating in a "
            "substantive theory grounded in, and directly traceable to, the data collected for this study."
        ),
    },
    "ethnography": {
        "design": (
            "An ethnographic research design was adopted, enabling sustained, in-depth engagement with the social "
            "or cultural group under study in order to understand {topic} from within its natural setting. This "
            "design was selected because the research objectives require an understanding of shared practices, "
            "meanings, and norms that are best accessed through prolonged immersion rather than brief contact.\n\n"
            "Fieldwork was conducted over a defined period within the study setting, with the researcher's role "
            "and degree of participation clarified at the outset and maintained consistently throughout."
        ),
        "sampling": (
            "Key informants were identified purposively from within the cultural or social group under study, "
            "selected for their knowledge of, and standing within, the group's practices and norms. Additional "
            "participants were recruited as fieldwork progressed, guided by relationships and opportunities that "
            "developed during prolonged engagement with the setting.\n\n"
            "This evolving, relationship-based approach to sampling is characteristic of ethnographic research and "
            "supports access to perspectives that might not be available through formal recruitment alone."
        ),
        "collection": (
            "Data were collected primarily through participant observation and detailed field notes, supplemented "
            "by informal and semi-structured interviews with key informants. Observations were recorded as soon "
            "as practicable after each period of fieldwork to preserve contextual detail.\n\n"
            "This combination of observation and interview data allowed the study to capture both what "
            "participants said about their practices and what was directly observed within the setting."
        ),
        "analysis": (
            "Field notes and interview transcripts were analysed thematically, with attention to recurring "
            "practices, meanings, and patterns of social interaction within the group studied. Analysis proceeded "
            "alongside data collection, allowing emerging interpretations to be checked against further "
            "observation.\n\n"
            "Thick description was used in reporting the findings, situating the patterns identified firmly within "
            "the cultural and contextual particulars of the setting studied."
        ),
    },
    "phenomenology": {
        "design": (
            "A phenomenological research design was adopted, aimed at uncovering the essential structure of "
            "participants' lived experience of {topic}. This design was selected because the research objectives "
            "are concerned with the meaning of the experience itself, rather than with explaining its causes or "
            "measuring its prevalence.\n\n"
            "The researcher's own assumptions about the phenomenon were examined and set aside as far as "
            "possible throughout data collection and analysis, consistent with the phenomenological emphasis on "
            "approaching participants' accounts with openness."
        ),
        "sampling": (
            "Participants were selected purposively on the basis of having directly lived the experience under "
            "investigation, with sufficient richness and variation in their accounts to support an in-depth "
            "exploration of the phenomenon. Recruitment continued until no substantially new aspects of the "
            "experience emerged from additional interviews.\n\n"
            "This criterion-based approach ensures that all participants are positioned to speak directly and "
            "meaningfully to the lived experience that is the focus of the study."
        ),
        "collection": (
            "Data were collected through in-depth, semi-structured interviews designed to elicit detailed, "
            "first-person descriptions of participants' lived experience, with broad, open questions used to "
            "minimise the imposition of the researcher's own assumptions. Interviews were audio-recorded with "
            "consent and transcribed verbatim.\n\n"
            "Sufficient time was allowed within each interview for participants to describe their experience in "
            "their own words and at their own pace."
        ),
        "analysis": (
            "Transcripts were analysed using a structured phenomenological approach, involving close reading of "
            "each transcript, identification of significant statements, and clustering of these statements into "
            "themes that capture the essential structure of the experience. Bracketing was used throughout to "
            "limit the influence of the researcher's preconceptions on the resulting themes.\n\n"
            "The resulting themes were synthesised into a composite description intended to convey the essence "
            "of the experience shared across participants' individual accounts."
        ),
    },
    "narrative_inquiry": {
        "design": (
            "A narrative inquiry research design was adopted, focused on understanding {topic} through the "
            "stories participants tell about their own experience. This design was selected because the research "
            "objectives are concerned with how participants make sense of and give meaning to their experience "
            "over time, rather than with isolated facts or events.\n\n"
            "The study followed a small number of participants in depth, prioritising the richness and coherence "
            "of each individual narrative over breadth across a larger sample."
        ),
        "sampling": (
            "A small number of participants were selected purposively on the basis of having a relevant and "
            "tellable story to share in relation to the research objectives. Selection prioritised diversity of "
            "experience and willingness to engage in extended narrative interviewing over statistical "
            "representativeness.\n\n"
            "This deliberately small, depth-oriented sample is consistent with narrative inquiry's focus on "
            "detailed individual accounts rather than aggregated patterns."
        ),
        "collection": (
            "Data were collected through extended, narrative-style interviews in which participants were invited "
            "to recount their experience in story form, with minimal interruption from the researcher. Interviews "
            "were audio-recorded with consent and transcribed verbatim, preserving the sequence and structure of "
            "each participant's account.\n\n"
            "Follow-up conversations were held where necessary to clarify or extend elements of a participant's "
            "narrative."
        ),
        "analysis": (
            "Transcripts were analysed using narrative analysis, attending to the structure, sequence, and "
            "meaning of each participant's story before identifying themes that cut across individual narratives. "
            "Restorying was used to present each account in a coherent chronological and thematic form.\n\n"
            "This approach preserves the integrity of each individual's story while still allowing broader "
            "patterns relevant to the research objectives to be identified across participants."
        ),
    },
    "content_analysis": {
        "design": (
            "A content analysis research design was adopted, involving the systematic examination of existing "
            "documents, texts, or media relevant to {topic}. This design was selected because the research "
            "objectives can be addressed through analysis of existing textual or recorded material rather than "
            "through the collection of new primary data from human participants.\n\n"
            "A coding frame was developed prior to analysis, informed by the research objectives and refined "
            "through initial review of a subset of the material."
        ),
        "sampling": (
            "A purposive sample of documents, texts, or media items was selected based on clearly defined "
            "inclusion criteria directly tied to the research objectives, with the sampling frame and selection "
            "period stated explicitly to support transparency and replication.\n\n"
            "Sample size was determined by the point at which additional material ceased to add substantively "
            "new categories to the coding frame."
        ),
        "collection": (
            "Relevant documents, texts, or media items were retrieved from clearly specified sources and "
            "compiled into a structured corpus for analysis. Each item was logged with relevant metadata "
            "(source, date, and type) to support systematic and auditable coding.\n\n"
            "Where items were not available in a directly analysable format, they were transcribed or converted "
            "prior to coding."
        ),
        "analysis": (
            "Material was analysed using a structured coding frame, with each unit of analysis classified "
            "according to predefined and emergent categories tied to the research objectives. Frequencies of "
            "categories were tabulated, and, where relevant, the latent meaning underlying manifest content was "
            "also considered.\n\n"
            "A subset of the material was independently double-coded to check the consistency of the coding "
            "frame's application."
        ),
    },
    "design_science": {
        "design": (
            "A design science research approach was adopted, structured around the iterative design, "
            "development, and evaluation of an artefact intended to address a defined problem within {topic}. "
            "This approach was selected because the research objectives are concerned with creating and "
            "evaluating a practical solution, rather than solely describing or explaining an existing "
            "phenomenon.\n\n"
            "The study proceeded through cycles of problem identification, artefact design, development, "
            "demonstration, and evaluation, with each cycle informing refinements to the artefact."
        ),
        "sampling": (
            "Participants in the evaluation phase were selected purposively from among intended users or domain "
            "experts, on the basis of their ability to assess the artefact's relevance, usability, and "
            "effectiveness against the problem it was designed to address.\n\n"
            "This evaluation-focused sampling approach reflects design science's emphasis on practical utility "
            "over statistical generalisation."
        ),
        "collection": (
            "Data were collected at each stage of the design cycle, including requirements data informing the "
            "artefact's design and evaluation data (gathered through testing, expert review, or user feedback) "
            "assessing its performance against the defined objectives.\n\n"
            "Evaluation data were collected using a combination of structured testing and qualitative feedback "
            "from users or domain experts engaging directly with the artefact."
        ),
        "analysis": (
            "Evaluation data were analysed against the design objectives defined at the outset of the study, "
            "assessing the extent to which the artefact addressed the identified problem. Qualitative feedback "
            "was analysed thematically, while any quantitative performance data were analysed descriptively.\n\n"
            "Findings from each evaluation cycle directly informed refinements carried into the subsequent design "
            "iteration, consistent with the iterative logic of design science research."
        ),
    },
    "experimental": {
        "design": (
            "A true experimental research design was adopted, involving the random assignment of participants "
            "to a treatment group and a control group, in order to test the causal effect of the intervention "
            "related to {topic}. This design was selected because the research objectives require a credible "
            "basis for causal inference, which random assignment is specifically intended to support.\n\n"
            "Extraneous variables were controlled as far as practicable through the experimental protocol, "
            "isolating the effect of the manipulated variable on the outcome of interest."
        ),
        "sampling": (
            "Eligible participants were recruited and then randomly assigned to either the treatment group or "
            "the control group, with random assignment used specifically to distribute potential confounding "
            "characteristics evenly between groups. Sample size was determined using a power analysis to ensure "
            "adequate sensitivity to detect the hypothesised effect.\n\n"
            "Random assignment, rather than random sampling alone, is the defining feature of this design and "
            "the basis for the causal claims the study is able to support."
        ),
        "collection": (
            "Data were collected using a pre-test/post-test protocol administered identically to both the "
            "treatment and control groups, with the treatment group additionally receiving the intervention "
            "under investigation between the two measurement points.\n\n"
            "Standardising the measurement protocol across both groups ensures that any difference observed "
            "between groups can be attributed to the intervention rather than to differences in how data were "
            "collected."
        ),
        "analysis": (
            "Pre-test and post-test scores were compared between the treatment and control groups using "
            "independent-samples t-tests or analysis of variance (ANOVA), with analysis of covariance (ANCOVA) "
            "used where pre-test scores needed to be statistically controlled for.\n\n"
            "Effect sizes were reported alongside significance tests to indicate the practical magnitude of any "
            "treatment effect identified, in addition to its statistical significance."
        ),
    },
    "quasi_experimental": {
        "design": (
            "A quasi-experimental research design was adopted, comparing an intervention group with a "
            "comparison group in relation to {topic}, without random assignment of participants to groups. This "
            "design was selected because random assignment was not practicable in the study setting, while the "
            "research objectives still require comparison of outcomes under different conditions.\n\n"
            "Pre-existing, naturally occurring groups were used, with statistical controls applied during "
            "analysis to account for the absence of random assignment."
        ),
        "sampling": (
            "Intact, naturally occurring groups were used for the intervention and comparison conditions, "
            "selected because they were already exposed (or not exposed) to the relevant intervention. Where "
            "possible, the comparison group was selected to closely match the intervention group on key "
            "background characteristics.\n\n"
            "Because assignment to groups was not random, relevant background characteristics were measured for "
            "later statistical control."
        ),
        "collection": (
            "Data were collected using a pre-test/post-test protocol administered to both the intervention and "
            "comparison groups, alongside measures of relevant background characteristics used to support "
            "statistical control for pre-existing group differences.\n\n"
            "The same instrument and administration procedure were used for both groups to ensure that "
            "observed differences reflect the intervention rather than measurement inconsistency."
        ),
        "analysis": (
            "Outcomes were compared between the intervention and comparison groups using analysis of covariance "
            "(ANCOVA) or multiple regression, with relevant background characteristics included as covariates "
            "to account for the absence of random assignment.\n\n"
            "Results are interpreted with appropriate caution regarding causal inference, consistent with the "
            "quasi-experimental design's reliance on statistical rather than randomised control."
        ),
    },
    # correlational and cross_sectional only need a "design" override — their sampling, data
    # collection, and analysis already match the generic quantitative-family text verbatim.
    "correlational": {
        "design": (
            "A correlational research design was adopted, examining the strength and direction of "
            "relationships between variables related to {topic} as they naturally occur, without manipulating "
            "any variable or assigning participants to conditions. This design was selected because the research "
            "objectives concern the relationships between variables rather than a causal effect of one on "
            "another.\n\n"
            "Because no variable is manipulated, the findings of this study describe association rather than "
            "causation, and are interpreted accordingly throughout this dissertation."
        ),
    },
    "cross_sectional": {
        "design": (
            "A cross-sectional survey research design was adopted, collecting data from the sample on {topic} "
            "at a single point in time. This design was selected because the research objectives require a "
            "snapshot of the relationships between variables as they currently stand, rather than tracking "
            "change over time.\n\n"
            "While efficient to administer, this design limits the ability to draw conclusions about how the "
            "variables of interest change over time, a limitation noted further in Chapter 5."
        ),
    },
}


def _specific_methodology_from_source(source: str) -> tuple[str | None, str | None]:
    """Detect a specific named methodology in already-lowercased combined source text.
    Returns (specific_label, implied_family) or (None, None) if nothing specific is named.
    """
    for label, hints, family in _SPECIFIC_METHODOLOGY_HINTS:
        if any(h in source for h in hints):
            return label, family
    return None, None


def _specific_methodology(message: str, topic: str, document: Document) -> str | None:
    source = " ".join([message or "", topic or "", _flatten_doc(document)]).lower()
    label, _ = _specific_methodology_from_source(source)
    return label


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
    _, implied_family = _specific_methodology_from_source(source)
    if implied_family:
        return implied_family
    if any(k in source for k in ["systematic review", "scoping review", "conceptual paper", "theoretical"]):
        return "non_empirical"
    return "quantitative"


# Heuristic keyword sets for _uses_human_respondents(). Not exhaustive — a topic can mix
# both signals (e.g. "user acceptance of a new robotic system") — so respondent hints are
# checked first and win on conflict, since a human-facing study still needs a respondent
# profile even when it also discusses a technical artifact.
_RESPONDENT_HINTS = (
    "survey", "questionnaire", "respondent", "interview", "participant", "perception",
    "awareness", "opinion", "attitude towards", "attitude of", "employee", "customer",
    "consumer", "student", "teacher", "patient", "citizen", "manager", "user experience",
    "stakeholder", "satisfaction", "smes", "small and medium enterpr",
)
_NO_RESPONDENT_HINTS = (
    "robot", "robotic", "drone", "uav", "embedded system", "firmware", "microcontroller",
    "arduino", "raspberry pi", "sensor", "actuator", "circuit", "pcb", "iot device",
    "algorithm", "neural network", "machine learning model", "deep learning model",
    "control system", "automation system", "autonomous", "simulation", "prototype",
    "structural analysis", "mechanical design", "hydraulic", "pneumatic",
    "software system", "network protocol", "encryption", "compiler", "operating system",
    "database performance", "api throughput", "latency", "signal processing",
)


def _uses_human_respondents(topic: str, message: str, objectives: list[str] | None = None) -> bool:
    """Decide whether Chapter 4 should include a survey-style Respondent Profile section
    (response rate, demographics). Defaults to True (the app's original assumption, which
    fits most business/social-science dissertations) unless the topic clearly reads as a
    technical/engineering build-and-test study with no human survey component.
    """
    text = " ".join([topic or "", message or "", " ".join(objectives or [])]).lower()
    if any(hint in text for hint in _RESPONDENT_HINTS):
        return True
    if any(hint in text for hint in _NO_RESPONDENT_HINTS):
        return False
    return True


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
        r"^(the\s+study\s+aims?\s+to\s+|this\s+study\s+(aims?\s+to|seeks?\s+to|will)\s+|"
        r"to\s+examine\s+|to\s+determine\s+|to\s+assess\s+|to\s+evaluate\s+|"
        r"to\s+investigate\s+|to\s+analyse\s+|to\s+analyze\s+|to\s+propose\s+|"
        r"to\s+recommend\s+|to\s+design\s+|to\s+develop\s+|to\s+explore\s+|"
        r"to\s+identify\s+|to\s+establish\s+|to\s+)",
        "",
        objective.strip(),
        flags=re.IGNORECASE,
    )
    text = text[0].upper() + text[1:] if text else objective
    if len(text) > 95:
        cut = text[:95].rsplit(" ", 1)[0].strip()
        # Strip trailing connector/stop words one at a time so the cut doesn't end mid-phrase.
        stopwords = {"of", "in", "on", "for", "to", "and", "the", "with", "by", "at", "an", "a"}
        words = cut.split(" ")
        while words and words[-1].lower().strip(",") in stopwords:
            words.pop()
        cut = " ".join(words).strip()
        return (cut or text[:95].strip()) + "…"
    return text


_ORG_NAME_RE = re.compile(
    r"\b[A-Z][\w&'.-]*(?:\s+[A-Z][\w&'.-]*){0,4}\s+"
    r"(?:Ltd|Limited|PLC|Plc|Inc|Corp|Corporation|Company|Co\.|Bank|Hospital|University|College|"
    r"Ministry|Authority|Agency|Group|Holdings)\b"
)


def _resolve_case_entity(
    document: Document, topic: str, research_design: str, objectives: list[str],
) -> dict[str, Any] | None:
    """Decide, once per document, whether this study should be framed as a case study of a
    single (fictional, since none was named) organization, and if so invent one consistent
    name/context for it. Cached on document.content so every chapter — even ones written in a
    separate request — converges on the SAME organization, rather than each chapter prompting
    the LLM independently and risking a different invented name each time.

    Skips invention entirely when the topic already names a specific organization (its own name
    is already carried into every prompt via the Research Topic line, so there's nothing to add).
    """
    cached = (document.content or {}).get("case_entity")
    if isinstance(cached, dict):
        return cached if cached.get("enabled") else None

    result: dict[str, Any] = {"enabled": False}
    if research_design != "non_empirical" and not _ORG_NAME_RE.search(topic or ""):
        obj_text = "\n".join(f"- {o}" for o in (objectives or [])[:5])
        prompt = (
            "Decide whether this dissertation should be framed as an in-depth case study of ONE "
            "specific organization. Return JSON only.\n"
            f"Topic: {topic}\n"
            f"Research design: {research_design}\n"
            f"Objectives:\n{obj_text or '(none stated)'}\n\n"
            "A single-organization case-study framing fits topics like 'effect of X on performance "
            "at a firm', a named-sector study (a bank, hospital, manufacturer, NGO, ministry, school), "
            "or any topic about practices/outcomes WITHIN an organization. It does NOT fit broad "
            "policy/macro topics, multi-organization/industry-wide surveys, literature syntheses, or "
            "engineering/lab builds with no organizational subject.\n"
            "If it fits, invent ONE plausible, clearly fictional organization (a name that does not "
            "match any real company) consistent with the topic's sector and give a 1-2 sentence "
            "context (industry, country/region, approximate size).\n"
            "Return ONLY this JSON shape: "
            '{"use_case_entity": true|false, "name": "...", "context": "..."}'
        )
        try:
            data = _extract_json_obj(generate_text(prompt))
            if data.get("use_case_entity") and str(data.get("name") or "").strip():
                result = {
                    "enabled": True,
                    "name": str(data["name"]).strip(),
                    "context": str(data.get("context") or "").strip(),
                }
        except Exception as exc:
            logger.warning("_resolve_case_entity failed, proceeding without a case entity: %s", exc)

    try:
        document.content["case_entity"] = result
        document.save(update_fields=["content"])
    except Exception as exc:
        logger.warning("_resolve_case_entity: failed to cache decision on document: %s", exc)

    return result if result.get("enabled") else None


def _extract_document_brief(
    document: Document, topic: str, research_design: str, specific_design: str | None = None,
) -> str:
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
    specific_label = _SPECIFIC_METHODOLOGY_LABELS.get(specific_design or "")
    if specific_label:
        design_label = f"{design_label} — specifically a {specific_label} design. Write methodology " \
            f"content (research design, sampling, data collection, analysis) framed explicitly around " \
            f"a {specific_label} approach, not generic survey/interview boilerplate."

    case_entity = _resolve_case_entity(document, topic, research_design, objectives)
    case_entity_line = ""
    if case_entity:
        entity_context = case_entity["context"].rstrip(".")
        case_entity_line = (
            f"Case-Study Organization: {case_entity['name']} — {entity_context}. This is a fictional "
            f"organization invented for this study. Refer to it by this SAME name consistently in every "
            f"chapter — never invent a different organization name elsewhere in the document. Frame the "
            f"population/respondents/interviewees, the background/problem context, and the findings as "
            f"specific to {case_entity['name']}, not a generic or industry-wide claim.\n"
        )

    return (
        "══ THIS STUDY'S BRIEF — ground ALL writing in these specifics ══\n"
        f"Document Title  : {doc_title}\n"
        f"Research Topic  : {topic}\n"
        f"Research Design : {design_label}\n"
        f"Research Objectives:\n{obj_text}\n"
        f"Research Questions:\n{q_text}\n"
        f"{case_entity_line}"
        "══ END OF STUDY BRIEF ══"
    )


# Keyword set for _objectives_are_relational(). A correlation/regression apparatus only
# makes sense when the study actually tests a relationship/effect between two or more
# measurable constructs — a purely descriptive objective (the level/extent/status of one
# thing) has nothing to correlate against and should not be forced into one.
_RELATIONAL_OBJECTIVE_RE = re.compile(
    r"\b(effect[s]? of|impact of|influence of|relationship between|relationship of|"
    r"association between|correlat\w*|link between|predict\w*|determinant[s]? of|"
    r"contribut\w* (?:of|to)|moderat\w*|mediat\w*|drivers? of|effect on|impact on|"
    r"influence on)\b",
    re.IGNORECASE,
)


def _objectives_are_relational(objectives: list[str]) -> bool:
    """True if at least one objective genuinely tests a relationship/effect between
    two or more constructs, rather than only describing the level/extent/status of one."""
    return any(_RELATIONAL_OBJECTIVE_RE.search(o or "") for o in (objectives or []))


def _chapter4_subsections(
    research_design: str,
    objectives: list[str],
    topic: str = "",
    message: str = "",
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = [
        {"title": "4.1 Introduction", "children": []},
    ]

    has_respondents = research_design in {"quantitative", "qualitative", "mixed"} and _uses_human_respondents(
        topic, message, objectives
    )
    if has_respondents:
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

    # Quantitative/mixed studies report Mean/SD/Min/Max for their measured constructs
    # right after the respondent profile and before any inferential analysis — the same
    # placement used in the reference dissertation standard this module targets.
    if research_design in {"quantitative", "mixed"} and objectives:
        nodes.append({
            "title": f"4.{obj_start} Descriptive Statistics of Study Variables",
            "kind": "table",
            "children": [],
            "meta": {"table_type": "descriptive_stats"},
        })
        obj_start += 1

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
    # Quantitative/mixed studies whose objectives genuinely test a relationship/effect
    # between measurable constructs get a real correlation + regression apparatus, not
    # just a vague promise of one — earlier versions of the Discussion/Chapter-Summary
    # fallback text claimed findings were "detailed in the correlation and regression
    # tables presented in this chapter" when no such tables actually existed anywhere in
    # the document. A purely descriptive study (the level/extent/status of one thing)
    # has nothing to correlate, so it should not be forced into this apparatus.
    if (
        research_design in {"quantitative", "mixed"}
        and len(objectives) >= 2
        and _objectives_are_relational(objectives)
    ):
        nodes.append({
            "title": f"4.{last_idx} Correlation Analysis",
            "kind": "table",
            "children": [],
            "meta": {"table_type": "correlation"},
        })
        nodes.append({
            "title": f"4.{last_idx + 1} Regression Analysis",
            "kind": "table",
            "children": [],
            "meta": {"table_type": "regression"},
        })
        last_idx += 2
    nodes.append({"title": f"4.{last_idx} Discussion of Findings", "children": []})
    nodes.append({"title": f"4.{last_idx + 1} Chapter Summary", "children": []})
    return [_normalize_subsection_node(n) for n in nodes]


def _plan_chapter4_structure(
    llm_sections: list[dict[str, Any]],
    research_design: str,
    objectives: list[str],
    topic: str,
    message: str = "",
) -> list[dict[str, Any]]:
    """LLM-first planner for Chapter 4 structure and its visuals.

    Rather than discarding the dissertation planner's own topic-tailored
    Chapter 4 section proposal, this reasons about which of those sections
    need a table/chart and what kind, based on the actual topic and research
    design (e.g. an engineering/technical study gets trial-based results
    visuals, not survey demographics). Falls back to the heuristic
    _chapter4_subsections when the LLM is unavailable or returns something
    unusable, so behaviour never regresses.
    """
    draft_titles = [str(s.get("title") or "").strip() for s in (llm_sections or []) if s.get("title")]
    has_respondents_hint = _uses_human_respondents(topic, message, objectives)
    objective_lines = "\n".join(f"- {o}" for o in objectives[:6]) or "- (infer from the topic)"
    titles_block = "\n".join(f"- {t}" for t in draft_titles) or "- (no draft titles provided; propose your own)"

    prompt = (
        "You are planning the data and visuals for Chapter 4 (Results/Analysis) of an academic "
        "dissertation. Return JSON only.\n"
        f"Topic: {topic}\n"
        f"Research design: {research_design}\n"
        f"Research objectives:\n{objective_lines}\n\n"
        "Draft section titles already proposed for this chapter:\n"
        f"{titles_block}\n\n"
        "Decide, section by section, what data visuals each section actually needs given the TOPIC "
        "and the kind of data this study would realistically produce. Do not assume a respondent "
        "survey unless the topic genuinely involves human participants — a technical/engineering study "
        "that builds or tests a system or device should report experimental/trial results, not survey "
        "demographics.\n\n"
        "Return ONLY valid JSON using this schema:\n"
        "{\n"
        '  "needs_respondent_profile": true|false,\n'
        '  "sections": [\n'
        '    {"title": "4.X ...", "objective": "<matching objective or empty>", '
        '"has_table": true|false, '
        '"table_type": "response_rate|demographics|descriptive_stats|generic|correlation|regression|none", '
        '"has_chart": true|false}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Cover the full chapter: an introduction, a respondent-profile section ONLY if "
        "needs_respondent_profile is true, one section per research objective, a discussion of "
        "findings, and a chapter summary.\n"
        "- Only set needs_respondent_profile=true if data is collected directly from human "
        "respondents (surveys/interviews/questionnaires).\n"
        "- Give each objective-results section at least a table; quantitative/mixed designs should "
        "usually pair it with a chart too. Qualitative designs should not use charts.\n"
        "- If the research design is quantitative or mixed, add one section right after the "
        "respondent profile (or right after the introduction if there is no respondent profile) "
        "titled like '4.X Descriptive Statistics of Study Variables' (table_type='descriptive_stats', "
        "has_table=true, has_chart=false), reporting Mean/SD/Min/Max for the constructs this study "
        "measures, before any of the per-objective results sections.\n"
        "- If the research design is quantitative or mixed AND at least one research objective "
        "genuinely tests a relationship, effect, or influence between two or more measurable "
        "constructs (e.g. 'the effect of X on Y', 'the relationship between X and Y') — NOT a purely "
        "descriptive objective like 'the level/extent/status/prevalence of X' — add exactly two more "
        "sections after the objective sections and before the discussion of findings: one titled like "
        "'4.X Correlation Analysis' (table_type='correlation', has_table=true, has_chart=false) and "
        "one titled like '4.X+1 Regression Analysis' (table_type='regression', has_table=true, "
        "has_chart=false). If every objective is purely descriptive, skip both sections entirely — do "
        "not invent a relationship to test. Never claim a correlation or regression table exists in "
        "the discussion/summary text unless you have actually added one of these two sections.\n"
        "- Use concise academic numbering (4.1, 4.2, ...).\n"
        "- Return ONLY the JSON object, no markdown fences."
    )

    try:
        data = _extract_json_obj(generate_text(prompt))
        sections = data.get("sections")
        if not isinstance(sections, list) or not sections:
            raise ValueError("no sections returned for chapter 4 plan")

        needs_respondents = bool(data.get("needs_respondent_profile", has_respondents_hint))
        allow_relational = _objectives_are_relational(objectives)
        nodes: list[dict[str, Any]] = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sec_title = str(sec.get("title") or "").strip()
            if not sec_title:
                continue
            if "respondent" in sec_title.lower() and not needs_respondents:
                continue

            sec_objective = str(sec.get("objective") or "").strip()
            table_type = str(sec.get("table_type") or "").strip().lower()
            if table_type in {"correlation", "regression"} and not allow_relational:
                # The objectives are purely descriptive — no relationship/effect to test —
                # so skip a correlation/regression section even if the LLM proposed one.
                continue
            children: list[dict[str, Any]] = []

            if sec.get("has_table"):
                children.append({
                    "title": f"{sec_title} — Table",
                    "kind": "table",
                    "children": [],
                    "meta": {
                        "table_type": table_type if table_type in {
                            "response_rate", "demographics", "descriptive_stats", "correlation", "regression",
                        } else None,
                        "objective": sec_objective,
                    },
                })
            if sec.get("has_chart") and research_design in {"quantitative", "mixed"}:
                children.append({
                    "title": f"{sec_title} — Chart",
                    "kind": "chart",
                    "children": [],
                    "meta": {
                        "chart_type": "demographics" if table_type == "demographics" else None,
                        "objective": sec_objective,
                    },
                })

            node: dict[str, Any] = {"title": sec_title, "children": children}
            if sec_objective:
                node["kind"] = "objective_findings"
                node["meta"] = {"objective": sec_objective}
            nodes.append(node)

        if nodes:
            return [_normalize_subsection_node(n) for n in nodes]
    except Exception as exc:
        logger.warning("plan_chapter4_structure LLM planning failed, using heuristic fallback: %s", exc)

    return _chapter4_subsections(research_design, objectives, topic, message)


def _truncate_label(text: str, max_len: int) -> str:
    """Trim a label to a word boundary, appending an ellipsis only when actually cut."""
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0].rstrip(",;:")
    return (cut or text[:max_len]) + "…"


def _parse_numeric_cell(cell: str) -> float | None:
    """Parse a table cell like '42', '41.7%', or '1,234' into a float, or None if not numeric."""
    try:
        return float(str(cell).replace("%", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _topic_terms(topic: str, n: int = 4) -> list[str]:
    """Split a topic phrase on connector words so each chunk is a plausible standalone
    term — works for any subject area since it derives terms from whatever topic the
    student supplied, not a fixed domain list.
    """
    raw = re.split(r"\s+(?:and|of|in|for|using|with|on|to|the|an?)\s+", topic or "", flags=re.IGNORECASE)
    terms = [t.strip() for t in raw if len(t.strip()) > 2]
    return (terms or [(topic or "the study variable").strip()])[:n]


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
        topic_label = _truncate_label(topic, 40)
        if _uses_human_respondents(topic, ""):
            return (
                "| Theme | Supporting Mentions | Representative Insight | Interpretation |\n"
                "|---|---|---|\n"
                f"| Theme 1 | {t1} | Participants highlighted concerns around {topic_label} | Indicates persistent implementation barriers |\n"
                f"| Theme 2 | {t2} | Respondents reported uneven institutional readiness | Suggests need for governance alignment |\n"
                f"| Theme 3 | {t3} | Stakeholders requested stronger policy direction | Supports a coordinated reform approach |"
            )
        return (
            "| Theme | Supporting Observations | Representative Insight | Interpretation |\n"
            "|---|---|---|\n"
            f"| Theme 1 | {t1} | Test logs highlighted recurring issues around {topic_label} | Indicates persistent design or calibration constraints |\n"
            f"| Theme 2 | {t2} | Trials showed uneven performance across configurations | Suggests need for tighter parameter tuning |\n"
            f"| Theme 3 | {t3} | Repeated runs pointed to a clear, addressable failure mode | Supports a targeted design refinement |"
        )
    objective_text = _truncate_label(objective or "the objective", 30)
    a = round(2.6 + (seed % 16) * 0.11, 2)
    b = round(3.1 + ((seed // 7) % 14) * 0.12, 2)
    c = round(2.4 + ((seed // 13) % 15) * 0.1, 2)
    return (
        "| Metric | Observation | Interpretation |\n"
        "|---|---:|---|\n"
        f"| Indicator A ({objective_text}) | {a} | Moderate performance with room for improvement |\n"
        f"| Indicator B | {b} | Stronger outcome where controls were applied |\n"
        f"| Indicator C | {c} | Weakest dimension and major constraint area |"
    )


def _infer_sample_size(document: Document) -> int:
    """Infer respondent/sample size from the current document, defaulting safely.

    Searches the FULL, untruncated content of every section written so far — not
    _full_context_for_generation's heavily compressed prompt context (which keeps only
    the first ~300 chars of older chapters) — so an explicit sample-size statement made
    anywhere in Chapter 3's methodology text is reliably found once Chapter 4 needs it,
    rather than silently falling through to the hardcoded default below.
    """
    sections = (document.content or {}).get("sections", [])
    context = "\n".join(str(sec.get("content") or "") for sec in sections) or _full_context_for_generation(document)
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


def _llm_table_data_is_consistent(
    headers: list[str], rows: list[list[str]], table_type: str | None, node_title: str, sample_size: int,
) -> bool:
    """Reject LLM-generated table data that numerically contradicts the document's own
    stated sample size. The generation prompt only asks the model to make frequencies
    "sum to the sample size" as a soft instruction with no enforcement, so a response-rate
    or demographics table can come back internally plausible-looking but arithmetically
    wrong (e.g. frequencies that sum to some other number entirely) — unlike the
    deterministic fallback below, which guarantees this by construction. When this check
    fails, the caller falls through to that guaranteed-consistent fallback instead.
    """
    title_lower = node_title.lower()
    needs_sum_check = table_type in {"response_rate", "demographics"} or (
        table_type is None and ("response rate" in title_lower or "demographic" in title_lower)
    )
    if not needs_sum_check or sample_size <= 0:
        return True
    tolerance = max(2, sample_size * 0.05)
    # Some tables include an explicit "Total"/"Administered" row equal to the sample size,
    # with the remaining rows being its breakdown (which themselves sum to that same total).
    # Others are a pure breakdown with no total row at all. Either layout is consistent, so
    # accept if any single cell already matches the sample size, or if any column's full sum does.
    all_values = [v for r in rows for v in (_parse_numeric_cell(c) for c in r) if v is not None]
    if any(abs(v - sample_size) <= tolerance for v in all_values):
        return True
    for col_idx in range(max((len(r) for r in rows), default=0)):
        values = [_parse_numeric_cell(r[col_idx]) for r in rows if len(r) > col_idx]
        if len(values) < 2 or any(v is None for v in values):
            continue
        total = sum(v for v in values if v is not None)
        if abs(total - sample_size) <= tolerance:
            return True
    return False


def _two_tailed_p_from_t(t: float) -> float:
    """Two-tailed p-value for a t/z statistic via the normal approximation. Accurate
    enough for the residual df typical of a dissertation-scale sample (>= ~30), and
    avoids needing a full t-distribution implementation for this seeded-but-plausible
    synthetic apparatus.
    """
    z = abs(t)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return max(0.0004, min(0.999, 2 * (1 - cdf)))


def _correlation_p(r: float, n: int) -> float:
    """Exact significance test for a Pearson r: convert to its t-transform
    (t = r·√(n-2) / √(1-r²), df = n-2), then the normal approximation above —
    a real textbook formula, not a magnitude heuristic.
    """
    n = max(int(n), 4)
    r = max(-0.999, min(0.999, r))
    t = r * math.sqrt(n - 2) / math.sqrt(max(1e-9, 1 - r ** 2))
    return _two_tailed_p_from_t(t)


def _format_p_value(p: float) -> str:
    """APA-style p-value: 3 decimals, no leading zero (e.g. '.034'), '.000' once it
    rounds that small — matching how dissertations actually report exact p-values
    in regression tables (as opposed to the p-bucket used in narrative prose).
    """
    return f"{p:.3f}".lstrip("0") or ".000"


def _format_r_value(r: float) -> str:
    """APA style for a correlation/beta coefficient: no leading zero below 1 (e.g.
    '.64'). Unlike B/SE B/t, which conventionally keep their leading zero and are
    formatted with plain f-strings at the call site.
    """
    if abs(r) >= 1:
        return f"{r:.2f}"
    sign = "-" if r < 0 else ""
    return sign + f"{abs(r):.2f}".lstrip("0")


def _sig_stars(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _significance_label(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    if p < 0.01:
        return "p < .01"
    if p < 0.05:
        return "p < .05"
    return "n.s."


def _significance_label_from_f(f_stat: float) -> str:
    """Rough p-value bucket for an overall regression F-statistic. The F-distribution's
    own CDF isn't worth implementing for fabricated data, so this stays a magnitude
    heuristic (unlike the r/t significance above, which now use real formulas).
    """
    if f_stat >= 8:
        return "p < .001"
    if f_stat >= 4:
        return "p < .01"
    if f_stat >= 2.5:
        return "p < .05"
    return "n.s."


_ABBREV_STOPWORDS = {"and", "of", "the", "in", "on", "for", "to", "with", "a", "an", "&"}


def _abbreviate_label(label: str, used: set[str]) -> str:
    """Short variable code for a correlation-matrix column header (e.g. 'Planning and
    Scheduling' -> 'PS'), disambiguated against codes already used in the same matrix.
    """
    words = [w for w in re.split(r"[\s/-]+", label.strip()) if w]
    letters = [w[0].upper() for w in words if w.lower().strip(",.") not in _ABBREV_STOPWORDS]
    if not letters:
        letters = [w[0].upper() for w in words] or ["X"]
    base = "".join(letters[:3]) or "X"
    abbrev, i = base, 2
    while abbrev in used:
        abbrev = f"{base}{i}"
        i += 1
    used.add(abbrev)
    return abbrev


def _regression_model_stats(
    topic: str,
    objectives: list[str] | None,
    research_design: str,
    sample_size: int,
) -> dict[str, Any]:
    """Deterministically derive an internally-consistent correlation/regression apparatus
    for Chapter 4 from nothing but the study's own topic/objectives/sample size: a full
    predictor-by-predictor correlation matrix (not just predictor-vs-outcome), and a full
    regression table (B, SE B, beta, t, p, plus the model fit) — mirroring what a real
    submission-ready dissertation reports rather than a single illustrative column.

    Per-predictor Pearson r values are seeded individually, but every other number is
    derived FROM those seeds using real textbook formulas (t-test for r, normal
    approximation for p, R² -> Adjusted R² -> F), so a reader who cross-checks one number
    against another finds them mutually consistent. This is a pure function of its
    inputs, so calling it again later — e.g. when Chapter 5 restates the headline finding —
    reproduces the exact same numbers without needing to thread state between chapters.
    """
    predictor_labels = [_objective_section_title(o) for o in (objectives or []) if o and o.strip()][:6]
    while len(predictor_labels) < 2:
        predictor_labels.append(f"Predictor {len(predictor_labels) + 1}")
    k = len(predictor_labels)
    n = max(int(sample_size or 0), k + 10)

    seed_base = sum(ord(c) for c in f"{topic}|{research_design}|{n}|{k}")
    used_abbrevs: set[str] = set()
    predictors: list[dict[str, Any]] = []
    r_values: list[float] = []
    for i, label in enumerate(predictor_labels):
        seed_i = seed_base + sum(ord(c) for c in f"{label}|{i}")
        r = round(0.30 + (seed_i % 46) * 0.01, 2)
        beta = round(max(0.05, r * 0.62 + ((seed_i // 7) % 11) * 0.01), 2)
        b = round(beta * (0.85 + (seed_i % 21) * 0.01), 2)
        t = max(0.3, round(1.5 + beta * 4.6 + (((seed_i // 11) % 9) - 4) * 0.18, 2))
        se_b = round(b / t, 3) if t else round(b / 2, 3)
        p = _two_tailed_p_from_t(t)
        r_p = _correlation_p(r, n)
        r_values.append(r)
        predictors.append({
            "label": label,
            "abbrev": _abbreviate_label(label, used_abbrevs),
            "r": r,
            "r_p": r_p,
            "r_sig": _significance_label(r_p),
            "r_stars": _sig_stars(r_p),
            "beta": beta,
            "b": b,
            "se_b": se_b,
            "t": t,
            "p": p,
            "p_display": _format_p_value(p),
            "beta_sig": _significance_label(p),
        })

    # Inter-predictor correlations — the rest of the full matrix beyond the
    # predictor-vs-outcome column above. Seeded per pair (i, j) so the matrix is
    # internally reproducible but distinct from the outcome correlations.
    matrix: dict[str, dict[str, Any]] = {}
    for i in range(k):
        for j in range(i):
            seed_ij = seed_base + sum(ord(c) for c in f"{i}-{j}-pair")
            r_ij = round(0.25 + (seed_ij % 46) * 0.01, 2)
            p_ij = _correlation_p(r_ij, n)
            matrix[f"{i}:{j}"] = {"r": r_ij, "p": p_ij, "stars": _sig_stars(p_ij)}

    mean_r2 = sum(r ** 2 for r in r_values) / len(r_values)
    r_squared = round(min(0.78, max(0.30, mean_r2 * 1.3)), 2)
    df1 = k
    df2 = max(n - k - 1, 5)
    adj_r_squared = round(1 - (1 - r_squared) * (n - 1) / df2, 2)
    f_stat = round((r_squared / (1 - r_squared)) * (df2 / df1), 2)

    const_b = round(0.30 + (seed_base % 40) * 0.01, 2)
    const_t = round(1.8 + (seed_base % 13) * 0.1, 2)
    const_se = round(const_b / const_t, 3) if const_t else round(const_b / 2, 3)
    const_p = _two_tailed_p_from_t(const_t)

    return {
        "predictors": predictors,
        "matrix": matrix,
        "outcome_label": "Overall Outcome",
        "constant": {
            "b": const_b, "se_b": const_se, "t": const_t,
            "p": const_p, "p_display": _format_p_value(const_p),
        },
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_stat": f_stat,
        "df1": df1,
        "df2": df2,
        "f_sig": _significance_label_from_f(f_stat),
        "n": n,
        "k": k,
    }


def _descriptive_stats_dataset(topic: str, objectives: list[str] | None, seed: int) -> dict[str, Any]:
    """Mean/SD/Min/Max table per measured dimension (one row per research objective, plus a
    composite row), the table type a quantitative/mixed dissertation reports right after
    its respondent profile and before any inferential analysis — present in the reference
    standard this module is matched against, but previously absent from this generator.
    """
    dims = [_objective_section_title(o) for o in (objectives or []) if o and o.strip()][:6]
    if not dims:
        dims = ["Primary Indicator", "Secondary Indicator"]
    rows: list[list[str]] = []
    means: list[float] = []
    for i, label in enumerate(dims):
        seed_i = seed + sum(ord(c) for c in f"{label}|{i}|desc")
        mean = round(2.2 + (seed_i % 18) * 0.05, 2)
        sd = round(0.55 + ((seed_i // 3) % 20) * 0.01, 2)
        lo = round(max(1.0, mean - sd * 1.7), 2)
        hi = round(min(5.0, mean + sd * 1.7), 2)
        means.append(mean)
        rows.append([label, f"{mean:.2f}", f"{sd:.2f}", f"{lo:.2f}", f"{hi:.2f}"])

    composite_mean = round(sum(means) / len(means), 2)
    composite_sd = round(sum(float(r[2]) for r in rows) / len(rows) * 0.85, 2)
    composite_lo = min(float(r[3]) for r in rows)
    composite_hi = max(float(r[4]) for r in rows)
    rows.append([
        "Composite Score", f"{composite_mean:.2f}", f"{composite_sd:.2f}",
        f"{composite_lo:.2f}", f"{composite_hi:.2f}",
    ])
    return {"headers": ["Dimension", "Mean", "Std. Dev.", "Min", "Max"], "rows": rows}


def _ai_table_dataset(
    node_title: str,
    research_design: str,
    topic: str,
    objective: str | None,
    sample_size: int,
    current_document_context: str,
    table_type: str | None = None,
    objectives: list[str] | None = None,
) -> dict[str, Any]:
    """Generate structured {headers, rows} table data, to be rendered as a real editable
    table (not an image). table_type (from the calling node's meta, e.g. "response_rate",
    "demographics", "key_terms") selects the deterministic fallback shape when the LLM is
    unavailable; falls back further to a node_title substring match when not provided.
    objectives (the full list, when available) feeds the "correlation"/"regression" shapes,
    which need every predictor dimension rather than the single `objective` other table
    types key off of.
    """
    seed = sum(ord(c) for c in f"{node_title}|{topic}|{objective or ''}|{research_design}")
    title_lower = node_title.lower()
    is_correlation = table_type == "correlation" or (table_type is None and "correlation" in title_lower)
    is_regression = table_type == "regression" or (
        table_type is None and "regression" in title_lower and "logistic" not in title_lower
    )
    is_descriptive_stats = table_type == "descriptive_stats" or (
        table_type is None and ("descriptive statistic" in title_lower or "descriptive analysis" in title_lower)
    )

    # These three table types carry numbers that must be mutually consistent (a
    # correlation matrix's significance stars must match its r values; a regression
    # table's R²/Adjusted R²/F must actually derive from its own betas) — exactly what
    # the deterministic apparatus below guarantees by construction via real textbook
    # formulas, and what an LLM call has no way to guarantee. So these always skip
    # straight to that apparatus rather than risking a numerically-incoherent table.
    if is_correlation or is_regression or is_descriptive_stats:
        return _deterministic_table_dataset(
            node_title, research_design, topic, objective, sample_size, table_type, objectives, seed, title_lower,
        )

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
        if headers and rows and _llm_table_data_is_consistent(headers, rows, table_type, node_title, sample_size):
            return {"headers": headers, "rows": rows}
    except Exception as exc:
        logger.warning("_ai_table_dataset fallback (%s): %s", node_title[:60], exc)

    return _deterministic_table_dataset(
        node_title, research_design, topic, objective, sample_size, table_type, objectives, seed, title_lower,
    )


def _deterministic_table_dataset(
    node_title: str,
    research_design: str,
    topic: str,
    objective: str | None,
    sample_size: int,
    table_type: str | None,
    objectives: list[str] | None,
    seed: int,
    title_lower: str,
) -> dict[str, Any]:
    """The guaranteed-consistent fallback shapes for _ai_table_dataset, factored out so the
    correlation/regression/descriptive-stats apparatus can be reached directly (bypassing
    the LLM call) as well as via the normal LLM-then-fallback path used by every other
    table type.
    """
    is_response_rate = table_type == "response_rate" or (table_type is None and "response rate" in title_lower)
    is_demographics = table_type == "demographics" or (table_type is None and "demographic" in title_lower)
    is_key_terms = table_type == "key_terms" or (table_type is None and "key terms" in title_lower)
    is_empirical_summary = table_type == "empirical_summary" or (
        table_type is None and "empirical studies" in title_lower
    )
    is_sampling = table_type == "sampling" or (table_type is None and "sampling frame" in title_lower)
    is_instruments = table_type == "instruments" or (
        table_type is None and "data collection instruments" in title_lower
    )
    is_reliability = table_type == "reliability" or (table_type is None and "reliability and validity" in title_lower)
    is_findings_summary = table_type == "findings_summary" or (
        table_type is None and "key findings by objective" in title_lower
    )
    is_correlation = table_type == "correlation" or (table_type is None and "correlation" in title_lower)
    is_regression = table_type == "regression" or (
        table_type is None and "regression" in title_lower and "logistic" not in title_lower
    )
    is_descriptive_stats = table_type == "descriptive_stats" or (
        table_type is None and ("descriptive statistic" in title_lower or "descriptive analysis" in title_lower)
    )

    if is_correlation:
        stats = _regression_model_stats(
            topic, objectives or ([objective] if objective else None), research_design, sample_size,
        )
        predictors = stats["predictors"]
        k = len(predictors)
        outcome_abbrev = "OO"
        headers = ["Variable"] + [f"{i + 1}. {p['abbrev']}" for i, p in enumerate(predictors)] + [
            f"{k + 1}. {outcome_abbrev}"
        ]
        rows = []
        for i, p in enumerate(predictors):
            row = [f"{i + 1}. {p['label']} ({p['abbrev']})"]
            for j in range(k):
                if j < i:
                    cell = stats["matrix"][f"{i}:{j}"]
                    row.append(f"{_format_r_value(cell['r'])}{cell['stars']}")
                elif j == i:
                    row.append("1.00")
                else:
                    row.append("")
            row.append(f"{_format_r_value(p['r'])}{p['r_stars']}")
            rows.append(row)
        outcome_row = [f"{k + 1}. {stats['outcome_label']} ({outcome_abbrev})"]
        for p in predictors:
            outcome_row.append(f"{_format_r_value(p['r'])}{p['r_stars']}")
        outcome_row.append("1.00")
        rows.append(outcome_row)
        return {"headers": headers, "rows": rows, "model_fit": stats}

    if is_regression:
        stats = _regression_model_stats(
            topic, objectives or ([objective] if objective else None), research_design, sample_size,
        )
        const = stats["constant"]
        headers = ["Predictor Variable", "B", "SE B", "β", "t", "p"]
        rows = [["Constant", f"{const['b']:.2f}", f"{const['se_b']:.2f}", "—", f"{const['t']:.2f}", const["p_display"]]]
        for p in stats["predictors"]:
            rows.append([
                p["label"], f"{p['b']:.2f}", f"{p['se_b']:.2f}", _format_r_value(p["beta"]),
                f"{p['t']:.2f}", p["p_display"],
            ])
        return {"headers": headers, "rows": rows, "model_fit": stats}

    if is_descriptive_stats:
        return _descriptive_stats_dataset(
            topic, objectives or ([objective] if objective else None), seed,
        )

    if is_key_terms:
        terms = _topic_terms(topic, 4)
        return {
            "headers": ["Term", "Operational Definition"],
            "rows": [
                [
                    t[:1].upper() + t[1:],
                    f"As used in this study, refers to the construct or process of {t.lower()} within the "
                    f"context of {_truncate_label(topic or 'the research focus', 60)}, operationally defined for the "
                    "purposes of data collection and analysis.",
                ]
                for t in terms
            ],
        }

    if is_empirical_summary:
        terms = _topic_terms(topic, 3)
        return {
            "headers": ["Source", "Focus of Study", "Key Finding", "Relevance to Current Study"],
            "rows": [
                [
                    f"Illustrative Study {chr(65 + i)}",
                    f"Examined {t.lower()} in a related context",
                    "Reported a measurable effect on the outcome of interest",
                    "Supports the relevance and timeliness of the current study",
                ]
                for i, t in enumerate(terms)
            ],
        }

    if is_sampling:
        population = max(sample_size * (3 + seed % 3), sample_size + 10)
        return {
            "headers": ["Component", "Value", "Basis"],
            "rows": [
                ["Target Population", str(population), "Estimated population relevant to the study context"],
                ["Sample Size (n)", str(sample_size), "Determined using a standard sample-size formula/justified allocation"],
                ["Sampling Approach", "Stratified / Purposive (per design)", "Selected to ensure adequate coverage of the study units"],
            ],
        }

    if is_instruments:
        terms = _topic_terms(topic, 3)
        return {
            "headers": ["Instrument/Method", "Purpose", "Role in the Study"],
            "rows": [
                [f"Instrument {i + 1}", f"Captures data on {t.lower()}", "Primary data source for objective-level analysis"]
                for i, t in enumerate(terms)
            ],
        }

    if is_reliability:
        if _uses_human_respondents(topic, ""):
            a = round(0.70 + (seed % 21) * 0.01, 2)
            b = round(0.72 + ((seed // 3) % 19) * 0.01, 2)
            c = round(0.68 + ((seed // 7) % 22) * 0.01, 2)

            def _alpha_note(v: float) -> str:
                return "Acceptable internal consistency" if v >= 0.7 else "Marginal internal consistency"

            return {
                "headers": ["Construct", "Cronbach's Alpha", "Interpretation"],
                "rows": [
                    ["Construct 1", str(a), _alpha_note(a)],
                    ["Construct 2", str(b), _alpha_note(b)],
                    ["Construct 3", str(c), _alpha_note(c)],
                ],
            }
        p1 = round(95.0 + (seed % 5) * 0.8, 1)
        p2 = round(93.5 + ((seed // 3) % 6) * 0.7, 1)
        return {
            "headers": ["Measurement/Test", "Repeatability (%)", "Interpretation"],
            "rows": [
                ["Trial-to-trial consistency", str(p1), "High repeatability across repeated trials"],
                ["Measurement/calibration check", str(p2), "Consistent readings within acceptable tolerance"],
            ],
        }

    if is_findings_summary:
        return {
            "headers": ["Objective", "Key Finding", "Implication"],
            "rows": [
                [
                    f"Objective {i + 1}",
                    "Findings showed uneven performance across the indicators measured for this objective",
                    "Highlights priority areas for improvement and further inquiry",
                ]
                for i in range(3)
            ],
        }

    if is_response_rate:
        returned = max(1, int(round(sample_size * (0.80 + (seed % 9) / 100))))
        not_returned = max(0, sample_size - returned)
        return {
            "headers": ["Category", "Frequency", "Percentage"],
            "rows": [
                ["Questionnaires Administered", str(sample_size), "100.0%"],
                ["Questionnaires Returned and Usable", str(returned), f"{(returned / sample_size) * 100:.1f}%"],
                ["Not Returned / Discarded", str(not_returned), f"{(not_returned / sample_size) * 100:.1f}%"],
            ],
        }

    if is_demographics:
        male = max(1, int(round(sample_size * (0.42 + (seed % 12) / 100))))
        female = max(1, sample_size - male)
        age1 = max(1, int(round(sample_size * (0.28 + (seed % 10) / 100))))
        age2 = max(1, int(round(sample_size * (0.34 + ((seed // 3) % 10) / 100))))
        age3 = max(1, sample_size - age1 - age2)
        return {
            "headers": ["Variable", "Category", "Frequency", "Percentage"],
            "rows": [
                ["Gender", "Male", str(male), f"{(male / sample_size) * 100:.1f}%"],
                ["Gender", "Female", str(female), f"{(female / sample_size) * 100:.1f}%"],
                ["Age", "18-29 years", str(age1), f"{(age1 / sample_size) * 100:.1f}%"],
                ["Age", "30-39 years", str(age2), f"{(age2 / sample_size) * 100:.1f}%"],
                ["Age", "40 years and above", str(age3), f"{(age3 / sample_size) * 100:.1f}%"],
            ],
        }

    if research_design == "qualitative":
        t1 = 3 + (seed % 5)
        t2 = 2 + ((seed // 5) % 5)
        t3 = 2 + ((seed // 9) % 4)
        topic_label = _truncate_label(topic, 28)
        if _uses_human_respondents(topic, ""):
            return {
                "headers": ["Theme", "Mentions", "Representative Excerpt", "Interpretation"],
                "rows": [
                    ["Theme 1", str(t1), f"Participants emphasized {topic_label}.", "Shows core experiential pattern"],
                    ["Theme 2", str(t2), "Respondents highlighted implementation constraints.", "Indicates operational barriers"],
                    ["Theme 3", str(t3), "Stakeholders requested stronger governance.", "Supports policy-focused recommendations"],
                ],
            }
        return {
            "headers": ["Theme", "Observations", "Representative Note", "Interpretation"],
            "rows": [
                ["Theme 1", str(t1), f"Test logs emphasized {topic_label}.", "Shows core performance pattern"],
                ["Theme 2", str(t2), "Trials highlighted configuration-related constraints.", "Indicates design or calibration barriers"],
                ["Theme 3", str(t3), "Repeated runs pointed to a consistent failure mode.", "Supports a targeted design refinement"],
            ],
        }

    a = round(2.7 + (seed % 17) * 0.12, 2)
    b = round(2.9 + ((seed // 7) % 14) * 0.11, 2)
    c = round(2.5 + ((seed // 11) % 15) * 0.10, 2)
    return {
        "headers": ["Metric", "Value", "Interpretation"],
        "rows": [
            ["Indicator A", str(a), "Moderate performance"],
            ["Indicator B", str(b), "Relatively stronger outcome"],
            ["Indicator C", str(c), "Priority improvement area"],
        ],
    }


_LIKERT_LABELS = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]


def _fallback_chart_labels(label_style: str, effective_n: int) -> list[str]:
    """Domain-appropriate placeholder labels for when neither category_labels nor the LLM
    are available. "likert" fits survey/respondent-based studies (5-point agreement scale);
    "trial" fits experimental/engineering studies (repeated test runs); "generic" is the
    least-informative last resort.
    """
    if label_style == "likert":
        if effective_n <= len(_LIKERT_LABELS):
            return _LIKERT_LABELS[:effective_n]
        return [_LIKERT_LABELS[i % len(_LIKERT_LABELS)] for i in range(effective_n)]
    if label_style == "trial":
        return [f"Trial {i + 1}" for i in range(effective_n)]
    return [f"Category {i + 1}" for i in range(effective_n)]


def _ai_chart_series(
    context: str,
    n_points: int = 8,
    category_labels: list[str] | None = None,
    label_style: str = "generic",
) -> dict[str, Any]:
    """Ask the LLM to produce realistic numeric data for a chart.

    Returns a dict with keys:
      - series: list[float]   — the data points
      - chart_type: str       — suggested chart type (bar/line/scatter/area/pie)
      - x_labels: list[str]   — short label for each point (SAME length as series)
      - unit: str             — measurement unit, e.g. "%", "score", "count"
    Falls back to seed-based data on any error so the pipeline never breaks. When the
    caller already knows the real category names (e.g. Gender/Age groups for a
    demographic chart), pass category_labels so both the LLM prompt and the fallback use
    those exact labels instead of inventing generic ones. Otherwise label_style picks a
    domain-appropriate fallback label set ("likert" for respondent-based studies, "trial"
    for experimental/engineering studies).
    """
    # Pie charts look cluttered with many slices; cap at 6 for pie-likely topics.
    pie_keywords = {"distribution", "composition", "proportion", "breakdown", "demographic", "share", "pie"}
    likely_pie = any(kw in context.lower() for kw in pie_keywords)
    effective_n = len(category_labels) if category_labels else (min(n_points, 6) if likely_pie else n_points)

    label_instruction = (
        f"3. x_labels: use EXACTLY these {effective_n} category labels, in this order: "
        f"{json.dumps(category_labels)}.\n"
        if category_labels else
        f"3. x_labels: provide EXACTLY {effective_n} short labels (1-4 words each) describing each data point "
        "(e.g. years like '2019', '2020'; categories like 'Urban', 'Rural'; groups like 'Group A').\n"
    )
    prompt = (
        "You are an academic data analyst generating chart data for a dissertation figure.\n"
        f"Chart topic / section title: \"{context}\"\n\n"
        "Instructions:\n"
        f"1. Generate EXACTLY {effective_n} data points that are academically plausible for this topic.\n"
        "2. Choose the BEST chart_type: 'bar' for comparisons/categories, 'line' for trends over time, "
        "'scatter' for correlation/relationship, 'area' for cumulative trends, 'pie' for proportions.\n"
        f"{label_instruction}"
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
        if category_labels:
            x_labels_raw = list(category_labels)[: len(series)]
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
            "x_labels": list(category_labels) if category_labels else _fallback_chart_labels(label_style, effective_n),
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
        "- After the front matter, generate exactly 6 main chapters: Introduction, Literature Review, "
        "Methodology, Results/Analysis, Conclusion, and a final 'References and Appendices' chapter "
        "with a 'References' section and an 'Appendices' section — this final chapter is mandatory, "
        "never omit it.\n"
        "- Chapter 1 must contain: Background, Statement of the Problem, Research Objectives, "
        "Research Questions, Significance, Scope & Delimitations, Definition of Key Terms, Chapter Summary.\n"
        "- Chapter 2 title and section headings must be SPECIFIC to the research topic — do NOT use "
        "generic placeholder titles like 'Related Work' or 'Review of Literature'. Instead, frame "
        "headings around the actual subject matter (e.g. if the topic is municipal finance, use "
        "'2.2 Revenue Mechanisms in Local Government', '2.3 Fiscal Sustainability Theories', etc.).\n"
        "- Chapter 2 must include exactly one section whose title literally contains the words "
        "'Conceptual Framework' (e.g. 'X.X Conceptual Framework of <topic-specific qualifier>') — this is "
        "where the conceptual framework diagram will be placed, so do not omit or rename it away from "
        "that exact phrase even while keeping other Chapter 2 headings topic-specific. Skip this only in "
        "the rare case the study has no variables/constructs whose relationships could be diagrammed (e.g. "
        "a pure historical or doctrinal literature synthesis).\n"
        "- Chapter 2 should ALSO include a separate section whose title literally contains the words "
        "'Theoretical Framework' (e.g. 'X.X Theoretical Framework: <Named Theory>'), distinct from the "
        "Conceptual Framework section, but ONLY when the study is genuinely anchored in one or more "
        "specific, named theories — name the actual theory, never a generic placeholder. If no single "
        "theory meaningfully grounds this particular study, omit this section rather than forcing one in.\n"
        "- Chapter 2 must end with a 'Chapter Summary' section.\n"
        "- Chapter 3 sections must reflect the stated research design "
        f"({research_design}): include appropriate data collection and analysis subsections. "
        "Chapter 3 must end with a 'Chapter Summary' section.\n"
        "- Chapter 4 must have results subsections that directly correspond to the research objectives "
        "or questions. Chapter 4 must end with a 'Chapter Summary' section.\n"
        "- Chapter 5 must include: Summary of Findings, Conclusions, Recommendations, Limitations, "
        "Areas for Further Research, Chapter Summary. For an applied/organizational/policy topic where an "
        "implementation timeline genuinely makes sense, organize the Recommendations section's content "
        "around Short-term, Medium-term, and Long-term actions — skip this structuring for purely "
        "theoretical, lab-based, or exploratory topics where a rollout timeline wouldn't make sense. "
        "Where it fits the study, you may also add a 'Contributions of the Study' section covering its "
        "Theoretical, Practical, and Methodological contributions.\n"
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
            {"title": "2.1 Introduction", "sections": []},
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


def llm_chapters_to_blueprints(
    chapters: list[dict[str, Any]],
    research_design: str = "quantitative",
    objectives: list[str] | None = None,
    topic: str = "",
    message: str = "",
) -> list[dict[str, Any]]:
    """Convert LLM plan chapters to the internal blueprint format used by _write_dissertation.

    Mirrors what the chat-driven single-chapter rewrite path already does via
    _chapter_nodes_for_generation: inject the standard table/chart visual nodes for
    Chapters 1/2/3/5 (_inject_standard_visuals) and plan the objective-aware results
    visuals for Chapter 4 from the LLM's own topic-tailored section proposal
    (_plan_chapter4_structure), so a freshly-generated full dissertation isn't all
    plain text and isn't forced into a one-size-fits-all survey template.
    """
    objectives = objectives or []
    blueprints = []
    for chapter in chapters:
        title = chapter.get("title", "Untitled Chapter")
        chapter_number = _chapter_number_from_title(title)
        nodes = _sections_to_nodes(chapter.get("sections", []))

        if chapter_number == 4:
            nodes = _plan_chapter4_structure(chapter.get("sections", []), research_design, objectives, topic, message)
        elif chapter_number == 2:
            nodes = _inject_standard_visuals(nodes, chapter_number, research_design)
            nodes = _ensure_framework_visual(nodes)
        elif chapter_number:
            nodes = _inject_standard_visuals(nodes, chapter_number, research_design)

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


def _default_framework_spec(topic: str, prompt: str, kind: str = "conceptual") -> dict[str, Any]:
    short_topic = (topic or "the study").strip()
    if kind == "theory":
        return {
            "title": f"Theoretical Framework: {short_topic}",
            "left_label": "Antecedent Constructs",
            "left_items": ["External factors", "Individual characteristics", "Contextual conditions"],
            "middle_label": "Core Theoretical Constructs",
            "middle_items": ["Perceived attributes", "Attitudes/beliefs"],
            "right_label": "Outcome Construct",
            "right_items": ["Behavioral intention / adoption"],
            "control_label": "Boundary Conditions",
            "control_items": ["Scope of theory", "Underlying assumptions"],
            "notes": (prompt or "").strip()[:180],
        }
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
        # Ordered most-specific-first, based on whichever framework type the prompt actually
        # names. Numbered prefixes are deliberately omitted: different templates number
        # Chapter 2 differently (DISSERTATION_TEMPLATE uses "2.2 Conceptual Review" / "2.3
        # Theoretical Framework", _fallback_dissertation_chapters uses "2.2 Theoretical
        # Framework" / "2.3 Conceptual Framework"), and find_section's numeric-token tier
        # would otherwise match on the number alone and pick the wrong section.
        wants_theoretical = "theoretical" in text and "conceptual" not in text
        if wants_theoretical:
            preferred_titles = [
                "Theoretical Framework",
                "Conceptual Framework",
                "Conceptual Review",
                "Literature Review",
                "Chapter 2",
            ]
            keyword_groups = (("theoretical framework",), ("conceptual framework",), ("conceptual review",), ("framework",))
        else:
            preferred_titles = [
                "Conceptual Framework",
                "Conceptual Review",
                "Theoretical Framework",
                "Literature Review",
                "Chapter 2",
            ]
            keyword_groups = (("conceptual framework",), ("conceptual review",), ("theoretical framework",), ("framework",))

        for query in preferred_titles:
            idx = find_section(document.content, query)
            if idx is not None:
                return idx

        for keywords in keyword_groups:
            for i, section in enumerate(sections):
                combined = f"{section.get('title', '')}\n{section.get('content', '')}".lower()
                if any(k in combined for k in keywords):
                    return i

    return len(sections) - 1


def _framework_spec_from_inputs(
    topic: str,
    objectives: list[str],
    local_title: str,
    local_content: str,
    full_context: str,
    prompt: str,
    document_title: str = "",
    kind: str = "conceptual",
) -> dict[str, Any]:
    """Ask the LLM for a structured boxes-and-arrows figure spec given already-located context.

    kind="conceptual" builds the study's own conceptual framework (its IV/mediator/DV/control
    variables). kind="theory" builds a diagram of the named theory discussed in a Theoretical
    Framework section instead (the theory's own antecedent/core/outcome constructs), since that
    section describes an existing theory rather than the study's variable model.

    Shared by _build_framework_spec (chat 'add image' command, document-located context) and the
    automatic Chapter 2 visual injected during full dissertation writing (in-progress chapter
    context, no document lookup needed).
    """
    objective_text = "\n".join(f"- {item}" for item in objectives[:4])

    if kind == "theory":
        task_line = (
            "Identify the specific theory/model named or implied in the section content below "
            "(e.g. Technology Acceptance Model, Diffusion of Innovation Theory) and build a "
            "specification for a diagram of THAT THEORY'S OWN structure (its antecedent, core, "
            "and outcome constructs) — not the study's variables."
        )
        shape_hint = (
            '"title":"Theoretical Framework: <Theory Name>",'
            '"left_label":"Antecedent Constructs",'
            '"left_items":["...","..."],'
            '"middle_label":"Core Theoretical Constructs",'
            '"middle_items":["...","..."],'
            '"right_label":"Outcome Construct",'
            '"right_items":["..."],'
            '"control_label":"Boundary Conditions",'
            '"control_items":["...","..."],'
            '"notes":"Short rationale under 180 chars"'
        )
    else:
        task_line = "Build a specification for the study's own conceptual framework figure."
        shape_hint = (
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
        )

    llm_prompt = (
        f"You are an academic research assistant. {task_line}\n\n"
        f"Document title: {document_title or topic}\n"
        f"Topic: {topic}\n"
        f"Target section title: {local_title or 'N/A'}\n"
        f"User request: {prompt}\n\n"
        "Research objectives:\n"
        f"{objective_text or '- N/A'}\n\n"
        "Relevant section content:\n"
        f"{local_content[:1400]}\n\n"
        "Whole document context:\n"
        f"{full_context[:5000]}\n\n"
        "Return JSON only with this exact shape:\n"
        "{" + shape_hint + "}\n"
        "Rules: keep labels short, items concrete and topic-aware, no markdown."
    )

    default_left = "Antecedent Constructs" if kind == "theory" else "Independent Variables"
    default_middle = "Core Theoretical Constructs" if kind == "theory" else "Mediating Variables"
    default_right = "Outcome Construct" if kind == "theory" else "Dependent Variable"
    default_control = "Boundary Conditions" if kind == "theory" else "Control Variables"
    default_title = f"Theoretical Framework: {topic}" if kind == "theory" else f"Conceptual Framework: {topic}"

    try:
        data = _extract_json_obj(generate_text(llm_prompt))
        return {
            "title": str(data.get("title") or default_title),
            "left_label": str(data.get("left_label") or default_left),
            "left_items": [str(x) for x in (data.get("left_items") or [])][:4],
            "middle_label": str(data.get("middle_label") or default_middle),
            "middle_items": [str(x) for x in (data.get("middle_items") or [])][:4],
            "right_label": str(data.get("right_label") or default_right),
            "right_items": [str(x) for x in (data.get("right_items") or [])][:4],
            "control_label": str(data.get("control_label") or default_control),
            "control_items": [str(x) for x in (data.get("control_items") or [])][:4],
            "notes": str(data.get("notes") or "")[0:180],
        }
    except Exception as exc:
        logger.warning("_framework_spec_from_inputs fallback: %s", exc)
        return _default_framework_spec(topic, prompt, kind=kind)


def _build_framework_spec(document: Document, target: str | None, prompt: str) -> dict[str, Any]:
    sections = (document.content or {}).get("sections", [])
    idx = _framework_target_index(document, target, prompt)
    local_title = ""
    local_content = ""
    if idx is not None and 0 <= idx < len(sections):
        local_title = str(sections[idx].get("title") or "")
        local_content = str(sections[idx].get("content") or "")

    full_context = _full_context_for_generation(document)
    topic = str((document.content or {}).get("topic") or document.title or "Study")
    objectives = _extract_objectives(document, topic)

    lowered = f"{local_title} {target or ''} {prompt or ''}".lower()
    kind = "theory" if ("theoretical" in lowered and "conceptual" not in lowered) else "conceptual"

    return _framework_spec_from_inputs(
        topic=topic,
        objectives=objectives,
        local_title=local_title,
        local_content=local_content,
        full_context=full_context,
        prompt=prompt,
        document_title=document.title,
        kind=kind,
    )


def _insert_block_marker(section_text: str, block_id: str, prompt: str) -> str:
    text = section_text or ""
    marker = f"[[BLOCK:{block_id}]]"
    if marker in text:
        return text

    lines = text.splitlines()
    lowered_prompt = (prompt or "").lower()
    framework_request = any(k in lowered_prompt for k in ["conceptual", "theoretical", "framework", "model"])

    if framework_request:
        # Prefer the most specific heading match: whichever framework type the prompt
        # actually names wins even if the other type's heading appears earlier in the
        # section (e.g. "2.2 Theoretical Framework" before "2.3 Conceptual Framework").
        wants_theoretical = "theoretical" in lowered_prompt and "conceptual" not in lowered_prompt
        if wants_theoretical:
            keyword_groups = (("theoretical framework",), ("conceptual framework",), ("framework",))
        else:
            keyword_groups = (("conceptual framework",), ("theoretical framework",), ("framework",))
        for keywords in keyword_groups:
            for i, line in enumerate(lines):
                lower_line = line.lower().strip()
                if any(kw in lower_line for kw in keywords):
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


_SYNTHETIC_DATA_NOTE = (
    "\nNote: these figures are AI-simulated placeholders for demonstration purposes only — "
    "they are not real survey/experimental results. Replace with your actual collected data before submission."
)


def _response_rate_quality(pct: float) -> str:
    if pct >= 85:
        return "an excellent"
    if pct >= 70:
        return "a strong"
    if pct >= 50:
        return "an adequate"
    return "a modest"


def _document_has_hypotheses(document: Document) -> bool:
    """Whether this dissertation actually commits to stated hypotheses (H1, H2, ...) anywhere
    in its written content — used to gate hypothesis-confirmation language in Chapter 4 so it
    only appears for studies that posed hypotheses in the first place, not every quantitative
    study (a fixed apparatus would force this language on studies that never asked for it).
    """
    sections = (document.content or {}).get("sections", [])
    for sec in sections:
        text = str(sec.get("content") or "")
        if re.search(r"\bH0\s*:|\bH1\s*:|hypothes", text, flags=re.IGNORECASE):
            return True
    return False


def _hypothesis_tie_back(predictors: list[dict[str, Any]]) -> str:
    """One clause per predictor stating whether its corresponding hypothesis (H1, H2, ...,
    numbered in the same order the predictors/objectives were declared in Chapter 1) is
    supported by this regression result. Only ever invoked when the document actually
    states hypotheses (see _document_has_hypotheses), so it never fabricates a hypothesis
    the dissertation never posed.
    """
    verdicts = [
        f"H{i} ({p['label']}) {'is supported' if p['p'] < 0.05 else 'is not supported'}"
        for i, p in enumerate(predictors, start=1)
    ]
    return "Against the hypotheses set out in Chapter 1: " + "; ".join(verdicts) + "."


def _table_discussion_text(
    node_title: str,
    research_design: str,
    objective: str | None = None,
    table_dataset: dict[str, Any] | None = None,
    has_hypotheses: bool = False,
) -> str:
    """Build interpretation/discussion text that cites the ACTUAL values inside
    table_dataset (real row/column data, not generic boilerplate), so the Chapter 4
    narrative stays consistent with the table the reader is looking at. Deliberately
    domain-agnostic — it reads whatever row labels and numbers are present, so it works
    the same way for a survey response-rate table as for an engineering test-results table.
    """
    table_dataset = table_dataset or {}
    headers = [str(h) for h in (table_dataset.get("headers") or [])]
    rows = [[str(cell) for cell in row] for row in (table_dataset.get("rows") or []) if row]
    title_lower = node_title.lower()
    num = _parse_numeric_cell

    if "response rate" in title_lower and rows:
        returned_row = next(
            (r for r in rows if "return" in r[0].lower() and "not" not in r[0].lower()), None
        )
        if returned_row:
            pct = num(returned_row[-1])
            count = returned_row[-2] if len(returned_row) >= 2 else returned_row[-1]
            quality = _response_rate_quality(pct) if pct is not None else "an acceptable"
            return (
                f"Interpretation: {count} of the distributed instruments were returned and usable "
                f"({returned_row[-1]}), representing {quality} response rate for this study.\n"
                "Discussion: This level of return supports the adequacy of the dataset for the analysis "
                "applied in the objective-level sections that follow, and reduces the risk that "
                "non-response materially biases the findings."
                f"{_SYNTHETIC_DATA_NOTE}"
            )

    if "demographic" in title_lower and rows:
        groups: dict[str, list[tuple[str, float]]] = {}
        for r in rows:
            if len(r) >= 3:
                pct = num(r[-1])
                if pct is not None:
                    groups.setdefault(r[0], []).append((r[1], pct))
        highlights = [
            f"{max(entries, key=lambda e: e[1])[0]} ({max(entries, key=lambda e: e[1])[1]:.1f}%) for {variable}"
            for variable, entries in groups.items()
        ]
        highlight_text = "; ".join(highlights) if highlights else "a broad spread of categories"
        return (
            f"Interpretation: The respondent profile is led by {highlight_text}, "
            "indicating where participation was concentrated.\n"
            "Discussion: This composition should be borne in mind when generalizing the objective-level "
            "findings, and is reported transparently so readers can judge the sample's representativeness."
            f"{_SYNTHETIC_DATA_NOTE}"
        )

    if "correlation" in title_lower:
        model_fit = table_dataset.get("model_fit") or {}
        predictors = model_fit.get("predictors") or []
        if predictors:
            strongest = max(predictors, key=lambda p: p["r"])
            weakest = min(predictors, key=lambda p: p["r"])
            return (
                f"Interpretation: {strongest['label']} shows the strongest association with the overall "
                f"outcome (r = {_format_r_value(strongest['r'])}, {strongest['r_sig']}), while {weakest['label']} "
                f"shows the weakest (r = {_format_r_value(weakest['r'])}, {weakest['r_sig']}).\n"
                "Discussion: All relationships are in the expected positive direction, consistent with the "
                "conceptual framework presented in Chapter 2; the regression analysis that follows assesses "
                "their combined and relative contribution to the outcome."
                f"{_SYNTHETIC_DATA_NOTE}"
            )

    if "regression" in title_lower and "logistic" not in title_lower:
        model_fit = table_dataset.get("model_fit") or {}
        if model_fit.get("r_squared") is not None:
            hypothesis_clause = (
                f" {_hypothesis_tie_back(model_fit.get('predictors') or [])}" if has_hypotheses else ""
            )
            return (
                f"Interpretation: The regression model explains {model_fit['r_squared'] * 100:.0f}% of the "
                f"variance in the overall outcome (R² = {_format_r_value(model_fit['r_squared'])}, Adjusted R² = "
                f"{_format_r_value(model_fit['adj_r_squared'])}), F({model_fit['df1']}, {model_fit['df2']}) = "
                f"{model_fit['f_stat']:.2f}, {model_fit['f_sig']}.\n"
                "Discussion: This indicates that the predictors considered jointly make a statistically "
                "meaningful contribution to the outcome, beyond what would be expected by chance; the "
                "standardized beta values above indicate each predictor's relative contribution once the "
                f"others are held constant.{hypothesis_clause}"
                f"{_SYNTHETIC_DATA_NOTE}"
            )

    if "descriptive statistic" in title_lower or "descriptive analysis" in title_lower:
        if rows and headers and len(headers) >= 2:
            data_rows = [r for r in rows if r and r[0].lower() != "composite score"]
            if data_rows:
                means = [(r[0], num(r[1])) for r in data_rows if num(r[1]) is not None]
                if means:
                    highest = max(means, key=lambda x: x[1])
                    lowest = min(means, key=lambda x: x[1])
                    return (
                        f"Interpretation: {highest[0]} received the highest mean rating (M = {highest[1]:.2f}), "
                        f"while {lowest[0]} received the lowest (M = {lowest[1]:.2f}).\n"
                        "Discussion: The spread across dimensions highlights which areas are already "
                        "comparatively strong and which represent the most pressing priorities for "
                        "improvement, a pattern explored further in the discussion of findings."
                        f"{_SYNTHETIC_DATA_NOTE}"
                    )

    if research_design == "qualitative" and rows and headers and "theme" in headers[0].lower():
        mention_idx = 1 if len(rows[0]) > 1 else 0
        top_row = max(rows, key=lambda r: num(r[mention_idx]) or 0) if mention_idx else rows[0]
        theme_name = top_row[0]
        mentions = top_row[mention_idx] if len(top_row) > mention_idx else "several"
        return (
            f"Interpretation: \"{theme_name}\" emerged as the most frequently referenced theme "
            f"({mentions} mentions), with the remaining themes showing comparatively fewer occurrences.\n"
            "Discussion: This pattern of emphasis provides a basis for the thematic interpretation and "
            "implications discussed later in this chapter."
            f"{_SYNTHETIC_DATA_NOTE}"
        )

    if rows and len(rows[0]) > 1:
        numeric_col_idx = None
        for col_idx in range(1, len(rows[0])):
            values = [num(r[col_idx]) for r in rows if len(r) > col_idx]
            if values and all(v is not None for v in values):
                numeric_col_idx = col_idx
                break
        if numeric_col_idx is not None:
            values = [num(r[numeric_col_idx]) for r in rows]
            best_idx = max(range(len(values)), key=lambda i: values[i])
            worst_idx = min(range(len(values)), key=lambda i: values[i])
            obj_label = _truncate_label(objective or node_title or "this objective", 70)
            if best_idx == worst_idx:
                return (
                    f"Interpretation: {rows[best_idx][0]} recorded a value of {values[best_idx]:.2f}, "
                    f"the only indicator measured for {obj_label}.\n"
                    "Discussion: This result is discussed further in the sections that follow."
                    f"{_SYNTHETIC_DATA_NOTE}"
                )
            return (
                f"Interpretation: {rows[best_idx][0]} recorded the strongest value ({values[best_idx]:.2f}), while "
                f"{rows[worst_idx][0]} recorded the weakest ({values[worst_idx]:.2f}), showing uneven performance "
                f"across the indicators measured for {obj_label}.\n"
                "Discussion: This spread highlights where attention is most needed and supports the "
                "interpretation offered in the discussion section of this chapter."
                f"{_SYNTHETIC_DATA_NOTE}"
            )

    obj = _truncate_label(objective or node_title or "the objective", 70)
    return (
        f"Interpretation: The table summarizes the evidence gathered for {obj}.\n"
        "Discussion: These results are discussed further in the sections that follow."
        f"{_SYNTHETIC_DATA_NOTE}"
    )


def _objective_findings_preview(table_dataset: dict[str, Any] | None, research_design: str) -> str:
    """Short, real-data preview sentence for an objective's findings narrative, derived from
    the SAME table_dataset its own table child will render below it — so the prose previews a
    finding the table actually backs up, instead of a generic claim disconnected from the real
    numbers. Returns "" when no usable data is available (caller should omit the sentence).
    """
    table_dataset = table_dataset or {}
    headers = [str(h) for h in (table_dataset.get("headers") or [])]
    rows = [[str(c) for c in r] for r in (table_dataset.get("rows") or []) if r]
    if not rows:
        return ""

    if research_design == "qualitative" and headers and "theme" in headers[0].lower():
        mention_idx = 1 if len(rows[0]) > 1 else 0
        top_row = max(rows, key=lambda r: _parse_numeric_cell(r[mention_idx]) or 0) if mention_idx else rows[0]
        return (
            f"Preliminary thematic analysis points to \"{top_row[0]}\" as the most prominent pattern emerging "
            "for this objective, with further detail presented in the theme matrix below."
        )

    if len(rows[0]) > 1:
        numeric_col_idx = None
        for col_idx in range(1, len(rows[0])):
            values = [_parse_numeric_cell(r[col_idx]) for r in rows if len(r) > col_idx]
            if values and all(v is not None for v in values):
                numeric_col_idx = col_idx
                break
        if numeric_col_idx is not None:
            values = [_parse_numeric_cell(r[col_idx]) for r in rows]
            best_idx = max(range(len(values)), key=lambda i: values[i])
            return (
                f"Data collected for this objective identify {rows[best_idx][0]} as the strongest-performing "
                f"indicator ({values[best_idx]:.2f}), as detailed in the summary table below."
            )
    return ""


def _chart_discussion_text(series: list[float], objective: str | None = None, node_title: str | None = None) -> str:
    avg = round(sum(series) / len(series), 2) if series else 0.0
    high = max(series) if series else 0.0
    low = min(series) if series else 0.0
    trend = "upward" if len(series) > 1 and series[-1] >= series[0] else "mixed"
    objective_label = _truncate_label(objective or node_title or "the subsection", 70)
    return (
        f"Interpretation: Figure trend is {trend}, with values ranging from {low:.2f} to {high:.2f} and an average of {avg:.2f}.\n"
        f"Discussion: For {objective_label}, the visual pattern reinforces the numerical evidence and clarifies priority areas for action."
        f"{_SYNTHETIC_DATA_NOTE}"
    )


def _chart_data_from_table(table_dataset: dict[str, Any] | None) -> dict[str, Any] | None:
    """Derive chart series/x_labels directly from a sibling table's own rows, so a chart and
    the table it accompanies report the same numbers instead of being generated independently.
    Returns None when no matching table was supplied or it has no usable numeric column (e.g. a
    purely textual table like Key Terms or Data Collection Instruments) so the caller can fall
    back to _ai_chart_series.
    """
    if not table_dataset:
        return None
    headers = [str(h) for h in (table_dataset.get("headers") or [])]
    rows = [[str(cell) for cell in row] for row in (table_dataset.get("rows") or []) if row]
    if len(rows) < 2 or not headers:
        return None

    headers_lower = [h.lower() for h in headers]
    if "category" in headers_lower and "frequency" in headers_lower:
        label_idx = headers_lower.index("category")
        value_idx: int | None = headers_lower.index("frequency")
    else:
        label_idx = 0
        value_idx = None
        for col_idx in range(1, len(headers)):
            values = [_parse_numeric_cell(r[col_idx]) for r in rows if len(r) > col_idx]
            if len(values) == len(rows) and all(v is not None for v in values):
                value_idx = col_idx
                break
    if value_idx is None:
        return None

    values = [_parse_numeric_cell(r[value_idx]) if len(r) > value_idx else None for r in rows]
    if any(v is None for v in values):
        return None
    # Drop a trailing "(objective excerpt)" parenthetical before truncating, so axis labels
    # read as "Indicator A" rather than a mid-word cut like "Indicator A (To…".
    labels = [
        _truncate_label(re.sub(r"\s*\([^()]*\)\s*$", "", r[label_idx]).strip(), 18) if len(r) > label_idx else ""
        for r in rows
    ]

    value_header_lower = headers_lower[value_idx]
    sample_cell = rows[0][value_idx] if rows and len(rows[0]) > value_idx else ""
    if "%" in sample_cell or "percent" in value_header_lower:
        unit = "%"
    elif value_header_lower in {"frequency", "mentions", "count"}:
        unit = "count"
    elif value_header_lower == "value":
        unit = ""
    else:
        unit = headers[value_idx]

    return {"series": values, "x_labels": labels, "chart_type": "bar", "unit": unit}


def _append_node_plan_steps(plan: list[dict[str, Any]], nodes: list[dict[str, Any]], depth: int = 1) -> None:
    indent = "  " * depth
    for node in nodes:
        kind = node.get("kind", "text")
        verb = "Writing"
        if kind == "table":
            verb = "Creating table for"
        elif kind == "chart" and (node.get("meta") or {}).get("chart_type") in {"framework", "theory_model"}:
            verb = "Creating diagram for"
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


_META_COMMENTARY_PHRASES: tuple[str, ...] = (
    "this document currently",
    "the document currently",
    "key strength is that",
    "main weakness is that",
    "main gap is that",
    "area of improvement",
    "areas of improvement",
    "areas for improvement",
    "i have reviewed",
    "upon reviewing",
    "as a reviewer",
    "as an academic reviewer",
    "overall, the document",
    "the writer should",
    "could be strengthened by",
    "i recommend revising",
    "this section is currently",
    "in summary, this document",
    "the strongest opportunity is",
    "my take on your document",
    "what is working is that",
)


def _looks_like_meta_commentary(text: str) -> bool:
    """True when generated body text reads like a review/critique of the document itself
    (e.g. the model treats a 'write the chapter summary' request as 'review the document')
    rather than genuine chapter prose. Content like this must never be saved as a section
    body — callers should treat it the same as a generation failure and fall back.
    """
    low = (text or "").lower()
    return any(p in low for p in _META_COMMENTARY_PHRASES)


def _retrieve_citation_pool(topic: str, document_id: int | None) -> list[Any]:
    """Fetch real papers from the open scholarly web (Crossref, arXiv, PubMed, SSRN,
    Semantic Scholar) to ground dissertation citations in verifiable sources.
    Fails soft — returns [] if retrieval is unavailable (e.g. offline)."""
    try:
        from .research_layer import retrieval_pipeline

        result = retrieval_pipeline(topic=topic or "research topic", document_id=document_id, top_k=20)
        return result.top_papers
    except Exception as exc:
        logger.warning("_retrieve_citation_pool: retrieval failed for topic=%s: %s", (topic or "")[:60], exc)
        return []


def _apa_reference_entry(paper: Any) -> str:
    authors = list(getattr(paper, "authors", None) or [])
    if authors:
        names: list[str] = []
        for a in authors[:6]:
            parts = a.split()
            if len(parts) >= 2:
                initials = "".join(f"{n[0]}." for n in parts[:-1])
                names.append(f"{parts[-1]}, {initials}")
            else:
                names.append(a)
        if len(authors) > 6:
            author_str = ", ".join(names) + ", et al."
        elif len(names) > 1:
            author_str = ", ".join(names[:-1]) + ", & " + names[-1]
        else:
            author_str = names[0]
    else:
        author_str = "Unknown Author"
    year = str(paper.year) if getattr(paper, "year", None) else "n.d."
    title = (paper.title or "Untitled").rstrip(".")
    venue = paper.journal or (paper.source or "").replace("_", " ").title()
    locator = f"https://doi.org/{paper.doi}" if getattr(paper, "doi", None) else (paper.url or "")
    entry = f"{author_str} ({year}). {title}. {venue}."
    if locator:
        entry += f" {locator}"
    return entry


def _format_reference_list(pool: list[Any], max_items: int = 15) -> str | None:
    """Build a real APA-style reference list from retrieved papers only — no LLM invention."""
    if not pool:
        return None
    seen: set[str] = set()
    entries: list[str] = []
    for paper in pool:
        if not getattr(paper, "title", None):
            continue
        key = (getattr(paper, "doi", None) or paper.title).lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append(_apa_reference_entry(paper))
        if len(entries) >= max_items:
            break
    if not entries:
        return None
    entries.sort()
    return "\n\n".join(entries)


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
    citation_pool: list[Any] | None = None,
) -> tuple[str, str, list[dict[str, str]]]:
    chunks: list[str] = []
    blocks: list[dict[str, str]] = []
    local_context = rolling_context

    # Tracks table datasets already generated in this call, keyed by objective and by
    # table_type, so a later chart sibling for the same objective/type can reuse the exact
    # same numbers instead of generating its own independently.
    tables_by_objective: dict[str, dict[str, Any]] = {}
    tables_by_type: dict[str, dict[str, Any]] = {}

    # Build study brief ONCE per call (shared by every node in this invocation)
    specific_design = _specific_methodology(user_instruction, topic, document)
    document_brief = _extract_document_brief(document, topic, research_design, specific_design)
    has_hypotheses = _document_has_hypotheses(document)
    if citation_pool:
        from .research_layer import build_citation_context

        document_brief = (
            f"{document_brief}\n\n{build_citation_context(citation_pool, max_items=12)}\n"
            "When attributing claims to prior research (e.g. 'Smith (2020) found...'), cite ONLY "
            "the verified sources listed above. Do not invent author names, years, or studies."
        )

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
            # Reuse the dataset already computed for this objective by the
            # objective_findings parent (if any) so the table and the findings
            # narrative above it report the same numbers instead of two
            # independently generated, potentially divergent datasets.
            table_dataset = meta.get("_precomputed_table_dataset") or _ai_table_dataset(
                node_title=title,
                research_design=research_design,
                topic=topic,
                objective=objective,
                sample_size=sample_size,
                current_document_context=current_document_context,
                table_type=meta.get("table_type"),
                objectives=(document.content or {}).get("research_objectives"),
            )
            dataset_path = save_dataset_json(table_dataset, prefix="table-data")
            block_id = f"tbl-{table_no}-{len(blocks) + 1}"
            blocks.append({
                "type": "table",
                "headers": table_dataset.get("headers", []),
                "rows": table_dataset.get("rows", []),
                "caption": table_caption,
                "block_id": block_id,
                "dataset_json": dataset_path,
            })
            if objective:
                tables_by_objective[objective.strip().lower()] = table_dataset
            table_type_key = str(meta.get("table_type") or "").strip().lower()
            if table_type_key:
                tables_by_type[table_type_key] = table_dataset
            body = (
                f"{table_caption}\n"
                f"[[BLOCK:{block_id}]]\n"
                f"{_table_discussion_text(title, research_design, objective, table_dataset, has_hypotheses)}"
            )
        elif kind == "chart" and meta.get("chart_type") in {"framework", "theory_model"}:
            is_theory = meta.get("chart_type") == "theory_model"
            figure_no = figure_counter[0]
            figure_counter[0] += 1
            objectives_list = (document.content or {}).get("research_objectives") or []
            diagram_prompt = (
                f"Theoretical framework diagram for: {topic}" if is_theory
                else f"Conceptual framework diagram for: {topic}"
            )
            framework_spec = _framework_spec_from_inputs(
                topic=topic,
                objectives=objectives_list,
                local_title=title,
                local_content=local_context[-1400:],
                full_context=current_document_context,
                prompt=diagram_prompt,
                document_title=document.title,
                kind="theory" if is_theory else "conceptual",
            )
            image_path = generate_image(diagram_prompt, framework_spec=framework_spec)
            figure_caption = f"Figure {figure_no}: {framework_spec.get('title') or title}"
            block_id = f"fig-{figure_no}-{len(blocks) + 1}"
            blocks.append({
                "type": "image",
                "src": image_path,
                "caption": figure_caption,
                "block_id": block_id,
            })
            framework_notes = framework_spec.get("notes") or ""
            interpretation = (
                "Interpretation: The diagram maps the antecedent, core, and outcome constructs of "
                "the theory underpinning this study."
                if is_theory else
                "Interpretation: The diagram maps how the independent, mediating, and control variables "
                "identified in this study relate to the dependent variable."
            )
            body = (
                f"{figure_caption}\n"
                f"[[BLOCK:{block_id}]]\n"
                f"{interpretation}\n"
                f"Discussion: {framework_notes or 'This framework guides the analysis and interpretation presented in subsequent chapters.'}"
            )
        elif kind == "chart":
            objective = str(meta.get("objective") or "") or None
            if research_design == "qualitative":
                body = (
                    "Qualitative design prioritizes narrative/theme interpretation for this subsection. "
                    "No quantitative chart was generated for this section.\n"
                    f"{_table_discussion_text(title, research_design, objective, has_hypotheses=has_hypotheses)}"
                )
            else:
                figure_no = figure_counter[0]
                figure_counter[0] += 1
                figure_caption = f"Figure {figure_no}: {title}"
                context_str = title + (f" — {objective}" if objective else "")
                is_demographics_chart = meta.get("chart_type") == "demographics" or any(
                    k in title.lower() for k in ["demographic", "response rate", "respondent"]
                )
                category_labels = (
                    ["Male", "Female", "18-29 years", "30-39 years", "40 years and above"]
                    if is_demographics_chart else None
                )
                label_style = "likert" if _uses_human_respondents(topic, user_instruction) else "trial"
                sample_size = _infer_sample_size(document)

                # Prefer the sibling table's own numbers (matched by objective, falling back
                # to table_type/chart_type, e.g. "demographics") so the chart never contradicts
                # the table it accompanies. Only generate independent data when no matching
                # table was processed earlier in this same call.
                matched_table = tables_by_objective.get((objective or "").strip().lower()) if objective else None
                if matched_table is None:
                    chart_type_key = str(meta.get("chart_type") or "").strip().lower()
                    if chart_type_key:
                        matched_table = tables_by_type.get(chart_type_key)
                ai_data = _chart_data_from_table(matched_table)
                if ai_data is not None:
                    if is_demographics_chart:
                        ai_data["chart_type"] = "pie"
                else:
                    ai_data = _ai_chart_series(
                        context_str,
                        n_points=8,
                        category_labels=category_labels,
                        label_style=label_style,
                    )
                    if is_demographics_chart:
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
            # Compute this objective's table data NOW (instead of when its table child node
            # is reached later in the recursion below) so the findings narrative we are about
            # to write can be grounded in the same real numbers the table itself will show,
            # rather than generating disconnected, generic prose above a table it never saw.
            table_child = next(
                (c for c in node.get("children", []) if isinstance(c, dict) and c.get("kind") == "table"),
                None,
            )
            table_dataset = None
            if table_child is not None:
                sample_size = _infer_sample_size(document)
                table_dataset = _ai_table_dataset(
                    node_title=table_child.get("title", ""),
                    research_design=research_design,
                    topic=topic,
                    objective=objective,
                    sample_size=sample_size,
                    current_document_context=current_document_context,
                    table_type=(table_child.get("meta") or {}).get("table_type"),
                    objectives=(document.content or {}).get("research_objectives"),
                )
                table_child.setdefault("meta", {})["_precomputed_table_dataset"] = table_dataset
            grounding = _objective_findings_preview(table_dataset, research_design)
            quote_guidance = (
                "Where it fits, ground the discussion with 1-2 short direct quotes presented as indented "
                "block quotes, each attributed to an anonymized informant role consistent with this study "
                "(e.g. \"(Informant 3, Operations Manager)\" or \"(Participant 5, Frontline Staff)\") rather "
                "than a generic 'respondents said' paraphrase — but only invent quotes that are plausible "
                "given the actual data/themes above, never quotes that contradict them.\n"
                if research_design in {"qualitative", "mixed"} else ""
            )
            try:
                body = generate_section_content(
                    title=title,
                    topic=topic,
                    context=(
                        f"{document_brief}\n\n"
                        f"Research objective: {objective}\n"
                        f"Research design: {research_design}\n"
                        f"Parent chapter: {section_title}\n"
                        + (f"Actual data collected for this objective: {grounding}\n" if grounding else "")
                        + f"Current document context:\n{current_document_context[-3000:]}\n\n"
                        "Write 2-3 paragraphs presenting findings specifically for this objective. "
                        "Discuss patterns, trends, and how findings directly address the objective. "
                        "Be specific and academically rigorous, and stay consistent with the actual data "
                        "noted above where provided.\n"
                        f"{quote_guidance}"
                    ),
                    word_count=default_word_count,
                )
                if _looks_like_meta_commentary(body):
                    raise ValueError("Generated body reads like document meta-commentary, not findings prose.")
            except Exception:
                body = _fallback_subsection_text(
                    topic, section_title, title,
                    objectives=(document.content or {}).get("research_objectives"),
                    target_words=default_word_count,
                    research_design=research_design,
                    specific_design=specific_design,
                    sample_size=_infer_sample_size(document),
                )
                if grounding:
                    body = f"{grounding} {body}"
        else:
            lowered_title = title.lower()
            real_references = (
                ("reference" in lowered_title or "bibliograph" in lowered_title)
                and _format_reference_list(citation_pool)
            )
            if real_references:
                # ── References/Bibliography: built directly from verified papers ──
                # retrieved from Crossref/arXiv/PubMed/SSRN/Semantic Scholar — never
                # asked of the LLM, so there is nothing here for it to hallucinate.
                body = (
                    "The following references were retrieved from open scholarly databases "
                    "(Crossref, arXiv, PubMed, Semantic Scholar) and ranked as directly relevant "
                    "to this study topic:\n\n" + real_references
                )
                body = _strip_leading_heading(body, title)
                body = _sanitize_body(body)
                chunks.append(f"{title}\n{body}")
                local_context = f"{local_context}\n\n{title}\n{body}".strip()
                _done(plan, step_idx)
                if callable(on_node_completed):
                    try:
                        on_node_completed("\n\n".join(chunks), list(blocks), title)
                    except Exception as exc:
                        logger.warning("Subsection progress callback failed for '%s': %s", title, exc)
                continue

            # ── Text node: multi-agent pipeline ──────────────────────────
            # PlannerAgent.generate_spec enriches the task with guidelines +
            # word count.  ContentGenerator → RuntimeSandbox → ErrorAnalyzer
            # → RepairAgent handle quality gating and automatic repair.
            from .pipeline import Pipeline
            from .planner import PlannerAgent, TaskSpec as PipelineTaskSpec, IntentSpec

            guidelines = _subsection_guidelines(title, topic)
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
            _doc_objectives = (document.content or {}).get("research_objectives")
            result = Pipeline().run_node(
                task=task_spec,
                document_brief=document_brief,
                rolling_context=local_context,
                generate_fn=generate_section_content,
                fallback_fn=lambda t, s, sub, wc: _fallback_subsection_text(
                    t, s, sub, objectives=_doc_objectives, target_words=wc,
                    research_design=research_design, specific_design=specific_design,
                    sample_size=_infer_sample_size(document),
                ),
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
                citation_pool,
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


def _extract_topic_phrase(text: str, lead_words: tuple[str, ...] = ("on", "about")) -> str | None:
    """Pull a topic phrase out of a free-form instruction like 'write a dissertation about X.
    Use a quantitative design.' Stops at the first sentence boundary and strips a trailing
    research-design clause so unpunctuated, voice-transcribed prompts (no period before
    'use a ... design') don't bleed the whole rest of the message into the topic.
    """
    pattern = r"\b(?:" + "|".join(lead_words) + r")\b\s+(.+?)(?:[.!?]|$)"
    match = re.search(pattern, text)
    if not match:
        return None
    topic = match.group(1).strip()
    topic = re.sub(
        r"\s+(?:use|using)\s+a\s+\w+(?:\s+\w+){0,2}\s+(?:research\s+)?design\s*$", "", topic
    ).strip()
    return topic or None


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
        topic = _extract_topic_phrase(text)
        return {"intent": "write_dissertation", "target_section": None, "topic": topic}

    if (
        "project" in text
        and any(k in text for k in ["full", "complete", "entire", "whole", "write", "create", "build", "do", "generate"])
    ):
        topic = _extract_topic_phrase(text)
        return {"intent": "write_dissertation", "target_section": None, "topic": topic}

    if "outline" in text:
        topic = _extract_topic_phrase(text, ("on",))
        return {"intent": "create_outline", "target_section": None, "topic": topic}

    _visual_verbs = {"add", "put", "insert", "include", "place", "attach", "generate", "create", "make", "draw", "build", "design"}

    # "generate the conceptual framework" — framework requests imply a diagram even
    # without an explicit "image"/"diagram"/"chart" keyword. _add_image already knows
    # how to locate the right section (_framework_target_index) and build a real
    # boxes-and-arrows spec (_build_framework_spec) rather than rewriting prose.
    _framework_phrases = ["conceptual framework", "theoretical framework"]
    if any(v in text for v in _visual_verbs) and any(p in text for p in _framework_phrases):
        return {"intent": "add_image", "target_section": target, "topic": None}

    # ── Visual + section-keyword detection (must come before plain add_chart/add_image) ──
    # "generate the conceptual framework image" / "add a chart to the empirical review"
    # → route to write_section so _write_section injects the visual node.
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
        topic = _extract_topic_phrase(text, ("on", "about", "for", "regarding"))
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

    # Strip common filler determiners so natural insertions like "remove THE ai"
    # or "get rid of THE plagiarism" still match the literal phrases below.
    _loose_text = re.sub(r"\b(the|a|an|this|that|my|your|its|it's)\b", " ", text)
    _loose_text = re.sub(r"\s+", " ", _loose_text).strip()

    # ── Humanise / de-AI ────────────────────────────────────────────────────
    _humanise_kw = {
        "humanise", "humanize", "humanise the", "humanize the",
        "make it sound human", "sound more human", "more human-like",
        "remove ai", "less ai", "bypass ai", "avoid ai detection",
        "rewrite ai", "humanise ai", "humanize ai",
        "natural voice", "natural writing", "make more natural",
        "get rid of ai", "take out ai", "eliminate ai", "scrub ai",
        "sound human", "less robotic", "more natural sounding",
        "make sound human", "make less ai",
    }
    if any(k in _loose_text for k in _humanise_kw):
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
        "get rid of plagiarism", "take out plagiarism", "eliminate plagiarism",
        "scrub plagiarism", "cut plagiarism", "clean up plagiarism",
    }
    if any(k in _loose_text for k in _plagiarism_reduce_kw):
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


def _objective_to_question(objective: str, topic: str) -> str:
    """Convert a research objective statement into an interrogative research question."""
    cleaned = objective.strip().rstrip(".")
    remainder, replaced = re.subn(r"^to\s+\w+\s+", "", cleaned, count=1, flags=re.IGNORECASE)
    if not replaced:
        return f"What is the relationship between {topic} and {cleaned}?"
    remainder = remainder.strip()
    subject_phrase = re.split(
        r"\s+(?:for|affecting|influencing|shaping|impacting|in|on|of|within|across|among)\s+",
        remainder,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    head_word = subject_phrase.strip().split(" ")[-1].lower().strip(",")
    is_plural = head_word.endswith("s") and not head_word.endswith(("ss", "us", "is", "his"))
    lead = "What are" if is_plural else "What is"
    return f"{lead} {remainder}".rstrip("?") + "?"


# Subsections that are inherently list-form (objectives, hypotheses, recommendations,
# ToC-style listings, definitions) — these must stay short and structured, so the
# word-count expansion pass below never pads them with prose elaboration.
_FALLBACK_NO_EXPAND_KEYWORDS: tuple[str, ...] = (
    "abstract",
    "table of contents", "list of figures", "list of tables", "abbreviation", "acronym",
    "reference", "bibliograph", "appendix", "appendices",
    "research objective", "objectives", "research question", "hypothes",
    "recommendation", "further research", "areas for future", "definition of key",
)


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w'-]+", text or ""))


# Generic, on-topic elaboration paragraphs used to pad deterministic fallback prose
# toward a chapter's word-count target. Iterated in order (never repeated within a
# single subsection) so static fallback text never duplicates a paragraph verbatim.
_FALLBACK_ELABORATIONS: tuple[str, ...] = (
    "Within the wider body of scholarship that informs {topic}, work examining comparable settings generally "
    "converges on the view that outcomes are shaped by an interaction of structural conditions and the specific "
    "choices made by practitioners or designers, rather than by any single determinant acting in isolation.",
    "Decision-makers operating in this space must weigh competing priorities; the available evidence suggests "
    "that the most durable outcomes for {topic} arise where strategic intent is matched by consistent "
    "operational or technical follow-through.",
    "The different dimensions of {topic} are not independent: institutional, operational, and individual-level "
    "factors interact in ways that a purely additive treatment would understate, which is why an integrated "
    "analytical lens is needed to make sense of the overall picture.",
    "Taken together, these considerations underscore the multidimensional nature of {topic}. No single explanation "
    "is sufficient on its own; the strongest accounts integrate technical, organisational, and contextual factors "
    "into a coherent explanatory framework capable of withstanding scrutiny across different settings.",
    "Conclusions about {topic} are strongest when they rest on a cumulative evidentiary base rather than an "
    "isolated data point or anecdote, which is why triangulating multiple sources of evidence remains important "
    "throughout this analysis.",
    "Alternative explanations proposed in the literature on {topic} differ in emphasis, but most converge on the "
    "conclusion that context-specific evidence is necessary to adjudicate between competing accounts.",
    "From an applied standpoint, practitioners, policymakers, and future researchers engaging with {topic} need "
    "to translate these insights into practice with attention to local context, available resources, and the "
    "specific institutional or technical arrangements that shape implementation.",
    "Findings of this kind are typically sensitive to scale, context, and the specific configuration under study, "
    "and {topic} is no exception — generalising beyond the conditions examined here should be done cautiously.",
    "Methodologically, the strength of any conclusion about {topic} depends on the quality of the underlying "
    "evidence base. Triangulating multiple sources and, where possible, independent verification strengthens "
    "confidence in the observed patterns.",
    "These dynamics surrounding {topic} are rarely static: they tend to evolve over time as practices mature, "
    "feedback loops take effect, and stakeholders adjust their behaviour in response to early outcomes.",
    "These dynamics also carry implications beyond the immediate scope of {topic}, particularly for related "
    "fields that share similar structural or contextual features. Cross-disciplinary engagement of this kind can "
    "surface analytical perspectives that a narrower, single-discipline treatment might overlook.",
    "Sample characteristics, measurement choices, and the specific context examined all bound the scope within "
    "which conclusions about {topic} should be interpreted, a limitation worth bearing in mind when generalising "
    "from any single body of evidence.",
    "From a policy perspective, {topic} speaks to ongoing debates about how institutions and regulators should "
    "respond to developments of this kind. Evidence-based policymaking in this area benefits from granular, "
    "context-specific findings rather than broad generalisation.",
    "The historical trajectory of research on {topic} provides useful context here: earlier work tended to "
    "emphasise different priorities than current scholarship, reflecting shifts in both the underlying phenomena "
    "and the analytical tools available to researchers.",
    "Stakeholder perspectives on {topic} are not uniform. Differences in priorities, incentives, and access to "
    "resources mean that the same evidence can support divergent practical conclusions depending on one's "
    "vantage point.",
    "Measurement and operationalisation choices materially affect how findings on {topic} are interpreted within "
    "the broader literature. Where different studies operationalise key constructs differently, comparing "
    "findings across that literature requires care.",
    "Resource and capacity constraints frequently mediate how these issues translate into practice for "
    "organisations engaging with {topic}. Even well-evidenced recommendations can founder where implementing "
    "capacity is lacking.",
    "These issues also intersect with broader questions of sustainability and long-term viability in the context "
    "of {topic}. Short-term gains that are not structurally sustained tend to erode, underscoring the importance "
    "of embedding any improvements within durable institutional arrangements.",
    "Risk considerations are relevant to a full appreciation of {topic}. Unanticipated risks — whether "
    "operational, reputational, or technical — can materially alter the practical calculus even where the "
    "underlying evidence is otherwise compelling.",
    "Comparative evidence from adjacent contexts can sharpen interpretation here. Where {topic} has been studied "
    "across multiple settings, convergent findings strengthen confidence in generalisability, while divergent "
    "findings point toward context-specific moderating factors worth investigating further.",
)


_FALLBACK_TECHNICAL_ELABORATIONS: tuple[str, ...] = (
    "Within the wider body of engineering work that informs {topic}, comparable implementations generally "
    "converge on the view that performance is shaped by an interaction of design choices and the specific "
    "operating conditions under test, rather than by any single component acting in isolation.",
    "Designers working in this space must weigh competing constraints such as cost, complexity, and power "
    "budget; the available evidence suggests that the most durable performance gains for {topic} arise where "
    "design intent is matched by consistent, disciplined testing.",
    "The different design considerations behind {topic} are not independent: mechanical, electrical, and "
    "control-level choices interact in ways that a purely component-by-component treatment would understate, "
    "which is why an integrated test-and-evaluation approach is needed to make sense of overall performance.",
    "Taken together, these considerations underscore the multidimensional nature of {topic}. No single design "
    "choice is sufficient on its own; the strongest implementations integrate hardware, control logic, and test "
    "methodology into a coherent design capable of performing reliably across different operating conditions.",
    "Conclusions about {topic} are strongest when they rest on a cumulative body of test evidence rather than an "
    "isolated measurement, which is why repeated, cross-referenced testing remains important throughout "
    "development.",
    "Alternative design approaches reported in the literature on {topic} differ in emphasis, but most converge "
    "on the conclusion that direct, repeatable test evidence is necessary to compare them fairly.",
    "From an applied standpoint, developers, students, and future researchers building on {topic} need to "
    "translate these insights into practice with attention to component tolerances, available test equipment, "
    "and the specific operating environment in which the system will be used.",
    "Results of this kind are typically sensitive to test setup, calibration, and the specific configuration "
    "under study, and {topic} is no exception — generalising beyond the conditions examined here should be done "
    "cautiously.",
    "Methodologically, the strength of any conclusion about {topic} depends on the quality of the underlying "
    "test evidence. Triangulating multiple measurement approaches and, where possible, independent verification "
    "strengthens confidence in the observed patterns.",
    "These performance characteristics are rarely static: they tend to shift over time as components age, "
    "operating conditions vary, and incremental design changes accumulate across iterations of {topic}.",
    "These design choices also have implications for the manufacturability and cost of scaling {topic} beyond a "
    "single prototype. Choices that perform well in a lab setting do not always translate cleanly into a "
    "cost-effective, mass-producible implementation.",
    "Sample size, test duration, and the specific operating envelope examined all bound the scope within which "
    "conclusions about {topic} should be interpreted, a limitation worth bearing in mind when generalising from "
    "a single test campaign.",
    "Safety considerations are relevant to a full appreciation of {topic}. Failure modes that are tolerable in a "
    "controlled test environment may carry materially different risk profiles once deployed in the field.",
    "Energy efficiency and power budget constraints frequently mediate how these design choices translate into a "
    "viable implementation of {topic}. Even a technically superior approach can be impractical where it exceeds "
    "the available power envelope.",
    "Maintainability and serviceability are relevant to the long-term viability of {topic}. A design that "
    "performs well when new but is difficult to inspect, calibrate, or repair carries hidden lifecycle costs "
    "that should factor into any overall assessment.",
    "Comparative evidence from alternative architectures sharpens interpretation here. Where {topic} has been "
    "approached using competing design philosophies, convergent performance findings strengthen confidence in "
    "the conclusions, while divergent findings point to context-specific trade-offs worth investigating further.",
    "Environmental robustness — performance under temperature extremes, vibration, humidity, or electrical "
    "noise — is a relevant consideration for {topic}. Results obtained under benign laboratory conditions do not "
    "automatically generalise to harsher operating environments.",
    "Integration considerations also bear on the practical significance of these results. A component or "
    "subsystem that performs well in isolation can still introduce unexpected interactions once integrated into "
    "the larger system developed for {topic}.",
    "Repeatability across units, not just across trials of a single unit, is relevant to these conclusions. "
    "Manufacturing and component tolerances can introduce unit-to-unit variation that a single-prototype "
    "evaluation of {topic} would not capture.",
    "These results sit alongside broader systems-engineering trade-offs inherent in {topic}, where improving one "
    "performance dimension routinely entails a measurable cost along another, and the optimal balance depends on "
    "the priorities of the specific application.",
)


_FALLBACK_PERSONAL_ELABORATIONS: tuple[str, ...] = (
    "Special thanks are due to family members whose patience, encouragement, and quiet sacrifices made the long "
    "hours behind this work possible. Their belief in this undertaking, even when progress was slow, was a "
    "constant source of motivation.",
    "Appreciation is also extended to friends, classmates, and colleagues who provided moral support, practical "
    "assistance, and honest feedback throughout the course of {topic}.",
    "Gratitude is owed to the institution, its staff, and its administration for providing access to the "
    "facilities, equipment, and resources without which this work could not have been completed.",
    "Above all, this stands as a tribute to everyone — named and unnamed — whose belief in this undertaking, even "
    "during its most difficult stretches, helped see it through to completion.",
)


def _seeded_pick(seed_text: str, options: tuple[str, ...]) -> str:
    """Deterministically choose one of options based on seed_text.

    Used so the highest-traffic deterministic fallback blocks (Background, Statement of
    the Problem, Discussion, Conclusion, Recommendations) vary in opening hook and
    paragraph structure across different topics/documents, rather than every document
    rendering the exact same skeleton with only the topic name swapped in.
    """
    return options[sum(ord(c) for c in seed_text) % len(options)]


# Background of the Study — multiple structurally distinct openings (trend-led,
# gap-led, historical-arc) so the fallback doesn't read as one fixed skeleton.
_BACKGROUND_VARIANTS_SURVEY: tuple[str, ...] = (
    (
        "The rapid growth of {topic} has emerged as a defining trend, prompting both practitioners and researchers "
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
    ),
    (
        "Although {topic} has attracted growing attention from researchers and practitioners alike, much of what informs "
        "current practice rests on assumption, anecdote, or evidence drawn from markedly different settings. This study "
        "begins from that gap: it situates the research problem within the specific socioeconomic and institutional "
        "conditions of the present context, rather than extrapolating from generalised models.\n\n"
        "Across the wider field, observers report meaningful shifts in how organisations manage performance, governance, "
        "and stakeholder relationships, driven in large part by developments in data analytics, digital infrastructure, "
        "and shifting regulatory expectations. Yet these shifts have not been matched by a comparable volume of "
        "context-specific empirical work — a shortfall that is especially pronounced in emerging-market settings, where "
        "institutional capacity and digital readiness differ markedly from the contexts most studies are drawn from.\n\n"
        "It is this shortfall that the present study sets out to address, by generating primary evidence grounded in the "
        "actual conditions of the study context rather than imported assumptions. In doing so, it aims to produce findings "
        "that speak directly to policy discourse, practitioner decision-making, and the wider scholarly conversation on "
        "{topic}."
    ),
    (
        "The trajectory of {topic} over recent years illustrates how quickly practice in this area can outpace the "
        "evidence base meant to guide it. What began as a relatively narrow concern has, over a comparatively short "
        "period, become a central preoccupation for institutions seeking to remain competitive, compliant, and "
        "responsive to stakeholder expectations.\n\n"
        "This evolution has not been uniform. Some organisations have moved decisively to embed new practices into their "
        "operating models, while others continue to grapple with constraints of capacity, governance, or readiness that "
        "slow adoption. The resulting variation is precisely the kind of pattern that warrants systematic empirical "
        "investigation rather than broad generalisation from a handful of visible cases.\n\n"
        "The present study responds to this need directly. By collecting and analysing primary data from within the "
        "specific context under investigation, it aims to clarify how, and under what conditions, {topic} translates into "
        "measurable outcomes — generating evidence that is transferable to policy discourse, practitioner decision-making, "
        "and ongoing scholarly debate."
    ),
)

_BACKGROUND_VARIANTS_TECHNICAL: tuple[str, ...] = (
    (
        "Interest in {topic} has grown substantially as advances in sensing, computation, and materials have made "
        "increasingly capable systems practical to design, build, and evaluate. The background of this study situates "
        "the research problem within this broader technical trajectory, drawing on prior engineering work and "
        "established design principles to justify the relevance and timing of the present effort.\n\n"
        "Across the field, developers have reported steady gains in performance, reliability, and efficiency, "
        "driven by improvements in component technology, control algorithms, and testing methodology. However, "
        "published evaluations vary widely in rigor and scope — many implementations are demonstrated under "
        "narrow conditions without systematic performance testing across realistic operating scenarios. This gap "
        "is particularly evident where cost, complexity, or environmental variability constrain what a thorough "
        "evaluation can practically cover.\n\n"
        "Against this backdrop, the present study addresses a clear technical need: to design, implement, and "
        "rigorously evaluate a solution for the problem at hand, generating measurable evidence of its performance "
        "under defined test conditions. By documenting its design choices and test results in detail, the study "
        "contributes findings that are transferable to further development, comparative benchmarking, and practical "
        "deployment decisions."
    ),
    (
        "Although {topic} sits at the intersection of several fast-moving technical fields, much of the published work "
        "describing similar systems stops short of a rigorous, repeatable performance evaluation. This study begins from "
        "that gap: rather than assuming a design will perform adequately because comparable designs have been reported "
        "elsewhere, it commits to building and testing one under clearly defined conditions.\n\n"
        "Across the wider field, practitioners report steady improvements in component capability, control sophistication, "
        "and overall system reliability, driven by advances that have made previously impractical designs newly feasible. "
        "Yet detailed, reproducible performance data describing exactly how these gains are achieved — and under what "
        "conditions they hold — remains comparatively scarce, particularly for designs built under realistic resource "
        "constraints rather than in a fully equipped research laboratory.\n\n"
        "It is this shortfall that the present study sets out to address, by producing a working implementation together "
        "with a transparent, structured record of its test performance. In doing so, it aims to generate findings that are "
        "directly transferable to further development, benchmarking, and practical deployment decisions."
    ),
    (
        "The development of systems addressing {topic} has accelerated noticeably as advances in sensing, computation, "
        "and materials have lowered the practical barriers to building increasingly capable designs. What was once "
        "feasible only with specialised laboratory equipment is now within reach of a much wider range of developers and "
        "researchers.\n\n"
        "This broadening of access has not been matched by an equivalent broadening of rigorous evaluation practice. Many "
        "reported implementations describe a single successful demonstration rather than systematic testing across the "
        "range of conditions a design is likely to encounter in practice, making it difficult to compare designs fairly "
        "or to anticipate where a given approach is likely to fail.\n\n"
        "The present study responds to this gap directly, committing to a design-and-test approach that documents not "
        "only what was built but precisely how it performed, under what conditions, and against what baseline. The "
        "resulting evidence is intended to be transferable to further development, comparative benchmarking, and "
        "practical deployment decisions."
    ),
)

# Statement of the Problem — varied framing (gap-led, cost-of-inaction-led, evidence-thinness-led).
_PROBLEM_VARIANTS_SURVEY: tuple[str, ...] = (
    (
        "Despite growing interest in {topic}, a critical gap persists in the empirical literature: there is insufficient "
        "context-specific evidence to explain how and under what conditions the key variables produce observed outcomes. "
        "Existing studies tend to rely on generalised frameworks that do not adequately capture the institutional "
        "and environmental specificity of the research context.\n\n"
        "This limitation has practical consequences — policymakers and practitioners operate with incomplete evidence, "
        "increasing the risk of misaligned interventions and suboptimal resource allocation. The problem, therefore, "
        "is not merely theoretical; it carries direct implications for how institutions design, implement, and evaluate "
        "strategies within this domain. This study is designed to address that gap directly by generating primary, "
        "context-grounded evidence on the relationships central to the research."
    ),
    (
        "A central difficulty facing researchers and practitioners working on {topic} is the scarcity of evidence that is "
        "both rigorous and specific to the context in which decisions actually have to be made. Much of the available "
        "literature draws on settings, samples, or institutional arrangements that differ materially from the present "
        "context, leaving a gap between what is generally known and what is actually true here.\n\n"
        "That gap is not merely an academic inconvenience. Where decision-makers lack context-specific evidence, they are "
        "left to rely on assumption, precedent, or evidence imported from elsewhere — any of which can lead to "
        "interventions that are poorly matched to local conditions, with resources allocated on a weaker evidential "
        "footing than the stakes would warrant. This study is designed to close that gap by generating primary, "
        "context-grounded evidence on the relationships central to the research."
    ),
    (
        "{topic} continues to attract considerable attention, yet the evidence base available to those who must act on "
        "it remains thinner than the level of interest would suggest. Existing studies frequently rely on frameworks "
        "developed elsewhere and applied to this setting without adequate adaptation, raising legitimate questions about "
        "how well their conclusions actually transfer.\n\n"
        "Left unaddressed, this evidentiary gap carries real costs: institutions and practitioners continue to make "
        "resourcing and strategic decisions without a clear, locally grounded understanding of how the relevant variables "
        "actually interact in this context, increasing the likelihood that well-intentioned interventions miss their "
        "mark. This study addresses that gap directly, generating primary, context-specific evidence rather than relying "
        "on inference from settings that may not be comparable."
    ),
)

_PROBLEM_VARIANTS_TECHNICAL: tuple[str, ...] = (
    (
        "Despite growing interest in {topic}, a practical gap persists: many existing implementations are described "
        "without a clearly documented design rationale or a rigorous, repeatable evaluation against defined performance "
        "criteria. Reported results are often anecdotal or limited to a single demonstration run rather than systematic "
        "testing across varied operating conditions.\n\n"
        "This limitation has real consequences — developers and adopters operate with incomplete evidence about how a "
        "given design performs outside the specific conditions under which it was first demonstrated, increasing the "
        "risk of unreliable behaviour when deployed more broadly. The problem, therefore, is not merely academic; it "
        "carries direct implications for how such systems should be designed, tested, and refined. This study is "
        "designed to address that gap directly by producing a working implementation and subjecting it to structured, "
        "repeatable performance testing."
    ),
    (
        "A recurring difficulty in published work related to {topic} is that designs are frequently described without a "
        "clearly documented rationale for the choices made, or without test results detailed enough to support a fair "
        "comparison against alternative approaches. A single successful demonstration is often presented as sufficient "
        "evidence of a design's merit.\n\n"
        "That gap matters in practice. Developers and adopters working from such reports are left to estimate how a "
        "design will behave outside the narrow conditions under which it was first shown to work, increasing the risk "
        "that a design judged adequate in one context performs unreliably in another. This study addresses that gap "
        "directly, producing a working implementation together with a structured, repeatable evaluation against clearly "
        "defined performance criteria."
    ),
    (
        "Despite the volume of work published on systems related to {topic}, comparatively little of it offers a level "
        "of testing detail sufficient to support confident replication or fair comparison. Reported figures are often "
        "presented without the operating conditions, tolerances, or trial counts needed to interpret them rigorously.\n\n"
        "The practical cost of this gap falls on exactly the people best placed to build on existing work: developers and "
        "researchers attempting to extend or compare designs are forced to re-derive baseline performance from scratch, "
        "slowing progress across the field. This study is designed to close that gap by subjecting a working "
        "implementation to systematic, clearly documented testing against defined performance criteria."
    ),
)

# Discussion of Findings (Chapter 4) — varied closing framing while keeping the same
# evidentiary content (consistency with the literature, plausible source of deviation).
_DISCUSSION_VARIANTS_SURVEY: tuple[str, ...] = (
    (
        "The findings are broadly consistent with the theoretical propositions advanced in the literature review. "
        "The observed relationships between the study variables confirm that {topic} is a multidimensional phenomenon shaped by "
        "institutional capacity, governance quality, and contextual readiness.\n\n"
        "Where deviations from prior research are observed, these may be attributed to sector-specific characteristics or differences "
        "in measurement approach. In particular, the relatively stronger effect sizes recorded in this study compared to earlier work "
        "may reflect the more advanced stage of adoption within the sampled organisations, or the more targeted sampling frame employed."
    ),
    (
        "On the whole, the pattern of results aligns with what the literature reviewed in Chapter 2 would lead one to "
        "expect, lending support to the conceptual framework that guided the analysis. {topic} emerges from this evidence "
        "as a multidimensional phenomenon — one in which institutional capacity, governance quality, and contextual "
        "readiness interact rather than operate independently.\n\n"
        "Where the present results diverge from earlier work, the most plausible explanation lies in features specific "
        "to this study's context: the particular stage of adoption reached within the sampled organisations, or "
        "differences in how key constructs were measured relative to prior studies. These are differences of degree "
        "rather than of kind, and they do not undermine the overall pattern so much as refine it."
    ),
)

_DISCUSSION_VARIANTS_TECHNICAL: tuple[str, ...] = (
    (
        "The findings are broadly consistent with the design expectations established in the literature review. "
        "The observed performance characteristics confirm that {topic} is a multidimensional engineering problem shaped by "
        "component selection, control strategy, and the range of operating conditions under which the system is tested.\n\n"
        "Where deviations from prior implementations are observed, these may be attributed to differences in hardware "
        "specification, test environment, or measurement methodology. In particular, the relatively stronger performance "
        "recorded in this study compared to earlier work may reflect refinements made during iterative testing, or the use "
        "of a more tightly controlled set of trial conditions."
    ),
    (
        "On the whole, the observed performance pattern aligns with what the design rationale set out in Chapter 3 would "
        "lead one to expect. {topic} emerges from this evidence as a multidimensional engineering problem — one in which "
        "component selection, control strategy, and operating-condition range interact rather than operate independently.\n\n"
        "Where the present results diverge from earlier implementations, the most plausible explanation lies in features "
        "specific to this study's test setup: refinements introduced during iterative testing, or a more tightly "
        "controlled set of trial conditions than comparable prior work. These are differences of degree rather than of "
        "kind, and they do not undermine the overall pattern so much as refine it."
    ),
)

# Conclusion (Chapter 5) — varied opening framing while preserving the evidentiary claim.
_CONCLUSION_VARIANTS_SURVEY: tuple[str, ...] = (
    (
        "Based on the empirical evidence presented, the study concludes that {topic} represents a significant and measurable influence "
        "on the outcomes under investigation. The data confirm that well-governed, adequately resourced environments yield substantially "
        "stronger results than those characterised by implementation gaps or resource constraints.\n\n"
        "The study's conclusions are grounded in primary evidence and corroborated by the existing literature, lending confidence to "
        "the interpretation that targeted investment in institutional capacity and process design is essential for maximising value."
    ),
    (
        "Taken as a whole, the evidence gathered in this study supports the conclusion that {topic} exerts a measurable "
        "and consistent influence on the outcomes under investigation. Where institutional environments are well governed "
        "and adequately resourced, the results obtained are markedly stronger than in settings marked by implementation "
        "gaps or constrained resources.\n\n"
        "This conclusion does not rest on the present data alone — it is consistent with, and reinforced by, the wider "
        "body of literature reviewed earlier in the dissertation. Together, they support the view that deliberate "
        "investment in institutional capacity and process design is a precondition for realising the full value available "
        "in this domain."
    ),
)

_CONCLUSION_VARIANTS_TECHNICAL: tuple[str, ...] = (
    (
        "Based on the test evidence presented, the study concludes that {topic} can be addressed by a design that delivers measurable, "
        "repeatable performance under the evaluated conditions. The data confirm that careful component selection, calibration, and "
        "control-strategy tuning yield substantially stronger and more consistent results than a naive or untuned implementation.\n\n"
        "The study's conclusions are grounded in primary test data and corroborated by comparable work in the literature, lending "
        "confidence to the interpretation that targeted attention to design detail and systematic testing is essential for achieving "
        "reliable system performance."
    ),
    (
        "Taken as a whole, the test evidence gathered in this study supports the conclusion that {topic} can be addressed "
        "through a design capable of measurable, repeatable performance under the conditions evaluated. Where component "
        "selection, calibration, and control-strategy tuning receive deliberate attention, the results obtained are "
        "markedly stronger and more consistent than those of a naive or untuned implementation.\n\n"
        "This conclusion does not rest on the present data alone — it is consistent with, and reinforced by, comparable "
        "work reviewed earlier in the dissertation. Together, they support the view that careful attention to design "
        "detail and systematic testing is a precondition for achieving reliable system performance."
    ),
)

# Recommendations (Chapter 5) — varied list framing/wording while keeping the same four
# recommendation slots (capacity, governance/strategy, regulatory/practice, future research).
_RECOMMENDATION_VARIANTS_SURVEY: tuple[str, ...] = (
    (
        "Based on the findings, the following recommendations are offered to practitioners, policymakers, and future researchers:\n\n"
        "1. Organisations should invest in structured capacity-building programmes to strengthen readiness for technology adoption.\n"
        "2. Governance frameworks should be reviewed to ensure alignment between strategic objectives and implementation mechanisms.\n"
        "3. Policymakers should develop enabling regulatory environments that incentivise responsible innovation while managing systemic risk.\n"
        "4. Future research should employ longitudinal designs to track outcome trajectories over time and across different institutional contexts."
    ),
    (
        "The evidence generated by this study points toward several concrete actions for practitioners, policymakers, and "
        "future researchers:\n\n"
        "1. Institutional leaders should prioritise structured capacity-building initiatives that directly target the "
        "readiness gaps identified in this study's findings.\n"
        "2. Existing governance frameworks should be reassessed to close the gap between stated strategic objectives and "
        "the mechanisms actually used to implement them.\n"
        "3. Regulators and policymakers should work toward enabling frameworks that reward responsible innovation without "
        "losing sight of systemic risk.\n"
        "4. Subsequent research should adopt longitudinal designs capable of tracking how these outcomes evolve over time "
        "and across differing institutional contexts."
    ),
)

_RECOMMENDATION_VARIANTS_TECHNICAL: tuple[str, ...] = (
    (
        "Based on the findings, the following recommendations are offered to developers, students, and future researchers:\n\n"
        "1. Future builds should invest in higher-precision components or additional calibration where the current design shows the "
        "greatest performance variance.\n"
        "2. The control strategy should be reviewed periodically to ensure it remains well matched to the operating conditions encountered "
        "in deployment.\n"
        "3. Developers should document test protocols and raw performance data in detail to support reproducibility and comparison "
        "with future designs.\n"
        "4. Future work should evaluate the system across a wider range of operating conditions and over longer continuous-operation "
        "periods to assess long-term reliability."
    ),
    (
        "The test evidence generated by this study points toward several concrete actions for developers, students, and "
        "future researchers:\n\n"
        "1. Subsequent builds should prioritise higher-precision components or additional calibration in exactly the "
        "areas where this design showed the greatest performance variance.\n"
        "2. The control strategy should be revisited periodically to confirm it remains well matched as operating "
        "conditions change.\n"
        "3. Developers should record test protocols and raw performance data in enough detail to support independent "
        "reproduction and fair comparison with future designs.\n"
        "4. Future work should extend testing to a wider range of operating conditions and longer continuous-operation "
        "periods to establish long-term reliability."
    ),
)


def _expand_fallback_text(
    body: str,
    topic: str,
    section_title: str,
    subsection: str,
    target_words: int,
    pool: tuple[str, ...] = _FALLBACK_ELABORATIONS,
) -> str:
    """Lightly pad deterministic fallback prose toward target_words with generic elaboration.

    Generic elaboration is a last resort, not a length-filling device. Chasing the full
    target_words gap with templated paragraphs produces subsections that are mostly
    boilerplate (e.g. a 250-word substantive body followed by 8-9 generic filler
    paragraphs) — that reads as padded/hallucinated, not submission-ready. So padding is
    capped at a small, fixed number of paragraphs and at no more words than the real body
    already contributed: undershooting target_words is preferable to diluting the
    subsection with repetitive boilerplate. `pool` lets personal/reflective sections
    (dedication, acknowledgements) pad with appropriately-toned text instead of
    academic-register elaboration.
    """
    base_words = _word_count(body)
    word_ceiling = int(target_words * 0.85)
    max_added_words = min(word_ceiling - base_words, base_words, 260)
    if max_added_words <= 0:
        return body
    # Rotate the pool's starting point per subsection so consecutive subsections (which all
    # draw from the same small pool) don't render the exact same paragraphs in the same
    # order — a contributor to the document reading as templated/repetitive.
    offset = sum(ord(c) for c in subsection) % len(pool)
    rotated = pool[offset:] + pool[:offset]
    max_paragraphs = 3
    parts = [body]
    added_words = 0
    for template in rotated[:max_paragraphs]:
        if added_words >= max_added_words:
            break
        paragraph = template.format(topic=topic, section_title=section_title, subsection=subsection)
        parts.append(paragraph)
        added_words += _word_count(paragraph)
    return "\n\n".join(parts)


def _fallback_subsection_text(
    topic: str,
    section_title: str,
    subsection: str,
    objectives: list[str] | None = None,
    target_words: int | None = None,
    research_design: str = "",
    specific_design: str | None = None,
    sample_size: int | None = None,
) -> str:
    """Return substantive academic fallback text when model generation fails.

    Tries one direct LLM call sized to target_words first; if that fails (or no LLM key
    is configured), falls back to deterministic per-topic text and pads it toward
    target_words so total document length stays consistent with the chapter's target —
    the deterministic branches alone are far too short to hit a 50-100 page dissertation.
    """
    sub_lower = subsection.strip().lower()
    is_personal = "dedication" in sub_lower or "acknowledgement" in sub_lower or "acknowledgment" in sub_lower

    words_request = target_words or 200
    if is_personal:
        prompt = (
            f"Write a brief, sincere '{subsection}' page for a dissertation on '{topic}'. "
            "Use warm, personal, first-person language — NOT academic or scholarly register, and do not "
            "mention literature, theory, or research methodology. "
            f"Produce approximately {words_request} words. "
            "Do NOT include the subsection heading in your response."
        )
    else:
        prompt = (
            f"Write a substantive academic subsection for '{subsection}' "
            f"in a dissertation on '{topic}'. Use formal scholarly language and concrete claims. "
            f"Produce approximately {words_request} words across 2-4 paragraphs. "
            "Do NOT include the subsection heading in your response."
        )
    try:
        candidate = (generate_text(prompt) or "").strip()
        # Reject weak/placeholder outputs and keep fallback quality consistent.
        if len(candidate) >= 140 and "this subsection addresses" not in candidate.lower():
            return candidate
    except Exception:
        pass

    body = _fallback_subsection_body(
        topic, section_title, subsection, objectives, research_design, specific_design, sample_size,
    )
    no_expand = any(k in sub_lower for k in _FALLBACK_NO_EXPAND_KEYWORDS)
    if no_expand or not target_words:
        return body
    if is_personal:
        pool = _FALLBACK_PERSONAL_ELABORATIONS
    elif _uses_human_respondents(topic, "", objectives):
        pool = _FALLBACK_ELABORATIONS
    else:
        pool = _FALLBACK_TECHNICAL_ELABORATIONS
    return _expand_fallback_text(body, topic, section_title, subsection, target_words, pool=pool)


def _fallback_subsection_body(
    topic: str,
    section_title: str,
    subsection: str,
    objectives: list[str] | None = None,
    research_design: str = "",
    specific_design: str | None = None,
    sample_size: int | None = None,
) -> str:
    """Deterministic, per-topic placeholder text for one dissertation subsection."""
    survey_based = _uses_human_respondents(topic, "", objectives)
    is_qualitative = research_design == "qualitative"
    is_mixed = research_design == "mixed"
    specific_label = _SPECIFIC_METHODOLOGY_LABELS.get(specific_design or "", "")
    sub = subsection.strip()
    sub_lower = sub.lower()
    sec = section_title.strip().lower()

    # ── Preliminary pages ───────────────────────────────────────────────────
    if "abstract" in sub_lower:
        if survey_based:
            if is_qualitative:
                return (
                    f"This study examines {topic}, with a focus on the meanings, experiences, and perspectives that "
                    "shape outcomes within the selected context. A qualitative research design was adopted, drawing on "
                    "primary data collected from a purposively selected group of participants.\n\n"
                    "Findings indicate recurring themes and patterns across participant accounts, with implications for "
                    "practice, policy, and future research. The study concludes by offering targeted recommendations and "
                    "identifying areas warranting further investigation."
                )
            if is_mixed:
                return (
                    f"This study examines {topic}, combining a quantitative strand testing relationships between key "
                    "variables with a qualitative strand exploring the meanings and experiences underlying those "
                    "relationships. A mixed-methods research design was adopted, drawing on survey data from a "
                    "statistically determined sample alongside semi-structured interviews with a purposively selected "
                    "subset of participants.\n\n"
                    "Quantitative findings indicate significant relationships between the study variables, while "
                    "qualitative findings surface recurring themes that help explain those relationships in context. "
                    "The study concludes by integrating both strands of evidence into targeted recommendations and "
                    "identifying areas warranting further investigation."
                )
            return (
                f"This study examines {topic}, with a focus on the mechanisms through which key variables interact "
                "to produce observed outcomes within the selected context. A quantitative research design was adopted, "
                "drawing on primary data collected from a purposive sample of respondents.\n\n"
                "Findings indicate significant relationships between the study variables, with implications for practice, "
                "policy, and future research. The study concludes by offering targeted recommendations and identifying "
                "areas warranting further investigation."
            )
        return (
            f"This study examines {topic}, focusing on the design, implementation, and evaluation of the system or "
            "process under investigation. A design-and-test research approach was adopted, combining technical "
            "development with structured experimental evaluation against defined performance criteria.\n\n"
            "Results indicate that the developed solution meets its core performance targets, with measurable gains "
            "over the baseline or prior approach considered. The study concludes by outlining the practical and "
            "technical implications of these results and identifying directions for further development."
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
        if survey_based:
            return (
                f"Figure 1: Conceptual Framework for {topic} ............. 12\n"
                "Figure 2: Research Design Overview ..................... 34\n"
                "Figure 3: Respondent Distribution by Category .......... 52\n"
                "Figure 4: Key Findings Summary ......................... 55\n"
                "Figure 5: Comparative Analysis Chart ................... 60"
            )
        return (
            f"Figure 1: System/Conceptual Architecture for {topic} ... 12\n"
            "Figure 2: Research and Development Design Overview ..... 34\n"
            "Figure 3: Performance Results Across Test Scenarios .... 52\n"
            "Figure 4: Key Findings Summary .......................... 55\n"
            "Figure 5: Comparative Analysis Chart .................... 60"
        )
    if "list of tables" in sub_lower:
        if survey_based:
            return (
                "Table 1: Summary of Reviewed Studies ................... 22\n"
                "Table 2: Demographic Profile of Respondents ............ 51\n"
                "Table 3: Descriptive Statistics for Key Variables ...... 54\n"
                "Table 4: Correlation Matrix ............................ 57\n"
                "Table 5: Regression Analysis Results ................... 62"
            )
        return (
            "Table 1: Summary of Reviewed Studies ................... 22\n"
            "Table 2: System/Test Configuration Summary ............. 51\n"
            "Table 3: Descriptive Statistics for Key Performance Metrics .. 54\n"
            "Table 4: Comparative Performance Matrix ................. 57\n"
            "Table 5: Statistical Test Results ....................... 62"
        )
    if "abbreviation" in sub_lower or "acronym" in sub_lower:
        if survey_based and not is_qualitative:
            lines = [
                "n   — Sample Size",
                "SD  — Standard Deviation",
                "df  — Degrees of Freedom",
                "CI  — Confidence Interval",
                "SPSS — Statistical Package for the Social Sciences",
                "IRB — Institutional Review Board",
            ]
        elif survey_based:
            lines = [
                "IRB — Institutional Review Board",
                "n   — Number of Participants",
            ]
        else:
            lines = [
                "n   — Number of Trials",
                "SD  — Standard Deviation",
                "CI  — Confidence Interval",
            ]
        return "\n".join(lines)

    if "hypoth" in sub_lower:
        if objectives:
            lines = []
            for i, obj in enumerate(objectives[:3], start=1):
                stem = re.sub(r"^to\s+\w+\s+", "", obj.strip().rstrip("."), count=1, flags=re.IGNORECASE)
                stem = stem or obj.strip().rstrip(".")
                lines.append(
                    f"{i}. H0: There is no statistically significant relationship between {stem} and the outcomes examined in this study.\n"
                    f"   H1: There is a statistically significant relationship between {stem} and the outcomes examined in this study."
                )
            return "\n".join(lines)
        return (
            f"1. H0: {topic[:1].upper()}{topic[1:]} has no statistically significant effect on the outcomes examined in this study.\n"
            f"   H1: {topic[:1].upper()}{topic[1:]} has a statistically significant positive effect on the outcomes examined in this study.\n"
            f"2. H0: There is no statistically significant relationship between the key variables associated with {topic}.\n"
            f"   H1: There is a statistically significant relationship between the key variables associated with {topic}."
        )

    # ── Chapter 1 Introduction subsections ─────────────────────────────────
    if "background" in sub_lower and ("chapter 1" in sec or "introduction" in sec):
        pool = _BACKGROUND_VARIANTS_SURVEY if survey_based else _BACKGROUND_VARIANTS_TECHNICAL
        return _seeded_pick(topic, pool).format(topic=topic)

    if "problem" in sub_lower and ("chapter 1" in sec or "introduction" in sec):
        pool = _PROBLEM_VARIANTS_SURVEY if survey_based else _PROBLEM_VARIANTS_TECHNICAL
        return _seeded_pick(f"{topic}|problem", pool).format(topic=topic)

    if "significance" in sub_lower:
        if survey_based:
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
        return (
            f"The significance of this study lies in its potential to generate a documented, tested reference design for "
            f"{topic}, contributing a working implementation and a transparent evaluation record to a field where many "
            "designs are reported without sufficient performance detail to be reproduced or compared.\n\n"
            "At the practical level, the findings offer a concrete starting point for engineers, students, and hobbyists "
            "seeking to build on or improve the design. At the technical level, the documented test results provide a "
            "performance baseline against which future modifications — to hardware, algorithms, or operating conditions — "
            "can be meaningfully compared."
        )

    if "scope" in sub_lower or "delimitation" in sub_lower:
        if survey_based:
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
        return (
            "The scope of this study is defined by three primary parameters: technical focus, test boundary, and "
            f"evaluation horizon. Technically, the study concentrates on the design and performance of the system "
            f"developed for {topic}, and excludes related but distinct design problems that fall outside its "
            "intended function. The test boundary is defined by the specific hardware configuration, test "
            "environment, and range of operating conditions under which the system is evaluated. Temporally, "
            "the study reports performance observed during the testing period, rather than long-term field "
            "deployment.\n\n"
            "These delimitations are deliberate design choices made to maintain technical focus, ensure feasibility "
            "within available time and equipment, and enhance the internal validity of the reported results. They "
            "do not represent shortcomings but rather principled boundaries that strengthen the coherence of the "
            "study's evaluation."
        )

    if "definition" in sub_lower or "key term" in sub_lower:
        if survey_based:
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
        return (
            "For the purpose of this study, key terms are operationally defined as follows:\n\n"
            "System Performance: The measured behaviour of the developed solution against defined technical "
            "criteria, such as accuracy, speed, throughput, or efficiency, recorded under controlled test conditions.\n\n"
            "Reliability: The consistency with which the system reproduces a given level of performance across "
            "repeated trials and varied operating conditions, expressed in terms of failure rate or variance in outcomes.\n\n"
            "Test Configuration: A specific, documented combination of hardware, software, and environmental settings "
            "under which the system is evaluated, used as the unit of comparison across trials.\n\n"
            "These definitions align with established usage in the relevant engineering literature and provide a "
            "consistent conceptual basis for measurement, analysis, and interpretation throughout the study."
        )

    if "research objective" in sub_lower or "research question" in sub_lower:
        is_question = "question" in sub_lower
        items = list(objectives[:6]) if objectives else [
            f"To determine the current state of {topic}.",
            f"To examine the key factors influencing {topic}.",
            f"To evaluate the relationship between {topic} and related outcomes in the study context.",
        ]
        if is_question:
            return "\n".join(f"{i}. {_objective_to_question(o, topic)}" for i, o in enumerate(items, start=1))
        return "\n".join(f"{i}. {o.strip().rstrip('.')}." for i, o in enumerate(items, start=1))

    if "summary" in sub_lower and ("chapter 1" in sec or "introduction" in sec):
        return (
            f"This chapter has introduced the study of {topic}, presenting the background and context that motivate "
            "the research, the problem the study addresses, and the research objectives and questions that guide it. "
            "It also outlined the significance of the study, defined its scope and delimitations, and clarified the "
            "key terms used throughout the dissertation.\n\n"
            "Building on this foundation, the next chapter reviews the relevant literature, situating the present "
            "study within existing theoretical and empirical work."
        )

    if "chapter 2" in sec or "literature review" in sec:
        if "introduction" in sub_lower:
            return (
                f"This chapter reviews the existing body of scholarship relevant to {topic}, situating the present study "
                "within established theoretical and empirical traditions. The review is organised around the conceptual "
                "review, the theoretical framework guiding the study, the empirical literature on the subject, and the "
                "research gap that this study addresses.\n\n"
                "The chapter begins by clarifying the core concepts central to the study, before moving to the theories "
                "that explain the relationships among the key variables. It then synthesises empirical findings from "
                "global, regional, and sector-specific studies, before concluding with an identification of gaps in "
                "current knowledge that justify the present inquiry."
            )
        if any(k in sub_lower for k in ["conceptual", "core concept", "dimension", "relationships between variables", "definition and conceptualisation"]):
            return (
                f"The conceptual review clarifies the key terms and constructs underpinning {topic}, establishing a shared "
                "vocabulary for the analysis that follows. Each construct is defined with reference to its dominant usage "
                "in the academic literature, and its dimensions, indicators, and measurement approaches are outlined "
                "to support the operationalisation adopted in Chapter 3.\n\n"
                "The review further maps the conceptual relationships among the study's key variables, illustrating how "
                "they are theorised to interact. This conceptual groundwork provides the basis for the theoretical "
                "framework and the empirical hypotheses examined later in the study."
            )
        if "theoretical" in sub_lower or "theory" in sub_lower or "theories" in sub_lower:
            if survey_based:
                return (
                    f"This section presents the theoretical foundations relevant to {topic}. Several theories offer "
                    "complementary lenses for understanding the relationships under investigation, including classical "
                    "and contemporary perspectives drawn from the relevant disciplinary literature.\n\n"
                    "Among the foundational theories considered, institutional theory explains how external normative and "
                    "coercive pressures shape organisational and individual behaviour, while resource-based perspectives "
                    "highlight the role of internal capabilities and resource endowments in determining outcomes. "
                    "Contemporary and emerging theoretical contributions extend these classical accounts to contexts "
                    "characterised by rapid technological, environmental, or socioeconomic change. The theory judged most "
                    "applicable to this study is justified on the grounds of its explanatory power for the specific "
                    "relationships and context examined here."
                )
            return (
                f"This section presents the theoretical and engineering foundations relevant to {topic}. Several "
                "established frameworks offer complementary lenses for understanding the system under investigation, "
                "including control-theoretic, systems-engineering, and signal-processing perspectives drawn from the "
                "relevant technical literature.\n\n"
                "Among the foundational frameworks considered, feedback control theory explains how a system senses "
                "its environment or internal state and adjusts its behaviour to track a desired target or avoid an "
                "undesired condition, while systems-engineering perspectives highlight the role of subsystem "
                "decomposition, interface design, and integration testing in determining overall performance. Where "
                "applicable, sensor and signal-processing models account for how raw measurements are filtered, fused, "
                "or interpreted before informing system decisions. The framework judged most applicable to this study "
                "is justified on the grounds of its explanatory power for the specific design and performance "
                "characteristics examined here."
            )
        if "global evidence" in sub_lower or ("empirical" in sub_lower and "global" in sub_lower):
            return (
                f"Global empirical evidence on {topic} reveals broadly consistent patterns across diverse settings, "
                "albeit with notable variation in magnitude and mechanism. Cross-country and cross-sectoral studies "
                "report measurable associations between the key variables identified in this study, with effect sizes "
                "shaped by contextual factors such as institutional maturity, resource availability, and policy "
                "environment.\n\n"
                "Taken together, the global evidence base establishes a strong prima facie case for the relationships "
                "this study investigates, while also highlighting the need for the context-specific evidence that "
                "this study seeks to generate."
            )
        if "developed econom" in sub_lower:
            return (
                f"Evidence from developed economies demonstrates relatively mature engagement with {topic}, supported "
                "by well-established institutional, regulatory, and infrastructural conditions. Studies conducted in "
                "these contexts often report stronger and more consistent outcomes, reflecting greater resource "
                "availability and longer histories of implementation.\n\n"
                "Nonetheless, even within developed-economy contexts, researchers note that outcomes vary by sector "
                "and by the quality of implementation, underscoring that institutional maturity alone does not "
                "guarantee favourable results."
            )
        if "emerging econom" in sub_lower:
            return (
                f"Studies situated in emerging economies provide important comparative evidence on {topic}, often "
                "revealing different constraints and enablers than those documented in developed-economy research. "
                "Infrastructural gaps, regulatory transition, and resource limitations frequently moderate the "
                "strength of observed relationships in these settings.\n\n"
                "Despite these constraints, emerging-economy studies report meaningful progress and, in some cases, "
                "outcomes that rival those of more developed contexts, particularly where targeted policy support "
                "and capacity-building interventions have been implemented."
            )
        if "developing econom" in sub_lower or "africa" in sub_lower:
            return (
                f"Evidence from developing economies and the African context highlights both the opportunities and "
                f"the structural challenges associated with {topic}. Studies in this context frequently emphasise "
                "infrastructural deficits, financing constraints, and capacity limitations as factors that condition "
                "the pace and depth of observed outcomes.\n\n"
                "At the same time, this literature documents innovative, context-adapted approaches that have achieved "
                "meaningful results despite resource constraints, offering valuable lessons for the present study's "
                "context and underscoring the importance of locally grounded evidence."
            )
        if "sectoral" in sub_lower or "industry" in sub_lower:
            return (
                f"Sectoral and industry-specific studies on {topic} reveal that outcomes are not uniform across "
                "industries, but instead reflect sector-specific operating conditions, regulatory regimes, and "
                "stakeholder priorities. Comparative sectoral analysis helps to clarify which of these conditions "
                "are most consequential for the relationships examined in this study.\n\n"
                "This sector-sensitive evidence base informs the contextual scope of the present study and supports "
                "the selection of an appropriate analytical frame for interpreting the primary data collected."
            )
        if "synthesis" in sub_lower or "critical appraisal" in sub_lower:
            return (
                f"Synthesising the empirical literature reviewed above, the evidence on {topic} is consistent in "
                "direction but heterogeneous in magnitude, with methodological differences accounting for a "
                "substantial share of the variation reported across studies. Many existing studies rely on "
                "secondary data, cross-sectional designs, or narrow sectoral samples, limiting the generalisability "
                "of their conclusions.\n\n"
                "This critical appraisal identifies methodological rigor, contextual specificity, and triangulation "
                "of data sources as priorities for closing the gaps identified, directly motivating the design "
                "adopted in the present study."
            )
        if "research gap" in sub_lower or "gap" in sub_lower:
            return (
                f"Despite the volume of literature on {topic}, this review identifies a clear and persistent gap: "
                "existing studies offer limited context-specific, primary evidence that directly links the key "
                "variables examined in this study to measurable outcomes within the present research setting. "
                "Much of the available evidence is either generalised across dissimilar contexts or focused on "
                "secondary data unsuited to establishing the specific relationships of interest here.\n\n"
                "The present study addresses this gap by generating original primary data within a clearly defined "
                "context, thereby contributing context-grounded evidence that extends and refines the existing "
                "body of knowledge."
            )
        if "summary" in sub_lower:
            return (
                f"This chapter has reviewed the conceptual, theoretical, and empirical literature relevant to "
                f"{topic}. It clarified the key constructs, outlined the theoretical perspectives guiding the study, "
                "synthesised empirical evidence across global, regional, and sectoral contexts, and identified the "
                "research gap that justifies the present inquiry.\n\n"
                "Building on this foundation, the next chapter presents the research methodology adopted to "
                "generate primary evidence addressing the identified gap."
            )
        return (
            f"This subsection contributes to the broader literature review on {topic}, situating the discussion "
            "within the relevant theoretical and empirical context established elsewhere in this chapter. "
            "It draws on the conceptual and empirical foundations introduced earlier to extend the analysis "
            "to the specific dimension addressed here.\n\n"
            "The discussion remains anchored in the study's research objectives, ensuring that the literature "
            "reviewed directly supports the analytical framework developed for subsequent chapters."
        )

    if "chapter 3" in sec or "methodology" in sec:
        if survey_based:
            if "introduction" in sub_lower:
                return (
                    f"This chapter describes the methodological approach used to investigate {topic}. It outlines the research "
                    "design, the population and sample, the data collection instruments and procedures, and the data analysis "
                    "techniques used to address the research objectives.\n\n"
                    "The chapter also describes the steps taken to establish the reliability and validity of the data collected, "
                    "and the ethical safeguards applied throughout the research process."
                )
            if "research design" in sub_lower or sub_lower.strip().endswith("design"):
                design_text = _SPECIFIC_DESIGN_TEXT.get(specific_design or "", {}).get("design")
                if design_text:
                    return design_text.format(topic=topic)
                if is_qualitative:
                    return (
                        "A qualitative research design was adopted, consistent with the interpretivist epistemological "
                        "stance that underpins the study. Qualitative approaches are well suited to the research "
                        "objectives because they enable an in-depth exploration of participants' experiences and "
                        "perspectives, surfacing meaning that structured numerical instruments would not capture.\n\n"
                        "The use of semi-structured engagement with participants further allows the inquiry to remain "
                        "responsive to emerging issues, while still being guided by the research objectives set out "
                        "in Chapter 1."
                    )
                if is_mixed:
                    return (
                        "A mixed-methods research design was adopted, combining a quantitative strand with a qualitative "
                        "strand within a single study. This explanatory sequential approach was selected because the "
                        "research objectives require both the statistical testing of hypothesised relationships and an "
                        "in-depth understanding of the experiences and perspectives underlying those relationships.\n\n"
                        "The quantitative strand, conducted first, establishes the strength and direction of relationships "
                        "between the study variables using a structured instrument; the qualitative strand then builds on "
                        "these results to explore the reasons behind them through semi-structured engagement with a "
                        "purposively selected subset of participants. Integrating both strands provides a more complete "
                        "account than either approach could offer in isolation."
                    )
                return (
                    "A quantitative research design was adopted, consistent with the positivist epistemological stance that "
                    "underpins the study. Quantitative approaches are well suited to the research objectives because they "
                    "enable the statistical testing of hypothesised relationships and support the generation of generalisable "
                    "findings.\n\n"
                    "The use of structured instruments further ensures that data collection is standardised, reducing the risk "
                    "of interviewer bias and enabling comparative analysis across respondent groups."
                )
            if "population" in sub_lower:
                return (
                    f"The target population consists of individuals or groups with direct relevance to {topic}, possessing "
                    "the knowledge or experience necessary to provide informed responses to the research instrument. Defining "
                    "the population clearly is a necessary precondition for selecting a sample that adequately represents the "
                    "group of interest.\n\n"
                    "Eligibility for inclusion in the population was based on criteria directly tied to the research objectives, "
                    "ensuring that the data collected speak meaningfully to the questions under investigation."
                )
            if "sampl" in sub_lower:
                seed = sum(ord(c) for c in topic)
                sampling_text = _SPECIFIC_DESIGN_TEXT.get(specific_design or "", {}).get("sampling")
                if sampling_text:
                    return sampling_text.format(topic=topic)
                if is_qualitative:
                    sample_n = 20 + (seed % 16)
                    return (
                        "A purposive sample was drawn from the target population, with participants selected on the "
                        "basis of their relevant knowledge or lived experience, in line with established qualitative "
                        "sampling conventions. Recruitment continued until thematic saturation was observed, yielding "
                        f"a final sample size of {sample_n} participants.\n\n"
                        "This sampling approach prioritises depth of insight over statistical representativeness, "
                        "balancing the need for rich, detailed accounts against the practical constraints of time and "
                        "resource availability."
                    )
                if is_mixed:
                    qual_n = 8 + (seed % 8)
                    quant_n = 80 + (seed % 161)
                    return (
                        "A two-phase sampling strategy was used, consistent with the mixed-methods design. For the "
                        "quantitative strand, a stratified random sample was drawn from the target population, with "
                        "sample size determined using Krejcie and Morgan's (1970) table for determining sample size "
                        f"from a known population, yielding a sample size of {quant_n} respondents sufficient to achieve "
                        "a 95% confidence level with a ±5% margin of error.\n\n"
                        f"For the qualitative strand, a purposive subsample of {qual_n} participants was then drawn from "
                        "respondents who completed the quantitative instrument, selected to represent a range of "
                        "response patterns observed in the quantitative data, in order to explore the reasons behind "
                        "the statistical relationships identified."
                    )
                sample_n = 80 + (seed % 161)
                return (
                    "A stratified random sample was drawn from the target population to ensure that respondents possess "
                    "the characteristics necessary to provide informed responses. Sample size was determined using "
                    f"Krejcie and Morgan's (1970) table for determining sample size from a known population, yielding "
                    f"a sample size of {sample_n} respondents, sufficient to achieve a 95% confidence level with "
                    "a ±5% margin of error.\n\n"
                    "The stratified approach was selected to ensure proportional representation across key subgroups "
                    "while balancing representativeness against the practical constraints of time and resource "
                    "availability, while still supporting the statistical procedures planned for Chapter 4."
                )
            if "collection" in sub_lower:
                collection_text = _SPECIFIC_DESIGN_TEXT.get(specific_design or "", {}).get("collection")
                if collection_text:
                    return collection_text.format(topic=topic)
                if is_qualitative:
                    return (
                        "Data were collected through semi-structured interviews guided by an interview protocol "
                        "developed from the research objectives, allowing participants to elaborate on their "
                        "experiences and perspectives in their own words. Interviews were audio-recorded with "
                        "participant consent and transcribed verbatim for analysis.\n\n"
                        "Interviews were conducted in a setting convenient to each participant, with sufficient "
                        "flexibility in the protocol to allow follow-up questions and probe emerging issues not "
                        "anticipated in the original interview guide."
                    )
                if is_mixed:
                    return (
                        "Data were collected in two phases, consistent with the mixed-methods design. In the first "
                        "phase, a structured questionnaire comprising closed-ended items measured on a five-point "
                        "Likert scale (1 = Strongly Disagree to 5 = Strongly Agree) was administered to the "
                        "quantitative sample, with items developed from validated instruments in the existing "
                        "literature.\n\n"
                        "In the second phase, semi-structured interviews were conducted with the qualitative "
                        "subsample, guided by an interview protocol informed by the quantitative results, allowing "
                        "participants to elaborate on the reasons behind their survey responses. Interviews were "
                        "audio-recorded with consent and transcribed verbatim for analysis."
                    )
                return (
                    "Data were collected via a structured questionnaire comprising closed-ended items measured on a "
                    "five-point Likert scale (1 = Strongly Disagree to 5 = Strongly Agree). The instrument was developed "
                    "from validated items in the existing literature, adapted to the specific context of this study.\n\n"
                    "Questionnaires were administered to respondents with clear completion instructions and a defined "
                    "window for completion and return."
                )
            if "analysis" in sub_lower or "analytic" in sub_lower:
                analysis_text = _SPECIFIC_DESIGN_TEXT.get(specific_design or "", {}).get("analysis")
                if analysis_text:
                    return analysis_text.format(topic=topic)
                if is_qualitative:
                    return (
                        "Interview transcripts were analysed using thematic analysis, following an iterative process "
                        "of familiarisation, coding, and theme development. Codes were grouped into broader themes "
                        "that were reviewed and refined against the original transcripts to ensure they accurately "
                        "represented participants' accounts.\n\n"
                        "Results of this analysis are presented and interpreted in Chapter 4 in relation to the study's "
                        "research questions, with representative excerpts used to illustrate each theme."
                    )
                if is_mixed:
                    return (
                        "Quantitative responses were coded, screened for missing or inconsistent data, and analysed "
                        "using SPSS v.28. Descriptive statistics were used to summarise the sample and key variables, "
                        "while inferential tests — including Pearson correlation and multiple linear regression — were "
                        "used to test the relationships specified in the research objectives.\n\n"
                        "Interview transcripts from the qualitative strand were analysed separately using thematic "
                        "analysis, following an iterative process of familiarisation, coding, and theme development. "
                        "The two sets of results were then merged at the interpretation stage, with qualitative themes "
                        "used to explain and contextualise the statistical relationships identified, consistent with "
                        "the explanatory sequential mixed-methods design."
                    )
                return (
                    "Completed responses were coded, screened for missing or inconsistent data, and analysed using SPSS "
                    "v.28. Descriptive statistics were used to summarise the sample and key variables, while inferential "
                    "tests — including Pearson correlation and multiple linear regression — were used to test the "
                    "relationships specified in the research objectives.\n\n"
                    "Results of this analysis are presented and interpreted in Chapter 4 in relation to the study's "
                    "research questions."
                )
            if "reliabil" in sub_lower or "validit" in sub_lower:
                if is_qualitative:
                    return (
                        "Trustworthiness of the findings was established using Lincoln and Guba's (1985) criteria of "
                        "credibility, transferability, dependability, and confirmability, in place of the reliability "
                        "and validity criteria associated with quantitative research. Credibility was supported "
                        "through member checking and prolonged engagement with participants.\n\n"
                        "Dependability and confirmability were supported by maintaining a clear audit trail of coding "
                        "decisions, while thick description of the context and findings supports the transferability "
                        "of conclusions to similar settings."
                    )
                if is_mixed:
                    return (
                        "For the quantitative strand, reliability of the measurement instrument was assessed using "
                        "Cronbach's Alpha, with values above 0.70 accepted as indicating satisfactory internal "
                        "consistency; the reliability table presented later in this chapter reports the alpha values "
                        "obtained for each construct. Validity was established through content review by domain "
                        "experts and, where applicable, confirmatory factor analysis.\n\n"
                        "For the qualitative strand, trustworthiness was established using Lincoln and Guba's (1985) "
                        "criteria of credibility, transferability, dependability, and confirmability, supported "
                        "through member checking and a clear audit trail of coding decisions. Triangulating both "
                        "strands further strengthens confidence in the overall study findings."
                    )
                return (
                    "Reliability of the measurement instrument was assessed using Cronbach's Alpha, with values above "
                    "0.70 accepted as indicating satisfactory internal consistency; the reliability table presented "
                    "later in this chapter reports the alpha values obtained for each construct. Validity was "
                    "established through content review by domain experts and, where applicable, confirmatory factor "
                    "analysis.\n\n"
                    "Pilot testing of the instrument was conducted prior to full administration to identify and resolve "
                    "any ambiguity in item wording."
                )
            if "ethic" in sub_lower:
                return (
                    "Ethical clearance was obtained from the relevant institutional review board prior to data "
                    "collection. Participation was voluntary, and informed consent was obtained from all respondents "
                    "after they were briefed on the purpose of the study, their right to withdraw, and the measures "
                    "taken to protect their confidentiality.\n\n"
                    "All data were stored securely and reported only in aggregate form, ensuring that no individual "
                    "respondent can be identified from the results presented in this study."
                )
            if "summary" in sub_lower:
                return (
                    "This chapter has described the research design, population and sample, data collection "
                    "instrument, and analysis techniques used to address the research objectives, along with the "
                    "measures taken to establish reliability, validity, and ethical compliance.\n\n"
                    "The next chapter presents and discusses the findings obtained using this methodology, organised "
                    "around the research questions set out in Chapter 1."
                )
            design_label = "qualitative" if is_qualitative else ("mixed-methods" if is_mixed else "quantitative")
            return (
                f"This section describes a further methodological aspect of the study on {topic}, situated within "
                f"the overall {design_label} research design adopted for this dissertation and consistent with the "
                "population, sampling, and data collection procedures described elsewhere in this chapter."
            )

        if "introduction" in sub_lower:
            return (
                f"This chapter describes the methodological approach used to design, implement, and evaluate the "
                f"system developed for {topic}. It outlines the research design, the development and test "
                "environment, the data collection and analysis procedures, and the quality-assurance measures "
                "applied throughout the study.\n\n"
                "The chapter is organised to give a transparent, reproducible account of how the system was built "
                "and tested, so that the results reported in Chapter 4 can be properly interpreted and, where "
                "appropriate, replicated."
            )
        if "research design" in sub_lower or sub_lower.strip().endswith("design"):
            return (
                "A design-and-build, experimental research approach was adopted, consistent with the engineering "
                "epistemological stance that underpins the study. This approach is well suited to the research "
                "objectives because it enables iterative prototyping, controlled performance testing, and the "
                "generation of reproducible, quantifiable results.\n\n"
                "The research proceeded through distinct phases — requirements definition, design and "
                "implementation, integration, and performance testing — with each phase informing refinements to "
                "the next. This iterative structure allowed design decisions to be evaluated empirically rather "
                "than assumed."
            )
        if "population" in sub_lower:
            return (
                f"In the absence of human survey respondents, the population relevant to this study is the full "
                f"range of operating conditions, test scenarios, and configurations under which the system "
                f"developed for {topic} could plausibly be deployed. This includes variation in relevant "
                "environmental factors, input conditions, and operational parameters.\n\n"
                "From this population, a representative subset of test scenarios and trial conditions was "
                "selected for detailed evaluation, chosen to cover both typical and edge-case operating "
                "conditions relevant to the research objectives."
            )
        if "sampl" in sub_lower:
            return (
                "Rather than a respondent sample, this study relies on a defined set of test trials and "
                "configurations to establish statistical confidence in the reported results. The number of trials "
                "per test scenario was selected to be large enough to characterise normal variability in system "
                "performance while remaining feasible within the available time and resources.\n\n"
                "Test configurations were selected purposively to exercise the system across the range of "
                "conditions identified in the target population above, ensuring that both common and boundary "
                "cases are represented in the evaluation."
            )
        if "collection" in sub_lower:
            return (
                "Performance data were collected directly from the system under test using instrumentation and "
                "logging appropriate to the metrics of interest (e.g. timing measurements, sensor logs, output "
                "accuracy checks). Each trial followed a standardised test protocol to ensure that conditions were "
                "comparable across runs.\n\n"
                "Where applicable, reference or ground-truth values were established in advance (e.g. via manual "
                "measurement or a known benchmark) so that system output could be compared against an objective "
                "standard rather than relying on subjective judgement."
            )
        if "analysis" in sub_lower or "analytic" in sub_lower:
            return (
                "Collected performance data were analysed using descriptive statistics (means, ranges, standard "
                "deviations) to characterise typical system behaviour, and comparative analysis to assess "
                "differences between design configurations or against baseline performance. Where appropriate, "
                "inferential tests were used to determine whether observed differences were statistically "
                "meaningful rather than attributable to chance variation.\n\n"
                "Results are presented and interpreted in Chapter 4 in relation to the performance thresholds and "
                "research objectives established earlier in this study."
            )
        if "reliabil" in sub_lower or "validit" in sub_lower:
            return (
                "Reliability in this context refers to the repeatability of measurements across trials, while "
                "validity refers to the extent to which the test conditions and metrics genuinely reflect the "
                "performance characteristics the study seeks to evaluate. The reliability table presented later "
                "in this chapter reports trial-to-trial measurement consistency.\n\n"
                "Validity was supported through calibration of measurement instruments, the use of recognised "
                "benchmarks or reference cases where available, and clear documentation of the test protocol so "
                "that results can be independently verified."
            )
        if "ethic" in sub_lower:
            return (
                "Although this study does not involve human participants, ethical practice remains relevant to "
                "the responsible conduct of engineering research. This includes accurate and honest reporting of "
                "results (including limitations and negative findings), proper attribution of any third-party "
                "components, designs, or datasets used, and consideration of the safety implications of the "
                "developed system, particularly where it interacts with people or shared environments.\n\n"
                "Where the system has potential safety, privacy, or environmental implications, these are "
                "identified and discussed so that the findings can be applied responsibly."
            )
        if "summary" in sub_lower:
            return (
                f"This chapter has described the research design, test scenarios and configurations, data "
                f"collection procedures, and analysis techniques used to evaluate the system developed for "
                f"{topic}, along with the measures taken to support the reliability, validity, and ethical "
                "conduct of the study.\n\n"
                "The next chapter presents and discusses the results obtained using this methodology, organised "
                "around the research objectives set out in Chapter 1."
            )
        return (
            f"This section describes a further methodological aspect of the system developed and evaluated for "
            f"{topic}, situated within the overall design-and-build, experimental research approach adopted for "
            "this dissertation and consistent with the test scenarios, data collection, and analysis procedures "
            "described elsewhere in this chapter."
        )

    if "chapter 4" in sec or "results" in sec or "discussion" in sec:
        if "introduction" in sub_lower or "overview" in sub_lower:
            if is_qualitative:
                return (
                    f"This chapter presents the findings of the study on {topic}. "
                    "Results are organised thematically, in accordance with the research objectives and questions set out in "
                    "Chapter 1. Themes identified through the analysis described in Chapter 3 are presented and discussed in "
                    "relation to the theoretical and empirical literature reviewed in Chapter 2.\n\n"
                    "The presentation follows a structured sequence: each theme is introduced, illustrated with representative "
                    "excerpts from participant accounts, and then discussed in relation to the existing body of knowledge."
                )
            if is_mixed:
                return (
                    f"This chapter presents the findings of the study on {topic}, drawing on both the quantitative and "
                    "qualitative strands of the mixed-methods design. Results are organised in accordance with the "
                    "research objectives and questions set out in Chapter 1, with quantitative results reported first "
                    "using descriptive and inferential statistics, followed by the qualitative themes that help explain "
                    "those statistical results.\n\n"
                    "The presentation follows a structured sequence: descriptive and inferential statistics are reported "
                    "first, followed by the qualitative themes identified through interview analysis, and then an "
                    "integrated discussion that triangulates both strands within the existing body of knowledge."
                )
            return (
                f"This chapter presents the empirical findings of the study on {topic}. "
                "Results are organised in accordance with the research objectives and questions set out in Chapter 1. "
                "Quantitative data are reported using descriptive and inferential statistics, and all findings are discussed "
                "in relation to the theoretical and empirical literature reviewed in Chapter 2.\n\n"
                "The presentation follows a structured sequence: descriptive statistics are reported first, followed by inferential analysis, "
                "and then an integrated discussion that contextualises each major finding within the existing body of knowledge."
            )
        if ("presentation" in sub_lower or "finding" in sub_lower) and "discussion" not in sub_lower:
            if survey_based and is_qualitative:
                return (
                    f"Thematic analysis of the interview data reveals several recurring patterns relating to {topic}. "
                    "Participants consistently emphasised the practical and contextual factors shaping their experiences, "
                    "with several themes echoed across the majority of accounts regardless of individual background.\n\n"
                    "Closer reading of the transcripts further surfaces points of divergence between participants, "
                    "offering a more nuanced picture than a single dominant narrative would suggest. "
                    "These themes provide a basis for the conceptual interpretation offered in the discussion that follows."
                )
            if survey_based and is_mixed:
                if len(objectives or []) >= 2:
                    stats = _regression_model_stats(topic, objectives, research_design, sample_size or 120)
                    inferential_sentence = (
                        "and inferential analysis confirms statistically significant relationships between the "
                        f"primary variables, with the regression model explaining {stats['r_squared'] * 100:.0f}% of "
                        f"the variance in the overall outcome (R² = {stats['r_squared']:.2f}), as detailed in the "
                        "correlation and regression tables presented in this chapter"
                    )
                else:
                    inferential_sentence = (
                        "and inferential analysis confirms a statistically significant relationship between the "
                        "primary variables, as detailed in the correlation table presented in this chapter"
                    )
                return (
                    f"Analysis of the quantitative strand reveals several significant patterns relating to {topic}. "
                    f"Descriptive statistics indicate that the majority of respondents reported moderate to high levels of "
                    f"exposure to the study variables, {inferential_sentence}.\n\n"
                    "Thematic analysis of the qualitative strand surfaces recurring patterns that help explain these "
                    "statistical relationships, with participants consistently emphasising the practical and contextual "
                    "factors underlying their survey responses. Triangulating both strands provides a more complete "
                    "account than either approach could offer in isolation."
                )
            if survey_based:
                if len(objectives or []) >= 2:
                    stats = _regression_model_stats(topic, objectives, research_design, sample_size or 120)
                    inferential_sentence = (
                        "Inferential analysis further confirms statistically significant relationships between the "
                        f"primary variables, with the regression model explaining {stats['r_squared'] * 100:.0f}% of "
                        f"the variance in the overall outcome (R² = {stats['r_squared']:.2f}, Adjusted R² = "
                        f"{stats['adj_r_squared']:.2f}), as detailed in the correlation and regression tables presented "
                        "in this chapter."
                    )
                else:
                    inferential_sentence = (
                        "Inferential analysis further confirms a statistically significant relationship between the "
                        "primary variables, as detailed in the correlation table presented in this chapter."
                    )
                return (
                    f"Analysis of primary data reveals several significant patterns relating to {topic}. "
                    "Descriptive statistics indicate that the majority of respondents reported moderate to high levels of exposure to the study variables, "
                    "with mean scores consistently above the midpoint of the measurement scale. "
                    "These initial findings suggest a broadly positive orientation toward the subject under investigation.\n\n"
                    f"{inferential_sentence} "
                    "These results provide empirical support for the conceptual framework outlined in Chapter 2."
                )
            return (
                f"Evaluation of the developed system reveals several significant patterns relating to {topic}. "
                "Performance measurements indicate that the system consistently met or exceeded the target thresholds "
                "defined in Chapter 3 across the majority of test scenarios, with measured values clustering closely "
                "around the expected performance range.\n\n"
                "Comparative analysis against the baseline configuration further confirms a measurable improvement on "
                "the primary performance metrics, with the developed approach achieving higher accuracy and throughput "
                "while reducing latency and error rate by a meaningful margin across repeated trials. "
                "These results provide empirical support for the design choices outlined in Chapter 3."
            )
        if "discussion" in sub_lower:
            pool = _DISCUSSION_VARIANTS_SURVEY if survey_based else _DISCUSSION_VARIANTS_TECHNICAL
            return _seeded_pick(f"{topic}|discussion", pool).format(topic=topic)
        if survey_based:
            return (
                "The results provide objective-level evidence on the observed patterns, highlighting both dominant trends and areas of divergence across indicators. "
                "Descriptive and comparative interpretation shows that some dimensions record stronger outcomes, while others reveal implementation and performance gaps "
                "that warrant targeted intervention.\n\n"
                "The discussion links these observed patterns to the study context and prior literature, explaining how institutional conditions, process maturity, "
                "and governance quality shape the magnitude and direction of outcomes. This interpretation provides an evidence base for practical recommendations "
                "and for refinement of future inquiry."
            )
        return (
            "The results provide objective-level evidence on the observed performance, highlighting both dominant trends and areas of divergence across test "
            "scenarios. Descriptive and comparative interpretation shows that some configurations record stronger outcomes, while others reveal performance "
            "gaps that warrant targeted refinement.\n\n"
            "The discussion links these observed patterns to the study context and prior implementations, explaining how component selection, test "
            "conditions, and design maturity shape the magnitude and direction of outcomes. This interpretation provides an evidence base for practical "
            "recommendations and for refinement of future development."
        )

    # ── Chapter 5 / Conclusion subsections ────────────────────────────────
    if "chapter 5" in sec or "conclusion" in sec or "recommendation" in sec:
        if "summary of finding" in sub_lower:
            if survey_based:
                if not is_qualitative and len(objectives or []) >= 2:
                    stats = _regression_model_stats(topic, objectives, research_design, sample_size or 120)
                    regression_clause = (
                        f"the regression model presented in Chapter 4 explained {stats['r_squared'] * 100:.0f}% of "
                        f"the variance in the overall outcome (R² = {stats['r_squared']:.2f}, Adjusted R² = "
                        f"{stats['adj_r_squared']:.2f})"
                    )
                else:
                    regression_clause = "a statistically significant positive relationship was confirmed between the independent and dependent variables"
                return (
                    f"The study set out to examine {topic}, guided by three primary research objectives. "
                    f"Analysis of primary data generated the following key findings: first, {regression_clause}; second, contextual enablers — particularly institutional "
                    "readiness and governance quality — moderated the strength of observed effects; and third, respondents in higher-engagement "
                    "categories consistently reported superior outcomes across all measured dimensions.\n\n"
                    "These findings collectively support the study's central argument and align with the theoretical propositions advanced in the conceptual framework."
                )
            return (
                f"The study set out to examine {topic}, guided by three primary research objectives. "
                "Evaluation of the developed system generated the following key findings: first, the system achieved its "
                "target performance criteria across the majority of test scenarios; second, specific design choices — "
                "particularly component selection and configuration tuning — measurably influenced the strength of observed "
                "performance gains; and third, performance remained consistent across repeated trials, indicating a reliable "
                "and reproducible result.\n\n"
                "These findings collectively support the study's central argument and align with the technical propositions "
                "advanced in the design framework presented in Chapter 3."
            )
        if "conclusion" in sub_lower:
            pool = _CONCLUSION_VARIANTS_SURVEY if survey_based else _CONCLUSION_VARIANTS_TECHNICAL
            return _seeded_pick(f"{topic}|conclusion", pool).format(topic=topic)
        if "recommendation" in sub_lower:
            pool = _RECOMMENDATION_VARIANTS_SURVEY if survey_based else _RECOMMENDATION_VARIANTS_TECHNICAL
            return _seeded_pick(f"{topic}|recommendation", pool).format(topic=topic)
        if "limitation" in sub_lower:
            if survey_based:
                return (
                    "This study is subject to several limitations that should be considered when interpreting its findings. "
                    "First, the cross-sectional research design limits causal inference; relationships identified are associative rather than definitively causal. "
                    "Second, the sample was drawn from a specific organisational context, which may limit generalisability to other sectors or regions.\n\n"
                    "Third, self-reported data are susceptible to response bias, despite the use of validated instruments and anonymity assurances. "
                    "Future studies should address these limitations through longitudinal panels, multi-sector samples, and mixed-methods triangulation."
                )
            return (
                "This study is subject to several limitations that should be considered when interpreting its findings. "
                "First, testing was conducted under a defined and necessarily limited set of conditions; performance under conditions outside "
                "this range cannot be assumed without further testing. Second, the evaluation relied on the specific hardware and component "
                "batch used during this study, which may limit generalisability to other units or component revisions.\n\n"
                "Third, measurement is subject to instrument and sensor tolerances, despite efforts to calibrate equipment and repeat trials. "
                "Future studies should address these limitations through testing across a broader range of conditions, multiple hardware units, "
                "and longer-duration trials."
            )
        if "further research" in sub_lower or "future research" in sub_lower:
            if survey_based:
                return (
                    "Several avenues merit further investigation beyond the scope of this study:\n\n"
                    f"1. Longitudinal studies tracking the evolution of {topic} over a five-to-ten-year period would yield valuable insights into causal dynamics.\n"
                    "2. Comparative cross-national research would test whether findings generalise across different regulatory and cultural environments.\n"
                    "3. Qualitative inquiry into lived experiences of practitioners would deepen understanding of the mechanisms and barriers identified here.\n"
                    "4. Studies incorporating objective performance metrics alongside perceptual data would strengthen construct validity."
                )
            return (
                "Several avenues merit further investigation beyond the scope of this study:\n\n"
                f"1. Extended trials evaluating {topic} under a wider range of environmental conditions would clarify the limits of reliable operation.\n"
                "2. Comparative studies against alternative component choices or control strategies would test whether the findings generalise "
                "across different design approaches.\n"
                "3. Long-duration endurance testing would deepen understanding of wear, drift, and failure modes not visible in short trials.\n"
                "4. Studies incorporating additional sensing modalities alongside the current instrumentation would strengthen measurement "
                "robustness and validity."
            )
        if "summary" in sub_lower:
            return (
                f"This chapter has presented the conclusions and recommendations arising from the study of {topic}. "
                "It summarised the key findings in relation to the original research objectives, drew conclusions grounded in the evidence "
                "presented in Chapter 4, and offered recommendations for practice and future research.\n\n"
                "Taken together, the contents of this chapter close the dissertation by linking the study's findings back to its original "
                "purpose and by identifying the contribution made to the wider body of knowledge in this area."
            )

    # ── References / Appendices ─────────────────────────────────────────────
    if "reference" in sub_lower:
        terms = _topic_terms(topic, 4) or [topic]
        field_term = re.sub(r"^(a|an|the)\s+", "", max(terms, key=len), flags=re.IGNORECASE)
        field_label = _truncate_label(field_term, 40)
        t = [_truncate_label(terms[i % len(terms)], 40) for i in range(5)]
        authors = ["A.", "B., & Author, C.", "D.", "E., & Author, F.", "G."]
        return (
            "Automatic source retrieval failed for this document (e.g. no network access), so no verified "
            "citations are available. The entries below are ILLUSTRATIVE PLACEHOLDERS ONLY, with fictitious "
            "author names generated for formatting purposes — they are NOT real sources and MUST be replaced "
            f"with verified literature on {topic} before submission.\n\n"
            f"[Placeholder] Author, {authors[0]} (2021). A review of {t[0]} and its implications for {field_label}. "
            f"Journal of {field_label} Research, 14(2), 45-67.\n"
            f"[Placeholder] Author, {authors[1]} (2022). Methodological approaches to studying {t[1]}. "
            f"International Journal of {field_label} Studies, 9(1), 12-30.\n"
            f"[Placeholder] Author, {authors[2]} (2020). {t[2][:1].upper()}{t[2][1:]}: A systematic review of the empirical literature. "
            f"Annual Review of {field_label}, 6, 101-128.\n"
            f"[Placeholder] Author, {authors[3]} (2023). Contemporary perspectives on {t[3]}. "
            f"{field_label} Quarterly, 18(3), 210-233.\n"
            f"[Placeholder] Author, {authors[4]} (2019). Theoretical foundations of {t[4]}. "
            f"Journal of Applied {field_label}, 11(1), 1-22."
        )
    if "appendix" in sub_lower or "appendices" in sub_lower:
        if survey_based and is_qualitative:
            return (
                "Appendix A: Interview Guide\n"
                "[Semi-structured interview protocol used to guide primary data collection.]\n\n"
                "Appendix B: Ethical Clearance Certificate\n"
                "[Clearance document issued by the institutional ethics review board prior to data collection.]\n\n"
                "Appendix C: Data Collection Authorization Letters\n"
                "[Formal letters of permission from participating organisations authorising access to staff and records.]\n\n"
                "Appendix D: Sample Interview Transcript\n"
                "[An anonymised, representative excerpt from one participant transcript, illustrating the coding process.]\n\n"
                "Appendix E: Informed Consent Form\n"
                "[Consent document provided to all research participants before data collection commenced.]"
            )
        if survey_based and is_mixed:
            return (
                "Appendix A: Research Questionnaire\n"
                "[Survey instrument used for the quantitative strand. Items measured on a five-point Likert scale: 1 = Strongly Disagree to 5 = Strongly Agree.]\n\n"
                "Appendix B: Interview Guide\n"
                "[Semi-structured interview protocol used for the qualitative strand.]\n\n"
                "Appendix C: Ethical Clearance Certificate\n"
                "[Clearance document issued by the institutional ethics review board prior to data collection.]\n\n"
                "Appendix D: Raw Statistical Output\n"
                "[SPSS/R output tables from regression, correlation, and reliability analysis.]\n\n"
                "Appendix E: Sample Interview Transcript\n"
                "[An anonymised, representative excerpt from one participant transcript, illustrating the coding process.]\n\n"
                "Appendix F: Informed Consent Form\n"
                "[Consent document provided to all research participants before data collection commenced.]"
            )
        if survey_based:
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
        return (
            "Appendix A: System Design Specifications\n"
            "[Detailed technical specifications, schematics, or architecture diagrams for the developed system.]\n\n"
            "Appendix B: Source Code / Implementation Listing\n"
            "[Key source code modules, configuration files, or build scripts used in the implementation.]\n\n"
            "Appendix C: Test Protocol and Procedures\n"
            "[Step-by-step test procedures, scenario definitions, and trial conditions used during evaluation.]\n\n"
            "Appendix D: Raw Performance Data\n"
            "[Raw measurement logs and statistical output from performance testing and analysis.]\n\n"
            "Appendix E: Calibration and Validation Records\n"
            "[Records of instrument calibration and validation against reference cases or benchmarks.]"
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
            if _looks_like_meta_commentary(subsection_text):
                raise ValueError("Generated body reads like document meta-commentary, not section prose.")
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


def _subsection_match_range(section_text: str, subsection_query: str) -> tuple[int, int] | None:
    """Locate the (start, end) character range of the heading matching `subsection_query`
    and everything up to the next heading. Returns None if no heading matches."""
    positions = _heading_positions(section_text)
    if not positions:
        return None

    query = (subsection_query or "").lower().strip()
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
    return start, end


def _replace_subsection_if_present(section_text: str, subsection_query: str, new_block: str) -> str | None:
    match_range = _subsection_match_range(section_text, subsection_query)
    if match_range is None:
        return None
    start, end = match_range
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


# ── Table of contents / preliminary-pages assembly ───────────────────────────
# _heading_positions() only matches decimal-numbered headings ("2.1 ...") or
# "Chapter N" lines. The Preliminary Pages front matter instead uses roman-
# numeral headings ("i. Abstract", "ii. Dedication", ...), which need their
# own matcher.
_ROMAN_HEADING_RE = re.compile(r"^([ivxlcdm]+)\.\s+(\S.*)$", re.IGNORECASE)

_WORDS_PER_PAGE = 275  # rough double-spaced academic-page estimate, for ToC page numbers


def _roman_heading_positions(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    positions: list[tuple[int, int, str]] = []
    cursor = 0
    for line in lines:
        stripped = line.strip()
        if _ROMAN_HEADING_RE.match(stripped):
            positions.append((cursor, cursor + len(line), stripped))
        cursor += len(line)
    return positions


def _to_roman(n: int) -> str:
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def _toc_line(title: str, page: str, width: int = 70) -> str:
    """Format one Table-of-Contents row with dot leaders, e.g.
    'Chapter 1: Introduction .................................. 1'."""
    title = title.strip()
    dots_len = max(3, width - len(title) - len(page))
    return f"{title} {'.' * dots_len} {page}"


def _insert_preliminary_page_breaks(content: str) -> str:
    """Insert a [[PAGEBREAK]] marker before every roman-numeral preliminary-page
    item except the first (Abstract, Dedication, Acknowledgements, ...) so each
    renders starting on a fresh page. Idempotent — strips existing markers first
    so it is safe to call again after edits."""
    content = content.replace("[[PAGEBREAK]]\n\n", "").replace("[[PAGEBREAK]]", "")
    positions = _roman_heading_positions(content)
    if len(positions) < 2:
        return content
    rebuilt = content[: positions[0][0]]
    for i, (start, _end, _heading) in enumerate(positions):
        chunk_end = positions[i + 1][0] if i + 1 < len(positions) else len(content)
        if i > 0:
            rebuilt += "[[PAGEBREAK]]\n\n"
        rebuilt += content[start:chunk_end]
    return rebuilt


def _build_table_of_contents(sections: list[dict[str, Any]]) -> str:
    """Compute a real Table of Contents from the actual generated sections —
    replaces the static placeholder (with fabricated page numbers) that used
    to be returned unconditionally for every dissertation."""
    lines: list[str] = []
    cumulative_words = 0
    roman_idx = 0
    page_number = 1

    for section in sections:
        title = (section.get("title") or "").strip()
        body = section.get("content") or ""
        if not title:
            continue

        if title.lower() == "preliminary pages":
            for _start, _end, heading in _roman_heading_positions(body):
                roman_idx += 1
                heading_text = re.sub(r"^[ivxlcdm]+\.\s+", "", heading, flags=re.IGNORECASE)
                lines.append(_toc_line(heading_text, _to_roman(roman_idx)))
            cumulative_words += len(body.split())
            continue

        lines.append(_toc_line(title, str(page_number)))
        for start, _end, heading in _heading_positions(body):
            words_before = len(body[:start].split())
            sub_page = page_number + words_before // _WORDS_PER_PAGE
            lines.append("    " + _toc_line(heading, str(sub_page), width=66))

        cumulative_words += len(body.split())
        page_number = max(page_number + 1, 1 + cumulative_words // _WORDS_PER_PAGE)

    return "\n".join(lines)


def _build_caption_list(sections: list[dict[str, Any]], kind: str) -> str:
    """Build a List of Figures/Tables from the real blocks embedded across the
    document — replaces the static placeholder that listed fabricated captions
    unrelated to anything actually generated."""
    token = "figure" if kind == "figure" else "table"
    entries: list[tuple[int, str, int]] = []
    cumulative_words = 0
    page_number = 1

    for section in sections:
        title = (section.get("title") or "").strip()
        body = section.get("content") or ""
        if title.lower() != "preliminary pages":
            for block in section.get("blocks", []) or []:
                caption = (block.get("caption") or "").strip()
                match = re.match(rf"{token}\s+(\d+)\s*:?\s*(.*)", caption, flags=re.IGNORECASE)
                if not match:
                    continue
                num = int(match.group(1))
                label = match.group(2).strip() or caption
                entries.append((num, label, page_number))
        cumulative_words += len(body.split())
        page_number = max(page_number + 1, 1 + cumulative_words // _WORDS_PER_PAGE)

    entries.sort(key=lambda e: e[0])
    if not entries:
        return f"No {token}s were generated in this document."
    return "\n".join(
        _toc_line(f"{token.capitalize()} {num}: {label}", str(page))
        for num, label, page in entries
    )


def _refresh_preliminary_pages(sections: list[dict[str, Any]]) -> None:
    """Replace the static Table of Contents / List of Figures / List of Tables
    placeholders with real content computed from the generated document, and
    (re)insert page breaks between each preliminary-page item. Mutates
    `sections` in place. Idempotent — safe to call again after the user edits
    the document (e.g. via a manual "Update Table of Contents" action)."""
    prelim = next(
        (s for s in sections if (s.get("title") or "").strip().lower() == "preliminary pages"),
        None,
    )
    if not prelim:
        return

    body = prelim.get("content") or ""
    positions = _roman_heading_positions(body)
    if not positions:
        return

    replacements = {
        "table of contents": _build_table_of_contents(sections),
        "list of figures": _build_caption_list(sections, "figure"),
        "list of tables": _build_caption_list(sections, "table"),
    }

    chunks: list[str] = [body[: positions[0][0]]]
    for idx, (start, _end, heading) in enumerate(positions):
        chunk_end = positions[idx + 1][0] if idx + 1 < len(positions) else len(body)
        heading_l = heading.lower()
        replacement = next((text for key, text in replacements.items() if key in heading_l), None)
        if replacement is not None:
            chunks.append(f"{heading}\n{replacement}\n\n")
        else:
            chunks.append(body[start:chunk_end])

    prelim["content"] = _insert_preliminary_page_breaks("".join(chunks))


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
    """Return a per-subsection word-count target (floor) for each dissertation chapter.

    These are minimum floors, not the primary length control — `_dissertation_word_budget`
    apportions the whole document's target page count (50-100 pages, flexible) across
    chapters and never drops below these floors, so quality stays consistent even for
    a sparse chapter outline or when no explicit page count is requested.
    """
    chapter_wc = {
        0: 260,   # Preliminary pages — dedication/acknowledgement/etc. ~half a page each
        1: 750,   # Introduction — rich contextualisation, objectives, significance
        2: 1000,  # Literature Review — thematic synthesis, densest chapter
        3: 750,   # Methodology — design rationale, sampling/build, instruments, analysis
        4: 800,   # Results & Discussion — findings tables, interpretation per objective
        5: 600,   # Conclusions — summary, recommendations, limitations, future research
        6: 350,   # References & Appendices — lists + structured entries
    }
    return chapter_wc.get(chapter_number, 450)


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


# A full dissertation should run 50-100 pages depending on topic/context complexity —
# 70 is a flexible midpoint default used only when the user gives no explicit length.
_DEFAULT_DISSERTATION_PAGES = 70
_WORDS_PER_PAGE = 400  # dense academic prose; sparse front-matter/table/figure pages add extra pages on top


def _dissertation_word_budget(
    chapter_blueprints: list[dict[str, Any]],
    instruction: str,
) -> dict[str, int]:
    """Apportion a whole-dissertation word budget across chapters.

    `_chapter_word_count_target` reapplies the SAME requested page count to every
    chapter independently, which — used per-chapter — would multiply a single
    "80 page dissertation" request by the number of chapters. This instead derives
    one document-wide budget (explicit page/word request, or the 50-100pp flexible
    default) and splits it across chapters in proportion to each chapter's natural
    weight (its base word-count floor times how many subsections it has), so chapters
    that should be denser (e.g. Literature Review) get a larger share automatically.
    Never drops a chapter below its own `_chapter_default_word_count` floor.
    """
    explicit_words = _requested_word_target(instruction)
    total_budget = explicit_words if explicit_words else _DEFAULT_DISSERTATION_PAGES * _WORDS_PER_PAGE

    weights: dict[str, float] = {}
    leaves: dict[str, int] = {}
    bases: dict[str, int] = {}
    for chapter in chapter_blueprints:
        title = chapter["title"]
        ch_num = _chapter_number_from_title(title)
        base = _chapter_default_word_count(ch_num)
        n_leaves = _leaf_node_count(chapter.get("nodes", []))
        leaves[title] = n_leaves
        bases[title] = base
        weights[title] = base * n_leaves

    total_weight = sum(weights.values()) or 1
    targets: dict[str, int] = {}
    for title, weight in weights.items():
        chapter_share = total_budget * weight / total_weight
        per_leaf = int(round(chapter_share / max(leaves[title], 1)))
        targets[title] = max(bases[title], per_leaf)
    return targets


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
    message: str = "",
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
            if chapter_number == 4:
                # Plan the table/chart visuals for the LLM's own proposed Chapter 4
                # sections instead of discarding them for a fixed survey template.
                return _plan_chapter4_structure(dynamic, research_design, objectives, topic, message)
            dynamic = _inject_standard_visuals(dynamic, chapter_number, research_design)
            return dynamic

    if "chapter 4" in chapter_title.lower():
        return _plan_chapter4_structure([], research_design, objectives, topic, message)

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
    """Detect passages overlapping with other workspace documents (and, unless disabled, the open web) and rewrite them to cut similarity."""
    from .ai_detector import _split_sentences
    from .plagiarism_detector import check_plagiarism, reduce_similarity
    from .web_plagiarism import check_external_plagiarism, external_check_enabled

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

    use_external = external_check_enabled()

    _done(plan, 0)  # Scanning document for matched/similar passages

    if not source_docs and not use_external:
        _all_done(plan)
        return (
            "No other documents exist in the workspace to compare against, so there is nothing to "
            "reduce — plagiarism similarity is only measured against other workspace documents.",
            False,
        )

    _done(plan, 1)  # Comparing against other documents and the open web

    # Run the (network-bound) web search exactly once for the whole document
    # rather than per section/per retry, so reducing similarity stays fast.
    external_hit_texts: set[str] = set()
    if use_external:
        full_text_before = "\n\n".join(
            (s.get("content") or "").strip() for s in sections if (s.get("content") or "").strip()
        )
        external = check_external_plagiarism(full_text_before)
        external_hit_texts = {h["text"] for h in external.get("hits", [])}

    def _flagged_for(content: str) -> set[str]:
        detection = check_plagiarism(content, source_docs)
        flagged = {s["text"] for s in detection.get("sentences", []) if s.get("label") != "original"}
        flagged |= {s.strip() for s in _split_sentences(content) if s.strip() in external_hit_texts}
        return flagged

    reduced = 0
    for i, section in enumerate(sections):
        content = (section.get("content") or "").strip()
        if not content:
            continue

        flagged = _flagged_for(content)
        if not flagged:
            continue

        new_content = reduce_similarity(content, flagged, seed=f"{document.id}:{i}")

        # Re-check; if anything is still flagged, retry once with a fresh seed
        # (mirrors the before/after verification pattern used by AI-humanisation).
        still_flagged = _flagged_for(new_content)
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
            f"Rewrote {reduced} section(s) to reduce overlap with other workspace documents"
            f"{' and the open web' if use_external else ''}. "
            f"Similarity dropped to **{after_pct}%** ({after_verdict.replace('_', ' ')}). "
            "Click **Check Plagiarism** to see the updated highlights.",
            True,
        )

    return (
        "No matched or similar passages were found against other workspace documents"
        f"{' or the open web' if use_external else ''} — your document "
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
                                    topic or "", _sec.get("title", ""), _subsec_name_clean or _heading_txt,
                                    objectives=(document.content or {}).get("research_objectives"),
                                    sample_size=_infer_sample_size(document),
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
        enhanced = _fallback_subsection_text(
            topic, sec_title, query,
            objectives=(document.content or {}).get("research_objectives"),
            sample_size=_infer_sample_size(document),
        )

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
        chapter_nodes = _chapter_nodes_for_generation(chapter_hint, design, objectives, topic, instruction)

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

        # Only fetch real sources when this section actually needs citations
        # (Literature Review / References) — avoids needless network latency
        # for purely structural sections like Objectives or Scope.
        citation_pool = _retrieve_citation_pool(topic, document.id) if chapter_hint in (2, 6) else []

        # Snapshot the chapter content BEFORE generation starts and locate the
        # target subsection's (start, end) range exactly once. Every progress
        # update (and the final write) splices against this fixed baseline —
        # never against content already mutated by a prior progress update —
        # so a heading match that only succeeds on the first callback (e.g.
        # "2.4 Empirical Review" gets replaced by "2.4.1 ...") can't cause
        # later callbacks to fall through to the append branch repeatedly and
        # duplicate already-written children.
        _original_chapter_content = chapter_payload.get("content", "")
        _match_range = (
            _subsection_match_range(_original_chapter_content, selected_node.get("title", query))
            if selected_node else None
        )

        def _splice_chapter_content(new_text: str) -> str:
            if not selected_node:
                return new_text if new_text.strip() else ""
            if _match_range:
                start, end = _match_range
                return _original_chapter_content[:start] + new_text.strip() + "\n\n" + _original_chapter_content[end:]
            return (_original_chapter_content.rstrip() + "\n\n" + new_text).strip()

        def _persist_subsection_progress(partial_text: str, partial_blocks: list[dict[str, str]], node_title: str) -> None:
            safe_node = re.sub(r"[^a-zA-Z0-9_.-]+", "-", node_title).strip("-")[:60] or "subsection"
            chapter_payload["content"] = _splice_chapter_content(partial_text)

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
            citation_pool=citation_pool,
        )

        if citation_pool:
            try:
                from .research_layer import repair_citations

                chapter_text = repair_citations(chapter_text, citation_pool, min_confidence=60).repaired_text
            except Exception as exc:
                logger.warning("_write_section: citation repair failed: %s", exc)

        # chapter_text already starts with the subsection title from _execute_subsection_nodes
        chapter_payload["content"] = _splice_chapter_content(chapter_text)

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
        if _looks_like_meta_commentary(content):
            raise ValueError("Generated body reads like document meta-commentary, not section prose.")
    except Exception:
        content = _fallback_subsection_text(
            topic, section_name, section_name,
            objectives=(document.content or {}).get("research_objectives"),
            sample_size=_infer_sample_size(document),
        )
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
        nodes = _chapter_nodes_for_generation(chapter_number, design, objectives, topic, instruction)
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

    # Only fetch real sources when a Literature Review or References chapter
    # is being rewritten — avoids needless network latency otherwise.
    citation_pool = (
        _retrieve_citation_pool(topic, document.id)
        if any(n in (2, 6) for n in chapter_numbers) else []
    )

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
            citation_pool=citation_pool,
        )

        if citation_pool:
            try:
                from .research_layer import repair_citations

                chapter_text = repair_citations(chapter_text, citation_pool, min_confidence=60).repaired_text
            except Exception as exc:
                logger.warning("_rewrite_chapter_batch: citation repair failed: %s", exc)

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
            nodes = _chapter4_subsections(design, [], topic, message)
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
            if _looks_like_meta_commentary(text):
                raise ValueError("Generated body reads like document meta-commentary, not section prose.")
        except Exception:
            text = _fallback_subsection_text(
                topic, doc_type.capitalize(), title,
                objectives=(document.content or {}).get("research_objectives"),
                research_design=design,
                sample_size=_infer_sample_size(document),
            )

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


def _review_and_revise_chapter(
    chapter_title: str,
    chapter_text: str,
    chapter_blocks: list[dict[str, Any]],
    topic: str,
    research_design: str,
    objectives: list[str],
) -> tuple[str, list[str]]:
    """Self-review pass run right after a chapter is drafted.

    Always runs two deterministic consistency checks regardless of LLM availability:
    strip any [[BLOCK:...]] marker that doesn't correspond to a generated figure/table
    (a broken reference), and re-attach any generated figure/table whose marker never
    made it into the body (so nothing produced gets silently dropped). When the LLM is
    available, also asks it to critique and tighten the chapter — that rewrite is only
    accepted if it keeps the exact same set of block markers and isn't a drastic
    truncation, so a "creative" rewrite can never break the figure/table linkage.
    """
    notes: list[str] = []
    known_block_ids = {b.get("block_id") for b in chapter_blocks if b.get("block_id")}
    text = chapter_text

    referenced_ids = set(re.findall(r"\[\[BLOCK:([^\]]+)\]\]", text))
    orphan_ids = referenced_ids - known_block_ids
    if orphan_ids:
        for bad_id in orphan_ids:
            text = re.sub(rf"\[\[BLOCK:{re.escape(bad_id)}\]\]\n?", "", text)
        notes.append(f"Removed {len(orphan_ids)} broken figure/table reference(s) in {chapter_title}.")

    referenced_ids = set(re.findall(r"\[\[BLOCK:([^\]]+)\]\]", text))
    unused = [b for b in chapter_blocks if b.get("block_id") and b["block_id"] not in referenced_ids]
    if unused:
        appendix = "\n\n".join(f"[[BLOCK:{b['block_id']}]]" for b in unused)
        text = f"{text}\n\n{appendix}" if text.strip() else appendix
        notes.append(f"Reattached {len(unused)} unreferenced figure/table block(s) in {chapter_title}.")

    try:
        objective_lines = "\n".join(f"- {o}" for o in objectives[:6]) or "- (none specified)"
        prompt = (
            "You are self-reviewing a dissertation chapter you just wrote. Read it for internal "
            "consistency, repetition, and whether it stays on-topic and tied to the stated "
            "objectives. Return ONLY the corrected full chapter text — no commentary, no markdown "
            "fences. Every line that looks like '[[BLOCK:some-id]]' is a placeholder for a figure "
            "or table; you may move surrounding prose around it but you must keep every one of "
            "those marker lines exactly as written, and must not add new ones.\n\n"
            f"Chapter: {chapter_title}\n"
            f"Topic: {topic}\n"
            f"Research design: {research_design}\n"
            f"Research objectives:\n{objective_lines}\n\n"
            "Chapter text:\n"
            f"{text[:8000]}"
        )
        revised = generate_text(prompt).strip()
        if revised.startswith("```"):
            revised = re.sub(r"^```[a-zA-Z]*\n?", "", revised).rstrip("`").strip()

        revised_marker_ids = set(re.findall(r"\[\[BLOCK:([^\]]+)\]\]", revised))
        original_marker_ids = set(re.findall(r"\[\[BLOCK:([^\]]+)\]\]", text))
        if revised and revised_marker_ids == original_marker_ids and len(revised) >= 0.6 * len(text):
            text = revised
            notes.append(f"Applied a self-review pass to {chapter_title} for consistency and flow.")
    except Exception as exc:
        logger.info("_review_and_revise_chapter: LLM self-review skipped for '%s': %s", chapter_title, exc)

    return text, notes


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

    chapter_blueprints = llm_chapters_to_blueprints(llm_chapters, design, objectives_early, topic, instruction)

    # ── Step 2: Build the flat step list ─────────────────────────────────
    for chapter in chapter_blueprints:
        plan.append({"step": f"Writing {chapter['title']}", "status": "pending"})
        _append_node_plan_steps(plan, chapter["nodes"], depth=1)
        plan.append({"step": f"  Reviewing {chapter['title']} for consistency", "status": "pending"})

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

    # ── Retrieve real literature ONCE for the whole dissertation ──────────
    # Grounds in-text citations and the References chapter in verifiable papers
    # from Crossref/arXiv/PubMed/SSRN/Semantic Scholar instead of LLM invention.
    citation_pool = _retrieve_citation_pool(topic, document.id)
    logger.info("_write_dissertation: retrieved %d candidate sources for citation grounding", len(citation_pool))

    # Apportion a whole-document length budget (50-100 flexible pages, or the user's
    # explicit request) across chapters by weight, rather than reapplying the same
    # document-wide page count to every chapter independently.
    chapter_word_budget = _dissertation_word_budget(chapter_blueprints, instruction)
    logger.info(
        "_write_dissertation: chapter word budget — %s",
        ", ".join(f"{t[:30]}={w}" for t, w in chapter_word_budget.items()),
    )

    review_notes: list[str] = []
    for chapter in chapter_blueprints:
        chapter_title = chapter["title"]
        ch_num = _chapter_number_from_title(chapter_title)
        ch_word_count = chapter_word_budget.get(
            chapter_title, _chapter_word_count_target(ch_num, instruction, chapter["nodes"]),
        )

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
            citation_pool=citation_pool,
        )

        if citation_pool:
            try:
                from .research_layer import repair_citations

                chapter_text = repair_citations(chapter_text, citation_pool, min_confidence=60).repaired_text
            except Exception as exc:
                logger.warning("_write_dissertation: citation repair failed for '%s': %s", chapter_title, exc)

        chapter_text, chapter_review_notes = _review_and_revise_chapter(
            chapter_title, chapter_text, chapter_blocks, topic, design, objectives,
        )
        review_notes.extend(chapter_review_notes)
        _done(plan, plan_cursor[0])
        plan_cursor[0] += 1

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

    # Now that every chapter has its final content, replace the static
    # Table of Contents / List of Figures / List of Tables placeholders in
    # Preliminary Pages with real entries computed from what was actually
    # written, and insert page breaks between each preliminary-page item.
    _refresh_preliminary_pages(sections)
    document.content = {
        "topic": topic,
        "research_design": design,
        "research_objectives": objectives,
        "sections": sections,
    }
    _save(document, "dissertation-step:table-of-contents")

    document.title = f"Dissertation: {topic}"
    document.save(update_fields=["title", "updated_at"])
    _all_done(plan)

    citation_note = (
        f" Citations were grounded in {len(citation_pool)} real sources retrieved from Crossref, "
        "arXiv, PubMed, and Semantic Scholar."
        if citation_pool else
        " No verified external sources could be retrieved (e.g. no network access) — "
        "the References chapter contains clearly-labelled illustrative placeholders that must be replaced."
    )
    review_note = f" Self-review pass: {' '.join(review_notes)}" if review_notes else ""
    reply = (
        f"Dissertation generation complete for '{topic}' using a {design.replace('_', ' ')} design. "
        "The AI generated a tailored chapter and section plan, wrote each section sequentially, then "
        "reviewed each chapter for consistency."
        f"{citation_note}{review_note}"
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
    figure_no = _next_caption_number(document, "figure")
    figure_caption = f"Figure {figure_no}: {_section_title}"
    chart_path = generate_chart(
        series=_ai["series"],
        chart_type=_ai["chart_type"],
        title=_section_title,
        x_labels=_ai.get("x_labels") or None,
        unit=_ai.get("unit") or None,
    )
    blocks = section.setdefault("blocks", [])
    block_id = f"fig-{figure_no}-{len(blocks) + 1}"
    blocks.append(
        {"type": "chart", "src": chart_path, "caption": figure_caption, "block_id": block_id}
    )
    addition = (
        f"\n\n{figure_caption}\n[[BLOCK:{block_id}]]\n"
        f"{_chart_discussion_text(_ai['series'], None, _section_title)}"
    )
    section["content"] = (section.get("content", "") or "").rstrip() + addition
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
