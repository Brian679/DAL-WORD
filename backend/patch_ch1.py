"""Patch autonomous.py:
1. Strip echoed heading from LLM body in _execute_subsection_nodes
2. Replace raw generate_text with generate_section_content for pointform sections
3. Make _fallback_subsection_text retry with a simple prompt
"""

path = "agent/autonomous.py"

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# ─────────────────────────────────────────────────────────────
# FIX 1: Add _strip_leading_heading helper before _execute_subsection_nodes
# ─────────────────────────────────────────────────────────────
HELPER = '''
def _strip_leading_heading(body: str, title: str) -> str:
    """Remove the section title if the LLM echoed it at the start of the body."""
    stripped = body.lstrip()
    clean_title = re.sub(r"^#+\\s*", "", title).strip().lower()
    first_line_raw = stripped.split("\\n", 1)[0] if stripped else ""
    first_line = re.sub(r"^#+\\s*|\\*+|_+", "", first_line_raw).strip().lower()
    # Match if the first line IS the title (exact or starts with it)
    if first_line == clean_title or first_line.startswith(clean_title):
        rest = stripped.split("\\n", 1)[1] if "\\n" in stripped else ""
        return rest.lstrip("\\n")
    return body

'''

marker = "def _execute_subsection_nodes("
assert marker in src, "marker not found: _execute_subsection_nodes"
idx = src.index(marker)
src = src[:idx] + HELPER + src[idx:]

# ─────────────────────────────────────────────────────────────
# FIX 2: Strip heading from body before appending chunk
# ─────────────────────────────────────────────────────────────
OLD_CHUNK = '        chunks.append(f"{title}\\n{body}")'
NEW_CHUNK = (
    '        body = _strip_leading_heading(body, title)\n'
    '        chunks.append(f"{title}\\n{body}")'
)
assert OLD_CHUNK in src, "chunk append marker not found"
src = src.replace(OLD_CHUNK, NEW_CHUNK, 1)

# ─────────────────────────────────────────────────────────────
# FIX 3: Replace is_pointform branch to use generate_section_content
# ─────────────────────────────────────────────────────────────
OLD_POINTFORM = (
                '                if is_pointform:\n'
                '                    try:\n'
                '                        body = generate_text(\n'
                '                            f"Write the \'{title}\' subsection for a research paper about: \'{topic}\'.\\n"\n'
                '                            f"Research design: {research_design}\\n"\n'
                '                            "Format as a numbered list ONLY (1. ... 2. ... 3. ...). "\n'
                '                            "Write 3-5 clear, specific, measurable points. "\n'
                '                            "Each item must be a complete standalone sentence. "\n'
                '                            "Do NOT write any prose paragraph. Do NOT add an introductory sentence before the list."\n'
                '                        )\n'
                '                    except Exception:\n'
                '                        body = _fallback_subsection_text(topic, section_title, title)'
)
NEW_POINTFORM = (
                '                if is_pointform:\n'
                '                    try:\n'
                '                        body = generate_section_content(\n'
                '                            title=title,\n'
                '                            topic=topic,\n'
                '                            context=(\n'
                '                                f"Parent chapter: {section_title}\\n"\n'
                '                                f"Research design: {research_design}\\n"\n'
                '                                f"Context from previous sections:\\n{local_context[-1500:]}\\n\\n"\n'
                '                                "Format the output as a clean numbered list (1. ... 2. ... 3. ...). "\n'
                '                                "Write 3-5 clear, specific, measurable items. "\n'
                '                                "Each item must be a complete, standalone academic sentence. "\n'
                '                                "Do NOT add an introductory paragraph before the list. "\n'
                '                                "Do NOT include the section heading in your response."\n'
                '                            ),\n'
                '                            word_count=120,\n'
                '                        )\n'
                '                    except Exception:\n'
                '                        body = _fallback_subsection_text(topic, section_title, title)'
)

assert OLD_POINTFORM in src, "pointform marker not found"
src = src.replace(OLD_POINTFORM, NEW_POINTFORM, 1)

# ─────────────────────────────────────────────────────────────
# FIX 4: Smarter _fallback_subsection_text that retries with a simple prompt
# ─────────────────────────────────────────────────────────────
OLD_FALLBACK = (
    'def _fallback_subsection_text(topic: str, section_title: str, subsection: str) -> str:\n'
    '    return (\n'
    '        f"This section discusses {subsection.lower()} in relation to {topic}. "\n'
    '        "It provides practical context, highlights key issues, and links the discussion to the overall study objectives. "\n'
    '        "The content is structured to maintain logical flow and support evidence-based academic writing."\n'
    '    )'
)
NEW_FALLBACK = (
    'def _fallback_subsection_text(topic: str, section_title: str, subsection: str) -> str:\n'
    '    """Try a simple one-shot prompt; return a placeholder only if that also fails."""\n'
    '    try:\n'
    '        return generate_text(\n'
    '            f"Write a concise academic paragraph for the \'{subsection}\' subsection "\n'
    '            f"of a research paper about \'{topic}\'. "\n'
    '            "Use formal, scholarly language. 120-180 words. "\n'
    '            "Do NOT include the subsection heading. Do NOT use phrases like \'in today\'s world\' or \'it is worth noting\'."\n'
    '        )\n'
    '    except Exception:\n'
    '        pass\n'
    '    clean = subsection.lower().replace("\\n", " ").strip()\n'
    '    return (\n'
    '        f"This subsection addresses {clean} within the context of {topic}. "\n'
    '        "Further analysis will be developed in accordance with the research objectives and empirical findings."\n'
    '    )'
)

assert OLD_FALLBACK in src, "fallback marker not found"
src = src.replace(OLD_FALLBACK, NEW_FALLBACK, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("All patches applied OK")
