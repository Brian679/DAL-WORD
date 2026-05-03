"""Patch autonomous.py to personalize plan step labels with the target section name."""

HELPER = '''
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
        elif "generating" in sl or "inserting" in sl:
            step["step"] = f"Generating content for '{name}'"


'''

path = "agent/autonomous.py"

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# 1. Insert helper before _enhance_section
marker = "def _enhance_section("
assert marker in src, "marker not found"
idx = src.index(marker)
src = src[:idx] + HELPER + src[idx:]

# 2. Add call in _enhance_section right after query is resolved
OLD_ENHANCE = (
    '    query = (target or _extract_subsection_phrase(instruction) or "").strip()\n'
    '    if _is_generic_section_query(query):\n'
    '        query = "Introduction"'
)
NEW_ENHANCE = (
    '    query = (target or _extract_subsection_phrase(instruction) or "").strip()\n'
    '    if _is_generic_section_query(query):\n'
    '        query = "Introduction"\n'
    '    _personalize_plan_steps(plan, query)'
)
assert OLD_ENHANCE in src, "enhance marker not found"
src = src.replace(OLD_ENHANCE, NEW_ENHANCE, 1)

# 3. Add call in _write_section right after query_l is defined
OLD_WRITE = (
    'def _write_section(\n'
    '    document: Document,\n'
    '    target: str | None,\n'
    '    topic: str,\n'
    '    instruction: str,\n'
    '    plan: list,\n'
    ') -> tuple[str, bool]:\n'
    '    query = (target or _extract_subsection_phrase(instruction) or "").strip()\n'
    '    query_l = query.lower()'
)
NEW_WRITE = (
    'def _write_section(\n'
    '    document: Document,\n'
    '    target: str | None,\n'
    '    topic: str,\n'
    '    instruction: str,\n'
    '    plan: list,\n'
    ') -> tuple[str, bool]:\n'
    '    query = (target or _extract_subsection_phrase(instruction) or "").strip()\n'
    '    query_l = query.lower()\n'
    '    _personalize_plan_steps(plan, query or instruction or "section")'
)
assert OLD_WRITE in src, "write marker not found"
src = src.replace(OLD_WRITE, NEW_WRITE, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("Patch applied OK")
