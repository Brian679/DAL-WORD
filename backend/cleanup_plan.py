"""Remove duplicate _personalize_plan_steps definitions and calls."""

path = "agent/autonomous.py"

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# Remove the second duplicate function definition block
DUPLICATE_DEF = (
    '\n\ndef _personalize_plan_steps(plan: list, section_name: str) -> None:\n'
    '    """Replace generic plan step labels with section-specific text."""\n'
    '    name = section_name.strip("\'").strip()\n'
    '    for step in plan:\n'
    '        s = step.get("step", "")\n'
    '        sl = s.lower()\n'
    '        if "locating" in sl:\n'
    '            step["step"] = f"Locating the section for \'{name}\'"\n'
    '        elif "analys" in sl and "content" in sl:\n'
    '            step["step"] = f"Analysing existing content in \'{name}\'"\n'
    '        elif "rewriting" in sl or ("clarity" in sl and "tone" in sl):\n'
    '            step["step"] = f"Rewriting \'{name}\' with improved clarity and academic tone"\n'
    '        elif "saving" in sl and "section" in sl:\n'
    '            step["step"] = f"Saving updated section \'{name}\'"\n'
    '        elif "generating" in sl or "inserting" in sl:\n'
    '            step["step"] = f"Generating content for \'{name}\'"\n'
)

count = src.count('def _personalize_plan_steps(')
print(f"Found {count} definitions of _personalize_plan_steps")

# Remove exactly one duplicate (the second occurrence of the whole block)
if count == 2:
    first = src.index('def _personalize_plan_steps(')
    second = src.index('def _personalize_plan_steps(', first + 1)
    # Find the end of the second block (blank line after last elif)
    end = src.index('\ndef _enhance_section(', second)
    # Remove the second block plus blank lines before it
    block_start = src.rindex('\n\n', first + 1, second)
    src = src[:block_start] + src[end:]
    print("Removed duplicate definition")

# Remove duplicate _personalize_plan_steps call in _enhance_section
OLD_DOUBLE_CALL = '    _personalize_plan_steps(plan, query)\n    _personalize_plan_steps(plan, query)'
NEW_SINGLE_CALL = '    _personalize_plan_steps(plan, query)'
if OLD_DOUBLE_CALL in src:
    src = src.replace(OLD_DOUBLE_CALL, NEW_SINGLE_CALL, 1)
    print("Fixed duplicate call in _enhance_section")

# Remove duplicate _personalize_plan_steps call in _write_section
OLD_DOUBLE2 = '    _personalize_plan_steps(plan, query or instruction or "section")\n    _personalize_plan_steps(plan, query or instruction or "section")'
NEW_SINGLE2 = '    _personalize_plan_steps(plan, query or instruction or "section")'
if OLD_DOUBLE2 in src:
    src = src.replace(OLD_DOUBLE2, NEW_SINGLE2, 1)
    print("Fixed duplicate call in _write_section")

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

count2 = src.count('def _personalize_plan_steps(')
print(f"After cleanup: {count2} definitions")
print("Done")
