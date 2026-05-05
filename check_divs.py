import re

content = open('frontend/src/components/DocumentEditorPage.jsx', encoding='utf-8').read()
lines = content.split('\n')

# Track <div> / </div> stack with a simple state machine
# Skip content inside {/* ... */} JSX comments
stack = []
i = 0
for lineno, line in enumerate(lines):
    # Count <div and </div> tokens
    stripped = line
    # Find <div> opens  
    for m in re.finditer(r'<div[\s>]', stripped):
        stack.append(lineno + 1)
    # Find </div> closes
    for m in re.finditer(r'</div>', stripped):
        if stack:
            stack.pop()
        else:
            print(f"Extra </div> at line {lineno+1}")

print(f"Unclosed <div> count: {len(stack)}")
for ln in stack:
    print(f"  Opened at line {ln}: {repr(lines[ln-1][:80])}")
