path = 'agent/autonomous.py'
with open(path, 'r', encoding='utf-8') as f:
    src = f.read()
old = 'content = f"{chapter_title}\\n\\n{partial_text}" if partial_text.strip() else chapter_title'
new = 'content = partial_text if partial_text.strip() else ""'
count = src.count(old)
print(f'Found {count} occurrences')
src = src.replace(old, new)
with open(path, 'w', encoding='utf-8') as f:
    f.write(src)
print('Done')
