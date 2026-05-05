content = open('frontend/src/components/DocumentEditorPage.jsx', encoding='utf-8').read()
lines = content.split('\n')
print("Total lines:", len(lines))
print("Last 10 lines:")
for i, line in enumerate(lines[-10:]):
    print(f"  {len(lines)-10+i+1}: {repr(line)}")

# Count parens
depth = 0
in_return = False
for i, line in enumerate(lines):
    if 'return (' in line:
        in_return = True
    if in_return:
        depth += line.count('(') - line.count(')')
        if depth <= 0 and i > 5:
            print(f"Return closed at line {i+1}")
            in_return = False
            depth = 0
