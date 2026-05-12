import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix mismatched quotes inside string expressions
# Pattern 1: '.....(" or '.....)" - single quote start, double quote end
fixes = [
    # Inside f-strings with expressions
    (r"'([^']*[^'\"])\"\)", r"'\1')"),
    (r"'([^']*[^'\"])\",", r"'\1',"),
    # Double quote strings with single quote end (less common)
    (r'"([^"]*[^"\']+)\'([\),])', r'"\1"\2'),
]

for pattern, repl in fixes:
    content = re.sub(pattern, repl, content)

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Applied additional fixes!")
