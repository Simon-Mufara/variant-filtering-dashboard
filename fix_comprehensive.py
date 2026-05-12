#!/usr/bin/env python3
import re

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find all syntax errors by trying to compile each section
fixed_lines = []

for i, line in enumerate(lines, 1):
    # Fix pattern: string starts with ' but ends with " before comma/paren
    # Examples: '<span...stuff",  or '<span...stuff")
    if "'" in line and '"' in line:
        # Very targeted regex for material-symbols-outlined patterns
        line = re.sub(r"'(<span[^>]*material-symbols-outlined[^>]*>.*?</span>[^']*)\",", r"'\1',", line)
        line = re.sub(r"'(<span[^>]*material-symbols-outlined[^>]*>.*?</span>[^']*)\"\)", r"'\1')", line)

        # Fix other quoted strings that have mismatches
        line = re.sub(r"'([^']*[^'\\\"])\",", r"'\1',", line)
        line = re.sub(r"'([^']*[^'\\\"])\"\)", r"'\1')", line)

        # Fix f-strings
        line = re.sub(r'f"([^"]*)"([^"]*)"([\),])', r'f"\1\2"\3', line)

    fixed_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Fixed mismatched quotes")
