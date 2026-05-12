import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix f-strings: f"...text' -> f"...text"
for i in range(len(lines)):
    # Fix f"....' pattern (f-string starting with " but ending with ')
    lines[i] = re.sub(r'f"([^"]*)"([^"]*)"([\'"])', r'f"\1\2"', lines[i])
    lines[i] = re.sub(r"f'([^']*)'([^']*)'", r"f'\1\2'", lines[i])

    # Fix f-strings that cross quotes
    lines[i] = re.sub(r"f\"([^\"]*)'([\),])", r'f"\1"\2', lines[i])
    lines[i] = re.sub(r"f'([^']*)\"([\),])", r"f'\1'\2", lines[i])

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed all quote mismatches!")
