"""Fix Unicode and encoding issues in fetch_audio_samples.py"""
import sys, os

fn = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                  "scripts", "fetch_audio_samples.py")

with open(fn, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode characters
content = content.replace('\u2192', '->')
content = content.replace('\u2014', '-')
content = content.replace('\u2013', '-')
content = content.replace('\u2713', '[OK]')
content = content.replace('\u2717', '[FAIL]')

# Remove trust_remote_code
content = content.replace(', trust_remote_code=True', '')
content = content.replace(',trust_remote_code=True', '')
content = content.replace('trust_remote_code=True,', '')
content = content.replace('trust_remote_code=True', '')

# Add sys import and encoding fix if not present
if 'sys.stdout.reconfigure' not in content:
    old_imports = 'import os\nimport json'
    new_imports = '''import os
import sys
import json

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")'''
    content = content.replace(old_imports, new_imports)

with open(fn, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Fixed: {fn}")
