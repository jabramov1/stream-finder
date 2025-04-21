#!/usr/bin/env python3
"""
wiki_cleaner.py

Reads an input JSON (default `init.json`), cleans the embedded Wikipedia summaries by removing
navigation clutter and extracting the main summary text starting after the first letter-surrounded period.
"""
import json
import re
import sys
from typing import Dict, Any

def clean_wiki_summary(raw: str) -> str:
    """
    Strip out navigation/menu lines, collapse whitespace, then drop everything
    up through the first letter-surrounded period and return the remaining text.
    """
    # Remove lines that look like navigation/menu headers (all-caps words)
    lines = raw.splitlines()
    filtered = [l for l in lines if not re.match(r'^[A-Z][A-Z\s\-"\']+$', l.strip())]
    # Join and collapse whitespace
    text = ' '.join(filtered)
    text = re.sub(r'\s+', ' ', text).strip()
    # Find first period surrounded by letters
    m = re.search(r'(?<=[A-Za-z])\.(?=[A-Za-z])', text)
    if m:
        return text[m.end():].lstrip()
    # Fallback to first period
    idx = text.find('.')
    if idx != -1 and idx < len(text) - 1:
        return text[idx+1:].lstrip()
    return text


def clean_all(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walks a data structure looking for 'wikipedia_summary' fields to clean.
    """
    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_clean(v) for v in obj]
        elif isinstance(obj, str):
            return clean_wiki_summary(obj)
        else:
            return obj
    
    return _clean(data)


def main(input_file: str = 'init.json', output_file: str = 'init.cleaned.json'):
    # Load JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Clean all summary fields
    cleaned = clean_all(data)

    # Write out cleaned structure
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"Cleaned Wikipedia summaries written to '{output_file}'")

if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
