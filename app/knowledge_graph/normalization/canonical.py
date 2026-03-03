# app/knowledge_graph/normalization/canonical.py
import re
import unicodedata

STOP = {"the", "a", "an"}

def canonical_key(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name).strip()
    s = s.replace("–", "-").replace("—", "-")
    s = s.lower()
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9\+\-\. ]+", " ", s)  # keep + - . for BLEU-4, F1, etc.
    s = re.sub(r"\s+", " ", s).strip()

    parts = s.split()
    while parts and parts[0] in STOP:
        parts = parts[1:]
    return " ".join(parts)