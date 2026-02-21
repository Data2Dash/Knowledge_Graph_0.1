# app/knowledge_graph/extraction/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import re
import hashlib

# ==========================================================
# Canonicalization
# ==========================================================

_CANON_WS = re.compile(r"\s+")
_MULTI_DASH = re.compile(r"[–—]+")
# Trim only *outer noise*, but do NOT aggressively strip internal structure.
_PUNCT_TRIM = re.compile(r"^[\s\"'`•]+|[\s\"'`•]+$")


def canon(s: str) -> str:
    """
    Canonical form for display + stable IDs.
    Keeps meaningful punctuation (hyphens, parentheses) but normalizes whitespace/dashes.
    """
    s = (s or "").strip()
    s = _MULTI_DASH.sub("-", s)
    s = _CANON_WS.sub(" ", s)
    s = _PUNCT_TRIM.sub("", s)
    return s


def norm_key(s: str) -> str:
    """
    Normalization key for matching (case-insensitive).
    Also strips trailing punctuation that commonly causes match failures.
    """
    x = canon(s).lower()
    # Remove common trailing punctuation safely
    while x.endswith((".", ",", ":", ";")):
        x = x[:-1].rstrip()
    return x


def stable_id(*parts: str) -> str:
    raw = "|".join(norm_key(p) for p in parts if p is not None)
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:16]


# ==========================================================
# Controlled Vocabularies
# ==========================================================

ALLOWED_ENTITY_TYPES = {
    "Paper", "Section", "Figure", "Table",
    "Method", "Model", "Component",
    "Dataset", "Task", "Metric",
    "Result", "Claim", "Baseline",
    "Hyperparameter", "Objective", "Ablation",
    "Domain", "Concept",
    "Author", "Institution", "Limitation", "Contribution", "Technique",
}

ALLOWED_PREDICATES = {
    "HAS_SECTION", "HAS_FIGURE", "HAS_TABLE",
    "PROPOSES", "INTRODUCES",
    "USES", "ADDRESSES",
    "EVALUATES_ON", "OPTIMIZES",
    "REPORTS", "ACHIEVES",
    "IMPROVES_OVER", "COMPARED_WITH",
    "ABLATION_OF", "SUPPORTS",
    "MENTIONS", "RELATED_TO",
    "WRITTEN_BY", "AFFILIATED_WITH",
    "HAS_LIMITATION", "CONTRIBUTES_TO",
    "TRAINED_ON", "REQUIRES",
    "OUTPERFORMS", "FINE_TUNED_ON",
    "IMPLEMENTED_WITH", "BASED_ON",
}

# ==========================================================
# Ontology Constraints (sane defaults)
# ==========================================================

VALID_EDGE_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
    "PROPOSES": [("Paper", "Method"), ("Paper", "Model")],
    "INTRODUCES": [("Paper", "Dataset"), ("Paper", "Task"), ("Paper", "Concept")],
    "WRITTEN_BY": [("Paper", "Author")],
    "AFFILIATED_WITH": [("Author", "Institution")],
    "USES": [
        ("Method", "Component"), ("Model", "Component"),
        ("Method", "Dataset"), ("Model", "Dataset"),
        ("Method", "Technique"), ("Model", "Technique"),
    ],
    "TRAINED_ON": [("Model", "Dataset")],
    "FINE_TUNED_ON": [("Model", "Dataset")],
    "EVALUATES_ON": [("Method", "Dataset"), ("Model", "Dataset")],
    "ACHIEVES": [("Model", "Metric"), ("Method", "Metric"), ("Model", "Result"), ("Method", "Result")],
    "IMPROVES_OVER": [("Method", "Baseline"), ("Model", "Baseline"), ("Model", "Model"), ("Method", "Method")],
    "OUTPERFORMS": [("Model", "Baseline"), ("Model", "Model"), ("Method", "Baseline"), ("Method", "Method")],
    "HAS_LIMITATION": [("Method", "Limitation"), ("Model", "Limitation")],
}


def is_valid_edge(head_type: str, predicate: str, tail_type: str) -> bool:
    patterns = VALID_EDGE_PATTERNS.get(predicate)
    if not patterns:
        return True  # allow unknown predicates to pass; validator may still filter later
    return (head_type, tail_type) in patterns


# ==========================================================
# Normalization
# ==========================================================

def normalize_entity_type(t: str) -> str:
    x = canon(t)
    if not x:
        return "Concept"

    x_low = x.lower()

    # exact match ignoring case
    for et in ALLOWED_ENTITY_TYPES:
        if x_low == et.lower():
            return et

    # substring match
    for et in ALLOWED_ENTITY_TYPES:
        if et.lower() in x_low:
            return et

    return "Concept"


_REL_MAP = [
    ("has section", "HAS_SECTION"),
    ("has figure", "HAS_FIGURE"),
    ("has table", "HAS_TABLE"),
    ("propos", "PROPOSES"),
    ("introduc", "INTRODUCES"),
    ("use", "USES"),
    ("address", "ADDRESSES"),
    ("evaluat", "EVALUATES_ON"),
    ("optimiz", "OPTIMIZES"),
    ("achiev", "ACHIEVES"),
    ("improv", "IMPROVES_OVER"),
    ("outperform", "OUTPERFORMS"),
    ("surpass", "OUTPERFORMS"),
    ("compare", "COMPARED_WITH"),
    ("report", "REPORTS"),
    ("support", "SUPPORTS"),
    ("mention", "MENTIONS"),
    ("written", "WRITTEN_BY"),
    ("author", "WRITTEN_BY"),
    ("affiliat", "AFFILIATED_WITH"),
    ("institution", "AFFILIATED_WITH"),
    ("limit", "HAS_LIMITATION"),
    ("contribut", "CONTRIBUTES_TO"),
    ("train", "TRAINED_ON"),
    ("fine tun", "FINE_TUNED_ON"),
    ("require", "REQUIRES"),
    ("implement", "IMPLEMENTED_WITH"),
    ("based on", "BASED_ON"),
    ("extend", "BASED_ON"),
]


def normalize_predicate(r: str) -> str:
    x = canon(r)
    if not x:
        return "RELATED_TO"

    x_upper = x.upper().replace(" ", "_")
    if x_upper in ALLOWED_PREDICATES:
        return x_upper

    x_low = x.lower()
    for needle, pred in _REL_MAP:
        if needle in x_low:
            return pred

    return "RELATED_TO"


# ==========================================================
# Evidence
# ==========================================================

@dataclass(frozen=True, slots=True)
class Evidence:
    text: str
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    source: Optional[str] = None
    section: Optional[str] = None


# ==========================================================
# Entity
# ==========================================================

@dataclass(frozen=True, slots=True)
class Entity:
    name: str
    type: str = "Concept"

    id: str = field(default="")
    normalized_name: str = field(default="")
    confidence: float = field(default=0.75)
    aliases: List[str] = field(default_factory=list)
    evidence: Optional[str] = None
    evidence_obj: Optional[Evidence] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        nn = canon(self.name)
        if not nn:
            raise ValueError("Entity.name must be non-empty")
        object.__setattr__(self, "normalized_name", nn)

        nt = normalize_entity_type(self.type)
        object.__setattr__(self, "type", nt)

        if not self.id:
            object.__setattr__(self, "id", stable_id(nt, nn))

        c = float(self.confidence) if self.confidence is not None else 0.75
        object.__setattr__(self, "confidence", max(0.0, min(1.0, c)))


# ==========================================================
# Relation
# ==========================================================

@dataclass(frozen=True, slots=True)
class Relation:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    evidence: Optional[str] = None

    id: str = field(default="")
    confidence: float = field(default=0.7)
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    source: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    predicate: str = field(default="")

    def __post_init__(self):
        h = canon(self.head)
        t = canon(self.tail)
        r = canon(self.relation)

        if not h or not t or not r:
            raise ValueError("Relation requires non-empty head/tail/relation")

        ht = normalize_entity_type(self.head_type)
        tt = normalize_entity_type(self.tail_type)
        pred = normalize_predicate(r)

        # If ontology says it's impossible, downgrade predicate (don’t drop the edge here)
        if not is_valid_edge(ht, pred, tt):
            pred = "RELATED_TO"

        object.__setattr__(self, "head_type", ht)
        object.__setattr__(self, "tail_type", tt)
        object.__setattr__(self, "predicate", pred)

        if not self.id:
            object.__setattr__(self, "id", stable_id(ht, h, pred, tt, t))

        c = float(self.confidence) if self.confidence is not None else 0.7
        object.__setattr__(self, "confidence", max(0.0, min(1.0, c)))


# ==========================================================
# Helper: entity name index for canonicalization/matching
# ==========================================================

def build_entity_name_index(entities: List[Entity]) -> Dict[str, str]:
    """
    Returns { normalized_key -> canonical_name } for robust matching.
    Also indexes aliases.
    """
    idx: Dict[str, str] = {}
    for e in entities:
        idx[norm_key(e.name)] = e.name
        for a in e.aliases or []:
            nk = norm_key(a)
            if nk and nk not in idx:
                idx[nk] = e.name
    return idx