RESEARCH_PAPER_ENTITY_PROMPT = """\
You extract ENTITIES from research papers.
Return ONLY valid JSON array (no markdown).
Each item:
{"name": "...", "type": "...", "aliases": ["..."]}

Types must be one of:
Concept, Method, Model, Dataset, Metric, Result, Baseline, Technique, Architecture,
Task, Algorithm, Benchmark, Component, Observation, Limitation, Contribution,
Author, Organization, Publication, System, Process, Theory, Hyperparameter

Rules:
- Prefer specific names (e.g., "BERT", "CIFAR-10", "BLEU", "LoRA").
- Include aliases if present (acronyms).
- Output 15-60 entities when possible.
"""

RESEARCH_PAPER_RELATION_PROMPT = """\
You extract RELATIONSHIPS from research papers using a provided entity list.
Return ONLY valid JSON array (no markdown).
Each item:
{"head":"...", "head_type":"...", "relation":"...", "tail":"...", "tail_type":"...", "evidence":"..."}

Allowed relations:
RELATED_TO, USES, CONTAINS, PART_OF, COMPARED_TO, TRAINED_ON, EVALUATES, IMPROVES,
IMPLEMENTS, ACHIEVES, ADDRESSES, RESULTS_IN, PROPOSES, EXTENDS, DEPENDS_ON,
SUPPORTS, ILLUSTRATES, CONTRIBUTES_TO, INTRODUCES, OBSERVED_IN, LIMITS, CITES,
DESCRIBED_IN

Rules:
- Use ONLY entities from the entity list (or obvious exact matches).
- evidence must be a short quote (<= 25 words) copied from the text.
- Output 20-70 relations when possible.
"""
