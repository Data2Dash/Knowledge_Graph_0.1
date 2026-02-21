RESEARCH_PAPER_ENTITY_PROMPT = """
You extract ENTITIES from a research paper chunk.

You MUST respond using ONLY a valid JSON object. 
Do NOT output anything other than JSON. 
Your final output MUST be strictly valid JSON.

STRICT RULES:
- Use ONLY the provided text — never infer or invent.
- Do NOT include vague phrases like "this paper", "our model", "the approach".
- Only include entities that are EXPLICITLY NAMED in the text.
- Do NOT duplicate entities with the same normalized name.

Normalization rules (dedupe only):
- Compare names case-insensitively.
- Ignore leading/trailing whitespace.
- Treat fancy quotes and normal quotes as equivalent.

You MUST return JSON with exactly this structure:

{
  "entities": [
    {
      "name": "string",
      "type": "string",
      "aliases": ["string"],
      "evidence": "short verbatim quote (<=25 words)",
      "confidence": 0.0,
      "meta": {}
    }
  ]
}

Allowed entity types (choose exactly ONE):
Paper, Section, Figure, Table,
Method, Model, Component,
Dataset, Task, Metric,
Result, Claim, Baseline,
Hyperparameter, Objective, Ablation,
Domain, Concept,
Author, Institution, Limitation, Contribution, Technique

Definitions:
- Author: human researcher (e.g., "Vaswani", "LeCun").
- Institution: university, lab, company (e.g., "Google Brain", "MIT").
- Method/Model: explicitly named technique (e.g., "ResNet-50", "BERT").
- Dataset: explicitly named dataset (e.g., "ImageNet", "SQuAD").
- Result: numeric finding.
- Claim: explicit statement of contribution.
- Contribution: novel stated element.
- Limitation: explicit weakness.
- Hyperparameter: explicit training setting (e.g., learning rate).
- Technique: general technique (e.g., attention).
- Concept: fallback ONLY if clearly defined.

Evidence rules:
- MUST be a verbatim quote from the text.
- MUST justify the entity.
- If evidence does not support it, SKIP the entity.

Confidence:
- 0.9–0.95 if explicit + strongly supported.
- 0.7–0.85 if reasonably supported.
- <0.7 if weak but explicit.
- Never use 1.0.

Quantity:
- Extract 10–50 entities when supported.
- Prefer precision over quantity.

Your output MUST be ONLY JSON. No explanations, no comments, no markdown.
"""


RESEARCH_PAPER_RELATION_PROMPT = """
You extract RELATIONS from a research paper chunk.

You MUST respond using ONLY a valid JSON array. 
The output MUST contain only JSON. No text, no markdown.

The provided Entity List is the authoritative list of allowed "head" and "tail" names.

STRICT MATCHING RULES:
- You MUST choose head and tail ONLY from the Entity List.
- Output entity names EXACTLY as they appear in the entity list.
- NEVER invent new entity names.
- Matching may ignore case, leading/trailing spaces, and simple punctuation.
- After matching, the canonical entity name MUST be used.

RELATION RULES:
- Extract ONLY relations explicitly stated in the text.
- NO inference, NO multi-hop reasoning.
- If not explicitly stated → SKIP it.

Return a JSON ARRAY strictly matching this schema:

[
  {
    "head": "string",
    "relation": "string",
    "tail": "string",
    "evidence": "short verbatim quote (<=25 words)",
    "confidence": 0.0
  }
]

Allowed relation types:
HAS_SECTION, HAS_FIGURE, HAS_TABLE,
PROPOSES, INTRODUCES,
USES, ADDRESSES,
EVALUATES_ON, OPTIMIZES,
REPORTS, ACHIEVES,
IMPROVES_OVER, COMPARED_WITH,
ABLATION_OF, SUPPORTS,
MENTIONS, RELATED_TO,
WRITTEN_BY, AFFILIATED_WITH,
HAS_LIMITATION, CONTRIBUTES_TO,
TRAINED_ON, REQUIRES,
OUTPERFORMS, FINE_TUNED_ON,
IMPLEMENTED_WITH, BASED_ON

Relation guidance:
- PROPOSES: Paper → Method/Model explicitly.
- WRITTEN_BY: Paper → Author.
- AFFILIATED_WITH: Author → Institution.
- USES: Method/Model uses Dataset/Component/Technique.
- TRAINED_ON: Model trained on Dataset.
- FINE_TUNED_ON: explicit fine-tuning.
- OUTPERFORMS: explicit outperforming.
- IMPROVES_OVER: explicit improvement.
- ACHIEVES: numeric performance tied to a model/method.
- EVALUATES_ON: explicit evaluation.
- HAS_LIMITATION: explicit limitation.
- REQUIRES: method requires hyperparameter/component.
- BASED_ON: model built on another model.
- IMPLEMENTED_WITH: method implemented using framework/library.
- RELATED_TO: ONLY if no specific type fits.

Evidence:
- MUST be an exact quote.
- MUST connect both head AND tail.
- If not clearly supported → SKIP.

Confidence:
- 0.9–0.95 explicit + numeric support.
- 0.7–0.85 if clearly stated.
- <0.7 if weak but explicit.
- Never 1.0.

If no relations exist, return [].
Your output MUST be ONLY JSON.
"""