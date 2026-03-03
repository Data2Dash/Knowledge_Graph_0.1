from __future__ import annotations
import json, re, logging
from typing import List, Tuple
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.knowledge_graph.llm.prompts import JOINT_EXTRACTION_PROMPT
from app.knowledge_graph.extraction.schema import Entity, Relation, JointExtractionResult
from app.knowledge_graph.extraction.validator import normalize_rel_type

LOGGER = logging.getLogger(__name__)

def _extract_json_to_pydantic(raw: str) -> JointExtractionResult:
    """Extracts raw JSON boundaries from LLM and strictly validates them via Pydantic."""
    raw = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return JointExtractionResult()
    try:
        parsed_dict = json.loads(raw[start:end+1])
        # Pydantic validation handles casting and checks schema strictly
        return JointExtractionResult.model_validate(parsed_dict)
    except Exception as e:
        LOGGER.error(f"Pydantic Validation or JSON Parsing Failed: {e}")
        return JointExtractionResult()

def extract_jointly(llm: ChatGroq, text: str, max_chars: int) -> Tuple[List[Entity], List[Relation]]:
    prompt = JOINT_EXTRACTION_PROMPT.replace("{{text_chunk}}", text[:max_chars])
    msg = llm.invoke([HumanMessage(content=prompt)])
    raw = msg.content if hasattr(msg, "content") else str(msg)
    
    # Strictly valid Pydantic Extraction
    validated_schema = _extract_json_to_pydantic(raw)
    
    ents: List[Entity] = []
    entity_type_map = {}
    
    # 1. Map Validated Entities
    for ext_ent in validated_schema.entities:
        eid = ext_ent.id.strip()
        etype = (ext_ent.type or "Concept").strip()
        if eid:
            ents.append(Entity(name=eid, type=etype))
            entity_type_map[eid.lower()] = etype

    # 2. Map Validated Relations
    rels: List[Relation] = []
    for ext_rel in validated_schema.relations:
        src = ext_rel.source.strip()
        tgt = ext_rel.target.strip()
        pred = normalize_rel_type(ext_rel.predicate or "RELATED_TO")
        evid = (ext_rel.evidence or "").strip() or None
        
        if src and tgt and pred:
            ht = entity_type_map.get(src.lower(), "Concept")
            tt = entity_type_map.get(tgt.lower(), "Concept")
            rels.append(Relation(
                head=src, 
                head_type=ht, 
                relation=pred, 
                tail=tgt, 
                tail_type=tt, 
                evidence=evid
            ))
                    
    return ents, rels
