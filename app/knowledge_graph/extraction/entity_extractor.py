"""
Entity Extraction Module

This module is responsible for parsing raw text and extracting structured scientific 
or conceptual entities using an LLM. It maps raw LLM text outputs into strict JSON sequences 
and ultimately into a list of standardized `Entity` objects.
"""
from __future__ import annotations
import json
import re
from typing import List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.knowledge_graph.llm.prompts import RESEARCH_PAPER_ENTITY_PROMPT
from app.knowledge_graph.extraction.schema import Entity

def _extract_json(raw: str) -> List[dict]:
    """
    Safely extract and parse a JSON array from a raw LLM response.
    
    The LLM might return conversational text surrounding the JSONblock.
    This function searches for markdown JSON code blocks and bounding brackets 
    to isolate and decode the actual payload.
    
    Args:
        raw: The raw text string returned by the LLM.
        
    Returns:
        A parsed list of dictionaries representing the extracted items. 
        Returns an empty list on failure or if no valid JSON array is found.
    """
    raw = (raw or "").strip()
    
    # Attempt to extract text nested inside markdown code blocks ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1).strip()
        
    # Isolate the list boundaries to exclude conversational filler
    start = raw.find("[")
    end = raw.rfind("]")
    
    # Guard against invalid substrings
    if start < 0 or end < 0 or end <= start:
        return []
        
    try:
        # Slice only the array portion and attempt JSON deserialization
        out = json.loads(raw[start:end+1])
        return out if isinstance(out, list) else []
    except Exception:
        # Silently fail and return empty if the JSON is garbled
        return []

def extract_entities(llm: ChatGroq, text: str, max_chars: int) -> List[Entity]:
    """
    Execute a prompt to extract entities from a chunk of text.
    
    Args:
        llm: The initialized Langchain ChatGroq model instance.
        text: The source text chunk to extract entities from.
        max_chars: Maximum characters to send to the LLM to prevent context overflow.
        
    Returns:
        A list of strongly-typed `Entity` objects. 
        Empty properties are defaulted (e.g., type="Concept") and invalid objects are dropped.
    """
    # 1. Format the payload, safely truncating text to max character limit
    prompt_content = f"{RESEARCH_PAPER_ENTITY_PROMPT}\n\nText:\n{text[:max_chars]}"
    
    # 2. Invoke the LLM with a single Human Message containing instructions and context
    msg = llm.invoke([HumanMessage(content=prompt_content)])
    
    # 3. Retrieve raw text output from response block
    raw = msg.content if hasattr(msg, "content") else str(msg)
    
    # 4. Parse output into raw Python dictionaries
    arr = _extract_json(raw)
    
    # 5. Build and validate Entity instances
    out: List[Entity] = []
    for it in arr:
        if isinstance(it, dict):
            name = (it.get("name") or "").strip()
            # Default to generic 'Concept' if type is omitted by LLM
            typ = (it.get("type") or "Concept").strip()
            
            # An entity must at least possess a name to be valid
            if name:
                out.append(Entity(name=name, type=typ))
                
    return out
