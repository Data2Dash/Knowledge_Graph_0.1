"""
pdf_loader.py  —  Enhanced PDF ingestion for AI research papers using SmolDocling
================================================================
Updates:
  1. Integrates SmolDocling vlm pipeline (Granite-Docling-258M)
  2. Parses documents using generated DocTags based natively on structural understanding
  3. Extracts <formula> tags cleanly into LaTeX representations
  4. Parses <table> tags strictly into structured Markdown
  5. Outputs a list of `Section` objects grounding texts with their inner formulas and tables
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Public dataclasses returned by the loader
# ─────────────────────────────────────────────────────────────

@dataclass
class Section:
    """A structured section extracted natively via the SmolDocling vlm pipeline."""
    heading: str
    text: str
    formulas: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# SmolDocling VLM Pipeline Extractor
# ─────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> List[Section]:
    """
    Load the research PDF and use the vlm pipeline (Granite-Docling-258M) 
    to generate DocTags. Extracts <formula> tags as LaTeX and <table> tags 
    as structured Markdown. Returns rich Section objects.
    """
    sections: List[Section] = []
    
    try:
        from smoldocling import DocumentLoader, Pipeline
    except ImportError:
        LOGGER.warning("smoldocling is not installed. Returning empty sections to fallback.")
        return []

    # Initialize the SmolDocling vlm pipeline leveraging Granite-Docling
    # Instructed to explicitly generate layout DocTags
    pipeline = Pipeline(model="Granite-Docling-258M", enable_doctags=True)
    
    try:
        doc = DocumentLoader(pdf_path).load(pipeline=pipeline)
    except Exception as e:
        LOGGER.error("Failed to parse PDF via SmolDocling: %s", e)
        return []

    current_heading = "Abstract"  # Default starting generic heading
    current_text = []
    current_formulas = []
    current_tables = []
    
    # Iterate through the generated semantic DocTags explicitly mapping formulas & tables
    for element in getattr(doc, "elements", []):
        tag = getattr(element, "tag_name", "").lower()
        
        if tag in ["heading", "title", "h1", "h2", "h3"]:
            # Flush the current section prior to entering a new heading block
            if current_text or current_formulas or current_tables:
                sections.append(Section(
                    heading=current_heading,
                    text="\n".join(current_text).strip(),
                    formulas=list(current_formulas),
                    tables=list(current_tables)
                ))
            
            # Reset buffers for the incoming section
            current_heading = getattr(element, "text", "Unknown heading").strip()
            current_text = []
            current_formulas = []
            current_tables = []
            
        elif tag == "formula":
            # Ensure formulas trigger LaTeX translation via SmolDocling methods
            latency_form = element.get_latex() if hasattr(element, "get_latex") else getattr(element, "text", "")
            if latency_form:
                current_formulas.append(latency_form)
                current_text.append(f"\n[Formula: {latency_form}]\n")
            
        elif tag == "table":
            # Extract the literal structural matrix as Markdown 
            md_table = element.get_markdown() if hasattr(element, "get_markdown") else getattr(element, "text", "")
            if md_table:
                current_tables.append(md_table)
                current_text.append(f"\n{md_table}\n")
                
        else:
            # Aggregate standard paragraphs, list elements, or captions
            txt = getattr(element, "text", "").strip()
            if txt:
                current_text.append(txt)

    # Clean up and append the final straggling section buffer
    if current_text or current_formulas or current_tables:
        sections.append(Section(
            heading=current_heading,
            text="\n".join(current_text).strip(),
            formulas=current_formulas,
            tables=current_tables
        ))

    LOGGER.info("Successfully extracted %d structural Sections via Granite-Docling-258M.", len(sections))
    return sections


# ─────────────────────────────────────────────────────────────
# Backward-compatible wrapper for unstructured extraction pipelines
# ─────────────────────────────────────────────────────────────

def load_pdf_text(pdf_path: str, with_page_markers: bool = True) -> str:
    """
    Wrapper bridging the new robust Section objects into the legacy 
    unstructured flat-text string expected by the core GraphRAG graph_pipeline.
    """
    sections = load_pdf(pdf_path)
    
    # Fallback to PyMuPDF if SmolDocling isn't properly functioning or installed
    if not sections:
        try:
            import fitz # type: ignore
            LOGGER.warning("Falling back to PyMuPDF for raw text extraction...")
            with fitz.open(pdf_path) as doc:
                raw_text = chr(10).join([page.get_text("text") for page in doc])
                return raw_text
        except ImportError:
            return ""

    # Flatten the visually structured segments
    text_parts = []
    for sec in sections:
        text_parts.append(f"\n\n## {sec.heading}\n\n{sec.text}")
    
    return "".join(text_parts).strip()