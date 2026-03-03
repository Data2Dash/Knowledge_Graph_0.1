"""
Main Data-to-Dashboard Graph Pipeline.

This module orchestrates the entire process of converting raw text or PDF documents 
into a structured Knowledge Graph. It handles:
- Document ingestion and preprocessing
- Semantic or custom chunking
- LLM-based entity and relationship extraction
- Node and edge construction
- Storage in Neo4j and local vector databases
"""
from __future__ import annotations

import asyncio
import re
import logging
from typing import Optional, Tuple, List, Dict, Callable, Awaitable

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_experimental.graph_transformers.llm import GraphDocument

from app.core.config import PipelineConfig
from app.core.logging import setup_logging
from app.knowledge_graph.llm.groq_client import build_llm

from app.knowledge_graph.ingestion.pdf_loader import load_pdf_text
from app.knowledge_graph.preprocessing.text_cleaner import (
    preprocess_text,
    split_by_sections,
    sliding_window_chunks,
    page_based_chunks,
)

from app.knowledge_graph.chunking.semantic_chunker import semantic_chunk, Chunk
from app.knowledge_graph.chunking.custom_chunker import custom_chunk
from app.knowledge_graph.chunking.chunk_ranker import rank_chunks

from app.knowledge_graph.extraction.async_runner import run_bounded
from app.knowledge_graph.extraction.joint_extractor import extract_jointly
from app.knowledge_graph.extraction.schema import Entity, Relation
from app.knowledge_graph.postprocess.cleaner import clean_entities_relations

from app.knowledge_graph.store.vector_store import InMemoryVectorStore
from app.knowledge_graph.store.neo4j_store import sync_graph_documents

setup_logging()  # Configure logging handlers globally once
LOGGER = logging.getLogger("data2dash.pipeline")

# Regex to detect strings that are strictly numeric (e.g., "123", "45.6")
_ONLY_NUMBER = re.compile(r"^\d+(\.\d+)?$")


def _chunks_fallback(text: str, cfg: PipelineConfig) -> List[str]:
    """
    Fallback chunking strategy that combines multiple chunking approaches
    when standard strategies are not specified or applicable.

    Args:
        text: The raw text to chunk.
        cfg: Pipeline configuration containing max chunk limits.

    Returns:
        A deduplicated list of text chunks up to `max_total_chunks`.
    """
    # Generate chunks using three diverse strategies
    page_chunks = page_based_chunks(text, min_page_chars=120)
    section_chunks = split_by_sections(text, max_chunk_size=2600, overlap=900)
    sw_chunks = sliding_window_chunks(text, window_size=2400, step=700, max_chunks=40)

    seen = set()
    out: List[str] = []
    
    # Iterate over all generated chunks sequentially
    for c in (page_chunks + section_chunks + sw_chunks):
        c = (c or "").strip()
        # Drop excessively short chunks
        if len(c) < 200:
            continue
        # Avoid duplicate chunks
        if c in seen:
            continue
        seen.add(c)
        out.append(c)

    # Return only up to the maximum allowed number of chunks
    return out[: cfg.max_total_chunks]


def _build_chunks(text: str, cfg: PipelineConfig) -> List[Chunk]:
    """
    Selects and executes the configured chunking strategy.

    Args:
        text: The complete input text to process.
        cfg: Configuration dictating strategy and chunk limits.

    Returns:
        A prioritized and truncated list of `Chunk` objects ready for extraction.
    """
    strat = (cfg.chunk_strategy or "semantic").lower().strip()

    # Route to appropriate chunking method based on configuration
    if strat == "semantic":
        # Semantic chunking via embedding similarity
        chunks = semantic_chunk(text, cfg)

    elif strat == "custom":
        # Custom regex-based and page-marker chunking tailored for structured PDFs
        target_words = getattr(cfg, "target_chunk_words", 900)
        overlap_words = getattr(cfg, "chunk_overlap_words", 150)
        drop_refs = getattr(cfg, "drop_references", True)

        chunks = custom_chunk(
            text_with_page_markers=text,
            target_words=target_words,
            overlap_words=overlap_words,
            drop_references=drop_refs,
        )

    elif strat == "sections":
        # Split purely by section metadata (headers/paragraphs)
        chunks = [Chunk(str(i + 1), c) for i, c in enumerate(split_by_sections(text, 2600, 900))]

    elif strat == "sliding":
        # Raw sliding window over the text corpus
        chunks = [Chunk(str(i + 1), c) for i, c in enumerate(sliding_window_chunks(text, 2400, 700, 40))]

    elif strat == "pages":
        # Split text strictly on page boundaries if available
        chunks = [Chunk(str(i + 1), c) for i, c in enumerate(page_based_chunks(text, 120))]

    else:
        # Fall back to combined chunker if strategy string is unknown
        chunks = [Chunk(str(i + 1), c) for i, c in enumerate(_chunks_fallback(text, cfg))]

    # Rank the resulting chunks to prioritize the most informative ones
    chunks = rank_chunks(chunks)
    
    # Enforce configured quantity bounds
    chunks = chunks[: cfg.prioritize_top_k]
    chunks = chunks[: cfg.max_total_chunks]
    
    return chunks


def _norm_text(s: str) -> str:
    """Normalize text by collapsing multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", (s or "").strip())


def _norm_rel(r: str) -> str:
    """Normalize relation names to uppercase with underscores replacing spaces."""
    return _norm_text(r or "RELATED_TO").upper().replace(" ", "_")



def _should_drop_node_id(node_id: str) -> bool:
    """
    Filter out meaningless or noisy node identifiers.
    
    Retains nodes containing letters (e.g., "L2.4 regularization" is kept),
    but filters purely numerical arrays, fragments, or empty nodes.
    
    Args:
        node_id: String identifier for the node to check.
    Returns:
        True if the node should be dropped/filtered, False otherwise.
    """
    nid = _norm_text(node_id)
    if not nid:
        return True
    # Drop numeric-only entries like "28.4" or "100"
    if _ONLY_NUMBER.match(nid):
        return True
    # Drop super short remnants like "1.", ".", or "A"
    if len(nid) <= 1:
        return True
    return False


def _build_nodes_and_edges(
    entities: List[Entity],
    relations: List[Relation],
) -> Tuple[List[Node], List[Relationship]]:
    """
    Converts raw entity and relation extract structures into valid Graph Nodes and Relationships.
    
    Crucially, it guarantees that every source and target endpoint of a relationship
    exists as a concrete Node. This prevents visualization libraries (like pyvis/vis.js)
    from crashing due to missing node references.
    
    Args:
        entities: List of isolated Entity objects found in the text.
        relations: List of Relationship objects connecting entities.
        
    Returns:
        A tuple of (List of clean Nodes, List of valid Relationships).
    """
    node_map: Dict[str, Node] = {}

    def add_node(node_id: str, node_type: str) -> None:
        """Helper to conditionally normalize and insert a node into the node pool."""
        nid = _norm_text(str(node_id))
        ntype = _norm_text(str(node_type or "Concept")) or "Concept"
        
        # Skip garbage nodes
        if _should_drop_node_id(nid):
            return
            
        key = nid.lower()
        if key not in node_map:
            node_map[key] = Node(id=nid, type=ntype)

    # 1) Base population: Register all explicitly declared entities as nodes
    for e in entities:
        add_node(e.name, e.type)

    # 2) Edge construction: Ensure endpoints exist for every relation, then map the edges
    edges: List[Relationship] = []
    for r in relations:
        h = _norm_text(r.head)
        t = _norm_text(r.tail)
        rt = _norm_rel(r.relation)

        # Skip relationships involving garbage nodes
        if _should_drop_node_id(h) or _should_drop_node_id(t):
            continue
            
        if not rt:
            rt = "RELATED_TO"

        # Dynamically inject the head and tail nodes if they weren't explicitly extracted
        # but are referenced by a valid relationship
        add_node(h, r.head_type)
        add_node(t, r.tail_type)

        hs = h.lower()
        ts = t.lower()
        
        # Guard check: If the node was dropped (e.g. numeric only), skip the edge
        if hs not in node_map or ts not in node_map:
            continue

        edges.append(
            Relationship(
                source=node_map[hs],
                target=node_map[ts],
                type=rt,
            )
        )

    return list(node_map.values()), edges


async def _extract_for_chunk(llm, text: str, cfg: PipelineConfig) -> Tuple[List[Entity], List[Relation]]:
    """Async wrapper to execute the joint entity-relationship extraction for a single text chunk."""
    return extract_jointly(llm, text, cfg.max_chunk_chars_for_llm)


def run_async(coro):
    """
    Synchronously runs an asynchronous coroutine.
    
    Robust against existing event loops (e.g., in Streamlit or Jupyter environments).
    If a loop already exists, we dispatch to it; otherwise we invoke `asyncio.run()`.
    
    Args:
        coro: The awaitable coroutine to run.
    Returns:
        The evaluated result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No active loop: Standard execution
        return asyncio.run(coro)
    # Active loop: Run until the task gets completed in the current thread loop
    return loop.run_until_complete(coro)


def _make_job(llm, chunk_text: str, cfg: PipelineConfig) -> Callable[[], Awaitable[Tuple[List[Entity], List[Relation]]]]:
    """
    Factory function to create deferred async extraction jobs.
    This encapsulation avoids immediate scheduling until passed to the async concurrency runner.
    """
    async def job():
        return await _extract_for_chunk(llm, chunk_text, cfg)
    return job


def generate_knowledge_graph(
    source: str,
    is_path: bool = True,
    cfg: Optional[PipelineConfig] = None,
) -> Tuple[InMemoryVectorStore, List[GraphDocument], Optional[bool]]:
    """
    Main entry point for generating a Knowledge Graph from a text source.

    Executes the comprehensive pipeline:
    Load -> Clean -> Chunk -> Index -> Extract -> Dedupe -> Graph Build -> Store

    Args:
        source: Path to the input file (if is_path=True) or literal text string.
        is_path: Indicates if `source` should be interpreted as a file path.
        cfg: The current system configuration and hyperparameter object.
        
    Returns:
        A tuple containing: 
        (In-memory VectorStore for RAG, List of built GraphDocuments, Status of optional Neo4J sync).
    """
    cfg = cfg or PipelineConfig()
    llm = build_llm(cfg)

    # 1) Ingestion phase: Import PDF or process raw string
    raw = load_pdf_text(source, with_page_markers=True) if is_path else (source or "")
    if not raw.strip():
        # Fallback if text is empty
        gd = GraphDocument(nodes=[], relationships=[], source=Document(page_content=""))
        return InMemoryVectorStore(), [gd], None

    # 2) Preprocessing phase: Remove boilerplate and sanitize tokens
    text = preprocess_text(raw)

    # 3) Chunking phase: Partition text with designated strategy
    chunks = _build_chunks(text, cfg)
    if not chunks:
        gd = GraphDocument(nodes=[], relationships=[], source=Document(page_content=""))
        return InMemoryVectorStore(), [gd], None

    LOGGER.info("Chunks: %d (strategy=%s)", len(chunks), cfg.chunk_strategy)

    # 4) Initial Retrieval Setup (GraphRAG hook): Store source chunks in a local vector base 
    # to facilitate traditional semantic search and evidence linking.
    vstore = InMemoryVectorStore()
    vstore.add_texts([c.id for c in chunks], [c.text for c in chunks])

    # 5) Semantic Extraction via LLM: Convert partitioned chunks into Graph structure (Entities, Relations)
    # Leverages asyncio to parallelize API requests within configured bounds
    jobs = [_make_job(llm, c.text, cfg) for c in chunks]
    results = run_async(run_bounded(cfg, jobs))

    # Aggregate batched results
    all_entities: List[Entity] = []
    all_relations: List[Relation] = []
    for ents, rels in results:
        all_entities.extend(ents)
        all_relations.extend(rels)

    # 6) Knowledge consolidation: Deduplicate overlapping entities and reconcile relational logic
    all_entities, all_relations = clean_entities_relations(all_entities, all_relations)

    # 7) Graph Schema construction: Formalize structures into LangChain graph primitives
    # ensuring no dangling edges
    nodes, rels = _build_nodes_and_edges(all_entities, all_relations)

    graph_doc = GraphDocument(
        nodes=nodes,
        relationships=rels,
        source=Document(page_content="Merged chunks"),
    )

    LOGGER.info("Nodes: %d | Relations: %d", len(nodes), len(rels))

    # 8) Storage phase: Conditionally export to persistent Neo4j backend
    sync_status = None
    if cfg.sync_neo4j:
        sync_status = sync_graph_documents(cfg, [graph_doc])

    return vstore, [graph_doc], sync_status