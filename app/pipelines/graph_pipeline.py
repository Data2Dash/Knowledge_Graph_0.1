from __future__ import annotations

import asyncio
import re
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
from app.knowledge_graph.chunking.chunk_ranker import rank_chunks

from app.knowledge_graph.extraction.async_runner import run_bounded
from app.knowledge_graph.extraction.entity_extractor import extract_entities
from app.knowledge_graph.extraction.relation_extractor import extract_relations
from app.knowledge_graph.extraction.schema import Entity, Relation
from app.knowledge_graph.postprocess.cleaner import clean_entities_relations

from app.knowledge_graph.store.vector_store import InMemoryVectorStore
from app.knowledge_graph.store.neo4j_store import sync_graph_documents

LOGGER = setup_logging("data2dash.pipeline")

_ONLY_NUMBER = re.compile(r"^\d+(\.\d+)?$")


def _chunks_fallback(text: str, cfg: PipelineConfig) -> List[str]:
    page_chunks = page_based_chunks(text, min_page_chars=120)
    section_chunks = split_by_sections(text, max_chunk_size=2600, overlap=900)
    sw_chunks = sliding_window_chunks(text, window_size=2400, step=700, max_chunks=40)

    seen = set()
    out: List[str] = []
    for c in (page_chunks + section_chunks + sw_chunks):
        c = (c or "").strip()
        if len(c) < 200:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)

    return out[: cfg.max_total_chunks]


def _build_chunks(text: str, cfg: PipelineConfig) -> List[Chunk]:
    strat = (cfg.chunk_strategy or "semantic").lower().strip()

    if strat == "semantic":
        chunks = semantic_chunk(text, cfg)
    elif strat == "sections":
        chunks = [Chunk(str(i + 1), c) for i, c in enumerate(split_by_sections(text, 2600, 900))]
    elif strat == "sliding":
        chunks = [Chunk(str(i + 1), c) for i, c in enumerate(sliding_window_chunks(text, 2400, 700, 40))]
    elif strat == "pages":
        chunks = [Chunk(str(i + 1), c) for i, c in enumerate(page_based_chunks(text, 120))]
    else:
        chunks = [Chunk(str(i + 1), c) for i, c in enumerate(_chunks_fallback(text, cfg))]

    chunks = rank_chunks(chunks)
    chunks = chunks[: cfg.prioritize_top_k]
    chunks = chunks[: cfg.max_total_chunks]
    return chunks


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _norm_rel(r: str) -> str:
    return _norm_text(r or "RELATED_TO").upper().replace(" ", "_")


def _should_drop_node_id(node_id: str) -> bool:
    """
    Filter obvious garbage node ids.
    We DO NOT drop if it contains letters (e.g., "L2.4 regularization" should be kept).
    """
    nid = _norm_text(node_id)
    if not nid:
        return True
    # Drop numeric-only like "28.4"
    if _ONLY_NUMBER.match(nid):
        return True
    # Drop super short like "1." or "."
    if len(nid) <= 1:
        return True
    return False


def _build_nodes_and_edges(
    entities: List[Entity],
    relations: List[Relation],
) -> Tuple[List[Node], List[Relationship]]:
    """
    Critical: ensure every Relationship endpoint exists as a Node.
    This prevents PyVis/vis.js error: "non existent node '...'"
    """
    node_map: Dict[str, Node] = {}

    def add_node(node_id: str, node_type: str) -> None:
        nid = _norm_text(str(node_id))
        ntype = _norm_text(str(node_type or "Concept")) or "Concept"
        if _should_drop_node_id(nid):
            return
        key = nid.lower()
        if key not in node_map:
            node_map[key] = Node(id=nid, type=ntype)

    # 1) Nodes from entities
    for e in entities:
        add_node(e.name, e.type)

    # 2) Ensure endpoints exist for every relation + build edges
    edges: List[Relationship] = []
    for r in relations:
        h = _norm_text(r.head)
        t = _norm_text(r.tail)
        rt = _norm_rel(r.relation)

        if _should_drop_node_id(h) or _should_drop_node_id(t):
            continue
        if not rt:
            rt = "RELATED_TO"

        add_node(h, r.head_type)
        add_node(t, r.tail_type)

        # If still missing (because got filtered), skip edge safely
        hs = h.lower()
        ts = t.lower()
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
    ents = extract_entities(llm, text, cfg.max_chunk_chars_for_llm)
    rels = extract_relations(llm, text, ents, cfg.max_chunk_chars_for_llm)
    return ents, rels


def run_async(coro):
    """
    Keep it simple. If Streamlit ever complains about event loop,
    we can move execution to a thread in the UI.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def _make_job(llm, chunk_text: str, cfg: PipelineConfig) -> Callable[[], Awaitable[Tuple[List[Entity], List[Relation]]]]:
    async def job():
        return await _extract_for_chunk(llm, chunk_text, cfg)
    return job


def generate_knowledge_graph(
    source: str,
    is_path: bool = True,
    cfg: Optional[PipelineConfig] = None,
) -> Tuple[InMemoryVectorStore, List[GraphDocument], Optional[bool]]:
    cfg = cfg or PipelineConfig()
    llm = build_llm(cfg)

    # 1) Load
    raw = load_pdf_text(source, with_page_markers=True) if is_path else (source or "")
    if not raw.strip():
        gd = GraphDocument(nodes=[], relationships=[], source=Document(page_content=""))
        return InMemoryVectorStore(), [gd], None

    # 2) Preprocess
    text = preprocess_text(raw)

    # 3) Chunk
    chunks = _build_chunks(text, cfg)
    if not chunks:
        gd = GraphDocument(nodes=[], relationships=[], source=Document(page_content=""))
        return InMemoryVectorStore(), [gd], None

    LOGGER.info("Chunks: %d (strategy=%s)", len(chunks), cfg.chunk_strategy)

    # 4) Vector store indexing (GraphRAG hook)
    vstore = InMemoryVectorStore()
    vstore.add_texts([c.id for c in chunks], [c.text for c in chunks])

    # 5) Extract (async bounded)
    jobs = [_make_job(llm, c.text, cfg) for c in chunks]
    results = run_async(run_bounded(cfg, jobs))

    all_entities: List[Entity] = []
    all_relations: List[Relation] = []
    for ents, rels in results:
        all_entities.extend(ents)
        all_relations.extend(rels)

    # 6) Clean/dedupe
    all_entities, all_relations = clean_entities_relations(all_entities, all_relations)

    # 7) Build GraphDocument (SAFE endpoints)
    nodes, rels = _build_nodes_and_edges(all_entities, all_relations)

    graph_doc = GraphDocument(
        nodes=nodes,
        relationships=rels,
        source=Document(page_content="Merged chunks"),
    )

    LOGGER.info("Nodes: %d | Relations: %d", len(nodes), len(rels))

    # 8) Optional Neo4j sync
    sync_status = None
    if cfg.sync_neo4j:
        sync_status = sync_graph_documents(cfg, [graph_doc])

    return vstore, [graph_doc], sync_status
