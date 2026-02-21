# app/pipelines/graph_pipeline.py
from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from typing import Optional, Tuple, List, Dict, Callable, Awaitable

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument

from app.core.config import PipelineConfig
from app.core.logging import setup_logging
from app.core.settings import get_settings

from app.knowledge_graph.llm.groq_client import build_llm
from app.knowledge_graph.ingestion.pdf_loader import load_pdf_text
from app.knowledge_graph.preprocessing.text_cleaner import make_chunks
from app.knowledge_graph.chunking.chunk_ranker import rank_chunks

from app.knowledge_graph.extraction.async_runner import run_bounded
from app.knowledge_graph.extraction.entity_extractor import extract_entities
from app.knowledge_graph.extraction.relation_extractor import extract_relations
from app.knowledge_graph.extraction.schema import Entity, Relation
from app.knowledge_graph.extraction.validator import dedupe_entities, dedupe_relations

from app.knowledge_graph.store.vector_store import InMemoryVectorStore
from app.knowledge_graph.store.neo4j_store import sync_graph_documents

LOGGER = setup_logging("knowledge_graph.pipeline")

_ONLY_NUMBER = re.compile(r"^\d+(\.\d+)?$")


# =========================
# Utility Helpers
# =========================

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _should_drop_node_id(node_id: str) -> bool:
    nid = _norm_text(node_id)
    if not nid:
        return True
    if _ONLY_NUMBER.match(nid):
        return True
    if len(nid) <= 1:
        return True
    return False


def _build_nodes_and_edges(
    entities: List[Entity],
    relations: List[Relation],
) -> Tuple[List[Node], List[Relationship]]:
    node_map: Dict[str, Node] = {}
    seen_edges = set()

    def add_node(node_id: str, node_type: str) -> None:
        nid = _norm_text(str(node_id))
        ntype = _norm_text(str(node_type or "Concept")) or "Concept"
        if _should_drop_node_id(nid):
            return
        key = nid.lower()
        if key not in node_map:
            node_map[key] = Node(id=nid, type=ntype)

    for e in entities:
        add_node(e.name, e.type)

    edges: List[Relationship] = []

    for r in relations:
        h = _norm_text(r.head)
        t = _norm_text(r.tail)
        rt = _norm_text(getattr(r, "predicate", None) or (r.relation or "RELATED_TO")).upper().replace(" ", "_")

        if _should_drop_node_id(h) or _should_drop_node_id(t):
            continue
        if h.lower() == t.lower():
            continue

        add_node(h, r.head_type)
        add_node(t, r.tail_type)

        hs, ts = h.lower(), t.lower()
        if hs not in node_map or ts not in node_map:
            continue

        edge_key = (hs, ts, rt)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        edges.append(Relationship(source=node_map[hs], target=node_map[ts], type=rt))

    return list(node_map.values()), edges


def run_async(coro):
    """
    Streamlit-safe runner:
    - if no running loop -> asyncio.run
    - if loop exists -> run in a separate thread
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import threading

    result_holder = {"value": None, "err": None}

    def _runner():
        try:
            result_holder["value"] = asyncio.run(coro)
        except Exception as e:
            result_holder["err"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()

    if result_holder["err"] is not None:
        raise result_holder["err"]
    return result_holder["value"]


def _merge_local_and_global_entities(
    local_ents: List[Entity],
    global_ents: List[Entity],
    *,
    max_total: int = 80,
) -> List[Entity]:
    out: List[Entity] = []
    seen = set()

    def add(e: Entity):
        key = (e.name.lower(), e.type)
        if key in seen:
            return
        seen.add(key)
        out.append(e)

    for e in local_ents:
        add(e)
        if len(out) >= max_total:
            return out

    for e in global_ents:
        add(e)
        if len(out) >= max_total:
            return out

    return out
def _make_job_entities(llm, chunk_text: str, cfg: PipelineConfig, context: dict):
    timeout_s = float(getattr(cfg, "request_timeout_s", 60.0))
    max_chars = int(getattr(cfg, "max_chunk_chars_for_llm", 6000))  # ✅ important

    async def job():
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: extract_entities(
                    llm=llm,
                    text=chunk_text,
                    max_chars=max_chars,   # ✅ FIX
                    cfg=cfg,
                    context=context,
                ),
            ),
            timeout=timeout_s,
        )

    return job

def _make_job_relations(llm, chunk_text: str, entities: List[Entity], cfg: PipelineConfig, context: dict):
    timeout_s = float(getattr(cfg, "request_timeout_s", 60.0))

    async def job():
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: extract_relations(
                    llm=llm,
                    chunk_text=chunk_text,
                    entities=entities,
                    cfg=cfg,
                    context=context,
                ),
            ),
            timeout=timeout_s,
        )

    return job


# =========================
# Main Pipeline
# =========================

def generate_knowledge_graph(
    source: str,
    is_path: bool = True,
    cfg: Optional[PipelineConfig] = None,
) -> Tuple[InMemoryVectorStore, List[GraphDocument], Optional[bool]]:
    start_time = time.time()

    settings = get_settings()
    settings.ensure_runtime_dirs()

    cfg = cfg or PipelineConfig()
    llm = build_llm(cfg, settings=settings)

    raw = load_pdf_text(source, with_page_markers=True) if is_path else (source or "")
    if not raw.strip():
        gd = GraphDocument(nodes=[], relationships=[], source=Document(page_content=""))
        return InMemoryVectorStore(), [gd], None

    chunks = make_chunks(raw, cfg.chunk_strategy, pipeline_cfg=cfg)
    if not chunks:
        gd = GraphDocument(nodes=[], relationships=[], source=Document(page_content=""))
        return InMemoryVectorStore(), [gd], None

    ranked = rank_chunks(chunks, cfg=cfg)

    # Respect both knobs deterministically
    max_keep = min(int(cfg.max_total_chunks), int(cfg.prioritize_top_k))
    ranked = ranked[:max_keep]

    chunk_ids = [str(c.id) for c in ranked]
    chunk_texts = [str(c.text) for c in ranked]

    LOGGER.info("Chunks selected", extra={"count": len(chunk_texts)})

    # Vector store
    vstore = InMemoryVectorStore()
    vstore.add_texts(
        ids=[f"chunk:{cid}" for cid in chunk_ids],
        texts=chunk_texts,
        cfg=cfg,
        metas=[{"kind": "chunk", "chunk_id": cid} for cid in chunk_ids],
    )

    # =========================
    # Stage A — Entities
    # =========================
    ent_jobs = []
    for cid, txt in zip(chunk_ids, chunk_texts):
        ctx = {"chunk_id": cid, "source": source}
        ent_jobs.append(_make_job_entities(llm, txt, cfg, ctx))

    ent_results = run_async(run_bounded(cfg, ent_jobs, return_exceptions=True, context={"stage": "entities"}))

    ent_lists: List[List[Entity]] = []
    for i, result in enumerate(ent_results):
        if isinstance(result, Exception):
            LOGGER.error("Entity extraction failed", extra={"chunk_id": chunk_ids[i], "error": repr(result)[:800]})
            ent_lists.append([])
        else:
            ent_lists.append(result)

    all_entities = [e for sub in ent_lists for e in sub]
    all_entities = dedupe_entities(all_entities)

    LOGGER.info("Entities extracted", extra={"count": len(all_entities)})

    # Select global entities by frequency across chunks (helps relation grounding)
    freq = Counter((e.name.lower(), e.type) for e in all_entities)
    global_entities = sorted(all_entities, key=lambda e: freq[(e.name.lower(), e.type)], reverse=True)[:200]

    # =========================
    # Stage B — Relations
    # =========================
    rel_jobs = []
    for cid, txt, local_ents in zip(chunk_ids, chunk_texts, ent_lists):
        ctx = {"chunk_id": cid, "source": source}
        ents_for_chunk = _merge_local_and_global_entities(
            local_ents,
            global_entities,
            max_total=int(getattr(cfg, "relation_max_entities_in_prompt", 80)),
        )
        rel_jobs.append(_make_job_relations(llm, txt, ents_for_chunk, cfg, ctx))

    rel_results = run_async(run_bounded(cfg, rel_jobs, return_exceptions=True, context={"stage": "relations"}))

    rel_lists: List[List[Relation]] = []
    for i, result in enumerate(rel_results):
        if isinstance(result, Exception):
            LOGGER.error("Relation extraction failed", extra={"chunk_id": chunk_ids[i], "error": str(result)[:500]})
            rel_lists.append([])
        else:
            rel_lists.append(result)

    all_relations = [r for sub in rel_lists for r in sub]

    # Canonicalize + type-infer using entity list, then dedupe
    all_relations = dedupe_relations(all_relations, all_entities)

    LOGGER.info("Relations extracted", extra={"count": len(all_relations)})

    # =========================
    # Build Graph
    # =========================
    nodes, rels = _build_nodes_and_edges(all_entities, all_relations)

    graph_doc = GraphDocument(
        nodes=nodes,
        relationships=rels,
        source=Document(page_content="Merged chunks"),
    )

    LOGGER.info("Final graph", extra={"nodes": len(nodes), "relationships": len(rels)})

    # =========================
    # Neo4j Sync (from Settings)
    # =========================
    sync_status: Optional[bool] = None
    if settings.SYNC_NEO4J:
        try:
            sync_status = sync_graph_documents(settings, [graph_doc])
        except Exception as e:
            LOGGER.error("Neo4j sync failed", extra={"error": str(e)[:700]})
            sync_status = False

    LOGGER.info("Pipeline completed", extra={"duration_s": round(time.time() - start_time, 2)})

    return vstore, [graph_doc], sync_status