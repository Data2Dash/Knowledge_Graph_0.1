# app/ui/streamlit_app.py
from __future__ import annotations

import os
import re
import time
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import streamlit.components.v1 as components

from langchain_community.graphs.graph_document import GraphDocument

from app.core.config import PipelineConfig
from app.core.settings import get_settings
from app.core.logging import configure_logging

from app.pipelines.graph_pipeline import generate_knowledge_graph
from app.knowledge_graph.visualization.pyvis_visualizer import visualize_graph

from app.knowledge_graph.graph_rag.query_engine import run_query, QueryConfig
from app.knowledge_graph.llm.groq_client import build_llm

from app.knowledge_graph.extraction.schema import Entity, Relation


# ---- filters to avoid showing hash-like junk labels
_HASH_LIKE_RE = re.compile(r"^[a-f0-9]{12,}$", re.IGNORECASE)
_PAGE_MARK_RE = re.compile(r"^\[PAGE:\d+\]$", re.IGNORECASE)


def _ensure_logging_once() -> None:
    if st.session_state.get("_logging_configured"):
        return

    settings = get_settings()
    settings.ensure_runtime_dirs()

    configure_logging(
        log_level=settings.LOG_LEVEL,
        log_dir=settings.resolved_log_dir(),
        log_json=settings.LOG_JSON,
        app_name=settings.APP_NAME,
    )
    st.session_state["_logging_configured"] = True


def _apply_neo4j_overrides_from_ui(
    *,
    enable: bool,
    url: str,
    user: str,
    password: str,
    database: str = "",
) -> None:
    os.environ["SYNC_NEO4J"] = "true" if enable else "false"
    os.environ["NEO4J_URL"] = url or "bolt://localhost:7687"
    os.environ["NEO4J_USER"] = user or "neo4j"
    os.environ["NEO4J_PASSWORD"] = password or ""
    os.environ["NEO4J_DATABASE"] = database or ""
    get_settings.cache_clear()


def _safe_node_name(node) -> str:
    """
    Prefer readable node name; fallback to id.
    Avoid hash-like + page markers where possible.
    """
    props = getattr(node, "properties", {}) or {}

    cand = (
        getattr(node, "name", None)
        or getattr(node, "label", None)
        or props.get("name")
        or props.get("title")
        or props.get("text")
        or str(getattr(node, "id", "") or "")
    )
    cand = str(cand).strip()

    # If name looks like hash, keep it but we can still show it (better than empty)
    return cand


def _graphdoc_to_entities_relations(gd: GraphDocument) -> Tuple[List[Entity], List[Relation]]:
    """
    Convert GraphDocument to Entity/Relation while filtering junk labels.
    """
    entities: List[Entity] = []
    relations: List[Relation] = []

    # --- nodes
    for node in getattr(gd, "nodes", []) or []:
        try:
            name = _safe_node_name(node)
            if not name or _PAGE_MARK_RE.match(name):
                continue
            # Drop pure hash-like nodes (usually chunk ids) to avoid ugly graphs
            if _HASH_LIKE_RE.match(name):
                continue

            node_type = getattr(node, "type", None) or (getattr(node, "properties", {}) or {}).get("type") or "Concept"
            entities.append(Entity(name=str(name), type=str(node_type)))
        except Exception:
            continue

    # quick index to improve head/tail types
    ent_type_by_name = {e.name: e.type for e in entities}

    # --- edges
    for rel in getattr(gd, "relationships", []) or []:
        try:
            src = getattr(rel, "source", None)
            tgt = getattr(rel, "target", None)
            if not src or not tgt:
                continue

            head = _safe_node_name(src)
            tail = _safe_node_name(tgt)

            if not head or not tail:
                continue
            if _HASH_LIKE_RE.match(head) or _HASH_LIKE_RE.match(tail):
                continue

            rel_type = getattr(rel, "type", None) or (getattr(rel, "properties", {}) or {}).get("type") or "RELATED_TO"
            relations.append(
                Relation(
                    head=str(head),
                    head_type=str(ent_type_by_name.get(str(head), getattr(src, "type", "Concept"))),
                    relation=str(rel_type),
                    tail=str(tail),
                    tail_type=str(ent_type_by_name.get(str(tail), getattr(tgt, "type", "Concept"))),
                    evidence=None,
                    confidence=float(getattr(rel, "confidence", 0.7) or 0.7),
                )
            )
        except Exception:
            continue

    return entities, relations


def run_app() -> None:
    _ensure_logging_once()
    settings = get_settings()
    settings.ensure_runtime_dirs()

    st.set_page_config(
        page_title="Data2Dash – Knowledge Graph Extractor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.session_state.setdefault("vstore", None)
    st.session_state.setdefault("graph_doc", None)
    st.session_state.setdefault("cfg", None)
    st.session_state.setdefault("last_html", None)
    st.session_state.setdefault("last_sync_status", None)

    st.title("Data2Dash")
    st.markdown("Knowledge Graph Extractor — AI Research Ontology Engine")

    # ---------------- Sidebar ----------------
    st.sidebar.title("Configuration")

    input_method = st.sidebar.selectbox("Input Source", ["Upload PDF/TXT", "Manual Text Input"])

    with st.sidebar.expander("Extraction Settings", expanded=False):
        chunk_strategy = st.sidebar.selectbox("Chunk Strategy", ["semantic", "sections", "sliding", "pages"], index=0)
        max_chunks = st.sidebar.slider("Max Chunks", 10, 80, 40, 2)
        top_k = st.sidebar.slider("Top Ranked Chunks", 10, 80, 28, 2)
        concurrency = st.sidebar.slider("Concurrency", 1, 12, 3, 1)  # ✅ default lower (more stable)

        model_name = st.sidebar.text_input("LLM Model", value=settings.MODEL_NAME)
        temperature = st.sidebar.slider("Temperature", 0.0, 2.0, float(settings.TEMPERATURE), 0.1)

    sync_neo4j = st.sidebar.checkbox("Sync to Neo4j", value=bool(settings.SYNC_NEO4J))

    neo4j_url = neo4j_user = neo4j_pass = neo4j_db = ""
    if sync_neo4j:
        neo4j_url = st.sidebar.text_input("Neo4j URL", value=os.getenv("NEO4J_URL", settings.NEO4J_URL))
        neo4j_user = st.sidebar.text_input("Neo4j Username", value=os.getenv("NEO4J_USER", settings.NEO4J_USER))
        neo4j_pass = st.sidebar.text_input("Neo4j Password", value=os.getenv("NEO4J_PASSWORD", ""), type="password")
        neo4j_db = st.sidebar.text_input("Neo4j Database (optional)", value=os.getenv("NEO4J_DATABASE", settings.NEO4J_DATABASE))

    # ---------------- Input Handling ----------------
    source = None
    is_path = False
    temp_path = None

    if input_method == "Upload PDF/TXT":
        uploaded_file = st.sidebar.file_uploader("Upload File", type=["pdf", "txt"])
        if uploaded_file:
            name = (uploaded_file.name or "").lower()
            if name.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    temp_path = tmp.name
                source = temp_path
                is_path = True
            else:
                raw = uploaded_file.read()
                try:
                    source = raw.decode("utf-8")
                except UnicodeDecodeError:
                    source = raw.decode("latin-1", errors="ignore")
                is_path = False
    else:
        source = st.sidebar.text_area("Paste Research Text", height=300)
        is_path = False

    # ---------------- Generate Graph ----------------
    if source and st.sidebar.button("Generate Knowledge Graph"):
        _apply_neo4j_overrides_from_ui(
            enable=sync_neo4j,
            url=neo4j_url,
            user=neo4j_user,
            password=neo4j_pass,
            database=neo4j_db,
        )
        settings = get_settings()
        settings.ensure_runtime_dirs()

        cfg = PipelineConfig(
            chunk_strategy=chunk_strategy,
            max_total_chunks=int(max_chunks),
            prioritize_top_k=int(top_k),
            max_concurrent_chunks=int(concurrency),
            model_name=model_name.strip() or settings.MODEL_NAME,
            temperature=float(temperature),
            request_timeout_s=float(settings.LLM_TIMEOUT_S),
            max_chunk_chars_for_llm=int(settings.MAX_CHUNK_CHARS_FOR_LLM),
            max_retries=int(settings.MAX_RETRIES),
            retry_base_delay_s=float(settings.RETRY_BASE_DELAY),
            retry_max_delay_s=float(settings.RETRY_MAX_DELAY),
        )
        st.session_state.cfg = cfg

        with st.spinner("Extracting structured knowledge..."):
            try:
                vstore, graph_docs, sync_status = generate_knowledge_graph(source, is_path=is_path, cfg=cfg)

                gd = graph_docs[0] if graph_docs else GraphDocument(nodes=[], relationships=[], source=None)
                st.session_state.vstore = vstore
                st.session_state.graph_doc = gd
                st.session_state.last_sync_status = sync_status

                st.success(f"Graph built with {len(gd.nodes)} nodes and {len(gd.relationships)} relationships.")

                # Convert + show quick debug
                entities, relations = _graphdoc_to_entities_relations(gd)
                st.caption(f"After UI conversion: {len(entities)} entities | {len(relations)} relations")
                if entities:
                    st.caption("Sample entities: " + ", ".join([e.name for e in entities[:8]]))

                run_id = str(int(time.time()))
                html_path = Path(settings.OUTPUT_DIR) / f"knowledge_graph_{run_id}.html"

                outpath = visualize_graph(entities=entities, relations=relations, output_file=str(html_path))
                st.session_state.last_html = outpath

            except Exception as e:
                st.error(f"Pipeline Error: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

    # ---------------- Show Graph ----------------
    if st.session_state.last_html and os.path.exists(st.session_state.last_html):
        with open(st.session_state.last_html, "r", encoding="utf-8", errors="ignore") as f:
            components.html(f.read(), height=800, scrolling=True)

        with open(st.session_state.last_html, "rb") as f:
            st.download_button("Download Graph HTML", data=f, file_name="knowledge_graph.html", mime="text/html")

    # ---------------- GraphRAG Query ----------------
    st.markdown("---")
    st.markdown("## Ask the Graph")

    if st.session_state.vstore is None or st.session_state.cfg is None:
        st.info("Generate a graph first.")
        return

    question = st.text_input("Question")

    colA, colB = st.columns(2)
    with colA:
        topk = st.slider("Top-K Chunks", 2, 12, 6, 1)
    with colB:
        max_chars = st.slider("Max Chars per Chunk", 300, 2000, 1200, 100)

    if question and st.button("Answer"):
        try:
            # IMPORTANT: Ask mode must be text-mode (no json_object)
            llm = build_llm(st.session_state.cfg, json_mode=False)

            qc = QueryConfig(top_k_chunks=int(topk), max_chunk_chars_each=int(max_chars))
            settings = get_settings()

            answer, retrieved, context = run_query(
                llm=llm,
                vstore=st.session_state.vstore,
                question=question,
                qc=qc,
                neo4j_url=settings.NEO4J_URL,
                neo4j_user=settings.NEO4J_USER,
                neo4j_password=settings.NEO4J_PASSWORD.get_secret_value(),
            )

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Retrieved Evidence"):
                for ch in retrieved:
                    st.markdown(f"**Chunk {ch.chunk_id}** (score={ch.score:.3f})")
                    st.write(ch.text[:1500] + ("..." if len(ch.text) > 1500 else ""))

            with st.expander("Full Context"):
                st.code(context)

        except Exception as e:
            st.error(f"Query Error: {e}")