from __future__ import annotations

import os
import time
import tempfile
import streamlit as st
import streamlit.components.v1 as components

from app.core.config import PipelineConfig
from app.pipelines.graph_pipeline import generate_knowledge_graph
from app.knowledge_graph.visualization.pyvis_visualizer import visualize_graph

from app.knowledge_graph.graph_rag.query_engine import run_query, QueryConfig
from app.knowledge_graph.llm.groq_client import build_llm


def _init_state() -> None:
    defaults = {
        "vstore": None,
        "graph_docs": None,
        "cfg": None,
        "last_html": None,
        "sync_status": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def run_app():
    st.set_page_config(
        page_title="Data2Dash – Knowledge Graph Extractor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()

    # ---------------- Custom CSS ----------------
    st.markdown(
        """
        <style>
        .main { background-color: #f8f9fa; color: #212529; }
        .stButton>button {
            width: 100%; border-radius: 8px; height: 3.5em;
            background-color: #4b6cb7; color: white; font-weight: 600; border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3a539b; box-shadow: 0 4px 12px rgba(75, 108, 183, 0.3); transform: translateY(-1px);
        }
        .stSidebar { background-color: #111111; color: #ffffff; }
        .stSidebar [data-testid="stMarkdownContainer"] p { color: #ffffff !important; }
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar label { color: #ffffff !important; }
        .stSidebar .stSelectbox label, .stSidebar .stRadio label, .stSidebar .stCheckbox label { color: #ffffff !important; }
        h1 { color: #1a1c23; font-family: 'Inter', sans-serif; font-weight: 700; }
        [data-testid="stAppViewContainer"] .main .block-container { padding-top: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Main Header ----------------
    st.title("📊 Data2Dash")
    st.markdown("**Knowledge Graph Extractor** — Transform research papers and text into interactive, relational knowledge graphs.")
    st.caption("Data2Dash project")

    # ---------------- Sidebar ----------------
    st.sidebar.title("Data2Dash")
    st.sidebar.caption("Configuration")

    input_method = st.sidebar.selectbox(
        "Select Input Source:",
        ["📄 Upload PDF/TXT", "✍️ Manual Text Input"],
    )
    st.sidebar.divider()

    # NOTE: Inside sidebar expander, use st.<widget> (not st.sidebar.<widget>)
    # otherwise widgets can appear outside the expander in some Streamlit versions. :contentReference[oaicite:2]{index=2}
    with st.sidebar.expander("⚙️ Extraction Settings", expanded=False):
        chunk_strategy = st.selectbox(
            "Chunk Strategy",
            ["custom", "semantic", "sections", "sliding", "pages"],
            index=0,
            help="custom = page markers + regex headings + paragraph windows",
        )
        max_chunks = st.slider("Max chunks (cost control)", 10, 60, 40, 2)
        top_k = st.slider("Top prioritized chunks", 10, 60, 28, 2)
        concurrency = st.slider("Concurrency (rate-limit risk)", 1, 12, 6, 1)
        min_rels = st.slider("Min relations target", 10, 80, 35, 1)

        # Optional knobs for custom chunker (safe even if PipelineConfig ignores them)
        if chunk_strategy == "custom":
            target_words = st.slider("Target words per chunk", 400, 1600, 900, 50)
            overlap_words = st.slider("Overlap words", 0, 400, 150, 10)
            drop_refs = st.checkbox("Drop References section", value=True)
        else:
            target_words, overlap_words, drop_refs = 900, 150, True

    # ---------- Neo4j ----------
    st.sidebar.divider()
    sync_neo4j = st.sidebar.checkbox("🔗 Sync to Neo4j Database", value=False)
    neo4j_url = neo4j_user = neo4j_pass = None
    if sync_neo4j:
        with st.sidebar.expander("Neo4j Credentials", expanded=True):
            neo4j_url = st.text_input("Neo4j URL", value=os.getenv("NEO4J_URL", "bolt://localhost:7687"))
            neo4j_user = st.text_input("Neo4j Username", value=os.getenv("NEO4J_USERNAME", "neo4j"))
            neo4j_pass = st.text_input("Neo4j Password", value=os.getenv("NEO4J_PASSWORD", ""), type="password")

    st.sidebar.divider()

    # ---------------- Input ----------------
    source = None
    is_path = False
    temp_pdf_path = None

    if "pdf" in input_method.lower():
        uploaded_file = st.sidebar.file_uploader("Upload Research Paper (PDF or TXT)", type=["pdf", "txt"])
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    temp_pdf_path = tmp.name
                source = temp_pdf_path
                is_path = True
            else:
                raw = uploaded_file.read()
                try:
                    source = raw.decode("utf-8")
                except UnicodeDecodeError:
                    source = raw.decode("latin-1", errors="ignore")
                is_path = False
    else:
        source = st.sidebar.text_area("Paste your research abstract or full text:", height=300)
        is_path = False

    # ---------------- Generate KG ----------------
    if source and st.sidebar.button("🚀 Generate Knowledge Graph"):
        cfg = PipelineConfig(
            chunk_strategy=chunk_strategy,
            max_total_chunks=max_chunks,
            prioritize_top_k=top_k,
            max_concurrent_chunks=concurrency,
            min_relationships_target=min_rels,
            sync_neo4j=sync_neo4j,
            neo4j_url=neo4j_url or "bolt://localhost:7687",
            neo4j_user=neo4j_user or "neo4j",
            neo4j_password=neo4j_pass or "",
        )

        # These are optional; only used if your PipelineConfig supports them
        # (won't crash if PipelineConfig ignores unknown attributes)
        try:
            cfg.target_chunk_words = target_words
            cfg.chunk_overlap_words = overlap_words
            cfg.drop_references = drop_refs
        except Exception:
            pass

        st.session_state.cfg = cfg

        with st.spinner("🧠 Extracting entities + relations (2-pass) ..."):
            try:
                vstore, graph_docs, sync_status = generate_knowledge_graph(source, is_path=is_path, cfg=cfg)
                st.session_state.vstore = vstore
                st.session_state.graph_docs = graph_docs
                st.session_state.sync_status = sync_status

                node_count = len(graph_docs[0].nodes or [])
                rel_count = len(graph_docs[0].relationships or [])
                st.success(f"✨ Knowledge graph generated with {node_count} nodes and {rel_count} relationships!")

                if sync_neo4j:
                    st.info("✅ Successfully synced to Neo4j." if sync_status else "⚠️ Neo4j sync failed.")

                run_id = str(int(time.time()))
                output_dir = os.path.join("output", "graphs")
                os.makedirs(output_dir, exist_ok=True)

                html_path = os.path.join(output_dir, f"knowledge_graph_{run_id}.html")
                outpath = visualize_graph(graph_docs, output_file=html_path)
                st.session_state.last_html = outpath

            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    try:
                        os.remove(temp_pdf_path)
                    except Exception:
                        pass

    # ---------------- Show Graph if exists ----------------
    if st.session_state.last_html and os.path.exists(st.session_state.last_html):
        # Showing saved HTML via components.html is a standard pattern. :contentReference[oaicite:3]{index=3}
        with open(st.session_state.last_html, "r", encoding="utf-8", errors="ignore") as f:
            components.html(f.read(), height=800, scrolling=True)

        with open(st.session_state.last_html, "rb") as f:
            st.download_button(
                "📥 Download HTML Graph",
                data=f,
                file_name="Data2Dash_knowledge_graph.html",
                mime="text/html",
            )
    else:
        st.info("👈 Please upload a file or enter text in the sidebar, then click **Generate Knowledge Graph**.")

    # ---------------- GraphRAG Query ----------------
    st.markdown("---")
    st.markdown("## 🔎 Ask the Graph (GraphRAG)")

    if st.session_state.vstore is None or st.session_state.cfg is None:
        st.warning("Generate a Knowledge Graph first — then you can query it here.")
        return

    question = st.text_input(
        "Ask a question about the paper:",
        placeholder="e.g., What dataset did the model train on and what metric improved?",
    )

    use_neo = st.checkbox("Use Neo4j graph facts (if synced)", value=False)

    colA, colB = st.columns(2)
    with colA:
        topk = st.slider("Top-K chunks", 2, 12, 6, 1)
    with colB:
        max_chars = st.slider("Max chars per chunk", 300, 2000, 1200, 100)

    if question and st.button("🧠 Answer with GraphRAG"):
        cfg = st.session_state.cfg
        vstore = st.session_state.vstore

        try:
            llm = build_llm(cfg)
            qc = QueryConfig(top_k_chunks=topk, max_chunk_chars_each=max_chars)

            answer, retrieved, context = run_query(
                llm=llm,
                vstore=vstore,
                question=question,
                qc=qc,
                neo4j_url=cfg.neo4j_url,
                neo4j_user=cfg.neo4j_user,
                neo4j_password=cfg.neo4j_password,
                use_neo4j=use_neo,
            )

            st.markdown("### ✅ Answer")
            st.write(answer)

            with st.expander("📎 Retrieved Evidence"):
                for ch in retrieved:
                    st.markdown(f"**Chunk {ch.chunk_id}** (score={ch.score:.3f})")
                    st.write(ch.text[:1500] + ("..." if len(ch.text) > 1500 else ""))

            with st.expander("🧱 Full Context sent to the LLM"):
                st.code(context)

        except Exception as e:
            st.error(f"❌ Query Error: {e}")

    st.markdown("---")
    st.markdown("**Data2Dash** — Research & Knowledge Extraction")