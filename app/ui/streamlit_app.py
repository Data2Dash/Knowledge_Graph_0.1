from __future__ import annotations

import os
import time
import tempfile
import streamlit as st
import streamlit.components.v1 as components

from app.core.config import PipelineConfig
from app.pipelines.graph_pipeline import generate_knowledge_graph
from app.knowledge_graph.visualization.pyvis_visualizer import visualize_graph

def run_app():
    st.set_page_config(
        page_title="Data2Dash – Knowledge Graph Extractor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
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
    """, unsafe_allow_html=True)

    st.title("📊 Data2Dash")
    st.markdown("**Knowledge Graph Extractor** — Transform research papers and text into interactive, relational knowledge graphs.")
    st.caption("Data2Dash project")

    st.sidebar.title("Data2Dash")
    st.sidebar.caption("Configuration")

    input_method = st.sidebar.selectbox("Select Input Source:", ["📄 Upload PDF/TXT", "✍️ Manual Text Input"])
    st.sidebar.divider()

    with st.sidebar.expander("⚙️ Extraction Settings", expanded=False):
        chunk_strategy = st.selectbox("Chunk Strategy", ["semantic", "sections", "sliding", "pages"], index=0)
        max_chunks = st.slider("Max chunks (cost control)", 10, 60, 40, 2)
        top_k = st.slider("Top prioritized chunks", 10, 60, 28, 2)
        concurrency = st.slider("Concurrency (rate-limit risk)", 1, 12, 6, 1)
        min_rels = st.slider("Min relations target", 10, 80, 35, 1)

    sync_neo4j = st.sidebar.checkbox("🔗 Sync to Neo4j Database", value=False)
    neo4j_url = neo4j_user = neo4j_pass = None
    if sync_neo4j:
        with st.sidebar.expander("Neo4j Credentials", expanded=True):
            neo4j_url = st.text_input("Neo4j URL", value=os.getenv("NEO4J_URL", "bolt://localhost:7687"))
            neo4j_user = st.text_input("Neo4j Username", value=os.getenv("NEO4J_USERNAME", "neo4j"))
            neo4j_pass = st.text_input("Neo4j Password", value=os.getenv("NEO4J_PASSWORD", ""), type="password")

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

    if source:
        if st.sidebar.button("🚀 Generate Knowledge Graph"):
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

            with st.spinner("🧠 Extracting entities + relations (2-pass) ..."):
                try:
                    vstore, graph_docs, sync_status = generate_knowledge_graph(source, is_path=is_path, cfg=cfg)
                    node_count = len(graph_docs[0].nodes or [])
                    rel_count = len(graph_docs[0].relationships or [])
                    st.success(f"✨ Knowledge graph generated with {node_count} nodes and {rel_count} relationships!")

                    if sync_neo4j:
                        if sync_status:
                            st.info("✅ Successfully synced to Neo4j.")
                        else:
                            st.warning("⚠️ Neo4j sync failed.")

                    # Visualize
                    run_id = str(int(time.time()))
                    expected_html = f"knowledge_graph_{run_id}.html"
                    outpath = visualize_graph(graph_docs, output_file=expected_html)

                    if outpath and os.path.exists(outpath):
                        with open(outpath, "r", encoding="utf-8", errors="ignore") as f:
                            components.html(f.read(), height=800, scrolling=True)
                        with open(outpath, "rb") as f:
                            st.download_button("📥 Download HTML Graph", data=f, file_name="Data2Dash_knowledge_graph.html", mime="text/html")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                finally:
                    if temp_pdf_path and os.path.exists(temp_pdf_path):
                        try:
                            os.remove(temp_pdf_path)
                        except Exception:
                            pass
    else:
        st.info("👈 Please upload a file or enter text in the sidebar to get started.")

    st.markdown("---")
    st.markdown("**Data2Dash** — Research & Knowledge Extraction")
