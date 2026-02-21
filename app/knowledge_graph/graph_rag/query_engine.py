# app/knowledge_graph/graph_rag/query_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

from pydantic import SecretStr
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.core.logging import setup_logging
from app.core.settings import Settings, get_settings
from app.knowledge_graph.store.vector_store import InMemoryVectorStore
from app.knowledge_graph.graph_rag.retriever import retrieve_chunks, RetrievedChunk, RetrieverConfig
from app.knowledge_graph.graph_rag.context_builder import build_context as build_context_md, ContextConfig

try:
    from langchain_community.graphs import Neo4jGraph
except Exception:
    Neo4jGraph = None

LOGGER = setup_logging("knowledge_graph.graphrag")


# ==========================================================
# Config
# ==========================================================

@dataclass(frozen=True, slots=True)
class QueryConfig:
    top_k_chunks: int = 6
    max_chunk_chars_each: int = 1200
    expand_hops: int = 1
    max_graph_facts: int = 60
    max_total_context_chars: int = 14000
    answer_max_chars: int = 1500
    include_reverse_edges: bool = True  # kept for API stability


# ==========================================================
# Prompt
# ==========================================================

ANSWER_PROMPT = """\
You are a research assistant using GraphRAG.

STRICT RULES:
- Use ONLY the provided context.
- If the context does not contain the answer, say what is missing.
- Do NOT invent facts.
- Cite sources using [Chunk X] or [Graph Fact].

User question:
{question}

Context:
{context}
"""


# ==========================================================
# Helpers
# ==========================================================

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _neo4j_enabled(settings: Settings) -> bool:
    # We treat Neo4j as "usable" if creds exist. SYNC_NEO4J is about writing, not reading.
    if Neo4jGraph is None:
        return False
    if not settings.NEO4J_URL or not settings.NEO4J_USER:
        return False
    if not settings.NEO4J_PASSWORD.get_secret_value().strip():
        return False
    return True


# ==========================================================
# Neo4j Facts (Safe + Limited)
# ==========================================================

def _fetch_graph_facts(
    settings: Settings,
    seed_terms: List[str],
    qc: QueryConfig,
) -> str:
    if not _neo4j_enabled(settings) or not seed_terms:
        return ""

    try:
        g = Neo4jGraph(
            url=settings.NEO4J_URL,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD.get_secret_value(),
            database=settings.NEO4J_DATABASE.strip() or None,
        )
    except Exception:
        return ""

    hops = max(1, min(int(qc.expand_hops), 2))
    limit = max(1, min(int(qc.max_graph_facts), 100))

    seed_terms = [t.strip() for t in seed_terms if t and t.strip()][:3]
    if not seed_terms:
        return ""

    where = " OR ".join(
        [f"toLower(coalesce(n.id,n.name,'')) CONTAINS toLower($t{i})" for i in range(len(seed_terms))]
    )
    params = {f"t{i}": seed_terms[i] for i in range(len(seed_terms))}

    cypher = f"""
    MATCH (n:Entity)
    WHERE {where}
    MATCH p=(n)-[*1..{hops}]-(m:Entity)
    WITH p LIMIT {limit}
    WITH nodes(p) AS ns, relationships(p) AS rs
    UNWIND range(0, size(rs)-1) AS i
    RETURN DISTINCT
      coalesce(ns[i].id, ns[i].name) AS head,
      type(rs[i]) AS rel,
      coalesce(ns[i+1].id, ns[i+1].name) AS tail
    LIMIT {limit}
    """

    try:
        rows = g.query(cypher, params)
    except Exception:
        return ""

    facts = []
    for row in rows or []:
        h, r, t = row.get("head"), row.get("rel"), row.get("tail")
        if h and r and t:
            facts.append(f"- [Graph Fact] {h} {r} {t}")

    if not facts:
        return ""

    return "# Graph Facts\n" + "\n".join(facts)


# ==========================================================
# Answer
# ==========================================================

def answer_query(
    llm: ChatGroq,
    question: str,
    context: str,
    qc: QueryConfig,
) -> str:
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    msg = llm.invoke([HumanMessage(content=prompt)])
    out = msg.content if hasattr(msg, "content") else str(msg)
    return _truncate(out, qc.answer_max_chars)


# ==========================================================
# Main Runner
# ==========================================================

def run_query(
    llm: ChatGroq,
    vstore: InMemoryVectorStore,
    question: str,
    qc: Optional[QueryConfig] = None,
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    use_neo4j: bool = False,
    cfg=None,
    settings: Optional[Settings] = None,
) -> Tuple[str, List[RetrievedChunk], str]:
    """
    Preferred:
      run_query(..., use_neo4j=True, settings=get_settings())

    Legacy:
      run_query(..., use_neo4j=True, neo4j_url=..., neo4j_user=..., neo4j_password=...)
    """
    qc = qc or QueryConfig()
    start = time.time()

    retrieved = retrieve_chunks(
        vstore,
        question,
        RetrieverConfig(
            top_k=qc.top_k_chunks,
            max_chunk_chars_each=qc.max_chunk_chars_each,
        ),
        cfg=cfg,  # kept for compatibility
    )

    # Build settings (prefer explicit Settings, else build from legacy params, else default)
    if settings is None:
        if neo4j_url and neo4j_user and neo4j_password:
            base = get_settings()
            settings = base.model_copy(
                update={
                    "NEO4J_URL": neo4j_url,
                    "NEO4J_USER": neo4j_user,
                    "NEO4J_PASSWORD": SecretStr(neo4j_password),
                    # Don't force SYNC_NEO4J here; reading facts is separate.
                }
            )
        else:
            settings = get_settings()

    graph_facts = ""
    if use_neo4j:
        graph_facts = _fetch_graph_facts(settings, seed_terms=[question], qc=qc)

    context = build_context_md(
        retrieved,
        graph_facts_text=graph_facts,
        cc=ContextConfig(
            max_chunk_chars_each=qc.max_chunk_chars_each,
            max_total_context_chars=qc.max_total_context_chars,
            include_scores=True,
            heading="Retrieved Chunks",
        ),
    )

    answer = answer_query(llm, question, context, qc)

    LOGGER.info(
        "GraphRAG query complete",
        extra={
            "retrieved_chunks": len(retrieved),
            "context_chars": len(context),
            "latency_s": round(time.time() - start, 3),
            "use_neo4j": bool(use_neo4j),
        },
    )

    return answer, retrieved, context