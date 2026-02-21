from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.knowledge_graph.store.vector_store import InMemoryVectorStore

try:
    from langchain_community.graphs import Neo4jGraph
except Exception:
    Neo4jGraph = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


@dataclass(frozen=True)
class QueryConfig:
    top_k_chunks: int = 6
    max_chunk_chars_each: int = 1200
    expand_hops: int = 1
    max_graph_facts: int = 60


ANSWER_PROMPT = """\
You are a research assistant using GraphRAG.
Answer the user using ONLY the provided context.
If context is insufficient, say what is missing.

Requirements:
- Provide a concise answer.
- Then provide "Evidence" with bullet points citing chunk IDs like [Chunk 3].
- Do not invent citations.

User question:
{question}

Context:
{context}
"""


def retrieve_chunks(vstore: InMemoryVectorStore, query: str, qc: QueryConfig) -> List[RetrievedChunk]:
    results = vstore.search(query, top_k=qc.top_k_chunks)
    out: List[RetrievedChunk] = []
    for cid, text, score in results:
        out.append(RetrievedChunk(chunk_id=str(cid), text=text, score=float(score)))
    return out


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n].rstrip() + "…"


def _format_chunks(chunks: List[RetrievedChunk], qc: QueryConfig) -> str:
    parts = ["# Retrieved Chunks"]
    for ch in chunks:
        parts.append(f"\n## [Chunk {ch.chunk_id}] score={ch.score:.3f}\n{_truncate(ch.text, qc.max_chunk_chars_each)}\n")
    return "\n".join(parts).strip()


def _fetch_graph_facts_from_neo4j(
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    seed_terms: List[str],
    qc: QueryConfig,
) -> str:
    if Neo4jGraph is None:
        return ""

    g = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)

    # Simple, safe query:
    # - match nodes whose id contains a seed term
    # - expand neighbors up to 1 hop (qc.expand_hops)
    # - return triples as text facts
    # Note: adjust labels/properties if your Neo4j schema differs.
    seed_terms = [t for t in seed_terms if t.strip()]
    if not seed_terms:
        return ""

    # Build WHERE clause with OR
    where = " OR ".join([f"toLower(n.id) CONTAINS toLower($t{i})" for i in range(len(seed_terms))])
    params = {f"t{i}": seed_terms[i] for i in range(len(seed_terms))}

    # 1 hop expansion is stable and fast
    cypher = f"""
    MATCH (n)
    WHERE {where}
    MATCH (n)-[r]->(m)
    RETURN n.id AS head, type(r) AS rel, m.id AS tail
    LIMIT {qc.max_graph_facts}
    """

    try:
        rows = g.query(cypher, params)
    except Exception:
        return ""

    facts = []
    for row in rows or []:
        h = row.get("head")
        r = row.get("rel")
        t = row.get("tail")
        if h and r and t:
            facts.append(f"- {h} {r} {t}")
    if not facts:
        return ""
    return "# Graph Facts (Neo4j)\n" + "\n".join(facts)


def build_graphrag_context(
    retrieved: List[RetrievedChunk],
    qc: QueryConfig,
    graph_facts_text: str = "",
) -> str:
    ctx = _format_chunks(retrieved, qc)
    if graph_facts_text:
        ctx = ctx + "\n\n" + graph_facts_text
    return ctx


def answer_query(
    llm: ChatGroq,
    question: str,
    context: str,
) -> str:
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    msg = llm.invoke([HumanMessage(content=prompt)])
    return msg.content if hasattr(msg, "content") else str(msg)


def run_query(
    llm: ChatGroq,
    vstore: InMemoryVectorStore,
    question: str,
    qc: Optional[QueryConfig] = None,
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    use_neo4j: bool = False,
) -> Tuple[str, List[RetrievedChunk], str]:
    qc = qc or QueryConfig()

    retrieved = retrieve_chunks(vstore, question, qc)

    graph_facts = ""
    if use_neo4j and neo4j_url and neo4j_user is not None and neo4j_password is not None:
        # seed terms from top chunks: cheap heuristic (first few capitalized words)
        seeds = []
        for ch in retrieved[:3]:
            seeds.append(ch.chunk_id)  # not used in graph match
        # better: use question keywords
        seeds = [w for w in question.split() if len(w) >= 4][:6]
        graph_facts = _fetch_graph_facts_from_neo4j(neo4j_url, neo4j_user, neo4j_password, seeds, qc)

    context = build_graphrag_context(retrieved, qc, graph_facts_text=graph_facts)
    answer = answer_query(llm, question, context)
    return answer, retrieved, context
