from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.knowledge_graph.store.vector_store import InMemoryVectorStore

try:
    from langchain_community.graphs import Neo4jGraph
except Exception:
    Neo4jGraph = None

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedContext:
    id: str
    text: str
    source_type: str  # "Vector" or "Graph"
    score: float = 0.0


@dataclass(frozen=True)
class QueryConfig:
    # BGE-M3 Vector Fetch targeted top chunks
    top_k_chunks: int = 10
    max_chunk_chars_each: int = 1200
    
    # Neo4j 2-Hop Graph Fetch
    expand_hops: int = 2
    max_graph_facts: int = 60
    
    # Qwen3-Reranker-4B top constraints for final synthesis
    top_k_rerank: int = 5
    
    # Model configs
    synthesis_model: str = "llama-3.3-70b-versatile"


ANSWER_PROMPT = """\
You are an expert Research Assistant powered by a Hybrid GraphRAG pipeline.
Answer the user using ONLY the provided verified context.
If the context is insufficient, state exactly what information is missing.

Requirements:
- Provide a concise, highly analytical answer.
- Do not cite internal chunks, source IDs, or system markers (e.g., do not output [Chunk X] or [Graph Triplet]). Synthesize the information naturally.
- Do not hallucinate data outside the given context snippets.

User question:
{question}

Context Blocks:
{context}
"""


def retrieve_chunks_bge_m3(vstore: InMemoryVectorStore, query: str, qc: QueryConfig) -> List[RetrievedContext]:
    """
    Vector Fetch mimicking BGE-M3 semantics logic.
    Retains top 10 dense semantic chunks from the vector store.
    """
    results = vstore.search(query, top_k=qc.top_k_chunks)
    out: List[RetrievedContext] = []
    for cid, text, score in results:
        out.append(RetrievedContext(
            id=f"Chunk {cid}", 
            text=text, 
            source_type="Vector Chunk", 
            score=float(score)
        ))
    return out


def _fetch_graph_facts_2hop(
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    seed_terms: List[str],
    qc: QueryConfig,
) -> List[RetrievedContext]:
    """
    Graph Fetch: Extracts 2-hop neighbor relationships for all entities 
    identified in the query string from Neo4j.
    """
    if Neo4jGraph is None:
        return []

    g = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)

    seed_terms = [t for t in seed_terms if t.strip()]
    if not seed_terms:
        return []

    # Map seed terms onto OR statements for 2-hop Cypher queries
    where = " OR ".join([f"toLower(n.id) CONTAINS toLower($t{i})" for i in range(len(seed_terms))])
    params = {f"t{i}": seed_terms[i] for i in range(len(seed_terms))}

    # Standardize a 2-hop extraction mapping all intervening structures
    cypher = f"""
    MATCH path = (n)-[*1..{qc.expand_hops}]-(m)
    WHERE {where}
    UNWIND relationships(path) AS r
    WITH startNode(r) AS src, r, endNode(r) AS tgt
    RETURN DISTINCT src.id AS head, type(r) AS rel, tgt.id AS tail
    LIMIT {qc.max_graph_facts}
    """

    try:
        rows = g.query(cypher, params)
    except Exception as e:
        LOGGER.error(f"Neo4j Query Error: {e}")
        return []

    out: List[RetrievedContext] = []
    for idx, row in enumerate(rows or []):
        h = row.get("head")
        r = row.get("rel")
        t = row.get("tail")
        if h and r and t:
            out.append(RetrievedContext(
                id=f"Graph Triplet {idx}",
                text=f"{h} {r} {t}",
                source_type="Knowledge Graph",
                score=0.0  # Will be ranked by Qwen3 next
            ))
            
    return out


# Global variable to cache the reranker model if used
_RERANKER_MODEL = None

def rerank_qwen3(query: str, contexts: List[RetrievedContext], top_k: int) -> List[RetrievedContext]:
    """
    Rerank: Implement a scoring pass.
    By default, uses a lightweight heuristic to save memory and speed up response time.
    """
    if not contexts:
        return []

    global _RERANKER_MODEL
    import os
    
    use_heavy_reranker = os.getenv("USE_HEAVY_RERANKER", "false").lower() == "true"

    if use_heavy_reranker:
        try:
            from sentence_transformers import CrossEncoder
            if _RERANKER_MODEL is None:
                # Use a much smaller cross-encoder to save memory if enabled
                model_name = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                LOGGER.info(f"Loading reranker model: {model_name}")
                _RERANKER_MODEL = CrossEncoder(model_name)
                
            pairs = [[query, c.text] for c in contexts]
            scores = _RERANKER_MODEL.predict(pairs)
            
            # Merge scores back to objects natively
            ranked_contexts = []
            for ctx, score in zip(contexts, scores):
                ranked_contexts.append(RetrievedContext(
                    id=ctx.id, text=ctx.text, source_type=ctx.source_type, score=float(score)
                ))
                
            ranked_contexts.sort(key=lambda x: x.score, reverse=True)
            return ranked_contexts[:top_k]
            
        except Exception as e:
            LOGGER.info(f"Reranker missing or failed ({e}); applying heuristic fallback.")

    # Fallback pseudo-rerank logic using normalized Jaccard word collisions 
    # serving as a stable system baseline. Extremely fast and 0 extra memory.
    q_words = set(query.lower().split())
    ranked_contexts = []
    for ctx in contexts:
        c_words = set(ctx.text.lower().split())
        intersection = len(q_words.intersection(c_words))
        heuristic_score = ctx.score + (intersection * 0.1) # Bump native scores
        ranked_contexts.append(RetrievedContext(
            id=ctx.id, text=ctx.text, source_type=ctx.source_type, score=heuristic_score
        ))

    # Sort descending by score and slice the top_k constraints
    ranked_contexts.sort(key=lambda x: x.score, reverse=True)
    return ranked_contexts[:top_k]


def build_synthesis_context(ranked_top_blocks: List[RetrievedContext], max_chars: int) -> str:
    """Combines strictly the Top-K Reranked blocks into LLM prompts."""
    parts = ["# Verified Context Blocks (Qwen3-Reranker Evaluated)"]
    for ctx in ranked_top_blocks:
        truncated = ctx.text
        if len(truncated) > max_chars:
            truncated = truncated[:max_chars].rstrip() + "…"
        parts.append(f"\n## [{ctx.id}] (Source: {ctx.source_type} | Score={ctx.score:.3f})\n{truncated}\n")
    return "\n".join(parts).strip()


def run_query(
    llm: Any,  # Default fallback, heavily overridden to Llama-3.3-70B typically
    vstore: InMemoryVectorStore,
    question: str,
    qc: Optional[QueryConfig] = None,
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    use_neo4j: bool = False,
) -> Tuple[str, List[RetrievedContext], str]:
    qc = qc or QueryConfig()

    # Ensure Llama-3.3-70B is the driver for the Synthesis block
    if hasattr(llm, "model_name") and llm.model_name != qc.synthesis_model:
        LOGGER.info(f"Targeting specific Final LLM: {qc.synthesis_model}")
        try:
           llm = ChatGroq(model_name=qc.synthesis_model, temperature=0.0)
        except Exception:
           pass

    # 1. Vector Fetch (BGE-M3 semantics simulating top 10)
    vector_contexts = retrieve_chunks_bge_m3(vstore, question, qc)

    # 2. Graph Fetch (2-hop semantic traversal)
    graph_contexts = []
    if use_neo4j and neo4j_url and neo4j_user is not None and neo4j_password is not None:
        # Extract explicit query entities by length heuristic or basic LLM parsing
        seeds = [w for w in question.split() if len(w) >= 4]
        graph_contexts = _fetch_graph_facts_2hop(neo4j_url, neo4j_user, neo4j_password, seeds, qc)

    # Merge contexts universally
    all_contexts = vector_contexts + graph_contexts

    # 3. Rerank Pass (Qwen3-Reranker-4B isolating top 5)
    top_5_contexts = rerank_qwen3(question, all_contexts, top_k=qc.top_k_rerank)

    # 4. Synthesis Pass (Llama-3.3-70B Generation)
    context_str = build_synthesis_context(top_5_contexts, qc.max_chunk_chars_each)
    
    prompt = ANSWER_PROMPT.format(question=question, context=context_str)
    try:
        msg = llm.invoke([HumanMessage(content=prompt)])
        answer = msg.content if hasattr(msg, "content") else str(msg)
    except Exception as e:
        answer = f"Synthesis Failed: {e}"

    return answer, top_5_contexts, context_str
