from __future__ import annotations
from typing import List, Optional
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers.llm import GraphDocument
from app.core.config import PipelineConfig

def sync_graph_documents(cfg: PipelineConfig, graph_documents: List[GraphDocument]) -> bool:
    try:
        g = Neo4jGraph(url=cfg.neo4j_url, username=cfg.neo4j_user, password=cfg.neo4j_password)
        g.add_graph_documents(graph_documents)
        return True
    except Exception:
        return False
