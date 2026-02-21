# app/knowledge_graph/store/neo4j_store.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from neo4j import GraphDatabase, Driver
from langchain_community.graphs.graph_document import GraphDocument

from app.core.logging import setup_logging
from app.core.settings import Settings

LOGGER = setup_logging("knowledge_graph.neo4j_store")

_DRIVER: Optional[Driver] = None


# ==========================================================
# Driver Singleton
# ==========================================================

def _get_driver(settings: Settings) -> Driver:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD.get_secret_value()),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
        )
    return _DRIVER


def close_driver() -> None:
    global _DRIVER
    if _DRIVER is not None:
        try:
            _DRIVER.close()
        finally:
            _DRIVER = None


# ==========================================================
# Constraints (idempotent)
# ==========================================================

def _ensure_constraints(driver: Driver, *, database: Optional[str] = None, create: bool = True) -> None:
    if not create:
        return

    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        # Unique ID constraint for Entity
        session.run(
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (n:Entity)
            REQUIRE n.id IS UNIQUE
            """
        )
        # Optional helpful index on name (non-unique)
        session.run(
            """
            CREATE INDEX entity_name_index IF NOT EXISTS
            FOR (n:Entity)
            ON (n.name)
            """
        )


# ==========================================================
# Bulk Insert
# ==========================================================

def _collect_payload(graph_documents: List[GraphDocument]) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    # Nodes dedup by id
    nodes_by_id: Dict[str, Dict] = {}

    # Relationships grouped by type for safe dynamic Cypher
    rels_by_type: Dict[str, List[Dict]] = {}

    for doc in graph_documents:
        for n in doc.nodes:
            nid = str(n.id)
            if nid not in nodes_by_id:
                nodes_by_id[nid] = {"id": nid, "name": nid, "type": str(getattr(n, "type", "Concept") or "Concept")}

        for r in doc.relationships:
            rel_type = str(getattr(r, "type", "RELATED_TO") or "RELATED_TO").upper().replace(" ", "_")
            rels_by_type.setdefault(rel_type, []).append(
                {"source": str(r.source.id), "target": str(r.target.id)}
            )

    return list(nodes_by_id.values()), rels_by_type


def _bulk_insert(driver: Driver, graph_documents: List[GraphDocument], *, database: Optional[str] = None) -> None:
    nodes_payload, rels_by_type = _collect_payload(graph_documents)

    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        # Nodes
        session.run(
            """
            UNWIND $nodes AS node
            MERGE (n:Entity {id: node.id})
            SET n.name = node.name,
                n.type = node.type
            """,
            {"nodes": nodes_payload},
        )

        # Relationships (typed). Relationship types cannot be parameterized, so we use safe dynamic Cypher
        # with strict sanitization: only A-Z0-9_ allowed.
        for rel_type, rels_payload in rels_by_type.items():
            safe = "".join(ch for ch in rel_type if (ch.isalnum() or ch == "_"))
            if not safe:
                safe = "RELATED_TO"

            cypher = f"""
            UNWIND $rels AS rel
            MATCH (a:Entity {{id: rel.source}})
            MATCH (b:Entity {{id: rel.target}})
            MERGE (a)-[r:{safe}]->(b)
            """

            session.run(cypher, {"rels": rels_payload})


# ==========================================================
# Public Sync
# ==========================================================

def sync_graph_documents(settings: Settings, graph_documents: List[GraphDocument]) -> bool:
    if not graph_documents:
        return True

    try:
        driver = _get_driver(settings)
        database = settings.NEO4J_DATABASE.strip() or None

        _ensure_constraints(
            driver,
            database=database,
            create=bool(settings.NEO4J_CREATE_CONSTRAINTS),
        )
        _bulk_insert(driver, graph_documents, database=database)

        LOGGER.info(
            "Neo4j sync complete",
            extra={
                "documents": len(graph_documents),
                "nodes": sum(len(d.nodes) for d in graph_documents),
                "relationships": sum(len(d.relationships) for d in graph_documents),
                "database": database or "(default)",
            },
        )
        return True

    except Exception as e:
        LOGGER.error("Neo4j sync failed", extra={"error": str(e)[:800]})
        return False