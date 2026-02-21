# app/knowledge_graph/graph_rag/community_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

from app.knowledge_graph.extraction.schema import Entity, Relation
from app.knowledge_graph.graph_rag.graph_summarizer import summarize_communities, CommunitySummary


class LLMInvoker(Protocol):
    def invoke(self, messages: list):  # langchain-style
        ...


@dataclass(frozen=True, slots=True)
class CommunityConfig:
    max_communities: int = 12
    max_edges_per_community: int = 80
    use_llm_summaries: bool = True


def build_communities(
    entities: List[Entity],
    relations: List[Relation],
    *,
    llm: Optional[LLMInvoker] = None,
    cc: Optional[CommunityConfig] = None,
) -> List[CommunitySummary]:
    """
    Build communities from relations and summarize them.

    - If llm is provided and use_llm_summaries=True => LLM summaries (grounded)
    - Otherwise => deterministic rule-based summaries
    """
    cc = cc or CommunityConfig()

    max_comms = max(1, int(cc.max_communities))
    max_edges = max(1, int(cc.max_edges_per_community))

    use_llm = bool(cc.use_llm_summaries and llm is not None)

    return summarize_communities(
        entities=entities,
        relations=relations,
        llm=llm if use_llm else None,
        max_communities=max_comms,
        max_edges_per_community=max_edges,
    )