from __future__ import annotations
from typing import List, Optional, Any
from collections import defaultdict
import os
import math

from pyvis.network import Network


# 🎨 Palette مطابق للصورة
TYPE_COLORS = {
    "Concept": "#AFC8E8",
    "Model": "#AFC8E8",
    "Method": "#AFC8E8",
    "Technique": "#AFC8E8",

    "Dataset": "#BFE7D3",
    "Task": "#BFE7D3",
    "Metric": "#BFE7D3",

    "Author": "#CFCFCF",
    "Institution": "#CFCFCF",
    "Paper": "#D8D2A8",
}

DEFAULT_NODE_COLOR = "#AFC8E8"

EDGE_COLOR = "#B8C2CF"
EDGE_LABEL_COLOR = "#6B7280"
FONT_COLOR = "#374151"
BORDER_COLOR = "#9CA3AF"


def visualize_graph(
    *,
    entities: List[Any],
    relations: List[Any],
    output_file: str = "knowledge_graph.html",
) -> Optional[str]:

    if not entities:
        return None

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    node_type = {}
    for e in entities:
        nid = str(getattr(e, "id", None) or e.get("id"))
        ntype = getattr(e, "type", None) or e.get("type") or "Concept"
        node_type[nid] = ntype

    norm_edges = []
    for r in relations:
        s = getattr(r, "head", None) or getattr(r, "source", None) or r.get("source")
        t = getattr(r, "tail", None) or getattr(r, "target", None) or r.get("target")
        rel = getattr(r, "type", None) or getattr(r, "predicate", None) or r.get("type") or "RELATED_TO"

        if hasattr(s, "id"):
            s = s.id
        if hasattr(t, "id"):
            t = t.id

        if s and t:
            norm_edges.append((str(s), str(t), rel))

    degree = defaultdict(int)
    for s, t, _ in norm_edges:
        degree[s] += 1
        degree[t] += 1

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#F5F6F8",   # نفس الخلفية الفاتحة
        font_color=FONT_COLOR,
    )

    # 🟢 Nodes
    for nid, ntype in node_type.items():
        d = degree[nid]
        size = 14 + min(8, math.log1p(d) * 3)

        color = TYPE_COLORS.get(ntype, DEFAULT_NODE_COLOR)

        net.add_node(
            nid,
            label=nid,
            shape="dot",
            size=size,
            color={
                "background": color,
                "border": BORDER_COLOR,
            },
            borderWidth=1,
            font={
                "size": 14,
                "color": FONT_COLOR,
                "face": "Arial"
            },
            shadow=False
        )

    # 🔵 Edges
    for s, t, rel in norm_edges:
        net.add_edge(
            s,
            t,
            label=rel,
            color=EDGE_COLOR,
            width=1.2,
            arrows={"to": {"enabled": True, "scaleFactor": 0.7}},
            font={
                "size": 13,
                "color": EDGE_LABEL_COLOR,
                "align": "middle"
            },
            smooth={"type": "continuous"},
        )

    # ⚙ Physics قريب جدًا من الصورة (نجمي)
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.03,
          "springLength": 180,
          "springConstant": 0.05,
          "damping": 0.85,
          "avoidOverlap": 1
        },
        "stabilization": {
          "enabled": true,
          "iterations": 250
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "zoomView": true
      },
      "nodes": { "shadow": false },
      "edges": { "shadow": false }
    }
    """)

    net.save_graph(output_file)
    return output_file