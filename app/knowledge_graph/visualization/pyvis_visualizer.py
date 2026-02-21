from __future__ import annotations
from typing import List, Optional
from pyvis.network import Network
from langchain_experimental.graph_transformers.llm import GraphDocument

def visualize_graph(graph_documents: List[GraphDocument], output_file: str = "knowledge_graph.html") -> Optional[str]:
    if not graph_documents:
        return None
    g = graph_documents[0]
    net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222")

    for n in g.nodes:
        net.add_node(n.id, label=n.id, title=getattr(n, "type", "Concept"), color="#b9d9ea")

    for r in g.relationships:
        net.add_edge(r.source.id, r.target.id, label=r.type, color="#97c2fc")

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 110,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "enabled": true, "iterations": 900 }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "hover": true,
        "zoomView": true
      }
    }
    """)
    net.save_graph(output_file)
    return output_file
