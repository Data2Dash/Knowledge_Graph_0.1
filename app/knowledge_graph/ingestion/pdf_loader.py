#C:\Users\Maryem\Knowledge_Graph_0.1\app\knowledge_graph\ingestion\pdf_loader.py
from __future__ import annotations
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_text(pdf_path: str, with_page_markers: bool = True) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        return ""
    if not with_page_markers:
        return "\n\n".join((d.page_content or "").strip() for d in docs if (d.page_content or "").strip())

    parts = []
    for i, d in enumerate(docs):
        page_num = (getattr(d, "metadata", {}) or {}).get("page", i + 1)
        content = (d.page_content or "").strip()
        if content:
            parts.append(f"\n\n--- Page {page_num} ---\n\n{content}")
    return "\n\n".join(parts).strip()
