# 🧠 Data2Dash GraphRAG Engine

> GraphRAG-powered Knowledge Graph Builder for Scientific Papers using LLMs, Semantic Chunking, and Neo4j.

Data2Dash GraphRAG Engine transforms unstructured research papers into structured, queryable knowledge graphs — enabling intelligent retrieval, relationship discovery, and explainable AI answers.

---

## 🚀 Features

✅ Automatic entity & relationship extraction  
✅ Semantic chunking for high-quality context  
✅ Knowledge Graph generation  
✅ GraphRAG query engine  
✅ Neo4j integration  
✅ Vector search for hybrid retrieval  
✅ Async extraction pipeline  
✅ Production-ready modular architecture  
✅ Interactive graph visualization  

---

## 🏗️ Architecture

PDF / Text
↓
Preprocessing
↓
Semantic Chunking
↓
LLM Extraction (Entities → Relations)
↓
Knowledge Graph
↓
Neo4j + Vector Store
↓
GraphRAG Query Engine


This design minimizes hallucinations while maximizing factual grounding.

---

## 📂 Project Structure

ai/knowledge_graph
│
├── app
│ ├── core
│ ├── pipelines
│ ├── knowledge_graph
│ │ ├── chunking
│ │ ├── extraction
│ │ ├── graph_rag
│ │ ├── ingestion
│ │ ├── preprocessing
│ │ ├── store
│ │ └── visualization
│ └── ui
│
├── data
├── outputs
└── requirements.txt


Built using clean architecture principles for scalability.

---

## ⚡ Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Data2Dash/Data2Dash-ai.git
cd Data2Dash-ai/ai/knowledge_graph
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Add Environment Variables
Create a .env file:

GROQ_API_KEY=your_api_key_here

# Optional
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
4️⃣ Run the app
streamlit run main.py
Open:

http://localhost:8501
🔎 GraphRAG Querying
After generating the knowledge graph, you can ask:

“What dataset was the model trained on?”
“Which method outperformed the baseline?”

The system retrieves relevant chunks, expands graph context, and produces grounded answers with evidence.

🧠 Tech Stack
LLM: Groq / Llama

Graph DB: Neo4j

Vector Store: In-memory (extensible)

Framework: LangChain

UI: Streamlit

Async Processing: Python asyncio

🎯 Use Cases
Research paper analysis

Literature reviews

Scientific discovery

AI-assisted research

Knowledge mining

Technical document understanding

🔮 Roadmap
Hybrid graph + vector reranking

Entity-aware retrieval

Persistent vector database

Community detection

Multi-document reasoning

API deployment (FastAPI)

Docker support

Distributed extraction

🤝 Contributing
We welcome contributions!

Fork the repo

Create your feature branch

Commit changes

Push and open a PR

📜 License
Apache 2.0 — feel free to use and modify.

🌟 About Data2Dash
Data2Dash builds intelligent data systems that convert complex information into actionable insights using AI.

⭐ If you find this project useful — consider giving it a star!
