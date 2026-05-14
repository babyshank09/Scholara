# Scholara — Research Paper RAG Application

A production-grade Retrieval-Augmented Generation (RAG) application 
specialized for querying and reasoning over research papers.

Built with an agentic LangGraph workflow, hybrid retrieval (vector + BM25 + 
Cohere reranking), multimodal ingestion (text, tables, images), and a 
Streamlit chat interface.   

## Features
- 🧠 Agentic graph — orchestrator, query rewriter, retriever, and response generation nodes
- 📄 Multimodal ingestion — extracts text, tables (HTML), and images (base64) from PDFs
- 🔍 Hybrid retrieval — dense vector search + BM25 + Cohere rerank
- 📊 RAGAS evaluation pipeline — faithfulness, answer relevancy, context precision, context recall
- 🚫 Abstention handling — refuses to answer when information is not in the documents
- 💬 Streamlit chat UI with persistent conversation memory via LangGraph checkpointer
- 🔭 LangSmith tracing for end-to-end observability

## Tech Stack
- **Orchestration:** LangGraph, LangChain
- **LLM:** OpenAI GPT-4o / GPT-4o-mini
- **Embeddings:** OpenAI text-embedding-3-small
- **Reranking:** Cohere Rerank
- **Vector Store:** ChromaDB
- **Evaluation:** RAGAS and abstention
- **Frontend:** Streamlit
- **Observability:** LangSmith, Langfuse
