# ⚖️ LegalMind: Agentic RAG System for Indian Criminal Law Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Pipeline-FF6B35?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00599C?style=flat)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-MiniLM_Embeddings-FFD21E?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**🔗 Live Demo:** [legalmind-agentic-rag.streamlit.app](https://legalmind-agentic-rag-system-for-indian-criminal-law.streamlit.app)

---

## 📖  1. Project Overview

LegalMind is an intelligent legal assistant powered by a **6-node LangGraph agentic pipeline** built over 6566 pages of Indian Criminal Law. Unlike traditional RAG systems that always retrieve from a database, LegalMind uses an **evaluator agent** that intelligently decides the best strategy for each query:

- Search the legal vector database (FAISS)
- Combine vector retrieval with LLM general legal knowledge
- Answer from general knowledge when the query is outside the legal corpus

Every answer is structured with **Section numbers**, **Chapter references**, and **Act attribution** — making responses legally precise and verifiable.


---

## 📉 2. Problem Statement

Indian Criminal Law spans three interconnected statutes — the Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), and Indian Evidence Act — totalling thousands of sections across 6566 pages. Legal professionals, students, and citizens face significant challenges:

- **Information Overload** — Finding a specific provision across thousands of pages is time-consuming
- **Cross-Reference Complexity** — A single legal question often spans multiple acts simultaneously
- **Lack of Accessible Tools** — Existing legal search tools rely on exact keyword matching, missing semantic meaning
- **No Citation Enforcement** — Generic LLMs answer legal questions without citing exact sections, making answers unverifiable

LegalMind solves all four problems with semantic vector search, agentic routing, and legally-structured output formatting.

---

## 🎯 3. Project Objective

| Objective | Implementation |
|-----------|---------------|
| Enable semantic legal search | FAISS vector index with HuggingFace embeddings |
| Enforce Section-level citations | Legal-aware RAG prompt templates |
| Handle multi-act queries | Merged IPC + CrPC + Evidence corpus with metadata tagging |
| Route intelligently per query | LangGraph conditional routing (yes / vd / no) |
| Provide source attribution | Per-chunk Act metadata (IPC / CrPC / Evidence) |
| Build accessible UI | Streamlit chat interface with glassmorphism theme |

---

## 📊 4.. Data Understanding

>### Dataset Source

| Act | Full Name | Year | Source |
|-----|-----------|------|--------|
| IPC | Indian Penal Code | 1860 | Government of India Legislative Department |
| CrPC | Code of Criminal Procedure | 1973 | Government of India Legislative Department |
| Evidence Act | Indian Evidence Act | 1872 | Government of India Legislative Department |

The three acts were merged into a single PDF corpus — `IPC_CrPC_Evidence.pdf` — for unified vector indexing.

>### Dataset Characteristics

| Property | Value |
|----------|-------|
| Total Pages | 6566 |
| Total Chunks (after splitting) | ~39,000+ |
| Chunk Size | 1000 characters |
| Chunk Overlap | 150 characters |
| Embedding Dimensions | 384 (MiniLM-L6-v2) |
| Search Type | MMR (Maximal Marginal Relevance) |
| Top-K Results | 5 per query |

>### Feature Breakdown

Each indexed chunk carries the following metadata:

```
{
    "page_content": "Section 154 — Every information relating to...",
    "metadata": {
        "source": "CrPC",       ← which Act (IPC / CrPC / Evidence)
        "page":   112,          ← original PDF page number
        "chunk_id": 445         ← unique chunk identifier
    }
}
```

**Act distribution across chunks:**

| Act | Approximate Chunks | Coverage |
|-----|--------------------|---------|
| CrPC | ~18,000 | Procedural law — largest portion |
| IPC | ~14,000 | Substantive criminal offences |
| Indian Evidence Act | ~7,000 | Evidence rules and admissibility |

---

## 🔍 5. Legal Domain Coverage

>LegalMind integrates the core pillars of the Indian criminal justice system:

***Indian Penal Code (IPC) 1860***
* **Offence Classification:** Defines criminal acts and mandated penalties.
* **Scope:** Covers crimes against the state, person, property, and public order.
* **Key Sections:** Technical mapping for murder (302), theft (378), fraud (420), and forgery.

***Code of Criminal Procedure (CrPC) 1973***
* **Police & FIR:** Protocols for registration and investigative powers (Sec. 154, 156).
* **Custody & Bail:** Regulations for arrests (Sec. 41, 70) and bailable/non-bailable provisions.
* **Judicial Process:** Trial procedures, magistrate jurisdiction, and rights of the accused.

***Indian Evidence Act 1872***
* **Admissibility:** Standards for factual relevance and burden of proof.
* **Witness Testimony:** Rules for examination, cross-examination, and re-examination.
* **Validation:** Legal frameworks for confessions, hearsay, and documentary evidence.

## ⚖️ 6. Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│         app.py                  │
│      Streamlit Chat UI          │
│  (glassmorphism legal theme)    │
└────────────┬────────────────────┘
             │ run_query()
             ▼
┌─────────────────────────────────┐
│         agent.py                │
│    LangGraph StateGraph         │
│    compiled workflow            │
└────────────┬────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│              LangGraph 6-Node Pipeline             │
│                                                    │
│  [1] query_preprocessor                            │
│       "crpc 154" → "What is Section 154 CrPC?"     │
│              │                                     │
│              ▼                                     │
│  [2] evaluate_agent                                │
│       Classifies: yes / vd / no                    │
│              │                                     │
│    ┌─────────┼──────────┐                          │
│   yes        vd         no                         │
│    │         │           │                         │
│    ▼         ▼           ▼                         │
│  [3] rag_retriever   [5] groq_general              │
│   FAISS MMR search    Groq LLaMA 3.3 70B           │
│    │         │                                     │
│    ▼         ▼                                     │
│  [4] citation_  [6] vd_node                        │
│     formatter    FAISS + Groq                      │
│    │         │           │                         │
│    └─────────┴───────────┘                         │
│                   │                                │
│                   ▼                                │
│            Final Answer                            │
│   Section No. + Chapter + Act + Key Provisions     │
└────────────────────────────────────────────────────┘
             │
             ▼
    Source Badges (IPC / CrPC / Evidence)
    Section Reference Tags (Section 154, Section 41...)
    Route Label (Answered from Legal Database)
```

**State flows through all nodes via TypedDict:**
```python
State = {
    question, cleaned_question,       # input
    route,                            # routing decision
    docs, sources,                    # retrieval results
    answer, section_refs,             # output
    error                             # error handling
}
```

---

## 📃 7. Example Queries

- "What is Section 154 CrPC?"
- "What is the punishment for murder under IPC?"
- "What are the rights of an arrested person?"
- "Explain bail provisions under CrPC"
- "What is cognizable offence?"
- "What is burden of proof under Evidence Act?"

---

## 📌 8. Key Concepts Implemented

**Agentic RAG vs Basic RAG**
```
Basic RAG:   always retrieves → always answers from docs → static pipeline
Agentic RAG: evaluates query → decides strategy → routes dynamically
             the agent THINKS before acting — not just retrieval + generation
```

**MMR Search (Maximal Marginal Relevance)**
```
Similarity search → top 5 most similar chunks (often repetitive)
MMR search        → top 5 relevant AND diverse chunks
                    fetch_k=15 candidates, MMR picks best 5
                    richer context, better answers
```

**Metadata Tagging per Chunk**
```
Every chunk tagged at ingestion: {"source": "CrPC", "page": 112}
Powers UI source badges and answer attribution
Users know exactly which Act the answer came from
```

**Legal-Aware Prompt Engineering**
```
EVALUATOR_PROMPT  → classifies query as yes/vd/no with legal domain rules
RAG_PROMPT        → enforces Section No. + Chapter + Explanation format
VD_PROMPT         → combines retrieved context with landmark case knowledge
PREPROCESSOR_PROMPT → normalizes "crpc 154" → "What is Section 154 CrPC?"
```

**LangGraph Conditional Routing**
```
evaluate_agent writes route → "yes" / "vd" / "no"
route_query() reads state["route"] → returns next node name
Two conditional edge stages: after evaluate_agent AND after rag_retriever
```

**Chunking Strategy for Legal Text**
```
RecursiveCharacterTextSplitter — tries paragraph → sentence → word splits
chunk_size=1000    → fits one complete legal section
chunk_overlap=150  → prevents cutting a section at boundary
```

---

## ⚙️ 9. Project Structure

```
LegalMind-Agentic-RAG/
│
├── app.py                      # Streamlit chat UI — entry point
├── requirements.txt            # Minimal pinned dependencies
├── .env                        # API keys (not in repo)
├── .gitignore
├── README.md
│
├── data/
│   └── IPC_CrPC_Evidence.pdf   # 6566-page legal corpus
│
├── vectorDatabase/             # FAISS index (auto-generated by ingestion)
│   ├── index.faiss             # vector index file
│   └── index.pkl               # chunk texts + metadata
│
├── src/
│   ├── __init__.py
│   ├── state.py                # LangGraph TypedDict state schema
│   ├── ingestion.py            # PDF → chunks → embeddings → FAISS
│   ├── retrieval.py            # MMR vector search + metadata extraction
│   ├── prompts.py              # All 5 LLM prompt templates
│   ├── nodes.py                # 6 LangGraph node functions
│   ├── agent.py                # Graph wiring, compilation, run_query()
│   └── utils.py                # Section extractor, act detector, helpers
│
├── config/
│   └── settings.py             # Centralized config — paths, models, params
│
└── assets/
    └── style.css               # Glassmorphism legal theme (gold + navy)
```

---

## 🎬 10. Deployment

>### Streamlit Cloud Deployment

 Deployed a  Streamlit interface on Streamlit Cloud
  with source-aware badges, section reference tags, query routing
  labels, and real-time session analytics

---

## 🚀 11. Tech Stack

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Agentic conditional routing graph |
| **LangChain** | LLM orchestration + document loading |
| **FAISS** | Vector similarity search index |
| **HuggingFace** | `all-MiniLM-L6-v2` embeddings (384-dim) |
| **Groq API** | `llama-3.3-70b-versatile` LLM inference |
| **Streamlit** | Interactive chat UI |
| **PyPDF** | PDF text extraction |
| **Python Regex** | Section number extraction |
---

## 👩‍💻 12. Author

**Shravani More**

*Computer Science & Electronics Student*

