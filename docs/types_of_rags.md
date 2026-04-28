# 10 — Types of RAG: Agentic RAG

## What Is Agentic RAG?

Agentic RAG is an advanced evolution of standard RAG where instead of a static retrieve-then-generate pipeline, an **agent** drives the process. The agent reasons, plans, selects tools, retrieves, and can retry or reflect to produce better-grounded answers.

---

## Notebooks

### `1-agenticrag.ipynb`

Three progressively more capable agentic RAG systems are built in this notebook.

---

#### Part 1 — Basic Agentic RAG with LangGraph

**What it does:** A minimal 2-node LangGraph graph that retrieves documents and generates an answer.

**Flow:**

```
START
  │
  ▼
[retriever]  ──  invoke FAISS retriever with the question
  │
  ▼
[responder]  ──  LLM generates answer from retrieved context
  │
  ▼
 END
```

**Key components:**
| Component | Detail |
|---|---|
| Documents | Web pages from `lilianweng.github.io` (agents, diffusion video) |
| Vector Store | FAISS + OpenAI Embeddings |
| LLM | `gpt-4o` via `init_chat_model` |
| State | `RAGState(question, retrieved_docs, answer)` |

---

#### Part 2 — ReAct Agent with Tools

**What it does:** Builds a ReAct (Reasoning + Acting) agent that can choose between a **RAG retriever tool** and a **Wikipedia tool** to answer a question.

**Flow:**

```
START
  │
  ▼
[react_agent]  ──  LLM reasons, picks a tool, observes output, repeats until done
  │
  ▼
 END
```

ReAct loop (internal to the agent):
```
Thought → Tool Call → Observation → Thought → ... → Final Answer
```

**Tools available:**
- `RAGRetriever` — searches the FAISS vectorstore built from the LangChain agents blog
- `WikipediaQueryRun` — fetches general knowledge from Wikipedia

---

#### Part 3 — Tool Creation for Multi-Source RAG Agents

**What it does:** Demonstrates a reusable pattern for creating retriever tools from any text file, combined with ArXiv and Wikipedia tools, all wired into a single ReAct agent.

**Flow:**

```
START
  │
  ▼
[agentic_rag]  ──  ReAct agent with 4 tools
  │
  ▼
 END
```

**Tools available:**
| Tool | Source |
|---|---|
| `InternalTechDocs` | `research_notes.txt` (FAISS) |
| `InternalResearchNotes` | `sample_docs.txt` (FAISS) |
| `ArxivSearch` | ArXiv papers via `ArxivLoader` |
| `WikipediaQueryRun` | Wikipedia |

**Generic tool factory pattern:**
```python
def make_retriever_tool_from_text(file, name, desc):
    # loads → splits → embeds → FAISS → retriever → Tool
```

---

### `2-agenticrags_project.ipynb`

A full project-style Agentic RAG system with **two separate vector stores** and a **document relevance grader** built into the agent loop.

**Flow:**

```
START
  │
  ▼
[agent]  ──  Groq LLaMA 3.1 reasons and calls retriever tools
  │
  ├── calls retriever_vector_db_blog (LangGraph docs)
  ├── calls retriever_vector_langchain_blog (LangChain docs)
  │
  ▼
[grade_documents]  ──  binary relevance check per document
  │
  ├── relevant → generate
  └── not relevant → rewrite query
       │
       ▼
    [generate]  ──  final answer
       │
       ▼
      END
```

**Key additions over Part 1:**
- Two independent FAISS vectorstores (LangGraph docs + LangChain docs)
- `GradeDocuments` Pydantic model for structured relevance scoring
- Question rewriting when documents are not relevant
- Uses **Groq LLaMA 3.1 8B** as the backbone LLM
