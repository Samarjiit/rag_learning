# 11 — Autonomous RAG

## What Is Autonomous RAG?

Autonomous RAG systems go beyond a single retrieve-generate cycle. The agent can **self-evaluate**, **plan sub-queries**, **iterate**, and **synthesize from multiple sources** — all without human intervention between steps.

---

## Notebooks

### `1-cotrag.ipynb` — Chain-of-Thought RAG

**What it does:** Uses Chain-of-Thought (CoT) prompting to make the LLM reason step-by-step before forming its final answer, combined with RAG retrieval.

**Flow:**

```
START → [retrieve] → [CoT generate] → END
```

The generate node prompts the LLM to "think step by step" before producing its answer, making reasoning explicit and more accurate.

---

### `2-selfreflection.ipynb` — Self-Reflection RAG

**What it does:** After generating an answer, a separate **reflector node** critiques it. If the answer is incomplete or incorrect, the agent loops back to retrieve again — up to a maximum of 2 attempts.

**Flow:**

```
START
  │
  ▼
[retriever]  ──  fetch relevant docs from FAISS
  │
  ▼
[responder]  ──  LLM generates answer from context
  │
  ▼
[reflector]  ──  LLM self-critiques: "Is this complete and correct? YES/NO"
  │
  ├── YES or attempts >= 2  ──▶  END
  └── NO                    ──▶  [retriever]  (loop back)
```

**State:**
```python
class RAGReflectionState(BaseModel):
    question: str
    retrieved_docs: List[Document]
    answer: str
    reflection: str       # "Reflection: YES/NO + Explanation"
    revised: bool         # True if reflection said NO
    attempts: int         # increments each generate cycle
```

**Termination condition:** `not revised` (satisfactory) OR `attempts >= 2` (max retries hit)

---

### `3-queryplanning-decomposition.ipynb` — Query Planning & Decomposition

**What it does:** Breaks a complex query into multiple sub-questions, retrieves for each independently, then synthesizes all answers into a final cohesive response.

**Flow:**

```
START
  │
  ▼
[planner]  ──  LLM decomposes question into N sub-questions
  │
  ▼
[retriever]  ──  retrieves docs for each sub-question
  │
  ▼
[synthesizer]  ──  merges all sub-answers into one final answer
  │
  ▼
 END
```

---

### `4-iterative-retrieval.ipynb` — Iterative Retrieval

**What it does:** Iteratively retrieves and verifies documents. If the retrieved context doesn't satisfy a verification check, the agent refines its query and retrieves again.

**State:**
```python
class IterativeRAGState(BaseModel):
    question: str
    refined_question: str     # updated on each iteration
    retrieved_docs: List[Document]
    answer: str
    verified: bool
    attempts: int
```

**Flow:**

```
START
  │
  ▼
[retrieve]  ──  uses refined_question (or original) to fetch docs
  │
  ▼
[generate]  ──  LLM produces an answer; increments attempts
  │
  ▼
[verify]  ──  checks if answer is grounded and sufficient
  │
  ├── verified OR attempts >= max  ──▶  END
  └── not verified                 ──▶  [refine_query] ──▶ [retrieve]
```

**Key difference from self-reflection:** The query itself is rewritten on each loop, not just the answer regenerated.

---

### `5-answerSynthesis.ipynb` — Answer Synthesis from Multiple Sources

**What it does:** Retrieves from **four different sources in sequence** and merges all context into a single synthesized answer via one final LLM call.

**Sources:**
| Source | How |
|---|---|
| Internal text docs | FAISS vectorstore from `sample_docs.txt` |
| YouTube transcript | Mocked document with hardcoded transcript text |
| Wikipedia | `WikipediaQueryRun(...).run(query)` |
| ArXiv | `ArxivLoader(query).load()` |

**Flow:**

```
START
  │
  ▼
[retrieve_text]  ──  FAISS retrieval from local docs
  │
  ▼
[retrieve_yt]  ──  FAISS retrieval from YouTube transcript mock
  │
  ▼
[retrieve_wiki]  ──  Wikipedia live search
  │
  ▼
[retrieve_arxiv]  ──  ArXiv paper search
  │
  ▼
[synthesize]  ──  LLM merges all 4 contexts into one final answer
  │
  ▼
 END
```

**State:**
```python
class MultiSourceRAGState(BaseModel):
    question: str
    text_docs: List[Document]
    yt_docs: List[Document]
    wiki_context: str
    arxiv_context: str
    final_answer: str
```
