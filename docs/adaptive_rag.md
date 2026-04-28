# 14 — Adaptive RAG

## What Is Adaptive RAG?

Adaptive RAG dynamically **routes each question** to the most appropriate retrieval strategy before doing anything else. Unlike Corrective RAG (which always retrieves first and corrects after), Adaptive RAG makes an upfront routing decision:

- Questions about known topics → **vectorstore retrieval**
- Questions about current events or out-of-scope topics → **web search**

After retrieval, it also checks the quality of the generated answer with **hallucination grading** and **answer grading**, looping back if needed.

---

## Notebook: `1-adaptiverag.ipynb`

---

## Components

### 1. Vector Index
Three blog posts from `lilianweng.github.io` (agents, prompt engineering, adversarial attacks) are loaded, chunked, embedded, and stored in FAISS.

---

### 2. Question Router
A structured LLM with a `RouteQuery` Pydantic model decides the retrieval path **before any retrieval happens**.

```python
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"]

question_router = route_prompt | structured_llm_router
```

Example decisions:
- `"What are the types of agent memory?"` → `vectorstore`
- `"Who won the Cricket World Cup 2023?"` → `web_search`

---

### 3. Retrieval Grader
After vectorstore retrieval, each document is scored `"yes"`/`"no"` for relevance (same pattern as CRAG).

```python
class GradeDocuments(BaseModel):
    binary_score: str  # "yes" or "no"
```

---

### 4. RAG Chain (Generator)
Local `ChatPromptTemplate` generates an answer from filtered relevant documents.

---

### 5. Hallucination Grader
Checks whether the generated answer is **grounded in the retrieved facts** (not hallucinated).

```python
class GradeHallucinations(BaseModel):
    binary_score: str  # "yes" = grounded, "no" = hallucinated
```

---

### 6. Answer Grader
Checks whether the generated answer actually **addresses the user's question**.

```python
class GradeAnswer(BaseModel):
    binary_score: str  # "yes" = answers question, "no" = does not
```

---

### 7. Question Re-writer
If the answer fails the answer grader check, rewrites the question optimized for vectorstore retrieval (not web search — distinct from CRAG's rewrite).

---

### 8. Web Search Tool
`TavilySearchResults(k=3)` for live web results when the router sends the question to web search.

---

## Full Graph Flow

```
START
  │
  ▼
[route_question]  ──  router decides: web_search or vectorstore?
  │
  ├── "web_search"
  │     │
  │     ▼
  │  [web_search]  ──  Tavily live search
  │     │
  │     ▼
  │  [generate]
  │
  └── "vectorstore"
        │
        ▼
     [retrieve]  ──  FAISS vectorstore retrieval
        │
        ▼
     [grade_documents]  ──  filter irrelevant docs
        │
        ├── docs remain   ──▶  [generate]
        └── no docs left  ──▶  [transform_query] ──▶ [retrieve]  (loop)

[generate]
  │
  ▼
[grade_generation_v_documents_and_question]
  │
  ├── hallucination check fails ("not supported")  ──▶  [generate]  (retry)
  ├── answer check fails ("not useful")             ──▶  [transform_query]
  └── both pass ("useful")                          ──▶  END
```

---

## State

```python
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
```

---

## Decision Functions

| Function | Input | Output |
|---|---|---|
| `route_question` | question | `"web_search"` or `"vectorstore"` |
| `decide_to_generate` | filtered documents | `"generate"` or `"transform_query"` |
| `grade_generation_v_documents_and_question` | docs + generation | `"useful"` / `"not useful"` / `"not supported"` |

---

## Adaptive RAG vs Corrective RAG

| Feature | Corrective RAG | Adaptive RAG |
|---|---|---|
| Routing decision | ❌ Always retrieves first | ✅ Routes before retrieving |
| Hallucination check | ❌ No | ✅ Yes |
| Answer quality check | ❌ No | ✅ Yes |
| Web search trigger | After failed grading | As initial routing option |
| Query rewrite | Optimized for web | Optimized for vectorstore |
| Complexity | Medium | High |
