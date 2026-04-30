# 16 — Cache-Augmented Generation (CAG)

## What Is Cache-Augmented Generation?

CAG is a retrieval-free approach that bypasses external knowledge queries at inference time. Instead of retrieving documents on every request, it **preloads relevant context** and reuses prior LLM responses when semantically similar questions are asked again — avoiding redundant LLM calls and reducing latency.

---

## Notebook: `1-cache_augmented_generation.ipynb`

Two CAG implementations are built progressively.

---

## Part 1 — Simple Dictionary Cache

A plain Python `dict` acts as an exact-match cache keyed by the query string.

```python
Model_Cache = {}

def cache_model(query):
    if Model_Cache.get(query):
        print("**Cache Hit**")
        return Model_Cache.get(query)
    else:
        response = llm.invoke(query)
        Model_Cache[query] = response
        return response
```

**Behaviour:**

- Same query string → cache hit, no LLM call
- Different query string (even slightly) → cache miss, LLM called and result stored
- Shows execution time difference between cache hit vs miss

**Limitation:** Exact string match only — `"hi"` and `"Hi"` are treated as different queries.

---

## Part 2 — Advanced Semantic CAG with LangGraph

Replaces exact matching with **semantic similarity search** using a FAISS vector store as the cache. Questions that are _semantically similar_ (not just identical) reuse prior answers.

### Configuration

```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim HuggingFace embeddings
CACHE_DISTANCE_THRESHOLD = 0.45   # L2 distance; lower = more similar
CACHE_TTL_SEC = 0                 # 0 = TTL disabled
CACHE_TOP_K = 3
RETRIEVE_TOP_K = 4
```

### State

```python
class RAGState(TypedDict):
    question: str
    normalized_question: str
    context_docs: List[Document]
    answer: Optional[str]
    citations: List[str]
    cache_hit: bool
```

### Graph Flow

```
START
  │
  ▼
[normalize_query]  ──  lowercases and strips the question
  │
  ▼
[semantic_cache_lookup]
  │  searches QA_CACHE (FAISS) for semantically similar prior Q&A
  │  checks L2 distance against CACHE_DISTANCE_THRESHOLD
  │  optionally checks TTL on the cached entry
  │
  ├── cache_hit = True  ──▶  [respond_from_cache]  ──▶  END
  │                            (returns stored answer + "(cache)" citation)
  │
  └── cache_hit = False ──▶  [retrieve]
                               │  fetches docs from RAG_STORE (FAISS)
                               ▼
                            [generate]
                               │  LLM answers with [doc-i] citations
                               ▼
                            [cache_write]
                               │  stores Q + answer in QA_CACHE for future reuse
                               ▼
                              END
```

### Two Separate FAISS Stores

| Store       | Purpose                                              |
| ----------- | ---------------------------------------------------- |
| `QA_CACHE`  | Stores past question→answer pairs for semantic reuse |
| `RAG_STORE` | Stores knowledge base documents for retrieval        |

### Cache Hit Logic

```python
hits = QA_CACHE.similarity_search_with_score(q, k=CACHE_TOP_K)
best_doc, dist = hits[0]
if dist <= CACHE_DISTANCE_THRESHOLD:
    state["answer"] = best_doc.metadata["answer"]
    state["cache_hit"] = True
```

### Demo Results

| Query                                  | Cache Hit?                 |
| -------------------------------------- | -------------------------- |
| `"What is LangGraph?"`                 | No (first time)            |
| `"Explain about LangGraph?"`           | Yes (semantically similar) |
| `"Explain about LangGraph agents?"`    | Yes (semantically similar) |
| `"Explain about agents in Langgraph?"` | Yes (semantically similar) |

---

## Key Benefits

| Feature                 | Benefit                                             |
| ----------------------- | --------------------------------------------------- |
| Semantic cache lookup   | Handles rephrased questions without re-invoking LLM |
| TTL support             | Optionally expire stale cache entries               |
| Citation tracking       | `[doc-i]` markers trace which documents were used   |
| LangGraph + MemorySaver | Full conversation checkpointing per thread          |
