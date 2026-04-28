# 13 — Corrective RAG (CRAG)

## What Is Corrective RAG?

Corrective RAG (CRAG) adds a **self-correction loop** to the standard RAG pipeline. After retrieving documents, each document is graded for relevance. If too many documents are irrelevant, the system **rewrites the query** and performs a **web search** to supplement or replace the original retrieval — before generating the final answer.

The core insight: *don't blindly trust what you retrieved; verify and correct first.*

---

## Notebook: `1-correctiveRag.ipynb`

---

## Components

### 1. Vector Index
Three blog posts from `lilianweng.github.io` are loaded, split into 500-token chunks, embedded with OpenAI Embeddings, and stored in a FAISS vectorstore.

```
WebBaseLoader(urls) → RecursiveCharacterTextSplitter → FAISS → retriever
```

---

### 2. Retrieval Grader
A structured LLM (`gpt-3.5-turbo` with `GradeDocuments` output) scores each retrieved document as `"yes"` (relevant) or `"no"` (irrelevant).

```python
class GradeDocuments(BaseModel):
    binary_score: str  # "yes" or "no"

retrieval_grader = grade_prompt | structured_llm_grader
```

---

### 3. RAG Chain (Generator)
Uses a local `ChatPromptTemplate` (no `hub.pull`) to generate an answer from filtered relevant documents.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using retrieved context..."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
rag_chain = prompt | llm | StrOutputParser()
```

---

### 4. Question Re-writer
When documents are not relevant, an LLM rewrites the original question to be better suited for web search.

```python
question_rewriter = re_write_prompt | llm | StrOutputParser()
```

---

### 5. Web Search Tool
Uses **Tavily Search** to fetch live web results when local retrieval fails.

```python
web_search_tool = TavilySearchResults(k=3)
```

---

## Full Graph Flow

```
START
  │
  ▼
[retrieve]  ──  fetch docs from FAISS vectorstore
  │
  ▼
[grade_documents]
  │  scores each doc: "yes" (relevant) or "no" (irrelevant)
  │  sets web_search = "Yes" if any doc is irrelevant
  │
  ├── all relevant (web_search="No")  ──▶  [generate]
  │
  └── some irrelevant (web_search="Yes")
         │
         ▼
      [transform_query]  ──  LLM rewrites question for web search
         │
         ▼
      [web_search_node]  ──  Tavily fetches live web results; appended to docs
         │
         ▼
      [generate]  ──  LLM answers using (filtered + web) docs
         │
         ▼
        END
```

---

## State

```python
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str        # "Yes" or "No"
    documents: List[str]   # filtered relevant docs + optional web results
```

---

## Key Decision: `decide_to_generate`

```python
def decide_to_generate(state):
    if state["web_search"] == "Yes":
        return "transform_query"   # go rewrite + web search
    else:
        return "generate"          # go straight to answer
```

---

## Summary

| Step | Purpose |
|---|---|
| Retrieve | Get initial candidate documents from FAISS |
| Grade | Filter irrelevant docs; flag if web search needed |
| Transform Query | Rewrite question optimized for web search |
| Web Search | Supplement with live Tavily results |
| Generate | Produce final answer from verified context |
