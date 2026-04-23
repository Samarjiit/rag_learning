# Query Enhancement Techniques

## Overview

Query enhancement improves the quality of queries sent to the retriever in a RAG pipeline. Since retrieval quality determines the LLM's final answer quality, better queries lead to better answers. This module covers three core techniques: **Query Expansion**, **Query Decomposition**, and **HyDE (Hypothetical Document Embeddings)**.

## Table of Contents

- [Why Query Enhancement?](#why-query-enhancement)
- [Technique 1 — Query Expansion](#technique-1--query-expansion)
- [Technique 2 — Query Decomposition](#technique-2--query-decomposition)
- [Technique 3 — HyDE](#technique-3--hyde)
- [Comparison](#comparison)
- [Full Pipeline Flow](#full-pipeline-flow)

---

## Why Query Enhancement?

Raw user queries are often:

- **Short or vague** — "LangChain memory"
- **Multi-part** — "How does LangChain memory work and how does CrewAI compare?"
- **Phrased differently** from document content — semantic mismatch

Query enhancement bridges the gap between what the user asks and what retriever finds.

---

## Technique 1 — Query Expansion

**File**: `1-query-expansion.ipynb`

### What It Does

Rewrites a short or ambiguous query into a richer version with synonyms, related terms, and useful context — before passing it to the retriever.

```
User query → LLM expands query → Expanded query → Retriever → LLM answers with original query
```

### Implementation

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Expansion chain
query_expansion_chain = (
    PromptTemplate.from_template("""
    Expand the following query to improve document retrieval by adding
    relevant synonyms, technical terms, and useful context.

    Original query: "{query}"
    Expanded query:
    """)
    | llm
    | StrOutputParser()
)

# Full RAG pipeline with expansion
rag_pipeline = (
    RunnableLambda(lambda x: {
        "context": "\n\n".join(
            doc.page_content for doc in retriever.invoke(
                query_expansion_chain.invoke({"query": x["input"]})
            )
        ),
        "input": x["input"]
    })
    | answer_prompt
    | llm
    | StrOutputParser()
)
```

### Example

| Original Query       | Expanded Query                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `"LangChain memory"` | `"LangChain memory management, ConversationBufferMemory, ConversationSummaryMemory, chat history, state retention in LLM applications"` |
| `"CrewAI agents?"`   | `"CrewAI multi-agent framework, autonomous agents, role-based agents, task delegation, agent collaboration"`                            |

### When to Use

- Short or keyword-style queries
- When users may not know exact terminology
- Broad topic searches

---

## Technique 2 — Query Decomposition

**File**: `2-query-decomposition.ipynb`

### What It Does

Breaks a complex, multi-part question into smaller atomic sub-questions. Each sub-question is retrieved and answered independently, enabling multi-hop reasoning.

```
Complex query → LLM decomposes → Sub-questions → [Retrieve + Answer each] → Combine results
```

### Implementation

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Decomposition chain
decomposition_chain = (
    PromptTemplate.from_template("""
    Decompose the following complex question into 2 to 4 smaller sub-questions
    for better document retrieval.

    Question: "{question}"
    Sub-questions:
    """)
    | llm
    | StrOutputParser()
)

# QA chain per sub-question
qa_chain = (
    PromptTemplate.from_template("""
    Use the context below to answer the question.
    Context: {context}
    Question: {input}
    """)
    | llm
    | StrOutputParser()
)

# Full pipeline
def full_query_decomposition_rag_pipeline(user_query):
    sub_qs_text = decomposition_chain.invoke({"question": user_query})
    sub_questions = [q.strip("-•1234567890. ").strip() for q in sub_qs_text.split("\n") if q.strip()]

    results = []
    for subq in sub_questions:
        docs = retriever.invoke(subq)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = qa_chain.invoke({"input": subq, "context": context})
        results.append(f"Q: {subq}\nA: {answer}")

    return "\n\n".join(results)
```

### Example

| Complex Query                                                    | Sub-questions Generated                                          |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| `"How does LangChain use memory and agents compared to CrewAI?"` | 1. What memory types does LangChain support?                     |
|                                                                  | 2. How do LangChain agents work?                                 |
|                                                                  | 3. What is CrewAI and how does it manage agents?                 |
|                                                                  | 4. How does CrewAI differ from LangChain in agent orchestration? |

### When to Use

- Multi-part questions with multiple concepts
- When a single retrieval pass misses parts of the question
- Multi-hop reasoning scenarios (answer depends on multiple facts)

---

## Technique 3 — HyDE

**File**: `3-hyde.ipynb`

### What It Does

**HyDE (Hypothetical Document Embeddings)** generates a hypothetical answer to the query using an LLM, then embeds _that hypothetical answer_ (not the original query) to search the vector store. This reduces the language gap between short queries and longer document content.

```
User query → LLM generates hypothetical answer → Embed hypothetical answer → Similarity search → LLM answers with real retrieved context
```

### Two Approaches

#### Approach A — Manual HyDE (Custom Implementation)

```python
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

def get_hyde_doc(query):
    template = """Imagine you are an expert writing a detailed explanation
    on the topic: '{query}'. Create a hypothetical answer for the topic."""

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template=template)
    ])
    messages = chat_prompt.format_prompt(query=query).to_messages()
    return llm.invoke(messages).content

# Retrieve using the hypothetical doc's embedding
matched_docs = base_retriever.invoke(get_hyde_doc(query))
```

#### Approach B — LangChain HyDE (LCEL pipeline)

```python
# HyDE chain — generates hypothetical answer, embeds it, retrieves
hyde_chain = (
    PromptTemplate.from_template(
        "Generate a concise hypothetical answer for this topic: {query}"
    )
    | llm
    | StrOutputParser()
)

# RAG chain
rag_chain = (
    PromptTemplate.from_template("""
    Use the context below to answer the question.
    Context: {context}
    Question: {input}
    """)
    | llm
    | StrOutputParser()
)

# Full HyDE pipeline
def hyde_rag_pipeline(query):
    hypothetical_doc = hyde_chain.invoke({"query": query})
    context = "\n\n".join(
        doc.page_content
        for doc in vectorstore.similarity_search(hypothetical_doc, k=4)
    )
    return rag_chain.invoke({"input": query, "context": context})
```

### Example

| Query                                     | Hypothetical Doc Generated                                                                           |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `"When was Steve Jobs fired from Apple?"` | `"Steve Jobs was removed from Apple in 1985 following a boardroom dispute with CEO John Sculley..."` |

The hypothetical doc's embedding is much closer to actual Wikipedia passages about Steve Jobs than the short query embedding.

### When to Use

- Queries are very short or use different vocabulary than documents
- Cross-domain retrieval (query phrasing ≠ document phrasing)
- When semantic similarity between query and documents is low

---

## Comparison

| Technique               | Best For                    | Retrieval Calls           | Complexity |
| ----------------------- | --------------------------- | ------------------------- | ---------- |
| **Query Expansion**     | Short/vague queries         | 1 (with expanded query)   | Low        |
| **Query Decomposition** | Multi-part complex queries  | N (one per sub-question)  | Medium     |
| **HyDE**                | Query-document language gap | 1 (with hypothetical doc) | Medium     |

---

## Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  ┌─────────────┐ ┌────────────┐ ┌─────────────┐
  │   Expand    │ │ Decompose  │ │    HyDE     │
  │   Query     │ │ into Sub-Qs│ │ Hypothetical│
  └──────┬──────┘ └─────┬──────┘ └──────┬──────┘
         │               │               │
         ▼               ▼               ▼
  ┌─────────────────────────────────────────────┐
  │              Vector Store Retrieval          │
  │         (FAISS / ChromaDB + MMR/similarity) │
  └────────────────────────┬────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  LLM Answers   │
                  │  with Context  │
                  └────────────────┘
```

---

## Setup

```python
# Required packages
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# LLM (Groq free tier)
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = init_chat_model("groq:llama-3.1-8b-instant")

# Embeddings (local, free)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

## Files in This Module

| File                           | Technique           | Description                                      |
| ------------------------------ | ------------------- | ------------------------------------------------ |
| `1-query-expansion.ipynb`      | Query Expansion     | Rewrites short queries with richer terms         |
| `2-query-decomposition.ipynb`  | Query Decomposition | Splits complex queries into sub-questions        |
| `3-hyde.ipynb`                 | HyDE                | Embeds hypothetical answers for better retrieval |
| `langchain_crewai_dataset.txt` | Dataset             | Sample text about LangChain and CrewAI           |

## Related Documentation

- [Hybrid Search Strategies](hybrid_search_strategies.md)
- [Semantic Chunking](semantic_chunking.md)
- [Vector Stores](vector_stores.md)
