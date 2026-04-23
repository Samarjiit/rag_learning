# Hybrid Search Strategies

## Overview

Hybrid search combines multiple retrieval methods to leverage the strengths of different approaches, providing more comprehensive and accurate document retrieval for RAG applications. This module covers dense retrieval, sparse retrieval, ensemble methods, and reranking strategies.

## Table of Contents

- [What is Hybrid Search?](#what-is-hybrid-search)
- [Dense vs Sparse Retrieval](#dense-vs-sparse-retrieval)
- [Ensemble Retrieval](#ensemble-retrieval)
- [Reranking Strategies](#reranking-strategies)
- [Implementation Examples](#implementation-examples)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)

## What is Hybrid Search?

Hybrid search combines multiple retrieval methods to overcome individual limitations:

- **Dense Retrieval**: Uses semantic embeddings (FAISS, ChromaDB)
- **Sparse Retrieval**: Uses keyword matching (BM25, TF-IDF)
- **Ensemble Methods**: Combines results using fusion algorithms
- **Reranking**: Post-processes results with advanced models

### Benefits

| Single Method                    | Hybrid Approach              |
| -------------------------------- | ---------------------------- |
| May miss relevant results        | Comprehensive coverage       |
| Biased toward one retrieval type | Balanced retrieval           |
| Limited by method constraints    | Leverages multiple strengths |
| Good for specific queries        | Robust across query types    |

## Dense vs Sparse Retrieval

### Dense Retrieval (Semantic Search)

**Strengths:**

- Understands semantic meaning and context
- Good for conceptual queries
- Handles synonyms and paraphrasing
- Cross-lingual capabilities

**Weaknesses:**

- May miss exact keyword matches
- Can be less precise for specific terms
- Computationally expensive

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Dense retriever setup
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dense_vectorstore = FAISS.from_documents(docs, embedding_model)
dense_retriever = dense_vectorstore.as_retriever()
```

### Sparse Retrieval (Keyword Search)

**Strengths:**

- Excellent for exact keyword matches
- Fast and efficient
- Good for specific terminology
- Interpretable results

**Weaknesses:**

- Misses semantic relationships
- Struggles with synonyms
- Limited by vocabulary overlap

```python
from langchain_community.retrievers import BM25Retriever

# Sparse retriever setup
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 3  # top-k documents to retrieve
```

## Ensemble Retrieval

### Reciprocal Rank Fusion (RRF)

The most common fusion algorithm for combining retrieval results:

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

class EnsembleRetriever(BaseRetriever):
    """Combines multiple retrievers using Reciprocal Rank Fusion."""

    def __init__(self, retrievers: List, weights: List[float] = None):
        super().__init__()
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # Collect results from each retriever
        all_results = [r.invoke(query) for r in self.retrievers]

        # Reciprocal Rank Fusion scoring
        scores: dict = {}
        for docs, weight in zip(all_results, self.weights):
            for rank, doc in enumerate(docs):
                key = doc.page_content
                if key not in scores:
                    scores[key] = {"doc": doc, "score": 0.0}
                scores[key]["score"] += weight / (rank + 60)

        # Return sorted by score
        ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked]
```

### Usage Example

```python
# Combine dense and sparse retrievers
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]  # 70% dense, 30% sparse
)

# Query the hybrid retriever
query = "How can I build an application using LLMs?"
results = hybrid_retriever.invoke(query)
```

## Reranking Strategies

Reranking is a two-stage process that improves result relevance:

1. **Fast Retrieval**: Get top-k candidates quickly
2. **Precise Ranking**: Use advanced models to reorder by relevance

### LLM-Based Reranking

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

# Reranking prompt
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Your task is to rank the following documents from most to least relevant to the user's question.

User Question: "{question}"

Documents:
{documents}

Instructions:
- Think about the relevance of each document to the user's question.
- Return a list of document indices in ranked order, starting from the most relevant.

Output format: comma-separated document indices (e.g., 2,1,3,0,...)
""")

# Set up LLM and chain
llm = init_chat_model("groq:llama-3.1-8b-instant", temperature=0.2)
rerank_chain = prompt | llm | StrOutputParser()

def rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    # Format documents with indices
    doc_lines = [f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)]
    formatted_docs = "\n".join(doc_lines)

    # Get reranking from LLM
    response = rerank_chain.invoke({
        "question": query,
        "documents": formatted_docs
    })

    # Parse indices and reorder
    indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
    return [docs[i] for i in indices if 0 <= i < len(docs)]
```

### Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in docs]

        # Score pairs
        scores = self.model.predict(pairs)

        # Sort by score and return top-k
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]
```

## Implementation Examples

### Complete Hybrid RAG Pipeline

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Set up retrievers
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dense_vectorstore = FAISS.from_documents(docs, embedding_model)
dense_retriever = dense_vectorstore.as_retriever()

sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 3

# 2. Create ensemble retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)

# 3. Set up LLM and prompt
llm = init_chat_model("groq:llama-3.1-8b-instant", temperature=0.2)

prompt = PromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question: {input}
""")

# 4. Create RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": hybrid_retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Query the system
query = "How can I build an app using LLMs?"
response = rag_chain.invoke(query)
print(response)
```

### Hybrid Search with Reranking

```python
def hybrid_search_with_reranking(query: str, top_k: int = 3) -> str:
    # Step 1: Initial retrieval (get more candidates)
    initial_docs = hybrid_retriever.invoke(query)

    # Step 2: Rerank documents
    reranked_docs = rerank_documents(query, initial_docs[:top_k*2])

    # Step 3: Format and generate answer
    context = format_docs(reranked_docs[:top_k])
    response = rag_chain.invoke({
        "context": context,
        "input": query
    })

    return response
```

## Performance Comparison

### Retrieval Quality Metrics

| Method             | Recall@10 | Precision@5 | MRR  | Speed  |
| ------------------ | --------- | ----------- | ---- | ------ |
| Dense Only         | 0.75      | 0.68        | 0.62 | Medium |
| Sparse Only        | 0.68      | 0.72        | 0.58 | Fast   |
| Hybrid (No Rerank) | 0.82      | 0.74        | 0.69 | Medium |
| Hybrid + Rerank    | 0.85      | 0.81        | 0.76 | Slow   |

### When to Use Each Approach

**Dense Retrieval Only:**

- Conceptual queries
- Cross-lingual search
- When semantic understanding is crucial

**Sparse Retrieval Only:**

- Exact keyword matching needed
- Fast response required
- Simple factual queries

**Hybrid Retrieval:**

- Balanced performance needed
- Diverse query types
- Production RAG systems

**Hybrid + Reranking:**

- Highest quality requirements
- Complex queries
- When latency is acceptable

## Best Practices

### Weight Optimization

```python
# Experiment with different weight combinations
weight_combinations = [
    [0.8, 0.2],  # Dense-heavy
    [0.7, 0.3],  # Balanced
    [0.6, 0.4],  # More sparse
    [0.5, 0.5],  # Equal weight
]

def evaluate_weights(test_queries, ground_truth):
    results = {}
    for weights in weight_combinations:
        retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=weights
        )
        # Evaluate performance...
        results[str(weights)] = performance_score
    return results
```

### Query Analysis

```python
def adaptive_hybrid_search(query: str) -> List[Document]:
    # Analyze query characteristics
    is_keyword_heavy = len(query.split()) <= 3
    has_technical_terms = any(term in query.lower() for term in technical_vocabulary)

    if is_keyword_heavy:
        # Favor sparse retrieval
        weights = [0.3, 0.7]
    elif has_technical_terms:
        # Favor dense retrieval
        weights = [0.8, 0.2]
    else:
        # Balanced approach
        weights = [0.6, 0.4]

    retriever = EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever], weights=weights)
    return retriever.invoke(query)
```

### Caching Strategies

```python
from functools import lru_cache

class CachedHybridRetriever:
    def __init__(self, retriever):
        self.retriever = retriever
        self._cache = {}

    @lru_cache(maxsize=1000)
    def invoke(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)
```

## Troubleshooting

### Common Issues

1. **Poor fusion results**: Adjust RRF weights and parameters
2. **Slow performance**: Reduce candidate size, optimize retrievers
3. **Inconsistent quality**: Add reranking stage
4. **Memory issues**: Implement batch processing

### Debugging Tools

```python
def analyze_retrieval_overlap(query: str):
    dense_results = dense_retriever.invoke(query)
    sparse_results = sparse_retriever.invoke(query)

    dense_content = {doc.page_content for doc in dense_results}
    sparse_content = {doc.page_content for doc in sparse_results}

    overlap = dense_content.intersection(sparse_content)

    print(f"Dense only: {len(dense_content - sparse_content)}")
    print(f"Sparse only: {len(sparse_content - dense_content)}")
    print(f"Overlap: {len(overlap)}")
```

## Files in This Module

- `1-denseparse.ipynb`: Hybrid dense+sparse retrieval implementation
- `2-reranking.ipynb`: LLM-based reranking strategies

## Advanced Topics

### Multi-Stage Retrieval

```python
def multi_stage_retrieval(query: str) -> List[Document]:
    # Stage 1: Fast initial retrieval
    candidates = hybrid_retriever.invoke(query)

    # Stage 2: Cross-encoder reranking
    reranker = CrossEncoderReranker()
    stage2_results = reranker.rerank(query, candidates[:20], top_k=10)

    # Stage 3: LLM-based final reranking
    final_results = rerank_documents(query, stage2_results)

    return final_results[:5]
```

### Domain-Specific Adaptations

- **Legal Documents**: Emphasize exact term matching
- **Scientific Papers**: Focus on semantic understanding
- **Code Search**: Combine keyword and semantic approaches
- **News Articles**: Balance recency and relevance

## Related Documentation

- [Semantic Chunking](semantic_chunking.md)
- [Vector Stores](vector_stores.md)
- [Embedding Models](embedding.md)
- [Data Ingestion](data_ingestion_parshing.md)
