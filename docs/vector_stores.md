# 🗄️ Vector Stores Guide

A comprehensive guide to the vector store implementations explored in this project, covering ChromaDB, FAISS, In-Memory vector stores, and Pinecone.

**Location**: `2-vector-store/`

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [ChromaDB](#1-chromadb)
3. [FAISS](#2-faiss)
4. [Other Vector Stores (In-Memory)](#3-other-vector-stores-in-memory)
5. [Pinecone](#4-pinecone-vector-db)
6. [Comparison Table](#comparison-table)
7. [Choosing the Right Vector Store](#choosing-the-right-vector-store)

---

## Overview

Vector stores are databases optimized for storing and retrieving high-dimensional vector embeddings. They are the backbone of RAG (Retrieval-Augmented Generation) systems, enabling fast semantic similarity search over large document collections.

### Common Workflow

```
Documents → Embeddings → Vector Store → Similarity Search → LLM → Response
```

---

## 1. ChromaDB

**Notebook**: `1-chromadb.ipynb`

ChromaDB is an open-source, AI-native embedding database designed for building LLM applications. It is the most widely used local vector store in LangChain-based RAG systems.

### What We Built

- **Full RAG Pipeline**: End-to-end document loading, chunking, embedding, storage, and retrieval
- **Conversational RAG**: Multi-turn chat with persistent conversation history using `RunnableWithMessageHistory`
- **Advanced Search**: Similarity search with relevance scores
- **CRUD Operations**: Add, update, and delete documents from the vector store

### Key Concepts Covered

#### Document Pipeline

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create and persist vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
    collection_name="rag_collection"
)
```

#### Standard RAG Chain (LCEL)

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

#### Conversational RAG with Message History

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

# Modern approach - replaces deprecated create_history_aware_retriever
conversational_rag_with_history = RunnableWithMessageHistory(
    conversational_rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Invoke with session-based history (auto-managed)
config = {"configurable": {"session_id": "session_1"}}
result = conversational_rag_with_history.invoke({"input": "question"}, config=config)
```

### Features Demonstrated

| Feature            | Description                                                             |
| ------------------ | ----------------------------------------------------------------------- |
| Document Loading   | `DirectoryLoader` with `TextLoader` for batch file loading              |
| Text Splitting     | `RecursiveCharacterTextSplitter` with configurable chunk size & overlap |
| Embeddings         | OpenAI `text-embedding-ada-002`                                         |
| Similarity Search  | L2 distance-based search with score retrieval                           |
| Persistence        | SQLite-backed persistent storage via `persist_directory`                |
| CRUD               | Add new documents, update metadata, delete by ID                        |
| Conversational RAG | Session-based multi-turn Q&A with `RunnableWithMessageHistory`          |

### ChromaDB Similarity Scores

ChromaDB uses **L2 distance** (Euclidean distance) by default:

- **Lower score = MORE similar** (closer in vector space)
- Score of `0` = identical vectors
- Typical range: `0` to `2`

---

## 2. FAISS

**Notebook**: `2-faiss.ipynb`

FAISS (Facebook AI Similarity Search) is a high-performance library from Meta for efficient similarity search and clustering of dense vectors. It is ideal for applications requiring very fast search over large datasets.

### What We Built

- **RAG System with FAISS**: Full retrieval pipeline using FAISS as the vector backend
- **Local Persistence**: Save and reload FAISS indexes from disk
- **Efficient Search**: Approximate nearest neighbor search for speed

### Key Concepts Covered

#### Creating a FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create FAISS index from documents
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)
```

#### RAG Chain with FAISS

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(prompt_template)
    | llm
    | StrOutputParser()
)
```

### FAISS Advantages

| Feature          | Detail                                                     |
| ---------------- | ---------------------------------------------------------- |
| Speed            | Extremely fast — benchmarked at millions of vectors/second |
| Memory Efficient | Optimized index structures for low memory footprint        |
| GPU Support      | Optional GPU acceleration for even faster search           |
| Index Types      | Flat (exact), IVF (approximate), HNSW (hierarchical)       |
| Scalability      | Handles millions of vectors efficiently                    |
| Local            | Fully offline — no external service required               |

### When to Use FAISS

- Large-scale document collections (millions of documents)
- On-premise or air-gapped deployments requiring no external services
- Applications where search latency is critical
- Research and prototyping with large datasets

---

## 3. Other Vector Stores (In-Memory)

**Notebook**: `3-Othervectorstores.ipynb`

Explores `InMemoryVectorStore` from `langchain_core` — a lightweight, dictionary-based vector store that uses cosine similarity for search. Perfect for prototyping, testing, and small-scale applications.

### What We Built

- **In-Memory RAG**: Lightweight RAG pipeline with no persistent storage
- **Quick Prototyping**: Instant setup without any external database
- **Cosine Similarity Search**: Semantic search using cosine similarity

### Key Concepts Covered

#### InMemoryVectorStore

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# Instant setup — no configuration required
vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings())

# Add documents
documents = [Document(page_content="...", metadata={"source": "tweet"}), ...]
vector_store.add_documents(documents)

# Search
results = vector_store.similarity_search("your query", k=3)
```

#### Metadata Filtering

```python
# Filter search results by metadata
results = vector_store.similarity_search(
    "your query",
    filter={"source": "tweet"}
)
```

### InMemoryVectorStore Characteristics

| Feature          | Detail                                    |
| ---------------- | ----------------------------------------- |
| Storage          | RAM only — lost on restart                |
| Search Algorithm | Cosine similarity via numpy               |
| Setup Time       | Instant — no configuration                |
| Dependencies     | Only `langchain_core` (no extra packages) |
| Best For         | Prototyping, testing, small datasets      |
| Scalability      | Limited by available RAM                  |

---

## 4. Pinecone Vector DB

**Notebook**: `5-PineconeVectorDB.ipynb`

Pinecone is a fully managed, production-ready cloud vector database. It is designed for enterprise-scale RAG applications requiring high availability, automatic scaling, and zero operational overhead.

### What We Built

- **Pinecone Index Setup**: Created a serverless index with cosine similarity
- **Cloud-Based RAG**: Full retrieval pipeline backed by Pinecone's managed infrastructure
- **Scalable Document Storage**: Added and queried documents at scale

### Key Concepts Covered

#### Setup and Index Creation

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key="your_api_key")

# Create a serverless index
pc.create_index(
    name="rag",
    dimension=1024,          # Must match embedding model dimensions
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("rag")
```

#### LangChain Integration

```python
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Configure embeddings to match index dimensions
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024
)

# Connect LangChain to Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Add documents
vector_store.add_documents(documents)

# Similarity search
results = vector_store.similarity_search("query", k=3)
```

### Pinecone Advantages

| Feature      | Detail                                         |
| ------------ | ---------------------------------------------- |
| Deployment   | Fully managed cloud — no infrastructure needed |
| Scaling      | Auto-scales to billions of vectors             |
| Availability | 99.9%+ uptime SLA                              |
| Index Types  | Dense (semantic), Sparse (keyword), Hybrid     |
| Regions      | Multi-cloud (AWS, GCP, Azure)                  |
| Pricing      | Free tier available; pay-as-you-scale          |

### When to Use Pinecone

- Production applications requiring high availability
- Multi-region or multi-cloud deployments
- Teams that want zero infrastructure management
- Applications that need to scale dynamically

---

## Comparison Table

| Feature               | ChromaDB               | FAISS             | InMemory          | Pinecone         |
| --------------------- | ---------------------- | ----------------- | ----------------- | ---------------- |
| **Type**              | Local DB               | Local Library     | In-RAM            | Cloud Managed    |
| **Setup**             | Simple                 | Simple            | Instant           | API Key required |
| **Persistence**       | ✅ SQLite              | ✅ Disk files     | ❌ RAM only       | ✅ Cloud         |
| **Scale**             | Medium                 | Large             | Small             | Massive          |
| **Speed**             | Fast                   | Very Fast         | Fast (small data) | Fast (managed)   |
| **Cost**              | Free                   | Free              | Free              | Free tier + paid |
| **Best For**          | Local RAG, prototyping | Large-scale local | Testing, demos    | Production apps  |
| **GPU Support**       | ❌                     | ✅                | ❌                | N/A              |
| **LangChain Support** | ✅                     | ✅                | ✅                | ✅               |

---

## Choosing the Right Vector Store

```
📦 Just prototyping / learning?
    → InMemoryVectorStore (zero setup)

🖥️ Local development / small-medium scale?
    → ChromaDB (easy persistence, great LangChain support)

⚡ Need maximum speed / large local dataset?
    → FAISS (best raw performance, GPU support)

🌐 Production / enterprise / multi-user?
    → Pinecone (fully managed, auto-scaling, high availability)
```

---

## Common Patterns Across All Vector Stores

All notebooks follow the same retrieval pattern via the LangChain interface:

```python
# 1. Create retriever (same API for all stores)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 2. Build RAG chain (LCEL — works with any vector store)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 3. Invoke
response = rag_chain.invoke("Your question here")
```

LangChain's unified interface means you can **swap vector stores with a single line change** — the rest of your RAG pipeline stays identical.

---

_Last Updated: April 2026_
