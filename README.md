# 🤖 GenAI - RAG System Development

A comprehensive repository for building and experimenting with **Retrieval-Augmented Generation (RAG)** systems using modern AI techniques.

## 🎯 Project Overview

This project contains modules and notebooks for developing end-to-end RAG applications, covering data ingestion, processing, embedding, retrieval, and generation workflows.

## 📚 Modules

### 📁 Data Ingestion & Parsing

**Location**: `0-data-ingestion-parsing/`

Complete data preprocessing pipeline that handles multiple file formats and converts them into searchable documents for RAG systems.

**Supported File Types**:

- 📄 Text files (`.txt`)
- 📋 PDF documents (`.pdf`)
- 📝 Word documents (`.docx`)
- 📊 CSV/Excel files (`.csv/.xlsx`)
- 🔗 JSON data (`.json/.jsonl`)
- 🗃️ SQL databases

**Key Features**:

- **3 Text Splitting Strategies**: Character, Recursive, and Token-based
- **Metadata Preservation**: Source tracking and filtering capabilities
- **LangChain Integration**: Compatible with vector databases
- **Batch Processing**: Handle multiple files efficiently

**Quick Start**:

```python
# Load and process text files
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load document
loader = TextLoader("data/text_files/sample.txt")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
```

📖 **Detailed Documentation**: [Data Ingestion Guide](docs/data_ingestion_parshing.md)

---

### 🧠 Vector Embeddings & Database

**Location**: `0-data-ingestion-parsing/1-vector-embedding-database/`

Comprehensive module for converting text into numerical vector representations and measuring semantic similarity for RAG retrieval systems.

**Supported Models**:

- 🤗 **HuggingFace Models**: Free, privacy-friendly, good quality
- 🔮 **OpenAI Models**: Premium quality, API-based, cost-effective

**Key Features**:

- **Multiple Providers**: HuggingFace and OpenAI embeddings
- **Model Comparison**: Performance vs quality analysis
- **Semantic Search**: Find documents by meaning, not keywords
- **Cosine Similarity**: Mathematical similarity measurement
- **Batch Processing**: Efficient multiple text embedding

**Popular Models**:

| Provider    | Model                    | Dimensions | Best For              |
| ----------- | ------------------------ | ---------- | --------------------- |
| HuggingFace | `all-MiniLM-L6-v2`       | 384        | Fast, general purpose |
| HuggingFace | `all-mpnet-base-v2`      | 768        | Highest quality       |
| OpenAI      | `text-embedding-3-small` | 1536       | Cost-effective        |
| OpenAI      | `text-embedding-3-large` | 3072       | Maximum accuracy      |

**Quick Start**:

```python
# HuggingFace (Free)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# OpenAI (Premium)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create embeddings
text = "Hello, world!"
vector = embeddings.embed_query(text)
print(f"Vector dimensions: {len(vector)}")
```

📖 **Detailed Documentation**: [Vector Embeddings Guide](docs/embedding.md)

---

### �️ Vector Stores

**Location**: `2-vector-store/`

Explores four major vector store solutions for RAG systems — from lightweight local databases to fully managed cloud-scale infrastructure.

**Vector Stores Covered**:

- 🟣 **ChromaDB**: Open-source local vector DB with persistence and full RAG + conversational AI
- ⚡ **FAISS**: Meta's high-performance similarity search library for large-scale local datasets
- 🧠 **InMemoryVectorStore**: Lightweight in-RAM store for prototyping and testing
- 🌐 **Pinecone**: Fully managed cloud vector DB for production-scale applications

**Key Features**:

- **Full RAG Pipelines**: Complete document → embed → retrieve → generate workflows
- **Conversational RAG**: Multi-turn chat with `RunnableWithMessageHistory` (modern session-based approach)
- **CRUD Operations**: Add, update, delete, and query documents across all stores
- **Similarity Search with Scores**: Retrieve documents with relevance ranking
- **Persistence**: Local disk (ChromaDB/FAISS) and cloud (Pinecone) storage
- **LangChain LCEL**: Unified chain syntax that works across all vector stores

**Quick Start**:

```python
# ChromaDB (local persistence)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Unified RAG chain — same for all vector stores
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

**Vector Store Comparison**:

| Store    | Type          | Scale   | Cost             | Best For                    |
| -------- | ------------- | ------- | ---------------- | --------------------------- |
| ChromaDB | Local DB      | Medium  | Free             | Local RAG, prototyping      |
| FAISS    | Local Library | Large   | Free             | Speed-critical applications |
| InMemory | In-RAM        | Small   | Free             | Testing, demos              |
| Pinecone | Cloud Managed | Massive | Free tier + paid | Production apps             |

📖 **Detailed Documentation**: [Vector Stores Guide](docs/vector_stores.md)

---

### 🧩 Advanced Chunking & Preprocessing Techniques

**Location**: `3-dvanced Chunking And Preprocessing Techniques/`

Advanced document preprocessing using semantic chunking that creates coherent chunks based on embedding similarity rather than arbitrary character limits.

**Key Features**:

- 🧠 **Semantic Chunking**: Groups sentences by semantic similarity using embeddings
- 🎯 **Threshold-Based Splitting**: Customizable similarity thresholds for chunk boundaries
- 🔧 **Custom Implementation**: `ThresholdSemanticChunker` class with configurable parameters
- 🏗️ **LangChain Integration**: Compatible with `SemanticChunker` from langchain-experimental
- ⚡ **RAG Pipeline Ready**: Seamless integration with vector stores and retrievers

**Chunking Approaches**:

| Method          | Boundary Logic           | Best For                        |
| --------------- | ------------------------ | ------------------------------- |
| Character-based | Fixed character count    | Simple, fast processing         |
| Token-based     | Fixed token count        | Model-specific limits           |
| **Semantic**    | **Embedding similarity** | **Coherent, meaningful chunks** |

**Quick Start**:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ThresholdSemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def split_documents(self, docs):
        # Creates semantically coherent chunks
        # Groups sentences with similarity >= threshold
        pass

# Usage
chunker = ThresholdSemanticChunker(threshold=0.7)
chunks = chunker.split_documents([document])
```

**Benefits over Traditional Chunking**:

- ✅ Preserves semantic coherence
- ✅ Avoids splitting mid-thought
- ✅ Better retrieval quality
- ✅ Context-aware boundaries

📖 **Detailed Documentation**: [Semantic Chunking Guide](docs/semantic_chunking.md)

---

### 🔄 Hybrid Search Strategies

**Location**: `4-hybrid-serach-strategy/`

Combines multiple retrieval methods to leverage the strengths of both semantic and keyword-based search, with advanced reranking for optimal results.

**Components**:

- ⚡ **Dense Retrieval**: FAISS + semantic embeddings for conceptual search
- 🔍 **Sparse Retrieval**: BM25 + keyword matching for exact term search
- 🎯 **Ensemble Retrieval**: Reciprocal Rank Fusion (RRF) to combine results
- 🏆 **LLM Reranking**: Advanced reordering using language models for relevance

**Retrieval Methods**:

| Method        | Strengths                        | Use Case                    |
| ------------- | -------------------------------- | --------------------------- |
| Dense (FAISS) | Semantic understanding, synonyms | Conceptual queries          |
| Sparse (BM25) | Exact keywords, fast             | Specific term search        |
| **Hybrid**    | **Best of both worlds**          | **Production RAG systems**  |
| + Reranking   | **Maximum relevance**            | **Highest quality results** |

**Quick Start**:

```python
# 1. Set up individual retrievers
dense_retriever = FAISS.from_documents(docs, embeddings).as_retriever()
sparse_retriever = BM25Retriever.from_documents(docs)

# 2. Create hybrid ensemble
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
)

# 3. Add LLM-based reranking
reranked_results = rerank_documents(query, initial_results)
```

**Performance Gains**:

- 📈 Higher recall and precision
- 🎯 Better relevance ranking
- 🔄 Robust across query types
- ⚖️ Balanced semantic + keyword matching

📖 **Detailed Documentation**: [Hybrid Search Strategies Guide](docs/hybrid_search_strategies.md)

---

### 🔎 Query Enhancement Techniques

**Location**: `5-queryEnhancement/`

Advanced query processing techniques that improve retrieval quality by transforming or enriching user queries before they reach the vector store.

**Techniques Covered**:

- 📝 **Query Expansion**: Rewrites short/vague queries with synonyms and related terms
- 🧩 **Query Decomposition**: Breaks complex multi-part questions into atomic sub-questions
- 🧠 **HyDE**: Generates a hypothetical answer and embeds it instead of the raw query

**Technique Comparison**:

| Technique           | Best For                    | Retrieval Calls      |
| ------------------- | --------------------------- | -------------------- |
| Query Expansion     | Short/vague queries         | 1 (expanded)         |
| Query Decomposition | Multi-part questions        | N (per sub-question) |
| HyDE                | Query-document language gap | 1 (hypothetical doc) |

**Quick Start**:

```python
# Query Expansion
expansion_chain = PromptTemplate.from_template(
    'Expand: "{query}" with synonyms and related terms'
) | llm | StrOutputParser()

# HyDE
hyde_chain = PromptTemplate.from_template(
    'Generate a hypothetical answer for: {query}'
) | llm | StrOutputParser()

# Retrieve using hypothetical doc embedding instead of raw query
docs = vectorstore.similarity_search(hyde_chain.invoke({"query": user_query}), k=4)
```

📖 **Detailed Documentation**: [Query Enhancement Guide](docs/query_enhancement.md)

---

### 🖼️ Multimodal RAG

**Location**: `6-multimodal_rag/`

Extends standard RAG to handle **both text and images** from PDF documents using CLIP unified embeddings and GPT-4 Vision for answer generation.

**Key Features**:

- 🔍 **CLIP Embeddings**: Unified 512-dim vector space for text and images
- 📄 **PDF Parsing**: Extracts text chunks and images per page using PyMuPDF
- 🗄️ **FAISS Index**: Precomputed multimodal embeddings for fast retrieval
- 🤖 **GPT-4 Vision**: Answers with both text context and base64 images
- 🔗 **Cross-modal Retrieval**: A text query can retrieve relevant images and vice versa

**Quick Start**:

```python
# Embed both text and images with CLIP
query_embedding = embed_text("What does the revenue chart show?")

# Retrieve top-k docs (text + images together)
results = vector_store.similarity_search_by_vector(query_embedding, k=5)

# Answer with GPT-4V using text + base64 images
response = llm.invoke([create_multimodal_message(query, results)])
```

📖 **Detailed Documentation**: [Multimodal RAG Guide](docs/multimodal_rag.md)

---

### 🔗 LangChain Updated — Modern Patterns

**Location**: `7-updatedlangchain/`

Covers the **modern LangChain 1.x API** — replacing older deprecated approaches with current patterns for model integration, tools, messages, structured output, and agent middleware.

**Notebooks**:

| Notebook                   | Focus                              | Key APIs                                                    |
| -------------------------- | ---------------------------------- | ----------------------------------------------------------- |
| `1-langchainintro.ipynb`   | Agent creation                     | `create_agent()`, `@tool`                                   |
| `2-modelintegration.ipynb` | Multi-model (OpenAI, Gemini, Groq) | `init_chat_model()`, `.stream()`, `.batch()`                |
| `3-tools.ipynb`            | Tool definition & execution loop   | `@tool`, `bind_tools()`, `tool_calls`                       |
| `4-messages.ipynb`         | Message types                      | `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage` |
| `5-structuredoutput.ipynb` | Typed LLM responses                | `.with_structured_output()`, `BaseModel`, `TypedDict`       |
| `6-middleware.ipynb`       | Agent control                      | `SummarizationMiddleware`, `HumanInTheLoopMiddleware`       |

**Quick Start**:

```python
# Universal model initialization
llm = init_chat_model("groq:llama-3.1-8b-instant")

# Structured output
class Result(BaseModel):
    answer: str
    confidence: float

result = llm.with_structured_output(Result).invoke("Is Python good for ML?")
```

📖 **Detailed Documentation**: [LangChain Updated Guide](docs/langchain_updated.md)

---

### �🔄 More Modules Coming Soon...

_This section will be updated as additional RAG components are added_

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd genai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
genai/
├── main.py                           # Main application entry point
├── requirements.txt                  # Python dependencies
├── pyproject.toml                   # Project configuration
├── README.md                        # This file
├── docs/                            # Documentation
│   ├── data_ingestion_parshing.md   # Data ingestion guide
│   ├── embedding.md                 # Vector embeddings guide
│   ├── vector_stores.md             # Vector stores guide
│   ├── semantic_chunking.md         # Semantic chunking guide
│   └── hybrid_search_strategies.md  # Hybrid search guide
├── 0-data-ingestion-parsing/        # Data preprocessing module
│   ├── 1-data-ingestion_text.ipynb
│   ├── 2-data-ingestion_pdf.ipynb
│   ├── 3-data-ingestion_word_doc.ipynb
│   ├── 4-data-ingestion_csv_excel.ipynb
│   ├── 5-data-ingestion-jsonparsing.ipynb
│   ├── 6-data-ingestion-sql-query.ipynb
│   └── data/                        # Sample data files
│       ├── databases/
│       ├── json_files/
│       ├── pdf/
│       ├── structured_files/
│       ├── text_files/
│       └── word_files/
├── 1-vector-embedding-database/      # Vector embeddings module
│   ├── embedding.ipynb               # HuggingFace embeddings
│   └── openaiEmbedding-model.ipynb   # OpenAI embeddings
├── 2-vector-store/                   # Vector stores module
│   ├── 1-chromadb.ipynb              # ChromaDB + conversational RAG
│   ├── 2-faiss.ipynb                 # FAISS similarity search
│   ├── 3-Othervectorstores.ipynb     # InMemoryVectorStore
│   ├── 4-Datastaxdb.ipynb            # DataStax Astra DB
│   ├── 5-PineconeVectorDB.ipynb      # Pinecone cloud vector DB
│   ├── chroma_db/                    # Persisted ChromaDB index
│   ├── faiss_index/                  # Persisted FAISS index
│   └── data/                        # Sample documents
├── 3-dvanced Chunking And Preprocessing Techniques/  # Advanced chunking
│   ├── 1-semantichunking.ipynb      # Semantic chunking implementation
│   └── langchain_intro.txt           # Sample text for testing
└── 4-hybrid-serach-strategy/        # Hybrid search strategies
│   ├── 1-denseparse.ipynb            # Dense + sparse hybrid retrieval
│   └── 2-reranking.ipynb             # LLM-based reranking
└── 5-queryEnhancement/              # Query enhancement techniques
    ├── 1-query-expansion.ipynb       # Query expansion with LLM
    ├── 2-query-decomposition.ipynb   # Query decomposition
    ├── 3-hyde.ipynb                  # HyDE — hypothetical document embeddings
    └── langchain_crewai_dataset.txt  # Sample dataset
6-multimodal_rag/                    # Multimodal RAG
    └── 1-multimodal.ipynb           # CLIP + GPT-4V multimodal pipeline
7-updatedlangchain/                  # Modern LangChain patterns
    ├── 1-langchainintro.ipynb       # Agents intro
    ├── 2-modelintegration.ipynb     # Multi-model integration
    ├── 3-tools.ipynb                # Tool definition and execution
    ├── 4-messages.ipynb             # Message types
    ├── 5-structuredoutput.ipynb     # Structured output schemas
    └── 6-middleware.ipynb           # Summarization & human-in-the-loop
```

## 🛠️ Usage

1. **Data Ingestion**: Start with notebooks in `0-data-ingestion-parsing/` to load and preprocess documents
2. **Vector Embeddings**: Convert text to numerical vectors using `1-vector-embedding-database/`
3. **Vector Stores**: Store and retrieve embeddings using `2-vector-store/`
4. **Semantic Chunking**: Create coherent document chunks using `3-dvanced Chunking And Preprocessing Techniques/`
5. **Hybrid Search**: Implement advanced retrieval strategies using `4-hybrid-serach-strategy/`
6. **Integration**: Combine all components into a complete RAG pipeline

## 📝 Documentation

- [Data Ingestion & Parsing Guide](docs/data_ingestion_parshing.md)
- [Vector Embeddings Guide](docs/embedding.md)
- [Vector Stores Guide](docs/vector_stores.md)
- [Semantic Chunking Guide](docs/semantic_chunking.md)
- [Hybrid Search Strategies Guide](docs/hybrid_search_strategies.md)
- [Query Enhancement Guide](docs/query_enhancement.md)
- [Multimodal RAG Guide](docs/multimodal_rag.md)
- [LangChain Updated Guide](docs/langchain_updated.md)

## 🤝 Contributing

This is an active development project. Feel free to contribute by:

- Adding new data ingestion methods
- Improving text splitting strategies
- Enhancing documentation
- Adding test cases

## 📄 License

[Add your license information here]

---

**Status**: 🚧 Under Active Development  
**Last Updated**: April 2026
