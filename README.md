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
│   └── vector_stores.md             # Vector stores guide
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
└── 2-vector-store/                   # Vector stores module
    ├── 1-chromadb.ipynb              # ChromaDB + conversational RAG
    ├── 2-faiss.ipynb                 # FAISS similarity search
    ├── 3-Othervectorstores.ipynb     # InMemoryVectorStore
    ├── 4-Datastaxdb.ipynb            # DataStax Astra DB
    ├── 5-PineconeVectorDB.ipynb      # Pinecone cloud vector DB
    ├── chroma_db/                    # Persisted ChromaDB index
    ├── faiss_index/                  # Persisted FAISS index
    └── data/                        # Sample documents
```

## 🛠️ Usage

1. **Data Ingestion**: Start with notebooks in `0-data-ingestion-parsing/`
2. **Vector Embeddings**: Convert text to numerical vectors using `1-vector-embedding-database/`
3. **Vector Stores**: Store and retrieve embeddings using `2-vector-store/`
4. **Processing**: Run individual notebooks or use batch processing
5. **Integration**: Use processed documents and embeddings in your RAG pipeline

## 📝 Documentation

- [Data Ingestion & Parsing Guide](docs/data_ingestion_parshing.md)
- [Vector Embeddings Guide](docs/embedding.md)
- [Vector Stores Guide](docs/vector_stores.md)

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
