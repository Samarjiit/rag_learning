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

| Provider | Model | Dimensions | Best For |
|----------|-------|------------|----------|
| HuggingFace | `all-MiniLM-L6-v2` | 384 | Fast, general purpose |
| HuggingFace | `all-mpnet-base-v2` | 768 | Highest quality |
| OpenAI | `text-embedding-3-small` | 1536 | Cost-effective |
| OpenAI | `text-embedding-3-large` | 3072 | Maximum accuracy |

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

### 🔄 More Modules Coming Soon...

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
│   └── embedding.md                 # Vector embeddings guide
└── 0-data-ingestion-parsing/        # Data preprocessing module
    ├── 1-data-ingestion_text.ipynb
    ├── 2-data-ingestion_pdf.ipynb
    ├── 3-data-ingestion_word_doc.ipynb
    ├── 4-data-ingestion_csv_excel.ipynb
    ├── 5-data-ingestion-jsonparsing.ipynb
    ├── 6-data-ingestion-sql-query.ipynb
    ├── 1-vector-embedding-database/  # Vector embeddings module
    │   ├── embedding.ipynb           # HuggingFace embeddings
    │   └── openaiEmbedding-model.ipynb # OpenAI embeddings
    └── data/                        # Sample data files
        ├── databases/
        ├── json_files/
        ├── pdf/
        ├── structured_files/
        ├── text_files/
        └── word_files/
```

## 🛠️ Usage

1. **Data Ingestion**: Start with notebooks in `0-data-ingestion-parsing/`
2. **Vector Embeddings**: Convert text to numerical vectors using `1-vector-embedding-database/`
3. **Processing**: Run individual notebooks or use batch processing
4. **Integration**: Use processed documents and embeddings in your RAG pipeline

## 📝 Documentation

- [Data Ingestion & Parsing Guide](docs/data_ingestion_parshing.md)
- [Vector Embeddings Guide](docs/embedding.md)
- More documentation coming soon...

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
