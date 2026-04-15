# 📁 Data Ingestion & Parsing Overview

The `0-data-ingestion-parsing` folder is a comprehensive **RAG (Retrieval-Augmented Generation) data preprocessing pipeline** that handles different file types and converts them into **searchable documents**.

## 🎯 Core Purpose

Each notebook follows the same pattern:

1. **Load** different file types → **Convert** to LangChain Documents → **Split** using 3 parsing techniques

## 📚 File-Specific Notebooks

| Notebook                             | File Type      | Purpose                              |
| ------------------------------------ | -------------- | ------------------------------------ |
| `1-data-ingestion_text.ipynb`        | `.txt` files   | Basic text processing                |
| `2-data-ingestion_pdf.ipynb`         | `.pdf` files   | PDF extraction with multiple loaders |
| `3-data-ingestion_word_doc.ipynb`    | `.docx` files  | Word document processing             |
| `4-data-ingestion_csv_excel.ipynb`   | `.csv/.xlsx`   | Structured data handling             |
| `5-data-ingestion-jsonparsing.ipynb` | `.json/.jsonl` | JSON data processing                 |
| `6-data-ingestion-sql-query.ipynb`   | Database       | SQL database extraction              |

## 🔧 The 3 Parsing Techniques

Each notebook uses these **text splitting strategies**:

### 1️⃣ CharacterTextSplitter

```python
CharacterTextSplitter(
    separator="\n",      # Split on newlines
    chunk_size=200,      # Max 200 characters
    chunk_overlap=20     # 20 character overlap
)
```

### 2️⃣ RecursiveCharacterTextSplitter ⭐ **(RECOMMENDED)**

```python
RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],  # Try these in order
    chunk_size=200,
    chunk_overlap=20
)
```

### 3️⃣ TokenTextSplitter

```python
TokenTextSplitter(
    chunk_size=50,       # 50 tokens (not characters)
    chunk_overlap=10
)
```

## 📋 Workflow Pattern

1. **Load** → Use appropriate loader (TextLoader, PyPDFLoader, etc.)
2. **Convert** → Create LangChain `Document` objects with metadata
3. **Split** → Apply one of the 3 text splitters
4. **Result** → Chunks ready for embedding in RAG system

## 🔍 Document Structure

Each processed document follows this structure:

```python
Document(
    page_content="Main text content for embedding and search",
    metadata={
        "source": "file_path",
        "page": 1,
        "author": "creator",
        "date_created": "timestamp",
        "custom_field": "any_value"
    }
)
```

## 📊 Splitting Strategy Comparison

| Strategy      | Best For       | Pros                               | Cons                   |
| ------------- | -------------- | ---------------------------------- | ---------------------- |
| **Character** | Simple text    | Fast, predictable                  | May break mid-sentence |
| **Recursive** | Most cases     | Smart splitting, preserves meaning | Slightly slower        |
| **Token**     | LLM processing | Token-aware, precise               | Requires tokenizer     |

## 💡 Why This Matters

- **Different file types** need different extraction methods
- **Proper chunking** improves RAG retrieval accuracy
- **Metadata preservation** enables filtering and source tracking
- **Consistent format** makes everything compatible with vector databases

## 🏗️ Data Pipeline Flow

```
Raw Files → Loaders → Documents → Text Splitters → Chunks → Vector DB → RAG System
```

This folder essentially serves as a **data preprocessing factory** for RAG systems, ensuring all content is properly formatted and searchable! 🏭

## 📁 Sample Data Structure

```
data/
├── databases/          # SQLite files
├── json_files/         # JSON/JSONL data
│   ├── company_data.json
│   └── events.jsonl
├── pdf/               # PDF documents
├── structured_files/   # CSV/Excel files
│   └── products.csv
├── text_files/        # Plain text files
│   ├── machine_learning.txt
│   └── python_intro.txt
└── word_files/        # Word documents
    └── proposal.txt
```
