# Semantic Chunking and Preprocessing Techniques

## Overview

Semantic chunking is an advanced document preprocessing technique that creates semantically coherent chunks based on embedding similarity rather than arbitrary character or token limits. This approach ensures that related content stays together, improving retrieval quality in RAG applications.

## Table of Contents

- [What is Semantic Chunking?](#what-is-semantic-chunking)
- [Benefits over Traditional Chunking](#benefits-over-traditional-chunking)
- [Implementation Approaches](#implementation-approaches)
- [Code Examples](#code-examples)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)

## What is Semantic Chunking?

Semantic chunking uses embedding models to measure similarity between sentences and groups them into chunks based on semantic coherence rather than fixed sizes. It works by:

1. **Splitting text into sentences**
2. **Computing embeddings** for each sentence using models like `all-MiniLM-L6-v2`
3. **Measuring cosine similarity** between consecutive sentences
4. **Grouping sentences** above a similarity threshold into the same chunk
5. **Creating new chunks** when similarity drops below the threshold

## Benefits over Traditional Chunking

| Traditional Chunking                  | Semantic Chunking                    |
| ------------------------------------- | ------------------------------------ |
| Fixed character/token limits          | Dynamic, content-aware boundaries    |
| Can split mid-sentence or mid-thought | Preserves semantic coherence         |
| May separate related content          | Groups related content together      |
| Simple but less intelligent           | More sophisticated and context-aware |
| Fast processing                       | Slower due to embedding computation  |

## Implementation Approaches

### 1. Custom Threshold-Based Chunker

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

class ThresholdSemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def split(self, text: str):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentences[i]]

        chunks.append(". ".join(current_chunk) + ".")
        return chunks

    def split_documents(self, docs):
        result = []
        for doc in docs:
            for chunk in self.split(doc.page_content):
                result.append(Document(page_content=chunk, metadata=doc.metadata))
        return result
```

### 2. LangChain Experimental SemanticChunker

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader

# Load documents
loader = TextLoader("document.txt")
docs = loader.load()

# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create semantic chunker
chunker = SemanticChunker(embedding)

# Split documents
chunks = chunker.split_documents(docs)
```

## Code Examples

### Basic Semantic Chunking Workflow

```python
# 1. Initialize the chunker
chunker = ThresholdSemanticChunker(threshold=0.7)

# 2. Prepare documents
doc = Document(page_content="Your text content here...")

# 3. Create semantic chunks
chunks = chunker.split_documents([doc])

# 4. Use in RAG pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()
```

### RAG Pipeline with Semantic Chunking

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

# LLM and prompt
llm = init_chat_model(model="groq:llama-3.1-8b-instant", temperature=0.4)

template = """Answer the question based on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create RAG chain
rag_chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Query the system
query = {"question": "What is LangChain used for?"}
result = rag_chain.invoke(query)
```

## Best Practices

### Threshold Selection

- **High threshold (0.8-0.9)**: Creates fewer, larger chunks with very similar content
- **Medium threshold (0.6-0.7)**: Balanced approach, good for most use cases
- **Low threshold (0.3-0.5)**: Creates many small chunks, more granular but may lose context

### Embedding Model Selection

| Model                                                         | Use Case                  | Performance                   |
| ------------------------------------------------------------- | ------------------------- | ----------------------------- |
| `all-MiniLM-L6-v2`                                            | General purpose, fast     | Good balance of speed/quality |
| `all-mpnet-base-v2`                                           | Higher quality embeddings | Better quality, slower        |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual content      | Good for non-English text     |

### Text Preprocessing

```python
def preprocess_text(text):
    # Clean and normalize text before chunking
    text = text.replace('\n', ' ')  # Handle line breaks
    text = ' '.join(text.split())   # Normalize whitespace
    return text
```

## Performance Considerations

### Memory Usage

- Embedding computation requires loading the model into memory
- Consider batch processing for large documents
- Use lighter models for resource-constrained environments

### Processing Time

- Semantic chunking is slower than character-based splitting
- Embedding computation is the bottleneck
- Consider caching embeddings for repeated processing

### Scalability Tips

```python
# Process documents in batches
def process_documents_batch(docs, batch_size=10):
    chunker = ThresholdSemanticChunker()
    all_chunks = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_chunks = chunker.split_documents(batch)
        all_chunks.extend(batch_chunks)

    return all_chunks
```

## Integration with Vector Stores

Semantic chunking works well with all vector store implementations:

- **ChromaDB**: Good for local development and small to medium datasets
- **FAISS**: Excellent performance for similarity search
- **Pinecone**: Managed solution, good for production
- **Weaviate**: Advanced features, good for complex use cases

## Troubleshooting

### Common Issues

1. **Empty chunks**: Adjust sentence splitting logic for your text format
2. **Very large chunks**: Lower the similarity threshold
3. **Too many small chunks**: Increase the similarity threshold
4. **Memory errors**: Use smaller batch sizes or lighter embedding models

### Debugging Tips

```python
# Debug similarity scores
def debug_similarity(text, threshold=0.7):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        print(f"Sentence {i-1} -> {i}: {sim:.3f} ({'Keep' if sim >= threshold else 'Split'})")
```

## Files in This Module

- `1-semantichunking.ipynb`: Complete implementation and examples
- `langchain_intro.txt`: Sample text file for testing

## Next Steps

After implementing semantic chunking, consider:

1. **Hybrid Search**: Combine with BM25 for better retrieval (see `hybrid_search_strategies.md`)
2. **Reranking**: Add LLM-based reranking for improved relevance
3. **Evaluation**: Test chunk quality with your specific use case
4. **Optimization**: Fine-tune threshold and model selection for your domain

## Related Documentation

- [Vector Stores](vector_stores.md)
- [Hybrid Search Strategies](hybrid_search_strategies.md)
- [Embedding Models](embedding.md)
