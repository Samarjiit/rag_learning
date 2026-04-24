# Multimodal RAG (PDF with Images)

## Overview

Multimodal RAG extends standard text-based RAG to handle both **text and images** from PDF documents. It uses **CLIP (Contrastive Language-Image Pretraining)** to generate a unified embedding space where text and images can be compared and retrieved together, then passes the retrieved context (text + images) to a vision-capable LLM like GPT-4 Vision.

## Table of Contents

- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Key Components](#key-components)
- [Pipeline Flow](#pipeline-flow)
- [Implementation](#implementation)
- [CLIP Embeddings](#clip-embeddings)
- [PDF Processing](#pdf-processing)
- [Retrieval](#retrieval)
- [Answer Generation](#answer-generation)
- [Common Issues & Fixes](#common-issues--fixes)

---

## Architecture

```
PDF Document
     │
     ├── Text Pages ──────────────────────────────────────────────────┐
     │   └── Chunks (RecursiveCharacterTextSplitter)                  │
     │       └── CLIP Text Embedding → FAISS index                   │
     │                                                                 ▼
     └── Images (PyMuPDF) ──────────────────────────────── Unified FAISS Vector Store
         ├── PIL Image → base64 (stored for GPT-4V)                  │
         └── CLIP Image Embedding → FAISS index                      │
                                                                      │
User Query → CLIP Text Embedding → Similarity Search                 │
                                        │                             │
                                   Top-k Results ◄────────────────────┘
                                   (text + images)
                                        │
                              GPT-4 Vision (multimodal message)
                                        │
                                   Final Answer
```

---

## How It Works

### Why CLIP?

CLIP creates a **shared embedding space** for both text and images — meaning a text query like _"bar chart showing revenue"_ will be close in vector space to an actual image of a bar chart. This enables cross-modal retrieval: one query can retrieve both relevant text passages and relevant images simultaneously.

| Standard RAG        | Multimodal RAG              |
| ------------------- | --------------------------- |
| Text only           | Text + Images               |
| Word embeddings     | CLIP unified embeddings     |
| GPT-3.5/4           | GPT-4 Vision                |
| PDF text extraction | PDF text + image extraction |

---

## Key Components

| Component      | Tool                                  | Purpose                                       |
| -------------- | ------------------------------------- | --------------------------------------------- |
| PDF parsing    | `PyMuPDF (fitz)`                      | Extract text and images per page              |
| Text splitting | `RecursiveCharacterTextSplitter`      | Chunk text into 500-token segments            |
| Embeddings     | `CLIP (openai/clip-vit-base-patch32)` | Unified text + image embeddings               |
| Vector store   | `FAISS`                               | Similarity search over precomputed embeddings |
| Vision LLM     | `GPT-4.1 (openai:gpt-4.1)`            | Answer questions using text + image context   |

---

## Pipeline Flow

```
Step 1: Load PDF (PyMuPDF)
         │
Step 2: Extract text per page → chunk → CLIP embed → store in FAISS
         │
Step 3: Extract images per page → PIL → base64 (for GPT-4V) + CLIP embed → store in FAISS
         │
Step 4: User query → CLIP text embed → FAISS similarity search → top-k docs
         │
Step 5: Separate retrieved text docs and image docs
         │
Step 6: Build multimodal message → [text context + base64 images]
         │
Step 7: GPT-4V generates answer from multimodal context
```

---

## Implementation

### 1. Imports and Setup

```python
import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import base64
import io
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

### 2. Initialize CLIP Model

```python
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
```

---

## CLIP Embeddings

Both text and images are embedded into the same 512-dimensional vector space using CLIP.

> **Note**: In newer versions of `transformers`, `get_text_features()` / `get_image_features()` may return a `BaseModelOutputWithPooling` object instead of a raw tensor. Always extract the tensor using `.pooler_output` if needed before normalizing.

```python
import torch.nn.functional as F

def _to_tensor(features):
    """Extract tensor from model output if needed."""
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features

def embed_image(image_data):
    """Embed image using CLIP."""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = _to_tensor(clip_model.get_image_features(**inputs))
        return F.normalize(features, dim=-1).squeeze().numpy()

def embed_text(text):
    """Embed text using CLIP."""
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77  # CLIP's max token length
    )
    with torch.no_grad():
        features = _to_tensor(clip_model.get_text_features(**inputs))
        return F.normalize(features, dim=-1).squeeze().numpy()
```

### Why `F.normalize` instead of `.norm()`?

`features / features.norm(dim=-1, keepdim=True)` fails when `features` is a `BaseModelOutputWithPooling` object. `F.normalize(features, dim=-1)` is the idiomatic, version-safe PyTorch approach.

---

## PDF Processing

```python
pdf_path = "multimodal_sample.pdf"
doc = fitz.open(pdf_path)

all_docs = []          # LangChain Document objects
all_embeddings = []    # Corresponding CLIP embeddings
image_data_store = {}  # base64 images for GPT-4V

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for i, page in enumerate(doc):
    # --- Text ---
    text = page.get_text()
    if text.strip():
        temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
        for chunk in splitter.split_documents([temp_doc]):
            all_embeddings.append(embed_text(chunk.page_content))
            all_docs.append(chunk)

    # --- Images ---
    for img_index, img in enumerate(page.get_images(full=True)):
        try:
            xref = img[0]
            image_bytes = doc.extract_image(xref)["image"]
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_id = f"page_{i}_img_{img_index}"

            # Store as base64 for GPT-4V
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            image_data_store[image_id] = base64.b64encode(buffered.getvalue()).decode()

            # CLIP embedding for retrieval
            all_embeddings.append(embed_image(pil_image))
            all_docs.append(Document(
                page_content=f"[Image: {image_id}]",
                metadata={"page": i, "type": "image", "image_id": image_id}
            ))
        except Exception as e:
            print(f"Error on page {i}, image {img_index}: {e}")

doc.close()
```

---

## Retrieval

```python
# Build FAISS index from precomputed CLIP embeddings
embeddings_array = np.array(all_embeddings)

vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
    embedding=None,  # precomputed — no embedding function needed
    metadatas=[doc.metadata for doc in all_docs]
)

def retrieve_multimodal(query, k=5):
    """Retrieve top-k text and image documents using CLIP query embedding."""
    query_embedding = embed_text(query)
    return vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)
```

---

## Answer Generation

```python
llm = init_chat_model("openai:gpt-4.1")

def create_multimodal_message(query, retrieved_docs):
    """Build a GPT-4V compatible message with text context + base64 images."""
    content = [{"type": "text", "text": f"Question: {query}\n\nContext:\n"}]

    text_docs = [d for d in retrieved_docs if d.metadata.get("type") == "text"]
    image_docs = [d for d in retrieved_docs if d.metadata.get("type") == "image"]

    # Add text context
    if text_docs:
        text_context = "\n\n".join(
            f"[Page {d.metadata['page']}]: {d.page_content}" for d in text_docs
        )
        content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})

    # Add images as base64
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({"type": "text", "text": f"\n[Image from page {doc.metadata['page']}]:\n"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[image_id]}"}
            })

    content.append({"type": "text", "text": "\nPlease answer the question based on the provided text and images."})
    return HumanMessage(content=content)

def multimodal_pdf_rag_pipeline(query):
    """End-to-end multimodal RAG pipeline."""
    context_docs = retrieve_multimodal(query, k=5)
    message = create_multimodal_message(query, context_docs)
    response = llm.invoke([message])
    return response.content
```

---

## Common Issues & Fixes

### `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`

**Cause**: Newer `transformers` versions return a model output object, not a raw tensor, from `get_text_features()` / `get_image_features()`.

**Fix**: Use `_to_tensor()` helper to extract `.pooler_output` before calling `F.normalize()`.

```python
def _to_tensor(features):
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features
```

---

### CLIP Token Limit

CLIP has a maximum input length of **77 tokens**. Always set `max_length=77, truncation=True` when processing text inputs.

---

### Large PDFs / Memory

For large PDFs with many images:

- Reduce `chunk_size` to lower memory per embedding
- Process pages in batches
- Use `pil_image.thumbnail((512, 512))` to reduce image resolution before embedding

---

## Files in This Module

| File                 | Description                            |
| -------------------- | -------------------------------------- |
| `1-multimodal.ipynb` | Complete multimodal RAG implementation |

## Related Documentation

- [Vector Stores](vector_stores.md)
- [Embedding Models](embedding.md)
- [Hybrid Search Strategies](hybrid_search_strategies.md)
