# 17 — RAG Evaluation

## What Is RAG Evaluation?

RAG Evaluation is the systematic process of measuring how well a RAG application performs across multiple dimensions — answer accuracy, relevance, groundedness, and retrieval quality. This notebook uses **LangSmith** as the evaluation platform and **LLM-as-a-Judge** as the evaluation approach.

---

## Notebook: `1-rag_eval.ipynb`

Two evaluation workflows are built: a **chatbot evaluation** and a **full RAG pipeline evaluation**.

---

## Part 1 — Chatbot Evaluation

### Dataset

A test dataset is created in LangSmith with 5 question–answer pairs covering LangChain, LangSmith, OpenAI, Google, and Mistral.

```python
client = Client()
dataset = client.create_dataset("Chatbots Evaluation")
client.create_examples(dataset_id=dataset.id, examples=[...])
```

### Metrics

**Correctness** — LLM-as-judge compares predicted answer vs reference answer:

```python
def correctness(inputs, outputs, reference_outputs) -> bool:
    # GPT-4o-mini grades as CORRECT or INCORRECT
```

**Concision** — Rule-based check; passes if response is less than 2× the reference length:

```python
def concision(outputs, reference_outputs) -> bool:
    return len(outputs["response"]) < 2 * len(reference_outputs["answer"])
```

### Running the Evaluation

Two model variants are evaluated and compared side-by-side in LangSmith:

```python
# Experiment 1
client.evaluate(ls_target, data=dataset_name,
    evaluators=[correctness, concision],
    experiment_prefix="openai-4o-mini-chatbot")

# Experiment 2 — different model
client.evaluate(ls_target, data=dataset_name,
    evaluators=[correctness, concision],
    experiment_prefix="openai-4-turbo-chatbot")
```

---

## Part 2 — RAG Pipeline Evaluation

### RAG Application Under Test

A `@traceable` RAG bot retrieves from 3 Lilian Weng blog posts (agents, prompt engineering, adversarial attacks) and answers concisely:

```python
@traceable()
def rag_bot(question: str) -> dict:
    docs = retriever.invoke(question)
    # ... LLM answers using retrieved docs
    return {"answer": ai_msg.content, "documents": docs}
```

### Dataset

3 domain-specific Q&A pairs on ReAct, few-shot biases, and adversarial attacks — topics directly covered in the indexed blogs.

### 4 Evaluation Metrics (LLM-as-Judge)

Each metric uses a structured output (`TypedDict`) with an `explanation` field that forces the judge LLM to reason before scoring.

---

#### 1. Correctness

**What:** Does the answer match the reference (ground-truth) answer factually?
**Requires:** Reference answer from dataset.

```python
class CorrectnessGrade(TypedDict):
    explanation: str
    correct: bool
```

---

#### 2. Relevance

**What:** Does the answer actually address the user's question?
**Requires:** Question + answer only (no reference needed).

```python
class RelevanceGrade(TypedDict):
    explanation: str
    relevant: bool
```

---

#### 3. Groundedness

**What:** Is the answer supported by the retrieved documents (no hallucination)?
**Requires:** Retrieved documents + answer.

```python
class GroundedGrade(TypedDict):
    explanation: str
    grounded: bool
```

---

#### 4. Retrieval Relevance

**What:** Are the retrieved documents relevant to the question?
**Requires:** Retrieved documents + question.

```python
class RetrievalRelevanceGrade(TypedDict):
    explanation: str
    relevant: bool
```

---

### Running the Full RAG Evaluation

```python
experiment_results = client.evaluate(
    target,                         # wraps rag_bot()
    data="RAG Test Evaluation",
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="rag-doc-relevance",
)
experiment_results.to_pandas()      # view results as a DataFrame
```

---

## Evaluation Coverage Summary

| Metric              | Inputs needed                 | What it catches         |
| ------------------- | ----------------------------- | ----------------------- |
| Correctness         | Question + answer + reference | Factually wrong answers |
| Relevance           | Question + answer             | Off-topic answers       |
| Groundedness        | Answer + retrieved docs       | Hallucinated facts      |
| Retrieval Relevance | Question + retrieved docs     | Bad retrieval           |

---

## Tools & Integrations

| Tool                                       | Role                                                          |
| ------------------------------------------ | ------------------------------------------------------------- |
| **LangSmith**                              | Dataset management, experiment tracking, result visualization |
| **LangSmith `Client`**                     | Create datasets, run evaluations programmatically             |
| `@traceable()` decorator                   | Automatically logs RAG bot runs to LangSmith                  |
| `ChatOpenAI` with `with_structured_output` | Forces judge LLM to return typed, validated scores            |
