# LangChain Updated — Modern Patterns & Features

## Overview

The `7-updatedlangchain/` module covers the **modern LangChain 1.x API** — the patterns, abstractions, and capabilities that replace older deprecated approaches. Each notebook focuses on a distinct area: model integration, tool use, message types, structured output, and middleware for agent control.

## Table of Contents

- [Module Structure](#module-structure)
- [Notebook 1 — LangChain Intro & Agents](#notebook-1--langchain-intro--agents)
- [Notebook 2 — Multi-Model Integration](#notebook-2--multi-model-integration)
- [Notebook 3 — Tools & Tool Execution](#notebook-3--tools--tool-execution)
- [Notebook 4 — Message Types](#notebook-4--message-types)
- [Notebook 5 — Structured Output](#notebook-5--structured-output)
- [Notebook 6 — Middleware](#notebook-6--middleware)
- [Quick Reference](#quick-reference)

---

## Module Structure

| File                       | Topic                             | Key APIs                                                    |
| -------------------------- | --------------------------------- | ----------------------------------------------------------- |
| `1-langchainintro.ipynb`   | Agent creation basics             | `create_agent()`, `@tool`                                   |
| `2-modelintegration.ipynb` | OpenAI, Gemini, Groq              | `init_chat_model()`, `.stream()`, `.batch()`                |
| `3-tools.ipynb`            | Tool definition & execution loop  | `@tool`, `bind_tools()`, `tool_calls`                       |
| `4-messages.ipynb`         | Message types & conversation      | `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage` |
| `5-structuredoutput.ipynb` | Typed LLM responses               | `.with_structured_output()`, `BaseModel`, `TypedDict`       |
| `6-middleware.ipynb`       | Agent control & human-in-the-loop | `SummarizationMiddleware`, `HumanInTheLoopMiddleware`       |

---

## Notebook 1 — LangChain Intro & Agents

**File**: `1-langchainintro.ipynb`

Introduction to LangChain 1.x and creating agents with tools.

### Key Concepts

- **Agent**: An LLM that decides which tools to use and when
- **Tool**: A function the agent can call to act on the world
- **Message-based invocation**: Agents receive and return message lists

### Creating an Agent

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

agent = create_agent(
    model="gpt-5",
    tools=[get_weather]
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}]
})
```

---

## Notebook 2 — Multi-Model Integration

**File**: `2-modelintegration.ipynb`

LangChain 1.x provides a **unified interface** across all major LLM providers — one API works for OpenAI, Google Gemini, and Groq.

### Universal Initialization

```python
from langchain.chat_models import init_chat_model

# Works for any provider
llm = init_chat_model("gpt-4.1")                    # OpenAI
llm = init_chat_model("gemini-2.5-flash")           # Google
llm = init_chat_model("groq:qwen/qwen3-32b")        # Groq
```

### Provider-Specific Classes

```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

llm_openai = ChatOpenAI(model="gpt-4.1")
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_groq   = ChatGroq(model="qwen/qwen3-32b")
```

### Invocation Modes

```python
# Single invocation
response = llm.invoke("Explain LangChain in one sentence.")
print(response.content)

# Streaming — get chunks as they arrive
for chunk in llm.stream("Write a short poem about Python."):
    print(chunk.content, end="", flush=True)

# Batch — run multiple queries with concurrency control
responses = llm.batch(
    ["What is FAISS?", "What is ChromaDB?", "What is Pinecone?"],
    config={"max_concurrency": 3}
)
```

### Supported Models

| Provider | Model Examples                              |
| -------- | ------------------------------------------- |
| OpenAI   | `gpt-4.1`, `gpt-4o-mini`                    |
| Google   | `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| Groq     | `qwen/qwen3-32b`, `llama-3.1-8b-instant`    |

---

## Notebook 3 — Tools & Tool Execution

**File**: `3-tools.ipynb`

Covers how to define tools, bind them to models, and implement the tool execution loop.

### Defining Tools with `@tool`

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)
```

### Binding Tools to a Model

```python
llm = init_chat_model("groq:qwen/qwen3-32b")
llm_with_tools = llm.bind_tools([search_web, calculate])
```

### The Tool Execution Loop

```python
from langchain_core.messages import HumanMessage, ToolMessage

messages = [HumanMessage(content="What is 42 * 1337?")]

# Step 1: Model decides to call a tool
response = llm_with_tools.invoke(messages)
messages.append(response)

# Step 2: Execute the tool calls
for tool_call in response.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id   = tool_call["id"]

    # Find and execute the tool
    tool_map = {"search_web": search_web, "calculate": calculate}
    result = tool_map[tool_name].invoke(tool_args)

    # Step 3: Append tool result as ToolMessage
    messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

# Step 4: Model generates final answer
final_response = llm_with_tools.invoke(messages)
print(final_response.content)
```

---

## Notebook 4 — Message Types

**File**: `4-messages.ipynb`

LangChain uses a structured message system where every interaction is typed. Understanding message types is essential for building conversations and tool-using agents.

### Message Types

| Type            | Role        | Used For                                 |
| --------------- | ----------- | ---------------------------------------- |
| `SystemMessage` | `system`    | Instructions/persona for the LLM         |
| `HumanMessage`  | `user`      | User input or questions                  |
| `AIMessage`     | `assistant` | LLM responses (may include `tool_calls`) |
| `ToolMessage`   | `tool`      | Result returned after executing a tool   |

### Usage Example

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

messages = [
    SystemMessage(content="You are a helpful data analyst."),
    HumanMessage(content="What's the total of 150 + 280?", name="Alice"),
]

response = llm.invoke(messages)  # → AIMessage
print(response.content)
print(response.usage_metadata)   # Token counts: input, output, total
```

### Tool Message Linking

`ToolMessage` must reference the `tool_call_id` from the `AIMessage` that requested it:

```python
ai_msg = llm_with_tools.invoke(messages)
# ai_msg.tool_calls = [{"name": "calculate", "args": {...}, "id": "call_abc123"}]

tool_result = ToolMessage(
    content="430",
    tool_call_id=ai_msg.tool_calls[0]["id"]  # Must match
)
```

### Message Metadata

```python
HumanMessage(
    content="Summarize this document.",
    name="Bob",       # Optional display name
    id="msg-001"      # Optional custom ID
)
```

---

## Notebook 5 — Structured Output

**File**: `5-structuredoutput.ipynb`

Force the LLM to return responses in a defined schema instead of free-form text. Useful for data extraction, classification, and any task requiring machine-readable output.

### Three Schema Approaches

#### 1. Pydantic BaseModel (recommended)

```python
from pydantic import BaseModel, Field
from typing import Optional

class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: float | None = Field(description="Rating from 0-10")
    summary: str = Field(description="Brief review summary")
    recommended: bool = Field(description="Whether to recommend it")

structured_llm = llm.with_structured_output(MovieReview)
result = structured_llm.invoke("Review the movie Inception")
print(result.title)    # str
print(result.rating)   # float
```

#### 2. TypedDict (lighter weight)

```python
from typing import TypedDict, Annotated

class PersonInfo(TypedDict):
    name: Annotated[str, "Full name of the person"]
    age:  Annotated[int, "Age in years"]
    city: Annotated[str, "City of residence"]

structured_llm = llm.with_structured_output(PersonInfo)
```

#### 3. Dataclass

```python
from dataclasses import dataclass

@dataclass
class SentimentResult:
    sentiment: str    # "positive", "negative", "neutral"
    confidence: float
    reasoning: str

structured_llm = llm.with_structured_output(SentimentResult)
```

### Include Raw Response

```python
# Get both the parsed structure AND the raw AIMessage
structured_llm = llm.with_structured_output(MovieReview, include_raw=True)
result = structured_llm.invoke("Review Inception")

parsed_result = result["parsed"]           # MovieReview object
raw_message   = result["raw"]              # AIMessage
```

### With Agents

```python
agent = create_agent(
    model="gpt-5",
    tools=[search_web],
    response_format=MovieReview   # Agent always returns this schema
)
```

---

## Notebook 6 — Middleware

**File**: `6-middleware.ipynb`

Middleware wraps agents to add capabilities like automatic summarization and human-in-the-loop approval — without changing the agent's core logic.

### Summarization Middleware

Automatically summarizes conversation history when it gets too long, keeping token usage under control.

```python
from langchain_agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_web],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            trigger=("messages", 10),  # Summarize after 10 messages
            keep=("messages", 3),      # Keep last 3 messages after summary
        )
    ]
)
```

**Trigger Options:**

| Trigger           | Meaning                                    |
| ----------------- | ------------------------------------------ |
| `("messages", N)` | Trigger after N messages                   |
| `("tokens", N)`   | Trigger after N tokens                     |
| `("fraction", X)` | Trigger after X fraction of context window |

### Human-in-the-Loop Middleware

Pauses the agent before executing specified tools and waits for human approval, rejection, or edit.

```python
from langchain_agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[send_email, delete_file],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True, "delete_file": True},
            allowed_decisions=["approve", "edit", "reject"]
        )
    ]
)

config = {"configurable": {"thread_id": "session-1"}}

# Agent pauses when it tries to call send_email
response = agent.invoke({"messages": [HumanMessage("Send a report to Alice")]}, config)

# Check if waiting for approval
if "__interrupt__" in response:
    # Approve the tool call
    final = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    # Or edit the tool call args before executing
    final = agent.invoke(Command(resume={"decisions": [{
        "type": "edit",
        "edited_action": {"args": {"recipient": "bob@example.com", "subject": "Updated Report"}}
    }]}), config)

    # Or reject it entirely
    final = agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), config)
```

### State Persistence with Checkpointers

Checkpointers allow multi-turn conversations with state preserved across calls:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_web],
    checkpointer=checkpointer
)

# Same thread_id maintains conversation history
config = {"configurable": {"thread_id": "user-session-42"}}

agent.invoke({"messages": [HumanMessage("My name is Alice")]}, config)
agent.invoke({"messages": [HumanMessage("What is my name?")]}, config)
# → "Your name is Alice"  (remembers context)
```

---

## Quick Reference

### Model Initialization

```python
# Universal (recommended)
llm = init_chat_model("groq:llama-3.1-8b-instant", temperature=0.2)

# Provider-specific
llm = ChatOpenAI(model="gpt-4.1")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatGroq(model="qwen/qwen3-32b")
```

### LCEL Chaining

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_template("Answer: {question}") | llm | StrOutputParser()
result = chain.invoke({"question": "What is RAG?"})
```

### Structured Output

```python
class Output(BaseModel):
    answer: str
    confidence: float

structured = llm.with_structured_output(Output)
result = structured.invoke("Is Python good for ML?")
```

### Tool Loop (3 Steps)

```python
# 1. Model generates tool_calls
# 2. Execute tools → ToolMessage
# 3. Model generates final answer
```

---

## Files in This Module

| File                       | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| `1-langchainintro.ipynb`   | Agent creation, basic tool use, message invocation                     |
| `2-modelintegration.ipynb` | OpenAI, Gemini, Groq — invoke, stream, batch                           |
| `3-tools.ipynb`            | `@tool` decorator, `bind_tools()`, execution loop                      |
| `4-messages.ipynb`         | All message types, metadata, conversation management                   |
| `5-structuredoutput.ipynb` | Pydantic, TypedDict, dataclass schemas, `with_structured_output()`     |
| `6-middleware.ipynb`       | `SummarizationMiddleware`, `HumanInTheLoopMiddleware`, `InMemorySaver` |

## Related Documentation

- [Vector Stores](vector_stores.md)
- [Semantic Chunking](semantic_chunking.md)
- [Hybrid Search Strategies](hybrid_search_strategies.md)
- [Query Enhancement](query_enhancement.md)
- [Multimodal RAG](multimodal_rag.md)
