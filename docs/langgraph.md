# LangGraph — Building Stateful AI Workflows

## Overview

The `8-langGraph/` module covers **LangGraph** — a framework for building stateful, graph-based AI workflows where nodes are Python functions and edges control the flow between them. Each notebook builds on the previous one, progressing from a simple graph to a full multi-tool chatbot.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Notebook 1 — Simple Graph Workflow](#notebook-1--simple-graph-workflow)
- [Notebook 2 — Simple Chatbot](#notebook-2--simple-chatbot)
- [Notebook 3 — State Schema Variants](#notebook-3--state-schema-variants)
- [Notebook 5 — Chains with Tools](#notebook-5--chains-with-tools)
- [Notebook 6 — Chatbot with Multiple Tools](#notebook-6--chatbot-with-multiple-tools)
- [Full Flow Summary](#full-flow-summary)
- [Quick Reference](#quick-reference)

---

## Core Concepts

| Concept              | Description                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| **State**            | A shared data object (TypedDict / dataclass / Pydantic) passed through all nodes                    |
| **Node**             | A plain Python function that reads state and returns updated state fields                           |
| **Edge**             | A directed connection between nodes — controls execution order                                      |
| **Conditional Edge** | Routes to different nodes based on a routing function's return value                                |
| **Reducer**          | A function that merges node output into state (e.g., `add_messages` appends instead of overwriting) |
| **START / END**      | Special built-in nodes — entry and exit points of the graph                                         |
| **ToolNode**         | Pre-built node that executes tool calls from an LLM's `AIMessage.tool_calls`                        |
| **tools_condition**  | Pre-built router — sends to `"tools"` if the LLM returned a tool call, else to `END`                |

### Graph Lifecycle

```
Define State → Define Nodes → Define Edges → Compile → Invoke / Stream
```

```python
graph = StateGraph(State)
graph.add_node("my_node", my_function)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)
graph_builder = graph.compile()
graph_builder.invoke({"key": "value"})
```

---

## Notebook 1 — Simple Graph Workflow

**File**: `1-langgraph_simple.ipynb`

Introduces the fundamental building blocks: State, Nodes, Edges, and conditional routing.

### State

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_info: str
```

### Nodes

Nodes are plain functions. They receive the current state and return a dict with updated fields:

```python
def start_play(state: State):
    return {"graph_info": state['graph_info'] + " I am planning to play"}

def cricket(state: State):
    return {"graph_info": state['graph_info'] + " Cricket"}

def badminton(state: State):
    return {"graph_info": state['graph_info'] + " Badminton"}
```

### Conditional Edge (Routing Function)

Returns the **name of the next node** as a string:

```python
import random
from typing import Literal

def random_play(state: State) -> Literal['cricket', 'badminton']:
    return "cricket" if random.random() > 0.5 else "badminton"
```

### Graph Construction

```python
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

graph = StateGraph(State)

graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)

graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play", random_play)  # routes to "cricket" or "badminton"
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)

graph_builder = graph.compile()
display(Image(graph_builder.get_graph().draw_mermaid_png()))
```

### Graph Flow

```
START → start_play → [cricket OR badminton] → END
                      ↑ decided by random_play()
```

### Invocation

```python
graph_builder.invoke({"graph_info": "Hey My name is Samarjit"})
# → {"graph_info": "Hey My name is Samarjit I am planning to play Cricket"}
#   (or Badminton — random each run)
```

---

## Notebook 2 — Simple Chatbot

**File**: `2-chatbot.ipynb`

Builds a single-node chatbot where an LLM processes a conversation. Introduces the `add_messages` reducer.

### The `add_messages` Reducer Problem

By default, a node's returned value **overwrites** the state field. For a chatbot, you want to **append** new messages to the history — not replace it.

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    #                         ↑ append new messages, don't replace
```

### Chatbot Node

```python
from langchain_groq import ChatGroq

llm_groq = ChatGroq(model="llama-3.1-8b-instant")

def superbot(state: State):
    return {"messages": [llm_groq.invoke(state['messages'])]}
```

### Graph

```python
graph = StateGraph(State)
graph.add_node("SuperBot", superbot)
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)
graph_builder = graph.compile()
```

```
START → SuperBot → END
```

### Invocation & Streaming

```python
# Single invocation
graph_builder.invoke({'messages': "Hi, My name is Samarjit and I like cricket"})

# Streaming — prints state snapshot after each node runs
for event in graph_builder.stream({"messages": "Hello My name is Samarjit"}, stream_mode="values"):
    print(event)
```

---

## Notebook 3 — State Schema Variants

**File**: `3-dataclassStateSchema.ipynb`

Demonstrates 3 different ways to define the State schema, with different tradeoffs for type safety.

### Comparison Table

| Schema Type          | Runtime Validation | Syntax                       | When to Use                          |
| -------------------- | ------------------ | ---------------------------- | ------------------------------------ |
| `TypedDict`          | ❌ No (hints only) | `state['key']` dict access   | Simple, lightweight                  |
| `@dataclass`         | ❌ No              | `state.key` attribute access | Cleaner syntax, no validation needed |
| `Pydantic BaseModel` | ✅ Yes             | `state.key` attribute access | When invalid input must be caught    |

### 1. TypedDict State

```python
from typing_extensions import TypedDict
from typing import Literal

class TypedDictState(TypedDict):
    name: str
    game: Literal["cricket", "badminton"]

# Access with dict syntax
def play_game(state: TypedDictState):
    return {"name": state['name'] + " want to play "}
```

### 2. Dataclass State

```python
from dataclasses import dataclass

@dataclass
class DataClassState:
    name: str
    game: Literal["badminton", "cricket"]

# Access with attribute syntax
def play_game(state: DataClassState):
    return {"name": state.name + " want to play "}

# Must instantiate with all fields when invoking
graph.invoke(DataClassState(name="Krish", game="cricket"))
```

### 3. Pydantic State (runtime validation)

```python
from pydantic import BaseModel

class State(BaseModel):
    name: str

graph.invoke({"name": "samar"})  # ✅ Valid — "samar" is a str
graph.invoke({"name": 123})      # ❌ ValidationError — 123 is an int, not str
```

### Graph Flow (same for all 3 variants)

```
START → playgame → [cricket OR badminton] → END
                    ↑ 50/50 random routing
```

---

## Notebook 5 — Chains with Tools

**File**: `5-ChainsLangGraph.ipynb`

Builds the complete LLM + Tool execution pattern in LangGraph. This is the foundation for all tool-calling agents.

### Key Concepts Covered

1. Chat messages as graph state
2. Chat models (LLMs) in graph nodes
3. Binding tools to LLMs
4. Executing tool calls in a `ToolNode`

### Step 1 — Message Types

```python
from langchain_core.messages import AIMessage, HumanMessage

messages = [
    AIMessage(content="How can I help?", name="LLMModel"),
    HumanMessage(content="I want to learn coding", name="Krish"),
    AIMessage(content="Which language?", name="LLMModel"),
    HumanMessage(content="Python", name="Krish"),
]

for msg in messages:
    msg.pretty_print()  # readable formatted output
```

### Step 2 — Define and Bind a Tool

A function becomes a tool when passed to `bind_tools`. The **docstring** is what the LLM reads to decide when to call it:

```python
def add(a: int, b: int) -> int:
    """Add a and b"""
    return a + b

llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools([add])

# LLM now responds with tool_calls when appropriate
response = llm_with_tools.invoke([HumanMessage(content="What is 2 plus 2?")])
print(response.tool_calls)
# → [{"name": "add", "args": {"a": 2, "b": 2}, "id": "call_abc..."}]
```

### Step 3 — add_messages Reducer

```python
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Demonstrates appending vs overwriting:
add_messages(existing_list, new_message)  # returns appended list
```

### Step 4 — Simple Graph (LLM only)

```python
def llm_tool(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("llm_tool", llm_tool)
builder.add_edge(START, "llm_tool")
builder.add_edge("llm_tool", END)
graph = builder.compile()
```

```
START → llm_tool → END
```

### Step 5 — Full Tool Execution Graph

Adds `ToolNode` to actually execute the tool calls the LLM requests:

```python
from langgraph.prebuilt import ToolNode, tools_condition

tools = [add]

builder = StateGraph(State)
builder.add_node("llm_tool", llm_tool)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "llm_tool")
builder.add_conditional_edges("llm_tool", tools_condition)
#   tools_condition → "tools" if AIMessage has tool_calls
#   tools_condition → END    if AIMessage has no tool_calls
builder.add_edge("tools", END)

graph_builder = builder.compile()
```

```
START → llm_tool ──(has tool_calls?)──► tools → END
                  └──────(no)──────► END
```

### Invocation

```python
# Tool call — LLM calls add(2,2), ToolNode executes it, result returned
messages = graph_builder.invoke({"messages": "What is 2 plus 2"})
# → HumanMessage, AIMessage(tool_calls=[add(2,2)]), ToolMessage(4), AIMessage("2+2=4")

# No tool call — LLM answers directly
messages = graph_builder.invoke({"messages": "What is Machine Learning"})
# → HumanMessage, AIMessage("Machine Learning is...")
```

---

## Notebook 6 — Chatbot with Multiple Tools

**File**: `6-chatbotswithmultipletools.ipynb`

Extends the pattern to a real-world chatbot with 3 external tools: **Arxiv**, **Wikipedia**, and **Tavily** web search.

### Tools Used

| Tool                  | Purpose                             | Package               |
| --------------------- | ----------------------------------- | --------------------- |
| `ArxivQueryRun`       | Search academic papers on arXiv.org | `langchain_community` |
| `WikipediaQueryRun`   | Search Wikipedia articles           | `langchain_community` |
| `TavilySearchResults` | Real-time web search                | `langchain_community` |

### Tool Setup

```python
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# ArxivAPIWrapper = raw API client (config + HTTP)
# ArxivQueryRun   = LangChain Tool wrapper (has .name, .description, .invoke())
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

tavily = TavilySearchResults()

tools = [arxiv, wiki, tavily]
```

### LLM + Tool Binding

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools)

# LLM decides which tool to call based on each tool's .description
llm_with_tools.invoke([HumanMessage(content="What is the recent AI News")]).tool_calls
```

### State Schema

```python
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### Full Graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```

```
START → tool_calling_llm ──(tool_calls?)──► tools → END
                          └────(none)────► END
```

### Invocation Examples

```python
# Search arxiv by paper ID
messages = graph.invoke({"messages": HumanMessage(content="1706.03762")})

# Real-time web search via Tavily
messages = graph.invoke({"messages": HumanMessage(content="Provide me the top 10 recent AI news for April 25 2026")})

# Wikipedia lookup
messages = graph.invoke({"messages": HumanMessage(content="What is machine learning")})

for m in messages['messages']:
    m.pretty_print()
```

---

## Full Flow Summary

### Progression Across Notebooks

```
Notebook 1 — Basic workflow
  State (TypedDict) → Nodes (functions) → Conditional edges → Compile → Invoke

Notebook 2 — LLM Chatbot
  + add_messages reducer → LLM node → stream support

Notebook 3 — Schema flexibility
  TypedDict vs Dataclass vs Pydantic (runtime validation)

Notebook 5 — LLM + Tools
  + bind_tools() → tool_calls on AIMessage → ToolNode → tools_condition

Notebook 6 — Production chatbot
  + External tools (Arxiv, Wikipedia, Tavily) → multi-tool routing
```

### Message Flow in Tool-Calling Graph

```
User: "What is 2 + 2?"
  ↓
HumanMessage → [graph starts]
  ↓
tool_calling_llm node:
  llm_with_tools.invoke(messages)
  → AIMessage(tool_calls=[{"name": "add", "args": {"a": 2, "b": 2}}])
  ↓
tools_condition → routes to "tools" node
  ↓
ToolNode executes add(a=2, b=2) → 4
  → ToolMessage(content="4", tool_call_id="call_abc...")
  ↓
[graph ends at END]

Final messages list:
  HumanMessage("What is 2 + 2?")
  AIMessage(tool_calls=[add(2,2)])
  ToolMessage("4")
```

---

## Quick Reference

### Build a Graph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("name", function)
builder.add_edge(START, "name")
builder.add_edge("name", END)
graph = builder.compile()
graph.invoke({"messages": "Hello"})
```

### Conditional Routing

```python
def router(state) -> Literal["node_a", "node_b"]:
    return "node_a" if condition else "node_b"

builder.add_conditional_edges("source_node", router)
```

### add_messages Reducer

```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
# New messages are appended, not replaced
```

### ToolNode + tools_condition

```python
from langgraph.prebuilt import ToolNode, tools_condition

builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("llm_node", tools_condition)
# tools_condition routes to "tools" if tool_calls exist, else END
builder.add_edge("tools", END)
```

### Visualize the Graph

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

---

## Files in This Module

| File                                | Key Concepts                                                          |
| ----------------------------------- | --------------------------------------------------------------------- |
| `1-langgraph_simple.ipynb`          | State, nodes, edges, conditional routing, graph compile               |
| `2-chatbot.ipynb`                   | `add_messages` reducer, LLM node, streaming                           |
| `3-dataclassStateSchema.ipynb`      | TypedDict vs dataclass vs Pydantic state schemas                      |
| `5-ChainsLangGraph.ipynb`           | `bind_tools`, `ToolNode`, `tools_condition`, full tool execution loop |
| `6-chatbotswithmultipletools.ipynb` | Arxiv, Wikipedia, Tavily tools, multi-tool chatbot                    |

## Related Documentation

- [LangChain Updated Guide](langchain_updated.md)
- [Query Enhancement](query_enhancement.md)
- [Hybrid Search Strategies](hybrid_search_strategies.md)
