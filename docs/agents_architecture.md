# Agents Architecture — ReAct, Streaming & LangGraph Studio Debugging

## Overview

The `9-agents-architecture/` module covers **agent patterns** built on LangGraph — from the foundational ReAct (Reason + Act) loop, to memory-enabled agents, streaming responses, and running agents inside **LangGraph Studio** for visual debugging.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Core Concept: ReAct Architecture](#core-concept-react-architecture)
- [Notebook 1 — ReAct Agent](#notebook-1--react-agent)
- [Notebook 2 — Streaming](#notebook-2--streaming)
- [Debugging Folder — LangGraph Studio](#debugging-folder--langgraph-studio)
- [Full Flow Summary](#full-flow-summary)
- [Quick Reference](#quick-reference)

---

## Folder Structure

```
9-agents-architecture/
├── 1-ReActAgent.ipynb          # ReAct agent with tools + memory
├── 2-streaming copy.ipynb      # Streaming modes: values, updates, astream_events
└── Debugging/
    ├── langgraph.json           # LangGraph Studio config (graph entry point + env)
    ├── openai_agent.py          # Agent definition exposed to Studio
    └── .langgraph_api/          # Auto-generated Studio checkpoints (do not edit)
```

---

## Core Concept: ReAct Architecture

**ReAct = Reason + Act** — an agent pattern where the LLM loops between three steps:

```
┌─────────────────────────────────────────────┐
│                ReAct Loop                    │
│                                              │
│  User Query                                  │
│      ↓                                       │
│  [REASON] LLM decides what to do             │
│      ↓                                       │
│  [ACT]    Call a tool (search, calculate...) │
│      ↓                                       │
│  [OBSERVE] Tool result fed back to LLM       │
│      ↓                                       │
│  Loop continues until LLM has final answer   │
│      ↓                                       │
│  Final Answer                                │
└─────────────────────────────────────────────┘
```

This is different from a simple chain — the agent **decides dynamically** which tool to call and how many times, based on the tool results it observes.

---

## Notebook 1 — ReAct Agent

**File**: `1-ReActAgent.ipynb`

Builds a full ReAct agent step by step — external search tools + custom math functions + memory.

### Tools Available

| Tool       | Type            | Purpose                             |
| ---------- | --------------- | ----------------------------------- |
| `arxiv`    | External API    | Search academic papers on arXiv.org |
| `wiki`     | External API    | Search Wikipedia articles           |
| `tavily`   | External API    | Real-time web search                |
| `add`      | Custom function | Add two integers                    |
| `multiply` | Custom function | Multiply two integers               |
| `divide`   | Custom function | Divide two numbers                  |

### Tool Setup

```python
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

tavily = TavilySearchResults()

# Custom math tools — docstrings are read by the LLM to know when to use them
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [arxiv, wiki, tavily, add, divide, multiply]
```

### LangChain Tracing (LangSmith)

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ReAct-agent"  # traces visible in LangSmith dashboard
```

### Agent Graph

```python
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition

llm = ChatGroq(model="qwen/qwen3-32b")
llm_with_tools = llm.bind_tools(tools)

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")  # ← loops BACK to LLM (not END)

graph = builder.compile()
```

**Key difference from earlier notebooks**: after `tools` runs, the edge goes back to `tool_calling_llm` (not `END`). This creates the **ReAct loop** — the LLM can call multiple tools in sequence:

```
START → tool_calling_llm ──(tool_calls?)──► tools ─┐
              ↑                                      │
              └─────────────────────────────────────┘
              └──────────(no tool calls)──────────► END
```

### Multi-step Example

```python
# Single query — agent calls Tavily + add + multiply tools in one run
messages = graph.invoke({"messages": HumanMessage(
    content="Get the top 10 recent AI news, add 5 plus 5 and then multiply by 10"
)})
```

The agent will:

1. Call `tavily` to get AI news
2. Call `add(5, 5)` → 10
3. Call `multiply(10, 10)` → 100
4. Return a combined final answer

---

### Memory with MemorySaver

Without memory, the agent forgets context between invocations:

```python
# No memory — agent doesn't know what "that" refers to
graph.invoke({"messages": HumanMessage("What is 5 plus 8")})
graph.invoke({"messages": HumanMessage("Divide that by 5")})  # ← "that" is lost
```

**Solution: `MemorySaver` + `thread_id`**

```python
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

memory = MemorySaver()
graph_memory = builder.compile(checkpointer=memory)

# Same thread_id = shared conversation history across invocations
config = {"configurable": {"thread_id": "1"}}

graph_memory.invoke({"messages": [HumanMessage("Add 12 and 13.")]}, config=config)
# → 25

graph_memory.invoke({"messages": [HumanMessage("add that number to 25")]}, config=config)
# → 50  (remembers 25 from previous turn)

graph_memory.invoke({"messages": [HumanMessage("then multiply that number by 2")]}, config=config)
# → 100  (remembers 50 from previous turn)
```

**How it works**: `MemorySaver` stores the full graph state (all messages) keyed by `thread_id`. On each call, it loads the prior state and appends the new messages before invoking.

---

## Notebook 2 — Streaming

**File**: `2-streaming copy.ipynb`

Demonstrates all streaming modes for getting real-time output from a LangGraph agent.

### Setup

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
llm_groq = ChatGroq(model="qwen/qwen3-32b")

def superbot(state: State):
    return {"messages": [llm_groq.invoke(state['messages'])]}

graph = StateGraph(State)
graph.add_node("SuperBot", superbot)
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)

graph_builder = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
```

### Streaming Modes

#### `.stream()` — Synchronous

Two modes control what each chunk contains:

| `stream_mode` | Each chunk contains                               | Use when                            |
| ------------- | ------------------------------------------------- | ----------------------------------- |
| `"updates"`   | Only the **delta** — what changed after this node | You want incremental state changes  |
| `"values"`    | The **full state** after this node                | You want a snapshot after each step |

```python
# stream_mode="updates" — get only what changed
for chunk in graph_builder.stream(
    {'messages': "Hi, My name is Krish"},
    config,
    stream_mode="updates"
):
    print(chunk)
# → {'SuperBot': {'messages': [AIMessage(content="Hello Krish!...")]}}

# stream_mode="values" — get full state snapshot
for chunk in graph_builder.stream(
    {'messages': "I also like football"},
    config,
    stream_mode="values"
):
    print(chunk)
# → {'messages': [HumanMessage(...), AIMessage(...), HumanMessage(...), AIMessage(...)]}
```

#### `.astream_events()` — Async token streaming

Streams individual **events** as they happen inside nodes — including token-by-token LLM output:

```python
async for event in graph_builder.astream_events(
    {"messages": ["Hi My name is Krish"]},
    config,
    version="v2"
):
    print(event)
# Each event dict has:
#   event:    type of event (e.g., "on_chat_model_stream")
#   name:     event name
#   data:     actual payload (e.g., token chunk)
#   metadata: {"langgraph_node": "SuperBot"}
```

**Use `astream_events` when** you want to stream tokens to a frontend UI as the LLM generates them, rather than waiting for the full response.

---

## Debugging Folder — LangGraph Studio

**Location**: `Debugging/`

### What is LangGraph Studio?

A visual IDE for LangGraph agents — lets you:

- See the graph topology as a live diagram
- Send messages and watch state flow through nodes in real time
- Inspect state at each step (messages, tool calls, results)
- Replay and debug specific graph runs

### Files

#### `langgraph.json` — Studio Configuration

```json
{
  "dependencies": ["."],
  "graphs": {
    "openai_agent": "./openai_agent.py:agent"
  },
  "env": "../../.env"
}
```

| Field                 | Meaning                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `dependencies`        | Install packages from current directory                                                  |
| `graphs.openai_agent` | Name → `file:variable` — the compiled graph to load                                      |
| `env`                 | Path to `.env` file (relative to this JSON file) — `../../.env` points to workspace root |

#### `openai_agent.py` — Agent Definition

Defines two graphs, exposes one to Studio:

**Graph 1 — `make_default_graph()`** (simple, no tools):

```
START → agent → END
```

**Graph 2 — `make_alternative_graph()`** (tool-calling loop):

```
START → agent ──(tool_calls?)──► tools → agent (loop)
              └────────(no)────► END
```

```python
# The graph Studio loads — referenced in langgraph.json as "openai_agent.py:agent"
agent = make_alternative_graph()
```

**Key detail**: `add_conditional_edges` requires an explicit path map for Studio to draw edges correctly:

```python
graph_workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}  # ← without this, Studio shows disconnected nodes
)
```

#### `.langgraph_api/` — Auto-generated (do not edit)

LangGraph Studio stores checkpoints and state here automatically:

- `.langgraph_checkpoint.*.pckl` — saved graph state at each step
- `.langgraph_ops.pckl` — operation history
- `store.pckl` / `store.vectors.pckl` — in-memory store data

---

## Full Flow Summary

### ReAct Agent Full Flow

```
User: "Get recent AI news, add 5+5, then multiply by 10"
  ↓
HumanMessage → graph starts
  ↓
tool_calling_llm node:
  LLM sees tools available: [arxiv, wiki, tavily, add, divide, multiply]
  LLM decides: call tavily("recent AI news")
  → AIMessage(tool_calls=[tavily(...)])
  ↓
tools_condition → "tools"
  ↓
ToolNode executes tavily → ToolMessage(news results)
  ↓
tool_calling_llm node (loop back):
  LLM sees news results, decides: call add(5, 5)
  → AIMessage(tool_calls=[add(5,5)])
  ↓
ToolNode executes add → ToolMessage("10")
  ↓
tool_calling_llm node (loop back):
  LLM sees 10, decides: call multiply(10, 10)
  → AIMessage(tool_calls=[multiply(10,10)])
  ↓
ToolNode executes multiply → ToolMessage("100")
  ↓
tool_calling_llm node (loop back):
  All info gathered — LLM generates final answer
  → AIMessage("Here are the top AI news... 5+5=10, 10×10=100")
  ↓
tools_condition → END
```

### Memory Flow

```
Thread "1", Turn 1:  [HumanMessage("Add 12 and 13")] → saved to MemorySaver
Thread "1", Turn 2:  [prev messages... + HumanMessage("add that to 25")]
                      ↑ loaded from MemorySaver by thread_id
Thread "1", Turn 3:  [prev messages... + HumanMessage("multiply by 2")]
                      ↑ loaded from MemorySaver — agent remembers 50
```

---

## Quick Reference

### ReAct Agent (with memory)

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

tools = [arxiv, wiki, tavily, add, multiply, divide]
llm_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def tool_calling_llm(state): return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")  # loop back!

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "session-1"}}
graph.invoke({"messages": [HumanMessage("...")]}, config=config)
```

### Streaming Modes

```python
# Incremental updates only
for chunk in graph.stream(input, config, stream_mode="updates"):
    print(chunk)

# Full state after each node
for chunk in graph.stream(input, config, stream_mode="values"):
    print(chunk)

# Token-by-token (async)
async for event in graph.astream_events(input, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

### LangGraph Studio Config

```json
{
  "dependencies": ["."],
  "graphs": { "my_agent": "./agent.py:agent" },
  "env": "../../.env"
}
```

---

## Files in This Module

| File                        | Key Concepts                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `1-ReActAgent.ipynb`        | ReAct loop, external tools (arxiv/wiki/tavily), custom tools, multi-step reasoning, `MemorySaver` |
| `2-streaming copy.ipynb`    | `.stream()` with `updates`/`values` modes, `.astream_events()` for token streaming                |
| `Debugging/langgraph.json`  | LangGraph Studio config — graph entry point, env path                                             |
| `Debugging/openai_agent.py` | Two graph variants, `should_continue` router, path map for Studio edge rendering                  |

## Related Documentation

- [LangGraph Guide](langgraph.md)
- [LangChain Updated Guide](langchain_updated.md)
