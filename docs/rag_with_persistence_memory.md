# 15 — RAG with Persistence Memory

## What Is It?

This notebook builds RAG systems that remember conversation history across multiple turns using LangGraph's `MemorySaver` checkpointer. Two approaches are demonstrated: a custom 3-node graph and a prebuilt ReAct agent — both with persistent memory attached to a thread ID.

---

## Notebook: `ragmemory.ipynb`

---

## Part 1 — Custom RAG Graph with Conversation Memory

### Components

**Vector Store**
Content from Lilian Weng's agents blog is loaded, parsed with BeautifulSoup (filtering `post-content`, `post-title`, `post-header`), split into 1000-token chunks, and stored in FAISS.

**Retrieve Tool**
Defined as a `@tool` so the LLM can call it as a function:

```python
@tool()
def retrieve(query: str):
    """Retrieve the information related to the query"""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    ...
```

### Graph Flow

```
START
  │
  ▼
[query_or_respond]
  │  LLM bound with tools decides:
  │  - call the retrieve tool (tool_call in response)
  │  - or answer directly
  │
  ├── tool_call present  ──▶  [tools]  ──  executes retrieve(), returns ToolMessage
  │                              │
  │                              ▼
  │                          [generate]  ──  builds prompt from tool messages + history
  │                              │
  │                              ▼
  │                             END
  │
  └── no tool_call  ──▶  END  (direct response)
```

### Persistence via MemorySaver

```python
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# All calls with the same thread_id share history
config = {"configurable": {"thread_id": "abc123"}}
```

Every invocation with the same `thread_id` resumes from where the conversation left off — the full message history is preserved in the checkpointer. Follow-up questions like _"Can you look up some common ways of doing it?"_ correctly reference the prior answer without re-stating context.

**Viewing conversation history:**

```python
chat_history = graph.get_state(config).values["messages"]
```

---

## Part 2 — ReAct Agent with Persistent Memory

Uses `create_react_agent` (prebuilt) with the same `retrieve` tool and `MemorySaver` — a simpler setup that achieves the same persistent memory behavior.

```python
memory = MemorySaver()
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
```

**Flow (internal ReAct loop):**

```
Thought → Tool Call (retrieve) → Observation → Thought → ... → Final Answer
```

The agent can handle multi-part questions in a single invocation:

> _"What is the standard method for Task Decomposition? Once you get the answer, look up common extensions of that method."_

It automatically retrieves twice — once per sub-question — and synthesizes a combined answer.

---

## State

Both approaches use `MessagesState` — a list of `BaseMessage` objects (HumanMessage, AIMessage, ToolMessage) that grows with each turn and is persisted by the checkpointer.

---

## Key Concepts

| Concept              | How it's implemented                            |
| -------------------- | ----------------------------------------------- |
| Persistent memory    | `MemorySaver` checkpointer + `thread_id` config |
| Tool-based retrieval | `@tool` decorator; LLM decides when to call it  |
| Multi-turn context   | Full message history replayed each invocation   |
| History inspection   | `graph.get_state(config).values["messages"]`    |
