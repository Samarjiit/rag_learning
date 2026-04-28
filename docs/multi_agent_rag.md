# 12 — Multi-Agent RAG

## What Is Multi-Agent RAG?

A Multi-Agent RAG system splits the RAG pipeline into **multiple specialized agents**, each responsible for a distinct role (e.g., researcher, writer, math expert). They collaborate — either peer-to-peer or through a central supervisor — to answer complex queries that no single agent could handle alone.

---

## Notebook: `1-multiagent.ipynb`

Three multi-agent architectures are built progressively.

---

### Part 1 — Multi-Agent Network (Peer-to-Peer Collaboration)

**What it does:** Two agents — a **Researcher** and a **Blog Writer** — collaborate in a turn-based loop. The researcher gathers information; the writer drafts a blog. Either agent can signal completion with `FINAL ANSWER`.

**Agents:**
| Agent | Tools | Role |
|---|---|---|
| `researcher` | `InternalResearchNotes` (FAISS), `TavilySearch` | Finds relevant information |
| `blog_generator` | None | Writes a detailed blog from research |

**Flow:**

```
START
  │
  ▼
[researcher]  ──  uses tools to gather info; wraps result as HumanMessage
  │
  ├── "FINAL ANSWER" in response  ──▶  END
  └── otherwise                   ──▶  [blog_generator]
         │
         ├── "FINAL ANSWER" in response  ──▶  END
         └── otherwise                   ──▶  [researcher]  (loop)
```

**Key pattern — `Command` routing:**
```python
def research_node(state) -> Command[Literal["blog_generator", END]]:
    # returns Command(update={messages}, goto=next_node)
```

Each agent wraps its AI response as a `HumanMessage` before passing to the next agent (required by some LLM providers that disallow AI messages at the end of input).

---

### Part 2 — Multi-Agent Supervisor

**What it does:** A central **Supervisor** agent orchestrates two specialized agents. The supervisor decides which agent to call for each step and never does any work itself.

**Agents:**
| Agent | Tools | Role |
|---|---|---|
| `research_agent` | `TavilySearch`, `InternalResearchNotes` (FAISS) | Research tasks only |
| `math_agent` | `add`, `multiply`, `divide` | Math calculations only |
| `supervisor` | handoff tools | Routes tasks; decides who does what |

**Flow:**

```
START
  │
  ▼
[supervisor]  ──  reads the task, decides which agent to call
  │
  ├──▶  [research_agent]  ──  fetches info, returns to supervisor
  │         │
  │         └──▶  [supervisor]  (re-evaluates)
  │
  ├──▶  [math_agent]  ──  computes result, returns to supervisor
  │         │
  │         └──▶  [supervisor]  (re-evaluates)
  │
  └── all tasks done  ──▶  END
```

Built using `langgraph_supervisor.create_supervisor(...)` with `output_mode="full_history"` so the full message chain is preserved.

---

### Part 3 — Hierarchical Agent Teams

**What it does:** Extends the supervisor pattern with a **hierarchy** — a top-level supervisor delegates to mid-level supervisors, who in turn manage their own worker agents. This handles large-scale, complex tasks.

**Structure:**

```
[Top-Level Supervisor]
  │
  ├──▶  [Research Team Supervisor]
  │         ├── web_search_agent
  │         └── rag_retrieval_agent
  │
  └──▶  [Writing Team Supervisor]
            ├── blog_writer_agent
            └── summarizer_agent
```

Each team is an independently compiled subgraph, composed together by the top-level supervisor.

---

## Shared Patterns Across All Three

**Generic retriever tool factory** (reused throughout):
```python
def make_retriever_tool_from_text(file, name, desc):
    docs = TextLoader(file).load()
    chunks = RecursiveCharacterTextSplitter(...).split_documents(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings())
    retriever = vs.as_retriever()
    def tool_func(query): ...
    return Tool(name=name, description=desc, func=tool_func)
```

**Shared state:** All agents communicate through `MessagesState` — a list of `BaseMessage` objects that grows as agents pass results to each other.
