# --- Imports ---
from typing import Annotated                          # used to attach reducer metadata to type hints
from typing_extensions import TypedDict               # defines state schema as a typed dictionary
from langchain_openai import ChatOpenAI               # OpenAI chat model (GPT-3.5/4)
from langgraph.graph import END, START                # special built-in entry (START) and exit (END) nodes
from langgraph.graph.state import StateGraph          # core graph builder class
from langgraph.graph.message import add_messages      # reducer: appends new messages instead of overwriting
from langgraph.prebuilt import ToolNode               # pre-built node that executes tool calls from AIMessage
from langchain_core.tools import tool                 # @tool decorator to register a function as an LLM tool
from langchain_core.messages import BaseMessage       # base class for HumanMessage, AIMessage, ToolMessage
import os
from dotenv import load_dotenv

# Load API keys from .env file into environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")       # required for ChatOpenAI
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") # required for LangSmith tracing in Studio


# --- State Schema ---
# Shared state object passed through every node in the graph.
# `add_messages` reducer ensures new messages are appended to the list, not replaced.
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Shared LLM instance — temperature=0 for deterministic (non-random) responses
model = ChatOpenAI(temperature=0)


# --- Graph 1: Simple agent with no tools ---
# Flow: START → agent → END  (single pass, no loops)
def make_default_graph():
    graph_workflow = StateGraph(State)

    # Node: sends all current messages to the LLM and returns its reply
    def call_model(state):
        return {"messages": [model.invoke(state['messages'])]}

    graph_workflow.add_node("agent", call_model)  # register the agent node
    graph_workflow.add_edge(START, "agent")        # entry point
    graph_workflow.add_edge("agent", END)          # exit point

    agent = graph_workflow.compile()               # validate graph structure and lock it
    return agent


# --- Graph 2: Tool-calling agent with a loop ---
# Flow: START → agent → (tool_calls?) → tools → agent → ... → END
def make_alternative_graph():
    """Make a tool-calling agent"""

    # Define a tool — the docstring is what the LLM reads to decide when to call it
    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    # ToolNode automatically executes whatever tool calls are in the last AIMessage
    tool_node = ToolNode([add])

    # Bind the tool to the model so the LLM knows it can call `add`
    model_with_tools = model.bind_tools([add])

    # Node: invoke the tool-aware LLM — may return a normal reply OR a tool call request
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    # Routing function: checks whether the LLM's last message contains tool calls
    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"  # LLM wants to call a tool → route to ToolNode
        else:
            return END      # LLM gave a final answer → exit the graph

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)   # LLM node
    graph_workflow.add_node("tools", tool_node)    # tool executor node

    graph_workflow.add_edge(START, "agent")         # always start at agent
    graph_workflow.add_edge("tools", "agent")       # after tool runs → loop back to agent

    # Conditional routing from agent: explicit path map needed for LangGraph Studio to draw edges
    # "tools" return value → tools node | END return value → graph exit
    graph_workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

    agent = graph_workflow.compile()
    return agent


# Expose the tool-calling agent as `agent` — referenced in langgraph.json as "openai_agent.py:agent"
agent = make_alternative_graph()

