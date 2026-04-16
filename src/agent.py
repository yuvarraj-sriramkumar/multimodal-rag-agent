"""
agent.py — Multimodal RAG agent built with LangGraph StateGraph.

Pipeline:
    START
      └─► retrieve_context  (ChromaDB similarity search)
            └─► vision_reasoning  (GPT-4o multimodal chain-of-thought)
                  └─► tool_reasoning  (GPT-4o + bound tools)
                        ├─► [tool_calls present] ToolNode ──► tool_reasoning
                        └─► [no tool_calls]       END

CLI:
    python src/agent.py --image test.jpg --query "what shapes are in this image?"
"""

import argparse
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# Local modules (src/ is on sys.path when running via CLI below)
from rag import format_context, retrieve
from tools import TOOLS
from vision import analyze_image

load_dotenv()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # full history, auto-appended
    image_path: str                                        # path to the input image
    query: str                                             # original user question
    rag_context: str                                       # formatted retrieval results
    final_answer: str                                      # vision node output


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
_llm_with_tools = _llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Node 1 — retrieve_context
# ---------------------------------------------------------------------------


def retrieve_context_node(state: AgentState) -> dict:
    """Query ChromaDB and format the top-3 relevant chunks."""
    hits = retrieve(state["query"])
    context = format_context(hits)
    return {"rag_context": context}


# ---------------------------------------------------------------------------
# Node 2 — vision_reasoning
# ---------------------------------------------------------------------------


def vision_reasoning_node(state: AgentState) -> dict:
    """Run GPT-4o vision with chain-of-thought, injecting the RAG context."""
    image_path = state["image_path"]
    query = state["query"]
    rag_context = state.get("rag_context", "")

    context_block = (
        f"Relevant background knowledge from the knowledge base:\n{rag_context}"
        if rag_context
        else None
    )

    vision_output = analyze_image(
        image_path=image_path,
        question=query,
        context=context_block,
    )

    # Store the full structured vision analysis and seed the message history
    return {
        "final_answer": vision_output,
        "messages": [
            HumanMessage(content=f"User query: {query}"),
            AIMessage(content=f"Visual analysis:\n{vision_output}"),
        ],
    }


# ---------------------------------------------------------------------------
# Node 3 — tool_reasoning
# ---------------------------------------------------------------------------

_TOOL_SYSTEM = SystemMessage(content=(
    "You are a multimodal reasoning assistant. "
    "You have already performed a visual analysis of the image supplied by the user. "
    "Use the visual analysis and the conversation history to answer the user's query. "
    "If the query requires computation or web lookup, call the appropriate tool. "
    "When you have a complete answer, respond directly without calling any more tools."
))


def tool_reasoning_node(state: AgentState) -> dict:
    """GPT-4o with tools bound; re-runs until it stops issuing tool_calls."""
    messages = [_TOOL_SYSTEM] + state["messages"]
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Conditional edge — should we call tools or stop?
# ---------------------------------------------------------------------------


def route_after_tools(state: AgentState) -> str:
    """Route to ToolNode if the last AI message has tool calls, else END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("vision_reasoning", vision_reasoning_node)
    graph.add_node("tool_reasoning", tool_reasoning_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "vision_reasoning")
    graph.add_edge("vision_reasoning", "tool_reasoning")
    graph.add_conditional_edges("tool_reasoning", route_after_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "tool_reasoning")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public run helper
# ---------------------------------------------------------------------------


def run_agent(image_path: str, query: str, verbose: bool = True) -> str:
    """
    Run the full multimodal RAG agent and return the final answer string.

    Args:
        image_path: Local path to the image file.
        query:      The user's question.
        verbose:    Print intermediate node outputs when True.

    Returns:
        The assistant's final text answer.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    app = build_graph()
    initial_state: AgentState = {
        "messages": [],
        "image_path": image_path,
        "query": query,
        "rag_context": "",
        "final_answer": "",
    }

    if verbose:
        print(f"\n{'='*64}")
        print(f"  Query : {query}")
        print(f"  Image : {image_path}")
        print(f"{'='*64}\n")

    final_state = app.invoke(initial_state)

    # The definitive answer is the last AI message that has no tool_calls
    last_ai = next(
        (m for m in reversed(final_state["messages"])
         if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    answer = last_ai.content if last_ai else final_state.get("final_answer", "")

    if verbose:
        print("\n── RAG Context ─────────────────────────────────────────────")
        print(final_state["rag_context"])
        print("\n── Vision Analysis ─────────────────────────────────────────")
        print(final_state["final_answer"])

        # Print any tool call/result pairs
        tool_pairs = [
            m for m in final_state["messages"]
            if isinstance(m, (AIMessage, ToolMessage))
            and (getattr(m, "tool_calls", None) or isinstance(m, ToolMessage))
        ]
        if tool_pairs:
            print("\n── Tool Calls ──────────────────────────────────────────────")
            for m in tool_pairs:
                if isinstance(m, AIMessage) and m.tool_calls:
                    for tc in m.tool_calls:
                        print(f"  → {tc['name']}({tc['args']})")
                elif isinstance(m, ToolMessage):
                    print(f"  ← {m.content[:300]}")

        print("\n── Final Answer ─────────────────────────────────────────────")
        print(answer)
        print()

    return answer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal RAG Agent — vision + retrieval + tools"
    )
    parser.add_argument("--image", required=True, metavar="PATH", help="Path to input image")
    parser.add_argument("--query", required=True, metavar="TEXT", help="Question about the image")
    parser.add_argument("--quiet", action="store_true", help="Suppress intermediate output")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_agent(args.image, args.query, verbose=not args.quiet)


if __name__ == "__main__":
    # Ensure src/ is importable regardless of working directory
    _src = str(Path(__file__).parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    main()
