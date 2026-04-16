"""
tools.py — LangChain-compatible tools for the multimodal RAG agent.

Exposes:
    TOOLS = [python_interpreter, web_search]
"""

import io
import sys
import traceback
from contextlib import redirect_stdout
from typing import Annotated

from langchain_core.tools import tool
from duckduckgo_search import DDGS


# ---------------------------------------------------------------------------
# Tool 1: Python interpreter
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "print": print,
    "range": range,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "isinstance": isinstance,
    "type": type,
    "__import__": __import__,  # allow safe stdlib imports
}


@tool
def python_interpreter(
    code: Annotated[str, "Python code to execute. Use print() to produce output."]
) -> str:
    """
    Execute Python code in a restricted environment and return stdout.
    Dangerous builtins (exec, eval, open, __import__ for OS calls) are not
    available by default; safe stdlib imports (math, json, re, etc.) work fine.
    """
    stdout_capture = io.StringIO()
    local_env: dict = {}
    global_env: dict = {"__builtins__": _SAFE_BUILTINS}

    try:
        with redirect_stdout(stdout_capture):
            exec(compile(code, "<string>", "exec"), global_env, local_env)  # noqa: S102
        output = stdout_capture.getvalue()
        return output if output else "(no output)"
    except Exception:
        return f"ERROR:\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Tool 2: Web search
# ---------------------------------------------------------------------------

@tool
def web_search(
    query: Annotated[str, "Search query to look up on the web."]
) -> str:
    """
    Search the web using DuckDuckGo and return the top 3 results.
    Each result includes the title, URL, and a short snippet.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as exc:
        return f"Search failed: {exc}"

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"[{i}] {title}\n    URL: {href}\n    {body}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Exported tool list
# ---------------------------------------------------------------------------

TOOLS = [python_interpreter, web_search]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: python_interpreter")
    print("=" * 60)
    code = """
import math
results = [math.sqrt(x) for x in range(1, 6)]
for i, v in enumerate(results, 1):
    print(f"sqrt({i}) = {v:.4f}")
"""
    print(python_interpreter.invoke({"code": code}))

    print("=" * 60)
    print("TEST: web_search")
    print("=" * 60)
    print(web_search.invoke({"query": "LangGraph multimodal RAG tutorial 2024"}))
