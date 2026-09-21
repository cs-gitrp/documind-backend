"""
agent.py — DocuMind v2  (Agentic Tool-Calling Loop)

Turns DocuMind from "query → retrieve → answer" into a true agent:

  Tools exposed to the LLM:
  ┌─────────────────────────┬────────────────────────────────────────────────────┐
  │ search_document         │ Retrieve passages from the current document        │
  │ search_all_documents    │ Retrieve passages across ALL uploaded documents     │
  │ calculate               │ Run a Python math expression safely                │
  │ extract_metadata        │ Return doc stats (pages, chunk count)              │
  │ summarize_section       │ Summarise a specific page range                    │
  └─────────────────────────┴────────────────────────────────────────────────────┘

The loop:
  1. User message + tool definitions → Groq
  2. If tool_calls in response → execute tool(s) → append results → repeat
  3. When no more tool calls → stream final answer

We use Groq's native tool-calling API (OpenAI-compatible format).
"""

import json
import logging
import math
import os
import re
from collections.abc import Generator
from typing import Any

from groq import Groq

from config import GROQ_API_KEY, INDEX_DIR
from services.query_rewriter import rewrite_and_expand
from services.retrieval import retrieve_hybrid

logger = logging.getLogger(__name__)
client = Groq(api_key=GROQ_API_KEY)

_AGENT_MODEL = "openai/gpt-oss-20b"   # tool-calling capable; falls back below
_FALLBACK_MODEL = "openai/gpt-oss-20b"


# ─────────────────────────────────────────────
#  Tool definitions (Groq / OpenAI format)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": (
                "Search within a specific document for passages relevant to a query. "
                "Always call this before attempting to answer factual questions about the document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The document ID to search in.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query — be specific.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of passages to retrieve (default 5).",
                        "default": 5,
                    },
                },
                "required": ["doc_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_all_documents",
            "description": (
                "Search across ALL indexed documents. Use when the user asks a cross-document question "
                "or when you don't know which document contains the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "top_k_per_doc": {
                        "type": "integer",
                        "description": "Results per document (default 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a Python mathematical expression. "
                "Use for arithmetic, percentages, conversions found in the document. "
                "Example: '(42 * 1.18)' or 'math.log(1000, 10)'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe Python math expression.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_metadata",
            "description": "Get metadata about a document: page count, chunk count, file name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The document ID.",
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_section",
            "description": "Return all chunks from a specific page range of a document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id":     {"type": "string"},
                    "page_start": {"type": "integer", "description": "First page (1-indexed)"},
                    "page_end":   {"type": "integer", "description": "Last page (inclusive)"},
                },
                "required": ["doc_id", "page_start", "page_end"],
            },
        },
    },
]


# ─────────────────────────────────────────────
#  Tool execution
# ─────────────────────────────────────────────

def _tool_search_document(doc_id: str, query: str, top_k: int = 5) -> str:
    """Run hybrid retrieval with query rewriting."""
    rewritten = rewrite_and_expand(query, use_hyde=True, use_multi=False)
    hyde_q    = rewritten["hyde_query"]
    results   = retrieve_hybrid(doc_id, hyde_q, top_k=top_k)
    if not results:
        return json.dumps({"error": "No passages found."})
    passages = [
        {
            "passage_id": i + 1,
            "page":       r["page"],
            "text":       r["chunk"][:800],
            "score":      round(r.get("rerank_score", r.get("rrf_score", 0)), 4),
        }
        for i, r in enumerate(results)
    ]
    return json.dumps({"doc_id": doc_id, "query": query, "passages": passages}, ensure_ascii=False)


def _tool_search_all_documents(query: str, top_k_per_doc: int = 3) -> str:
    """Search across all indexed documents."""
    meta_files = [f for f in os.listdir(INDEX_DIR) if f.endswith("_meta.json")]
    all_results = []
    for meta_file in meta_files:
        doc_id = meta_file.replace("_meta.json", "")
        try:
            results = retrieve_hybrid(doc_id, query, top_k=top_k_per_doc)
            for r in results:
                all_results.append({
                    "doc_id": doc_id,
                    "page":   r["page"],
                    "text":   r["chunk"][:500],
                    "score":  round(r.get("rerank_score", r.get("rrf_score", 0)), 4),
                })
        except Exception as e:
            logger.warning("Search failed for doc %s: %s", doc_id, e)

    all_results.sort(key=lambda x: -x["score"])
    return json.dumps({"query": query, "results": all_results[:top_k_per_doc * 3]})


def _tool_calculate(expression: str) -> str:
    """Safe math evaluator — only allows math module and basic ops."""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"math": math, "abs": abs, "round": round, "min": min, "max": max})
    # Block anything that looks like a call to dangerous builtins
    if re.search(r"\b(import|exec|eval|open|os|sys|__)\b", expression):
        return json.dumps({"error": "Expression blocked for safety."})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_extract_metadata(doc_id: str) -> str:
    meta_path = os.path.join(INDEX_DIR, f"{doc_id}_meta.json")
    if not os.path.exists(meta_path):
        return json.dumps({"error": f"Document {doc_id} not found."})
    with open(meta_path) as f:
        meta = json.load(f)
    chunks = meta.get("chunks", [])
    pages  = meta.get("pages", [])
    return json.dumps({
        "doc_id":      doc_id,
        "chunk_count": len(chunks),
        "page_count":  max(pages) if pages else 0,
    })


def _tool_summarize_section(doc_id: str, page_start: int, page_end: int) -> str:
    meta_path = os.path.join(INDEX_DIR, f"{doc_id}_meta.json")
    if not os.path.exists(meta_path):
        return json.dumps({"error": f"Document {doc_id} not found."})
    with open(meta_path) as f:
        meta = json.load(f)
    chunks = meta.get("chunks", [])
    pages  = meta.get("pages", [1] * len(chunks))
    section_chunks = [
        chunks[i] for i, p in enumerate(pages) if page_start <= p <= page_end
    ]
    return json.dumps({
        "page_range":   f"{page_start}-{page_end}",
        "chunk_count":  len(section_chunks),
        "text":         "\n\n".join(section_chunks[:10]),  # cap at 10 chunks
    })


TOOL_REGISTRY = {
    "search_document":     _tool_search_document,
    "search_all_documents": _tool_search_all_documents,
    "calculate":           _tool_calculate,
    "extract_metadata":    _tool_extract_metadata,
    "summarize_section":   _tool_summarize_section,
}


def _execute_tool(name: str, args: dict) -> str:
    fn = TOOL_REGISTRY.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e)
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────
#  Agent loop
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are DocuMind, an intelligent document analysis assistant.

You have access to tools to search documents, perform calculations, and extract metadata.
ALWAYS use search_document before answering any question about document content.
If the user asks about numbers or calculations, use the calculate tool.
Be concise but thorough. Cite page numbers from retrieved passages.
If a question cannot be answered from the document, say so clearly."""


def run_agent(
    query:       str,
    doc_id:      str,
    history:     list[dict[str, Any]] | None = None,
    max_turns:   int = 6,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """
    Agentic loop. Yields tokens for streaming.
    Stops when model produces a non-tool-call response or max_turns reached.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    # Inject doc_id context
    user_msg = f"[Document ID: {doc_id}]\n\nUser question: {query}"
    messages.append({"role": "user", "content": user_msg})

    for turn in range(max_turns):
        try:
            response = client.chat.completions.create(
                model=_AGENT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error("Agent LLM call failed: %s", e)
            yield f"\n[Error: {e}]"
            return

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # ── Tool call branch ──
        if finish_reason == "tool_calls" and msg.tool_calls:
            # Append the assistant's tool_call message to history
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
            # Execute each tool and append results
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                logger.info("Agent calling tool: %s(%s)", tool_name, list(tool_args.keys()))
                tool_result = _execute_tool(tool_name, tool_args)
                messages.append({
                    "role":        "tool",
                    "tool_call_id": tc.id,
                    "name":        tool_name,
                    "content":     tool_result,
                })
            # Yield a status token so the frontend can show "searching…"
            yield f"\n[🔧 Used tool: {', '.join(tc.function.name for tc in msg.tool_calls)}]\n"
            continue  # next turn

        # ── Final answer branch ──
        final_text = msg.content or ""
        # Stream token by token (simulated — Groq non-streaming used above for tool calls)
        for token in final_text.split(" "):
            yield token + " "
        return

    yield "\n[Agent reached max turns — partial answer above.]"
