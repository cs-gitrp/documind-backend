"""
query_rewriter.py — DocuMind v2

Two complementary strategies called before retrieval:

  1. HyDE (Hypothetical Document Embeddings)
     Generate a short hypothetical answer, embed IT instead of the raw query.
     This moves the query vector into the "answer space" of the embedding model,
     dramatically improving recall for factoid and definition questions.

  2. Multi-Query Expansion
     Generate N alternative phrasings of the same question.
     Retrieve for each → union → deduplicate by chunk_id.
     Catches chunks that one phrasing misses.

Both call Groq (same API key already in config.py) — zero extra cost.
"""

import logging

from groq import Groq

from config import GROQ_API_KEY

logger = logging.getLogger(__name__)
client = Groq(api_key=GROQ_API_KEY)

_MODEL = "openai/gpt-oss-20b"   # fast & cheap; swap to llama-3.3-70b for quality


# ─────────────────────────────────────────────
#  HyDE
# ─────────────────────────────────────────────

HYDE_SYSTEM = (
    "You are a technical document expert. "
    "Given a question, write ONE SHORT PARAGRAPH (3-4 sentences) that directly answers it "
    "as if you had found the answer in a technical document. "
    "Do not mention uncertainty. Output ONLY the paragraph."
)

def hyde_rewrite(query: str) -> str:
    """
    Returns a hypothetical document passage for the query.
    Falls back to original query on error.
    """
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user",   "content": query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        hyde_text = resp.choices[0].message.content.strip()
        logger.debug("HyDE expansion: %s → %s", query[:60], hyde_text[:80])
        return hyde_text
    except Exception as exc:
        logger.warning("HyDE failed, falling back to original query: %s", exc)
        return query


# ─────────────────────────────────────────────
#  Multi-Query Expansion
# ─────────────────────────────────────────────

MULTI_QUERY_SYSTEM = (
    "You are a search query expert. "
    "Given a user question, generate {n} distinct alternative phrasings that could retrieve "
    "the same information from a document. "
    "Output ONLY a numbered list, one query per line, no explanations."
)

def multi_query_expand(query: str, n: int = 3) -> list[str]:
    """
    Returns a list of [original_query] + [n alternative phrasings].
    Always includes the original so callers can use it as a union.
    """
    try:
        system_prompt = MULTI_QUERY_SYSTEM.format(n=n)
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            temperature=0.5,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        # Parse numbered list — "1. ...", "2. ...", etc.
        queries = []
        for line in raw.splitlines():
            line = line.strip()
            if line and line[0].isdigit():
                # Strip leading "1. " or "1) "
                cleaned = line.split(".", 1)[-1].split(")", 1)[-1].strip()
                if cleaned:
                    queries.append(cleaned)
        if not queries:
            queries = [query]
        logger.debug("Multi-query expansion: %d alternatives", len(queries))
        return [query] + queries[:n]
    except Exception as exc:
        logger.warning("Multi-query expansion failed: %s", exc)
        return [query]


# ─────────────────────────────────────────────
#  Combined helper used by the agent/chat router
# ─────────────────────────────────────────────

def rewrite_and_expand(query: str, use_hyde: bool = True, use_multi: bool = True) -> dict:
    """
    Returns:
        {
          "hyde_query":   str   — HyDE passage (or original),
          "sub_queries":  List[str] — multi-query alternatives,
        }
    """
    hyde_q   = hyde_rewrite(query)     if use_hyde  else query
    sub_qs   = multi_query_expand(query) if use_multi else [query]
    return {"hyde_query": hyde_q, "sub_queries": sub_qs}
