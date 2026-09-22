"""
eval_pipeline.py — DocuMind v2  (RAG Evaluation Framework)

Three-part evaluation:

  1. Golden Set Generation
     Auto-generate Q&A pairs from the document chunks using the LLM.
     Output: JSON file with {question, expected_answer, source_chunk_ids}.

  2. Retrieval Evaluation
     Metric: Hit@k — for each golden question, did the retriever surface
     at least one of the expected source chunks in top-k results?
     Also reports: MRR (Mean Reciprocal Rank).

  3. Answer Faithfulness Evaluation (LLM-as-judge)
     For each golden Q, generate an answer with the pipeline, then ask
     an LLM judge (Groq) whether the answer is faithful to the reference.
     Scores: faithful / partially_faithful / unfaithful per question.

Usage (CLI):
    python evals/eval_pipeline.py \
        --doc_id <your_doc_id> \
        --golden_path evals/golden_set.json \
        --generate_golden      # only on first run
        --output_path evals/results.json

The eval router (routers/eval.py) wraps this for the API.
"""

import argparse
import json
import logging
import os
import re as _re
import sys
import tempfile
import time
from typing import Any

import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq

from config import GROQ_API_KEY, INDEX_DIR
from services.llm import build_prompt, stream_response
from services.retrieval import retrieve_hybrid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

client = Groq(api_key=GROQ_API_KEY)
_EVAL_MODEL  = "openai/gpt-oss-20b"   # used for generation
_JUDGE_MODEL = "openai/gpt-oss-120b" # larger model as judge


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and nested containers to JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json_atomic(path: str, value: Any) -> None:
    """Serialize completely before replacing the destination file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".eval-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


# ─────────────────────────────────────────────
#  1. Golden Set Generation
# ─────────────────────────────────────────────

GOLDEN_GEN_PROMPT = """You are creating a RAG evaluation dataset.

Given the following document chunk, generate {n} diverse question-answer pairs.
STRICT RULES:
- Questions must be answerable ONLY from the text in this chunk
- Do NOT generate questions about author names, affiliations, institutions, acknowledgements, or references
- Do NOT generate questions that require information from other chunks
- Focus on: methods, findings, definitions, numerical results, experimental details, conclusions
- Answers must be specific and extractable directly from the chunk text

Output STRICTLY as a JSON array (no markdown, no extra text):
[
  {{"question": "...", "answer": "...", "question_type": "factual|definition|numerical|reasoning"}},
  ...
]

Chunk (chunk_id={chunk_id}, page={page}):
{chunk_text}"""


def generate_golden_set(
    doc_id:        str,
    n_questions:   int = 20,
    q_per_chunk:   int = 2,
    output_path:   str = "evals/golden_set.json",
) -> list[dict]:
    """
    Sample chunks from the document, generate Q&A pairs for each,
    return and save to output_path.
    """
    meta_path = os.path.join(INDEX_DIR, f"{doc_id}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    chunks = meta["chunks"]
    pages  = meta.get("pages", [1] * len(chunks))

    # Sample evenly across the document
    import random
    random.seed(42)
    n_chunks_needed = min(math.ceil(n_questions / q_per_chunk), len(chunks))
    step = max(1, len(chunks) // n_chunks_needed)
    sampled_indices = list(range(0, len(chunks), step))[:n_chunks_needed]

    golden_set: list[dict] = []

    for idx in sampled_indices:
        chunk_text = chunks[idx]
        page       = pages[idx]
        prompt     = GOLDEN_GEN_PROMPT.format(
            n=q_per_chunk, chunk_id=idx, page=page, chunk_text=chunk_text[:1500]
        )
        try:
            resp = client.chat.completions.create(
                model=_EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            pairs = json.loads(raw)
            for pair in pairs:
                pair["source_chunk_id"] = idx
                pair["source_page"]     = page
                pair["doc_id"]          = doc_id
                golden_set.append(pair)
            logger.info("Generated %d Q&A pairs from chunk %d", len(pairs), idx)
        except Exception as e:
            logger.warning("Failed to generate Q&A for chunk %d: %s", idx, e)
            continue
        # Intentional delay to reduce free-tier rate-limit pressure.
        time.sleep(3.0)

    # Filter out metadata/attribution questions that are unfair retrieval targets
    skip_keywords = [
        "author", "affiliation", "institution", "acknowledge",
        "correspond", "department", "university", "email"
    ]

    golden_set = [
        q for q in golden_set
        if not any(kw in q["question"].lower() for kw in skip_keywords)
    ]

    golden_set = golden_set[:n_questions]
    _write_json_atomic(output_path, golden_set)
    logger.info("Golden set saved: %d questions → %s", len(golden_set), output_path)
    return golden_set


# ─────────────────────────────────────────────
#  2. Retrieval Evaluation
# ─────────────────────────────────────────────

def _first_relevant_rank(retrieved_ids: list[int], relevant_ids: list[int]) -> int | None:
    """Return the one-based rank of the first relevant ID, or None on a miss."""
    relevant = set(relevant_ids)
    for index, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant:
            return index + 1
    return None

def evaluate_retrieval(
    golden_set: list[dict],
    top_k:      int = 5,
) -> dict[str, Any]:
    """
    For each golden question, run retrieval and check if the source chunk
    appears in the top-k results.

    Returns:
        {
          "hit_rate_at_k": float,   # fraction of questions where source chunk in top-k
          "mrr":           float,   # Mean Reciprocal Rank
          "top_k":         int,
          "n_questions":   int,
          "per_question":  List[Dict]
        }
    """
    hits  = 0
    rr_sum = 0.0
    per_q  = []

    for item in golden_set:
        doc_id  = item["doc_id"]
        query   = item["question"]
        src_cid = item["source_chunk_id"]
        relevant_ids = item.get("relevant_chunk_ids", [src_cid])

        retrieval = retrieve_hybrid(doc_id, query, top_k=top_k, diagnostics=True)
        if isinstance(retrieval, dict):
            results = retrieval["results"]
        else:
            # Keep evaluation compatible with callers that provide the legacy list API.
            results = retrieval
            retrieval = {
                "results": results,
                "dense_results": results,
                "bm25_results": results,
                "rrf_candidates": results,
                "reranked_results": results,
            }
        ret_cids = [r["chunk_id"] for r in results]

        rank = _first_relevant_rank(ret_cids, relevant_ids)
        hit = rank is not None
        rr   = 1.0 / rank if rank else 0.0

        hits    += int(hit)
        rr_sum  += rr

        per_q.append({
            "question":        query,
            "source_chunk_id": src_cid,
            "relevant_chunk_ids": relevant_ids,
            "hit":             hit,
            "rank":            rank,
            "retrieved_ids":   ret_cids,
            "dense_retrieved_ids": [r["chunk_id"] for r in retrieval["dense_results"]],
            "bm25_retrieved_ids": [r["chunk_id"] for r in retrieval["bm25_results"]],
            "rrf_candidate_ids": [r["chunk_id"] for r in retrieval["rrf_candidates"]],
            "reranked_ids": [r["chunk_id"] for r in retrieval["reranked_results"]],
            "final_top_k_ids": ret_cids,
            "gold_chunk_stage_presence": {
                "dense": src_cid in [r["chunk_id"] for r in retrieval["dense_results"]],
                "bm25": src_cid in [r["chunk_id"] for r in retrieval["bm25_results"]],
                "rrf": src_cid in [r["chunk_id"] for r in retrieval["rrf_candidates"]],
                "reranked": src_cid in [r["chunk_id"] for r in retrieval["reranked_results"]],
                "final_top_k": hit,
            },
            "manual_review_candidate_ids": [cid for cid in ret_cids if cid != src_cid],
            "gold_chunk_only_evaluation": True,
        })
        logger.debug("Q: %s | hit=%s rank=%s", query[:60], hit, rank)

    n  = len(golden_set)
    return {
        "hit_rate_at_k": round(hits / n, 4) if n else 0,
        "mrr":           round(rr_sum / n, 4) if n else 0,
        "top_k":         top_k,
        "n_questions":   n,
        "hits":          hits,
        "per_question":  per_q,
    }


# ─────────────────────────────────────────────
#  3. Answer Faithfulness (LLM-as-judge)
# ─────────────────────────────────────────────

FAITHFULNESS_JUDGE_PROMPT = """You are an impartial judge evaluating RAG system outputs.

Question: {question}
Reference Answer: {reference}
System Answer: {system_answer}

Evaluate whether the System Answer is faithful to the Reference Answer and does not
contain hallucinations (facts not in the reference).

Respond with ONLY a JSON object:
{{"verdict": "faithful"|"partially_faithful"|"unfaithful", "reason": "one sentence"}}"""


def _generate_answer(doc_id: str, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """Run full retrieval + LLM pipeline. Returns (answer, chunks)."""
    chunks = retrieve_hybrid(doc_id, question, top_k=top_k)
    prompt = build_prompt(question, chunks, response_mode="balanced")
    tokens = list(stream_response(prompt, temperature=0.2))
    answer = "".join(tokens)
    return answer, chunks



def _judge_faithfulness(question: str, reference: str, system_answer: str) -> dict:
    """Ask the judge LLM to score faithfulness. Retries once on parse failure."""
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        question=question, reference=reference, system_answer=system_answer
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict JSON-only evaluator. Output ONLY a valid JSON object. No markdown, no explanation, no code fences."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or ""
            # Strip markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()
            # Extract first JSON object if model added text around it
            match = _re.search(r'\{.*?\}', raw, _re.DOTALL)
            if match:
                raw = match.group(0)
            result = json.loads(raw)
            if "verdict" in result:
                return result
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                return {"verdict": "error", "reason": f"Parse failed after 3 attempts: {e}"}
            time.sleep(2)
        except Exception as e:  # noqa: BLE001
            return {"verdict": "error", "reason": str(e)}
    return {"verdict": "error", "reason": "Max retries exceeded"}


def evaluate_faithfulness(
    golden_set: list[dict],
    top_k:      int = 5,
) -> dict[str, Any]:
    """
    Generate answers for each golden question, judge faithfulness.
    """
    verdicts: dict[str, int] = {"faithful": 0, "partially_faithful": 0, "unfaithful": 0, "error": 0}
    per_q = []

    for i, item in enumerate(golden_set):
        logger.info("Faithfulness eval %d/%d: %s", i + 1, len(golden_set), item["question"][:60])
        sys_answer = ""
        try:
            sys_answer, _ = _generate_answer(item["doc_id"], item["question"], top_k)
            judgment = _judge_faithfulness(item["question"], item["answer"], sys_answer)
        except Exception as exc:
            judgment = {"verdict": "error", "reason": str(exc)}
        verdict       = judgment.get("verdict", "error")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        per_q.append({
            "question":      item["question"],
            "reference":     item["answer"],
            "system_answer": sys_answer,
            "verdict":       verdict,
            "reason":        judgment.get("reason", ""),
        })
        time.sleep(2.5)  # rate limit buffer

    n = len(golden_set)
    faithfulness_score = round((verdicts["faithful"] + 0.5 * verdicts["partially_faithful"]) / n, 4) if n else 0
    return {
        "faithfulness_score": faithfulness_score,
        "verdict_counts":     verdicts,
        "n_questions":        n,
        "per_question":       per_q,
    }


# ─────────────────────────────────────────────
#  Master runner
# ─────────────────────────────────────────────

import math  # imported here so it's available in generate_golden_set


def run_full_eval(
    doc_id:          str,
    golden_path:     str  = "evals/golden_set.json",
    generate_golden: bool = False,
    n_questions:     int  = 20,
    top_k:           int  = 5,
    output_path:     str  = "evals/results.json",
) -> dict[str, Any]:
    """
    End-to-end eval. Pass generate_golden=True on first run.
    """
    # Load or generate golden set
    if generate_golden or not os.path.exists(golden_path):
        logger.info("Generating golden set (%d Q&A)…", n_questions)
        golden_set = generate_golden_set(doc_id, n_questions=n_questions, output_path=golden_path)
    else:
        with open(golden_path) as f:
            golden_set = json.load(f)
        logger.info("Loaded golden set: %d questions from %s", len(golden_set), golden_path)

    logger.info("Running retrieval evaluation…")
    retrieval_results = evaluate_retrieval(golden_set, top_k=top_k)

    logger.info("Running faithfulness evaluation (LLM-as-judge)…")
    faithfulness_results = evaluate_faithfulness(golden_set, top_k=top_k)

    results = {
        "doc_id":       doc_id,
        "n_questions":  len(golden_set),
        "top_k":        top_k,
        "retrieval":    retrieval_results,
        "faithfulness": faithfulness_results,
        "summary": {
            "hit_rate_at_k":      retrieval_results["hit_rate_at_k"],
            "mrr":                retrieval_results["mrr"],
            "faithfulness_score": faithfulness_results["faithfulness_score"],
        },
    }

    _write_json_atomic(output_path, results)
    logger.info("Eval results saved → %s", output_path)
    logger.info(
        "SUMMARY | Hit@%d: %.2f | MRR: %.2f | Faithfulness: %.2f",
        top_k,
        results["summary"]["hit_rate_at_k"],
        results["summary"]["mrr"],
        results["summary"]["faithfulness_score"],
    )
    # Log to MLflow
    try:
        from services.experiment_tracker import log_eval_run
        run_id = log_eval_run(results)
        results["mlflow_run_id"] = run_id
        logger.info("MLflow run logged: %s", run_id)
    except Exception:  # noqa: BLE001
        logger.warning("MLflow logging skipped (not configured)")
    return results


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocuMind RAG Evaluation Pipeline")
    parser.add_argument("--doc_id",          required=True)
    parser.add_argument("--golden_path",     default="evals/golden_set.json")
    parser.add_argument("--generate_golden", action="store_true")
    parser.add_argument("--n_questions",     type=int, default=20)
    parser.add_argument("--top_k",           type=int, default=5)
    parser.add_argument("--output_path",     default="evals/results.json")
    args = parser.parse_args()

    run_full_eval(
        doc_id=args.doc_id,
        golden_path=args.golden_path,
        generate_golden=args.generate_golden,
        n_questions=args.n_questions,
        top_k=args.top_k,
        output_path=args.output_path,
    )
