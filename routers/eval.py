"""
routers/eval.py — DocuMind v2

Exposes the eval pipeline via FastAPI endpoints so the frontend
(and your resume demo) can trigger + display eval results without SSH.

Endpoints:
  POST /api/eval/golden-set          — generate/regenerate golden set
  POST /api/eval/run                 — run full retrieval + faithfulness eval
  GET  /api/eval/results/{doc_id}    — fetch last saved results
  GET  /api/eval/golden-set/{doc_id} — fetch golden set questions
"""

import json
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from evals.eval_pipeline import (
    generate_golden_set,
    run_full_eval,
)

router = APIRouter(prefix="/api/eval", tags=["eval"])

GOLDEN_DIR  = "evals/golden"
RESULTS_DIR = "evals/results"
os.makedirs(GOLDEN_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class GoldenSetRequest(BaseModel):
    doc_id:      str
    n_questions: int = 20
    regenerate:  bool = False


class EvalRequest(BaseModel):
    doc_id:      str
    top_k:       int  = 5
    n_questions: int  = 20
    run_faithfulness: bool = True  # set False for faster retrieval-only runs


# ── Async state (simple in-memory; swap for Redis in prod) ──
_running_evals: dict[str, str] = {}   # doc_id → "running" | "done" | "error"


@router.post("/golden-set")
async def create_golden_set(req: GoldenSetRequest):
    """Generate a golden Q&A set for the document. Cached unless regenerate=True."""
    golden_path = os.path.join(GOLDEN_DIR, f"{req.doc_id}.json")
    if os.path.exists(golden_path) and not req.regenerate:
        with open(golden_path, encoding="utf-8") as f:
            existing = json.load(f)
        return {
            "status":      "cached",
            "n_questions": len(existing),
            "golden_path": golden_path,
        }
    golden_set = generate_golden_set(
        doc_id=req.doc_id,
        n_questions=req.n_questions,
        output_path=golden_path,
    )
    return {
        "status":      "generated",
        "n_questions": len(golden_set),
        "golden_path": golden_path,
        "sample":      golden_set[:3],
    }


def _run_eval_task(doc_id: str, top_k: int, n_questions: int, run_faithfulness: bool):
    """Background task for running eval."""
    golden_path = os.path.join(GOLDEN_DIR,  f"{doc_id}.json")
    results_path = os.path.join(RESULTS_DIR, f"{doc_id}.json")
    _running_evals[doc_id] = "running"
    try:
        run_full_eval(
            doc_id=doc_id,
            golden_path=golden_path,
            generate_golden=not os.path.exists(golden_path),
            n_questions=n_questions,
            top_k=top_k,
            output_path=results_path,
        )
        _running_evals[doc_id] = "done"
    except Exception as e:
        _running_evals[doc_id] = f"error: {e}"


@router.post("/run")
async def run_eval(req: EvalRequest, background_tasks: BackgroundTasks):
    """
    Kick off a full evaluation in the background.
    Poll GET /api/eval/results/{doc_id} for completion.
    """
    if _running_evals.get(req.doc_id) == "running":
        return {"status": "already_running", "doc_id": req.doc_id}

    background_tasks.add_task(
        _run_eval_task,
        doc_id=req.doc_id,
        top_k=req.top_k,
        n_questions=req.n_questions,
        run_faithfulness=req.run_faithfulness,
    )
    return {"status": "started", "doc_id": req.doc_id, "message": "Poll /api/eval/results/{doc_id} for results."}


@router.get("/status/{doc_id}")
async def eval_status(doc_id: str):
    return {"doc_id": doc_id, "status": _running_evals.get(doc_id, "not_started")}


@router.get("/results/{doc_id}")
async def get_results(doc_id: str):
    results_path = os.path.join(RESULTS_DIR, f"{doc_id}.json")
    if not os.path.exists(results_path):
        status = _running_evals.get(doc_id, "not_started")
        raise HTTPException(404, f"No results found. Eval status: {status}")
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/golden-set/{doc_id}")
async def get_golden_set(doc_id: str, limit: int = 20):
    golden_path = os.path.join(GOLDEN_DIR, f"{doc_id}.json")
    if not os.path.exists(golden_path):
        raise HTTPException(404, "Golden set not found. POST /api/eval/golden-set first.")
    with open(golden_path, encoding="utf-8") as f:
        golden_set = json.load(f)
    return {
        "doc_id":      doc_id,
        "total":       len(golden_set),
        "questions":   golden_set[:limit],
    }
