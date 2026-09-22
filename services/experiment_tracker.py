"""
Lightweight MLflow wrapper.
Logs each eval run as an MLflow experiment so you have
before/after numbers tracked with git SHA.
"""
import json
import os
import tempfile

import mlflow

EXPERIMENT_NAME = "documind-rag-eval"

def log_eval_run(results: dict, run_name: str = "eval") -> str:
    """Log eval results to MLflow. Returns run_id."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        summary = results.get("summary", {})
        mlflow.log_metrics({
            "hit_rate_at_k":      summary.get("hit_rate_at_k", 0),
            "mrr":                summary.get("mrr", 0),
            "faithfulness_score": summary.get("faithfulness_score", 0),
        })
        mlflow.log_params({
            "top_k":       results.get("top_k", 5),
            "n_questions": results.get("n_questions", 0),
            "doc_id":      results.get("doc_id", ""),
            "retriever":   "bm25_dense_rrf_flashrank",
            "judge_model": os.getenv("JUDGE_MODEL", "gpt-oss-120b"),
        })
        # Log full results as artifact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f, indent=2)
            mlflow.log_artifact(f.name, "eval_results")
        return run.info.run_id