"""
tests/test_retrieval.py

Unit tests for the hybrid retrieval pipeline.
These run without hitting any external API.

Run: pytest tests/ -v
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

# Allow import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Minimal stubs so we don't need actual FAISS indexes in CI ───────────────

class MockFaissIndex:
    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32)

    def search(self, query_vec, k):
        # Return distances and indices for first k vectors
        n = min(k, len(self._data))
        return np.zeros((1, n)), np.arange(n, dtype=np.int64).reshape(1, n)


# ─────────────────────────────────────────────────────────────────────────────

class TestRRFFusion(unittest.TestCase):
    """Test Reciprocal Rank Fusion logic in isolation."""

    def setUp(self):
        # Import here so we can patch INDEX_DIR
        from services.retrieval import _rrf_fuse
        self._rrf_fuse = _rrf_fuse

    def test_rrf_prefers_items_appearing_in_both_lists(self):
        """A chunk that ranks well in both dense AND BM25 should win."""
        dense = [
            {"chunk_id": 0, "chunk": "a", "page": 1, "score": 0.9},
            {"chunk_id": 1, "chunk": "b", "page": 1, "score": 0.8},
        ]
        bm25 = [
            {"chunk_id": 1, "chunk": "b", "page": 1, "score": 10.0},  # chunk 1 appears in both
            {"chunk_id": 2, "chunk": "c", "page": 2, "score": 9.0},
        ]
        fused = self._rrf_fuse(dense, bm25)
        # chunk_id 1 should be ranked first because it ranks well in both
        self.assertEqual(fused[0]["chunk_id"], 1)

    def test_rrf_output_contains_rrf_score(self):
        dense = [{"chunk_id": 0, "chunk": "x", "page": 1, "score": 1.0}]
        bm25  = [{"chunk_id": 0, "chunk": "x", "page": 1, "score": 5.0}]
        fused = self._rrf_fuse(dense, bm25)
        self.assertIn("rrf_score", fused[0])
        self.assertGreater(fused[0]["rrf_score"], 0)

    def test_rrf_empty_inputs(self):
        result = self._rrf_fuse([], [])
        self.assertEqual(result, [])

    def test_rrf_single_list(self):
        dense = [{"chunk_id": 0, "chunk": "a", "page": 1, "score": 1.0}]
        fused = self._rrf_fuse(dense, [])
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["chunk_id"], 0)


class TestBM25Retrieve(unittest.TestCase):
    """Test BM25 keyword matching."""

    def test_bm25_ranks_keyword_match_first(self):
        from services.retrieval import _bm25_retrieve

        meta = {
            "chunks": [
                "The capital of France is Paris.",
                "Quantum entanglement is a physics phenomenon.",
                "Paris is famous for the Eiffel Tower.",
            ],
            "pages": [1, 2, 3],
        }
        results = _bm25_retrieve(meta, "Paris capital France", top_k=3)
        # First two results should contain "Paris"
        top_chunk = results[0]["chunk"]
        self.assertIn("Paris", top_chunk)

    def test_bm25_returns_expected_count(self):
        from services.retrieval import _bm25_retrieve

        meta = {"chunks": [f"chunk {i}" for i in range(10)], "pages": [1] * 10}
        results = _bm25_retrieve(meta, "chunk 3", top_k=5)
        self.assertLessEqual(len(results), 5)


class TestCalculateTool(unittest.TestCase):
    """Test the safe calculator tool used by the agent."""

    def test_basic_arithmetic(self):
        from services.agent import _tool_calculate
        result = json.loads(_tool_calculate("2 + 2"))
        self.assertEqual(result["result"], 4)

    def test_math_functions(self):
        from services.agent import _tool_calculate
        result = json.loads(_tool_calculate("round(math.pi, 4)"))
        self.assertAlmostEqual(result["result"], 3.1416)

    def test_blocks_dangerous_code(self):
        from services.agent import _tool_calculate
        result = json.loads(_tool_calculate("__import__('os').system('ls')"))
        self.assertIn("error", result)

    def test_blocks_exec(self):
        from services.agent import _tool_calculate
        result = json.loads(_tool_calculate("exec('x=1')"))
        self.assertIn("error", result)


class TestEvalGoldenSetFormat(unittest.TestCase):
    """Test that the eval pipeline handles golden set JSON correctly."""

    def _make_golden_item(self, chunk_id=0, doc_id="test_doc"):
        return {
            "question":       "What is the purpose of DocuMind?",
            "answer":         "DocuMind is a RAG-based document Q&A system.",
            "question_type":  "definition",
            "source_chunk_id": chunk_id,
            "source_page":    1,
            "doc_id":         doc_id,
        }

    def test_golden_item_has_required_keys(self):
        item = self._make_golden_item()
        for key in ("question", "answer", "source_chunk_id", "doc_id"):
            self.assertIn(key, item)

    def test_retrieval_eval_hit_calculation(self):
        """Verify Hit@k logic without hitting actual FAISS."""
        from evals.eval_pipeline import evaluate_retrieval
        from unittest.mock import patch

        golden = [self._make_golden_item(chunk_id=2)]

        # Mock retrieve_hybrid to return chunk_id=2 in first position
        mock_results = [
            {"chunk_id": 2, "chunk": "DocuMind is a RAG system.", "page": 1, "score": 0.9},
            {"chunk_id": 5, "chunk": "Other chunk.", "page": 2, "score": 0.7},
        ]
        with patch("evals.eval_pipeline.retrieve_hybrid", return_value=mock_results):
            results = evaluate_retrieval(golden, top_k=5)

        self.assertEqual(results["hits"], 1)
        self.assertEqual(results["hit_rate_at_k"], 1.0)
        self.assertAlmostEqual(results["mrr"], 1.0)

    def test_retrieval_eval_miss_calculation(self):
        from evals.eval_pipeline import evaluate_retrieval
        from unittest.mock import patch

        golden = [self._make_golden_item(chunk_id=99)]  # chunk 99 won't be in results

        mock_results = [
            {"chunk_id": 0, "chunk": "Different chunk.", "page": 1, "score": 0.9},
        ]
        with patch("evals.eval_pipeline.retrieve_hybrid", return_value=mock_results):
            results = evaluate_retrieval(golden, top_k=5)

        self.assertEqual(results["hits"], 0)
        self.assertEqual(results["hit_rate_at_k"], 0.0)
        self.assertEqual(results["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
