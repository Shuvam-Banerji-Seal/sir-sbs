"""Unit tests for TrialEvaluator and ndcg_at_k — no real retrievers or models."""
import pytest

from ragtune.tuning.evaluator import EvalDataset, EvalQuery, TrialEvaluator, ndcg_at_k
from ragtune.tuning.pruners import CostPruner


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────

class TestNdcgAtK:
    def test_perfect_ranking_returns_one(self):
        qrels = {"d1": 2, "d2": 1, "d3": 0}
        # Rank docs in ideal order
        ranked = ["d1", "d2", "d3"]
        score = ndcg_at_k(ranked, qrels, k=3)
        assert abs(score - 1.0) < 1e-9

    def test_all_irrelevant_returns_zero(self):
        qrels = {"d1": 1, "d2": 2}
        ranked = ["d3", "d4", "d5"]
        assert ndcg_at_k(ranked, qrels, k=3) == 0.0

    def test_empty_qrels_returns_zero(self):
        assert ndcg_at_k(["d1", "d2"], {}, k=5) == 0.0

    def test_empty_ranking_returns_zero(self):
        assert ndcg_at_k([], {"d1": 1}, k=5) == 0.0

    def test_k_limits_consideration(self):
        qrels = {"d1": 1, "d2": 1}
        # d2 is relevant but outside k=1 window
        ranked = ["d0", "d2"]  # d0 irrelevant, d2 at rank 2
        score_k1 = ndcg_at_k(ranked, qrels, k=1)
        score_k2 = ndcg_at_k(ranked, qrels, k=2)
        assert score_k1 < score_k2

    def test_partial_match_between_zero_and_one(self):
        qrels = {"d1": 1, "d2": 1, "d3": 1}
        ranked = ["d1", "x1", "x2"]  # only first doc relevant
        score = ndcg_at_k(ranked, qrels, k=3)
        assert 0 < score < 1.0


# ── EvalDataset ───────────────────────────────────────────────────────────────

class TestEvalDataset:
    def _make_dataset(self):
        queries = [
            {"query_id": "q1", "query": "covid transmission"},
            {"query_id": "q2", "query": "mask efficacy"},
        ]
        qrels = {
            "q1": {"d1": 2, "d2": 1},
            "q2": {"d3": 1},
        }
        return EvalDataset.from_dicts("test", queries, qrels)

    def test_from_dicts_populates_queries(self):
        ds = self._make_dataset()
        assert len(ds.queries) == 2

    def test_qrels_mapped_correctly(self):
        ds = self._make_dataset()
        q1 = next(q for q in ds.queries if q.query_id == "q1")
        assert q1.qrels["d1"] == 2

    def test_iter_queries_respects_limit(self):
        ds = self._make_dataset()
        results = list(ds.iter_queries(limit=1))
        assert len(results) == 1

    def test_missing_qrels_gives_empty_dict(self):
        queries = [{"query_id": "q99", "query": "orphan"}]
        ds = EvalDataset.from_dicts("test", queries, {})
        assert ds.queries[0].qrels == {}


# ── TrialEvaluator ────────────────────────────────────────────────────────────

class _FakeTrial:
    def __init__(self):
        self._reported: list = []
        self._attrs: dict = {}

    def report(self, value, step):
        self._reported.append((value, step))

    def set_user_attr(self, key, val):
        self._attrs[key] = val


class _FakeController:
    """Controller that returns a fixed ranked list and zero rerank cost."""

    def __init__(self, ranked_ids=None, rerank_docs=0):
        from ragtune.core.types import ScoredDocument, ControllerTrace
        self._docs = [
            ScoredDocument(id=did, content="text", score=float(i))
            for i, did in enumerate(reversed(ranked_ids or []))
        ]
        self._cost = rerank_docs

    def run(self, query):
        from ragtune.core.types import ControllerOutput, ControllerTrace
        return ControllerOutput(
            query=query,
            documents=self._docs,
            trace=ControllerTrace(),
            final_budget_state={"rerank_docs": float(self._cost)},
        )


class TestTrialEvaluator:
    def _make_dataset(self, n=3):
        queries = [{"query_id": f"q{i}", "query": f"query {i}"} for i in range(n)]
        qrels = {f"q{i}": {f"d{i}": 1} for i in range(n)}
        return EvalDataset.from_dicts("test", queries, qrels)

    def test_evaluate_returns_objectives(self):
        ds = self._make_dataset()
        # Controller returns d0, d1, d2 for every query; q0 has d0 as relevant
        controller = _FakeController(ranked_ids=["d0", "d1", "d2"])
        evaluator = TrialEvaluator(dataset=ds, n_eval_queries=3)
        result = evaluator.evaluate(controller, _FakeTrial())
        assert 0.0 <= result.ndcg_at_10 <= 1.0
        assert result.queries_evaluated == 3
        assert result.rerank_docs == 0.0

    def test_evaluate_calls_pruner_report(self):
        """The evaluator's custom pruners receive a report call for each query."""
        from ragtune.tuning.pruners import CostPruner

        ds = self._make_dataset(n=5)
        controller = _FakeController(rerank_docs=0)
        pruner = CostPruner(max_mean_rerank_docs=9999.0, warmup_steps=1)
        evaluator = TrialEvaluator(dataset=ds, n_eval_queries=5, pruners=[pruner], report_interval=1)
        evaluator.evaluate(controller, _FakeTrial())
        # Pruner should have seen all 5 steps
        assert pruner._step == 5

    def test_evaluate_raises_trial_pruned_when_cost_pruner_fires(self):
        import optuna

        ds = self._make_dataset(n=10)
        # Controller uses 1000 rerank docs per query — far above limit
        controller = _FakeController(rerank_docs=1000)
        pruner = CostPruner(max_mean_rerank_docs=5.0, warmup_steps=1)
        evaluator = TrialEvaluator(dataset=ds, n_eval_queries=10, pruners=[pruner])

        with pytest.raises(optuna.TrialPruned):
            evaluator.evaluate(controller, _FakeTrial())

    def test_evaluate_respects_n_eval_queries_limit(self):
        ds = self._make_dataset(n=20)
        controller = _FakeController()
        evaluator = TrialEvaluator(dataset=ds, n_eval_queries=5)
        result = evaluator.evaluate(controller, _FakeTrial())
        assert result.queries_evaluated == 5

    def test_evaluate_handles_controller_exception_gracefully(self):
        """A crashing controller logs the error and continues rather than failing the whole trial."""
        ds = self._make_dataset(n=3)

        class CrashController:
            def run(self, query):
                raise RuntimeError("model error")

        evaluator = TrialEvaluator(dataset=ds, n_eval_queries=3)
        trial = _FakeTrial()
        result = evaluator.evaluate(CrashController(), trial)
        # All queries crashed → ndcg 0.0 but no exception
        assert result.ndcg_at_10 == 0.0
        assert result.queries_evaluated == 3
        assert any("error_step" in k for k in trial._attrs)
