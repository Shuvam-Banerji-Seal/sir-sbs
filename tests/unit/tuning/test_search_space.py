"""Unit tests for RAGtuneSearchSpace — no real models, no API calls."""
import pytest
from optuna.trial import FixedTrial

from ragtune.tuning.search_space import RAGtuneSearchSpace


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trial(space: RAGtuneSearchSpace) -> FixedTrial:
    """Build a FixedTrial that satisfies every suggest_* call in sample()."""
    return FixedTrial({
        # Discrete component selection
        "reranker_type": space.reranker_types[0],
        "reformulator_type": "identity",
        "estimator_type": "baseline",
        "scheduler_type": space.scheduler_types[0],
        "feedback_type": "none",
        # Always-active numerical
        "original_query_depth": 10,
        "depth_per_reformulation": 5,
        "max_pool_size": 50,
        "near_duplicate_threshold": 0.8,
        "assembler_max_docs": 10,
        "budget_rerank_docs": 30,
        "budget_reformulations": 1,
        "scheduler_batch_size": 5,
        "gd_llm_limit": 3,
        "gd_ce_limit": 10,
        # Conditional (sampled unconditionally)
        "ce_model": space.ce_models[0],
        "monot5_model": space.monot5_models[0],
        "monot5_batch_size": "16",
        "reformulator_model": space.reformulator_models[0],
        "reformulator_n_variants": 3,
        "similarity_model": space.similarity_models[0],
        "min_reranked_for_regression": 3,
        "budget_stop_token_threshold": 0.9,
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSearchSpaceCardinality:
    def test_positive(self):
        ss = RAGtuneSearchSpace()
        assert ss.get_cardinality() > 0

    def test_restricted_menu_reduces_cardinality(self):
        full = RAGtuneSearchSpace()
        restricted = RAGtuneSearchSpace(reranker_types=["noop"])
        assert restricted.get_cardinality() < full.get_cardinality()


class TestSampleReturnsAllKeys:
    EXPECTED_KEYS = {
        "reranker_type", "reformulator_type", "estimator_type",
        "scheduler_type", "feedback_type",
        "original_query_depth", "depth_per_reformulation", "max_pool_size",
        "near_duplicate_threshold", "assembler_max_docs",
        "budget_rerank_docs", "budget_reformulations",
        "scheduler_batch_size", "gd_llm_limit", "gd_ce_limit",
        "ce_model", "monot5_model", "monot5_batch_size",
        "reformulator_model", "reformulator_n_variants",
        "similarity_model", "min_reranked_for_regression",
        "budget_stop_token_threshold",
    }

    def test_all_keys_present(self):
        ss = RAGtuneSearchSpace()
        trial = _make_trial(ss)
        params = ss.sample(trial)
        missing = self.EXPECTED_KEYS - set(params.keys())
        assert not missing, f"Missing keys: {missing}"


class TestRetrievalOverrides:
    def test_keys_match_controller_config_paths(self):
        ss = RAGtuneSearchSpace()
        trial = _make_trial(ss)
        params = ss.sample(trial)
        overrides = ss.to_retrieval_overrides(params)
        assert set(overrides) == {
            "retrieval.original_query_depth",
            "retrieval.depth_per_reformulation",
            "retrieval.max_pool_size",
            "retrieval.near_duplicate_threshold",
        }

    def test_values_match_sampled_params(self):
        ss = RAGtuneSearchSpace()
        trial = _make_trial(ss)
        params = ss.sample(trial)
        overrides = ss.to_retrieval_overrides(params)
        assert overrides["retrieval.original_query_depth"] == params["original_query_depth"]
        assert overrides["retrieval.max_pool_size"] == params["max_pool_size"]


class TestBuildController:
    """build_controller should succeed for every reranker/reformulator/estimator type
    that does NOT require model downloads (noop, identity, baseline)."""

    def _noop_params(self, ss: RAGtuneSearchSpace) -> dict:
        trial = _make_trial(ss)
        return ss.sample(trial)

    def test_noop_reranker_identity_reformulator(self):
        import ragtune.components  # noqa — populate registry
        from ragtune.components.retrievers import InMemoryRetriever
        from ragtune.core.types import ScoredDocument

        docs = [ScoredDocument(id=f"d{i}", content=f"doc {i}", score=float(i)) for i in range(5)]
        retriever = InMemoryRetriever(docs)

        ss = RAGtuneSearchSpace(
            reranker_types=["noop"],
            reformulator_types=["identity"],
            estimator_types=["baseline"],
            scheduler_types=["active-learning"],
            feedback_types=["none"],
        )
        params = self._noop_params(ss)
        controller = ss.build_controller(params, retriever)
        assert controller is not None

    def test_unknown_reranker_raises(self):
        import ragtune.components  # noqa
        from ragtune.components.retrievers import InMemoryRetriever
        from ragtune.core.types import ScoredDocument

        docs = [ScoredDocument(id="d0", content="doc", score=1.0)]
        retriever = InMemoryRetriever(docs)

        ss = RAGtuneSearchSpace(reranker_types=["noop"])
        params = {
            "reranker_type": "nonexistent-reranker",
            "reformulator_type": "identity",
            "estimator_type": "baseline",
            "scheduler_type": "active-learning",
            "feedback_type": "none",
            "original_query_depth": 10,
            "depth_per_reformulation": 5,
            "max_pool_size": 50,
            "near_duplicate_threshold": 0.8,
            "assembler_max_docs": 10,
            "budget_rerank_docs": 30,
            "budget_reformulations": 1,
            "scheduler_batch_size": 5,
            "gd_llm_limit": 3,
            "gd_ce_limit": 10,
            "ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "monot5_model": "castorini/monot5-base-msmarco",
            "monot5_batch_size": "16",
            "reformulator_model": "gpt-4o-mini",
            "reformulator_n_variants": 3,
            "similarity_model": "all-MiniLM-L6-v2",
            "min_reranked_for_regression": 3,
            "budget_stop_token_threshold": 0.9,
        }
        with pytest.raises(ValueError, match="not in registry"):
            ss.build_controller(params, retriever)

    def test_graceful_degradation_scheduler_params_wired(self):
        import ragtune.components  # noqa
        from ragtune.components.retrievers import InMemoryRetriever
        from ragtune.core.types import ScoredDocument
        from ragtune.components.schedulers import GracefulDegradationScheduler

        docs = [ScoredDocument(id=f"d{i}", content=f"doc {i}", score=float(i)) for i in range(3)]
        retriever = InMemoryRetriever(docs)

        ss = RAGtuneSearchSpace(
            reranker_types=["noop"],
            reformulator_types=["identity"],
            estimator_types=["baseline"],
            scheduler_types=["graceful-degradation"],
            feedback_types=["none"],
        )
        trial = _make_trial(ss)
        params = ss.sample(trial)
        params["scheduler_type"] = "graceful-degradation"
        params["gd_llm_limit"] = 7
        params["gd_ce_limit"] = 15

        controller = ss.build_controller(params, retriever)
        assert isinstance(controller.scheduler, GracefulDegradationScheduler)
        assert controller.scheduler.llm_limit == 7
        assert controller.scheduler.cross_encoder_limit == 15
