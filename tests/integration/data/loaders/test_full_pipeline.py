"""
Integration test for the full tool retrieval pipeline.

Tests the complete flow:
    DataLoaderFactory → IndexerFactory → ScenarioFactory → RetrievalEvaluator

This single test verifies all components work together end-to-end.
"""

import os
import pytest
from ragtune.data.loaders import DataLoaderFactory
from ragtune.data.constants import Benchmark
from ragtune.components.indexers import IndexerFactory
from ragtune.components.scenarios import ScenarioSpec, build_controller
from ragtune.evaluation.RetrievalEvaluator import RetrievalEvaluator


class TestFullPipeline:
    """End-to-end integration test: load → index → retrieve → evaluate."""

    def test_toolret_pipeline(self):
        """Test full pipeline on ToolRet apibank (smallest subset)."""
        # 1. Load data via DataLoaderFactory
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="apibank",
            benchmark_name=Benchmark.TOOLRET,
            n_queries=5,
        )
        corpus, queries, qrels = loader.load()
        assert len(corpus) > 0, "Corpus should not be empty"
        assert len(queries) > 0, "Queries should not be empty"
        assert len(qrels) > 0, "Qrels should not be empty"

        # 2. Build retriever via IndexerFactory (BM25)
        indexer = IndexerFactory.create("bm25", num_results=10)
        retriever = indexer.build(corpus)
        assert retriever is not None, "Retriever should not be None"

        # 3. Build scenarios via ScenarioSpec
        scenarios = ScenarioSpec.from_env()
        assert len(scenarios) >= 1, "Should have at least 1 scenario"

        # 4. Create baseline controller (noop reranker, creates RAGtuneContext internally)
        baseline_spec = ScenarioSpec(name="baseline", reranker="noop", budget_docs=0)
        controller_baseline = build_controller(baseline_spec, retriever)

        # 5. Run baseline
        evaluator = RetrievalEvaluator(k_values=[10])
        results = {}
        for qid, qtext in list(queries.items())[:3]:
            output = controller_baseline.run(qtext)
            results[qid] = {
                doc.id: 1.0 / (rank + 1) for rank, doc in enumerate(output.documents)
            }
        assert len(results) > 0, "Should have results"

        # 5. Evaluate
        metrics = evaluator.evaluate(qrels, results, k_values=[10])
        assert "ndcg" in metrics, "Should have NDCG metric"
        assert "recall" in metrics, "Should have Recall metric"
        assert f"NDCG@10" in metrics["ndcg"], "Should have NDCG@10"
        assert metrics["ndcg"]["NDCG@10"] >= 0, "NDCG should be non-negative"

        # 6. Build and run a controller scenario
        spec = scenarios[0]
        controller = build_controller(spec, retriever)
        assert controller is not None, "Controller should not be None"

        # 7. Run controller on one query
        qid, qtext = list(queries.items())[0]
        output = controller.run(qtext)
        assert output is not None, "Controller output should not be None"
        assert len(output.documents) > 0, "Should return documents"

        print(
            f"Pipeline test passed: {len(corpus)} docs, {len(queries)} queries, "
            f"NDCG@10={metrics['ndcg']['NDCG@10']:.4f}"
        )

    def test_skillret_pipeline(self):
        """Test full pipeline on SkillRet test split (limited)."""
        # 1. Load
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="test",
            benchmark_name=Benchmark.SKILLRET,
            n_queries=5,
        )
        corpus, queries, qrels = loader.load()
        assert len(corpus) > 0
        assert len(queries) > 0
        assert len(qrels) > 0

        # 2. Index
        indexer = IndexerFactory.create("bm25", num_results=10)
        retriever = indexer.build(corpus)
        assert retriever is not None

        # 3. Evaluate
        evaluator = RetrievalEvaluator(k_values=[10])
        baseline_spec = ScenarioSpec(name="baseline", reranker="noop", budget_docs=0)
        controller_baseline = build_controller(baseline_spec, retriever)
        results = {}
        for qid, qtext in list(queries.items())[:3]:
            output = controller_baseline.run(qtext)
            results[qid] = {
                doc.id: 1.0 / (rank + 1) for rank, doc in enumerate(output.documents)
            }
        metrics = evaluator.evaluate(qrels, results, k_values=[10])
        assert metrics["ndcg"]["NDCG@10"] >= 0

        print(
            f"SkillRet pipeline test passed: {len(corpus)} skills, "
            f"NDCG@10={metrics['ndcg']['NDCG@10']:.4f}"
        )

    def test_sra_pipeline(self):
        """Test full pipeline on SRA-Bench toolqa (limited)."""
        # 1. Load
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="toolqa",
            benchmark_name=Benchmark.SRA_BENCH,
            n_queries=5,
        )
        corpus, queries, qrels = loader.load()
        assert len(corpus) > 0
        assert len(queries) > 0
        assert len(qrels) > 0

        # 2. Index
        indexer = IndexerFactory.create("bm25", num_results=10)
        retriever = indexer.build(corpus)
        assert retriever is not None

        # 3. Evaluate
        evaluator = RetrievalEvaluator(k_values=[10])
        baseline_spec = ScenarioSpec(name="baseline", reranker="noop", budget_docs=0)
        controller_baseline = build_controller(baseline_spec, retriever)
        results = {}
        for qid, qtext in list(queries.items())[:3]:
            output = controller_baseline.run(qtext)
            results[qid] = {
                doc.id: 1.0 / (rank + 1) for rank, doc in enumerate(output.documents)
            }
        metrics = evaluator.evaluate(qrels, results, k_values=[10])
        assert metrics["ndcg"]["NDCG@10"] >= 0

        print(
            f"SRA-Bench pipeline test passed: {len(corpus)} skills, "
            f"NDCG@10={metrics['ndcg']['NDCG@10']:.4f}"
        )

    def test_scenario_specs_from_env(self):
        """Test that ScenarioSpec.from_env works with defaults and custom JSON."""
        import json

        # Default
        os.environ.pop("SCENARIOS", None)
        specs = ScenarioSpec.from_env()
        assert len(specs) == 3
        assert specs[0].reranker == "noop"

        # Custom
        os.environ["SCENARIOS"] = json.dumps(
            [{"name": "custom", "reranker": "cross-encoder", "budget_docs": 15}]
        )
        specs = ScenarioSpec.from_env()
        assert len(specs) == 1
        assert specs[0].name == "custom"
        assert specs[0].budget_docs == 15
        os.environ.pop("SCENARIOS")

    def test_indexer_factory_all_types(self):
        """Test that IndexerFactory creates correct indexer for each type."""
        bm25 = IndexerFactory.create("bm25")
        assert type(bm25).__name__ == "BM25Indexer"

        dense = IndexerFactory.create("dense", model_name="test")
        assert type(dense).__name__ == "DenseIndexer"

        with pytest.raises(ValueError):
            IndexerFactory.create("nonexistent")

    def test_data_loader_factory_cache_dir(self):
        """Test that DataLoaderFactory forwards cache_dir to loaders."""
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="toolqa",
            benchmark_name=Benchmark.SRA_BENCH,
            n_queries=3,
            cache_dir="/tmp/test_cache",
        )
        assert loader.cache_dir == "/tmp/test_cache"
