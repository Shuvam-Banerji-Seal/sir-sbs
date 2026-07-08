"""
Unit tests for IndexerFactory and ScenarioFactory.
"""

import pytest
from ragtune.components.indexers import (
    IndexerFactory,
    BM25Indexer,
    DenseIndexer,
    BaseIndexer,
)
from ragtune.components.scenarios import ScenarioSpec, build_controller


class TestIndexerFactory:
    def test_create_bm25(self):
        indexer = IndexerFactory.create("bm25", num_results=10)
        assert isinstance(indexer, BM25Indexer)
        assert indexer.num_results == 10

    def test_create_dense(self):
        indexer = IndexerFactory.create("dense", model_name="test-model")
        assert isinstance(indexer, DenseIndexer)
        assert indexer.model_name == "test-model"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown indexer type"):
            IndexerFactory.create("invalid")

    def test_register_custom(self):
        class CustomIndexer(BaseIndexer):
            def build(self, corpus):
                return None

        IndexerFactory.register("custom", CustomIndexer)
        indexer = IndexerFactory.create("custom")
        assert isinstance(indexer, CustomIndexer)

    def test_from_env_default(self):
        import os

        os.environ.pop("INDEX_TYPE", None)
        indexer = IndexerFactory.from_env()
        assert isinstance(indexer, BM25Indexer)

    def test_from_env_dense(self):
        import os

        os.environ["INDEX_TYPE"] = "dense"
        os.environ["EMBEDDING_MODEL"] = "test-model"
        indexer = IndexerFactory.from_env()
        assert isinstance(indexer, DenseIndexer)
        os.environ.pop("INDEX_TYPE")
        os.environ.pop("EMBEDDING_MODEL")


class TestScenarioSpec:
    def test_default_scenarios(self):
        import os

        os.environ.pop("SCENARIOS", None)
        specs = ScenarioSpec.from_env()
        assert len(specs) == 3
        assert specs[0].reranker == "noop"
        assert specs[1].reranker == "cross-encoder"

    def test_custom_scenarios(self):
        import os

        os.environ["SCENARIOS"] = '[{"name":"test","reranker":"noop","budget_docs":5}]'
        specs = ScenarioSpec.from_env()
        assert len(specs) == 1
        assert specs[0].name == "test"
        assert specs[0].budget_docs == 5
        os.environ.pop("SCENARIOS")

    def test_build_controller(self):
        from ragtune.components.indexers import BM25Indexer

        indexer = BM25Indexer(num_results=10)

        spec = ScenarioSpec(name="test", reranker="noop", budget_docs=5)
        controller = build_controller(spec, None)
        assert controller is not None
