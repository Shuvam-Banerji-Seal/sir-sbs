"""
Unit tests for IndexFactory (Mandeep's) and ScenarioFactory.
"""

import pytest
from ragtune.indexing import IndexFactory, PyTerrierIndexer
from ragtune.registry import registry
from ragtune.components.scenarios import ScenarioSpec, build_controller


class TestIndexFactory:
    def test_create_pyterrier(self):
        indexer = IndexFactory.create("pyterrier")
        assert isinstance(indexer, PyTerrierIndexer)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown indexer type"):
            IndexFactory.create("invalid_nonexistent")

    def test_registered_indexers(self):
        all_indexers = registry.list_all().get("indexer", {})
        assert "pyterrier" in all_indexers
        assert "faiss" in all_indexers

    def test_from_config_sparse(self):
        from ragtune.config.models import IndexConfig

        config = IndexConfig(type="sparse")
        indexer = IndexFactory.from_config(config)
        assert isinstance(indexer, PyTerrierIndexer)


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
        spec = ScenarioSpec(name="test", reranker="noop", budget_docs=5)
        controller = build_controller(spec, None)
        assert controller is not None
