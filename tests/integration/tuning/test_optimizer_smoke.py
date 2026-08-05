"""
Integration smoke test for the MOBO optimizer.

Runs a 5-trial study using InMemoryRetriever + synthetic qrels.
No model downloads, no API calls.  Verifies:
  1. Study completes without exception.
  2. study.best_trials is non-empty.
  3. extract_pareto_configs writes YAML files.
  4. Each written YAML is a valid PipelineConfig dict (has required keys).
"""
import os
import tempfile

import pytest

import ragtune.components  # noqa — populate registry
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.core.types import ScoredDocument
from ragtune.tuning.evaluator import EvalDataset
from ragtune.tuning.optimizer import extract_pareto_configs, run_study
from ragtune.tuning.search_space import RAGtuneSearchSpace
from ragtune.tuning.study_config import DatasetConfig, TuningStudyConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_dataset():
    """20 queries; each query has one relevant doc (d<i>) at rank 1."""
    n = 20
    queries = [{"query_id": f"q{i}", "query": f"what is topic {i}"} for i in range(n)]
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(n)}
    return EvalDataset.from_dicts("synthetic", queries, qrels)


@pytest.fixture
def fake_retriever():
    """Returns d0-d19 for every query (no actual ranking)."""
    docs = [
        ScoredDocument(id=f"d{i}", content=f"document about topic {i}", score=float(20 - i))
        for i in range(20)
    ]
    return InMemoryRetriever(docs)


@pytest.fixture
def smoke_study_config(tmp_path):
    return TuningStudyConfig(
        name="smoke-test",
        dataset=DatasetConfig(name="synthetic"),
        n_trials=5,
        n_startup_trials=3,
        n_parallel_workers=1,
        n_eval_queries=10,
        seed=0,
        storage_url=None,          # in-memory storage
        max_mean_rerank_docs=200.0,
        max_trial_seconds=300.0,
        pareto_warmup_trials=100,  # disable Pareto pruner (not enough warmup trials)
        output_dir=str(tmp_path / "pareto_out"),
        search_space_overrides={
            # Only use components that don't require model downloads
            "reranker_types": ["noop"],
            "reformulator_types": ["identity"],
            "estimator_types": ["baseline"],
            "scheduler_types": ["active-learning"],
            "feedback_types": ["none"],
        },
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_study_completes(smoke_study_config, fake_retriever, synthetic_dataset):
    study = run_study(smoke_study_config, fake_retriever, synthetic_dataset)

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    assert len(completed) == smoke_study_config.n_trials


def test_best_trials_non_empty(smoke_study_config, fake_retriever, synthetic_dataset):
    study = run_study(smoke_study_config, fake_retriever, synthetic_dataset)
    assert len(study.best_trials) >= 1


def test_extract_pareto_configs_writes_yaml(smoke_study_config, fake_retriever, synthetic_dataset):
    import yaml

    study = run_study(smoke_study_config, fake_retriever, synthetic_dataset)
    search_space = RAGtuneSearchSpace(**smoke_study_config.search_space_overrides)
    paths = extract_pareto_configs(study, search_space, smoke_study_config.output_dir)

    assert len(paths) >= 1
    for path in paths:
        assert os.path.exists(path), f"Expected file at {path}"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "pipeline" in data
        assert "components" in data["pipeline"]
        assert "budget" in data["pipeline"]


def test_pareto_yaml_has_all_component_keys(smoke_study_config, fake_retriever, synthetic_dataset):
    import yaml

    study = run_study(smoke_study_config, fake_retriever, synthetic_dataset)
    search_space = RAGtuneSearchSpace(**smoke_study_config.search_space_overrides)
    paths = extract_pareto_configs(study, search_space, smoke_study_config.output_dir)

    required_keys = {"retriever", "reformulator", "reranker", "estimator", "scheduler", "assembler"}
    for path in paths:
        with open(path) as f:
            data = yaml.safe_load(f)
        components = data["pipeline"]["components"]
        missing = required_keys - set(components.keys())
        assert not missing, f"{path} is missing component keys: {missing}"
