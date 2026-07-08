"""
Scenario Factory
================
Declarative scenario configuration for benchmark experiments.

Usage:
    scenarios = ScenarioSpec.from_env()
    for spec in scenarios:
        controller = build_controller(spec, retriever)

    # Or via environment:
    SCENARIOS='[{"name":"baseline","reranker":"noop"}]' python scripts/run_benchmark.py
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.components.rerankers import NoOpReranker, CrossEncoderReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator


@dataclass
class ScenarioSpec:
    """Declarative specification for a benchmark scenario."""

    name: str
    reranker: str = "noop"
    reranker_model: Optional[str] = None
    estimator: str = "baseline"
    batch_size: int = 1
    budget_docs: int = 10
    budget_tokens: int = 100_000
    budget_latency_ms: int = 60_000
    max_docs: int = 100

    @staticmethod
    def from_env() -> list:
        """Parse SCENARIO_CONFIG env var (JSON array) or return defaults."""
        raw = os.environ.get("SCENARIOS", "")
        if raw:
            return [ScenarioSpec(**s) for s in json.loads(raw)]
        return [
            ScenarioSpec(
                name="BM25 (baseline)", reranker="noop", budget_docs=0, max_docs=100
            ),
            ScenarioSpec(
                name="Static Rerank",
                reranker="cross-encoder",
                estimator="baseline",
                batch_size=20,
                budget_docs=20,
            ),
            ScenarioSpec(
                name="RAGtune (budget=10)",
                reranker="cross-encoder",
                estimator="similarity",
                batch_size=2,
                budget_docs=10,
            ),
        ]


def build_controller(spec: ScenarioSpec, retriever) -> RAGtuneController:
    """Build a RAGtuneController from a ScenarioSpec."""
    reranker_map = {
        "noop": NoOpReranker,
        "cross-encoder": lambda: CrossEncoderReranker(
            spec.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
    }
    estimator_map = {
        "baseline": BaselineEstimator,
        "similarity": SimilarityEstimator,
    }
    reranker = reranker_map[spec.reranker]()
    estimator = estimator_map[spec.estimator]()

    return RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(max_docs=spec.max_docs),
        scheduler=ActiveLearningScheduler(batch_size=spec.batch_size),
        estimator=estimator,
        budget=CostBudget.simple(
            docs=spec.budget_docs,
            tokens=spec.budget_tokens,
            latency=spec.budget_latency_ms,
        ),
    )
