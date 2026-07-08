"""
RAGtune Tool & Skill Retrieval Benchmark Runner
================================================
Evaluates RAGtune against BM25 baseline and static reranking on
ToolRet, SkillRet, and SRA-Bench datasets.

Usage:
    python scripts/run_tool_retrieval.py
    BENCHMARK=toolret SUBSET=apibank python scripts/run_tool_retrieval.py
    BENCHMARK=sra QUERIES=30 python scripts/run_tool_retrieval.py
"""

import os
import tempfile
import time
from typing import Dict, List, Tuple

import pandas as pd
from rich.console import Console

from ragtune.data.loaders import DataLoaderFactory
from ragtune.data.constants import Benchmark, TOOLRET_SUBSETS, SRA_BENCH_SUBSETS
from ragtune.evaluation.RetrievalEvaluator import RetrievalEvaluator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import NoOpReranker, CrossEncoderReranker
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.utils.config import config

_console = Console()


def print_header(msg):
    _console.print(f"[bold blue]{msg}[/bold blue]")


def print_step(msg):
    _console.print(f"[dim]{msg}[/dim]")


def print_success(msg):
    _console.print(f"[bold green]{msg}[/bold green]")


# --- Configuration (via environment variables) ---

BENCHMARK: str = os.environ.get("BENCHMARK", "toolret")  # toolret, skillret, sra
SUBSET: str = os.environ.get("SUBSET", "")  # specific subset name
QUERIES_PER_TASK: int = int(os.environ.get("QUERIES", "0"))  # 0 = all
CANDIDATES_TOP_K: int = int(os.environ.get("TOP_K", "100"))
EVAL_K: int = int(os.environ.get("EVAL_K", "10"))  # NDCG@k cutoff

_evaluator = RetrievalEvaluator(k_values=[EVAL_K])

# --- Data Loading ---


def load_task(
    benchmark: str, subset: str
) -> Tuple[
    Dict[str, str],  # corpus: {doc_id: {"text": str, "title": str}}
    Dict[str, str],  # queries: {query_id: query_text}
    Dict[str, Dict[str, int]],  # qrels:   {query_id: {doc_id: label}}
]:
    """Loads corpus, queries, and qrels via DataLoaderFactory."""
    print_step(f"Loading [{benchmark}] {subset}...")
    benchmark_name = {
        "toolret": Benchmark.TOOLRET,
        "skillret": Benchmark.SKILLRET,
        "sra": Benchmark.SRA_BENCH,
    }.get(benchmark)
    if benchmark_name is None:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    factory = DataLoaderFactory()
    loader = factory.create_dataloader(
        dataset_name=subset,
        benchmark_name=benchmark_name,
        n_queries=QUERIES_PER_TASK,
    )
    return loader.get_corpus(), loader.get_queries(), loader.get_qrels()


# --- Index Building ---


def build_retriever(
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
):
    """Builds a BM25 index over the corpus using PyTerrier."""
    import pyterrier as pt

    if not pt.java.started():
        pt.java.init()

    corpus_list = [{"docno": did, "text": d["text"]} for did, d in corpus.items()]
    print_step(f"Indexing {len(corpus_list)} documents...")
    idx_dir = os.path.join(tempfile.mkdtemp(), "idx")
    indexer = pt.IterDictIndexer(
        idx_dir, overwrite=True, meta={"docno": 128, "text": 4096}
    )
    index_ref = indexer.index(iter(corpus_list))
    bm25 = pt.terrier.Retriever(
        index_ref,
        wmodel="BM25",
        metadata=["docno", "text"],
        num_results=CANDIDATES_TOP_K,
    )

    from ragtune.adapters.pyterrier import PyTerrierRetriever

    return PyTerrierRetriever(bm25), bm25


# --- Evaluation ---


def score_results(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """Computes macro-averaged NDCG@10, Recall@10."""
    metrics = _evaluator.evaluate(qrels, results)
    return {
        "NDCG@10": round(metrics["ndcg"].get("NDCG@10", 0.0), 4),
        "Recall@10": round(metrics["recall"].get("Recall@10", 0.0), 4),
    }


# --- Scenario Execution ---


def build_scenarios(retriever) -> List[Tuple[str, RAGtuneController]]:
    """Builds list of (name, controller) scenarios for comparison."""
    return [
        (
            "BM25 (baseline)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=NoOpReranker(),
                assembler=GreedyAssembler(max_docs=CANDIDATES_TOP_K),
                scheduler=ActiveLearningScheduler(batch_size=1),
                estimator=BaselineEstimator(),
                budget=CostBudget.simple(docs=0, tokens=100_000, latency=60_000),
            ),
        ),
        (
            "Static Rerank (budget=20)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=CrossEncoderReranker(),
                assembler=GreedyAssembler(max_docs=CANDIDATES_TOP_K),
                scheduler=ActiveLearningScheduler(batch_size=20),
                estimator=BaselineEstimator(),
                budget=CostBudget.simple(docs=20, tokens=100_000, latency=600_000),
            ),
        ),
        (
            "RAGtune (budget=10)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=CrossEncoderReranker(),
                assembler=GreedyAssembler(max_docs=CANDIDATES_TOP_K),
                scheduler=ActiveLearningScheduler(batch_size=2),
                estimator=SimilarityEstimator(),
                budget=CostBudget.simple(docs=10, tokens=100_000, latency=600_000),
            ),
        ),
        (
            "RAGtune (budget=20)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=CrossEncoderReranker(),
                assembler=GreedyAssembler(max_docs=CANDIDATES_TOP_K),
                scheduler=ActiveLearningScheduler(batch_size=5),
                estimator=SimilarityEstimator(),
                budget=CostBudget.simple(docs=20, tokens=100_000, latency=600_000),
            ),
        ),
    ]


def run_controller_scenario(
    name: str,
    controller: RAGtuneController,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Runs a controller over all queries. Returns (results, avg_latency_ms)."""
    print_step(f"  Running [{name}]...")
    results: Dict[str, Dict[str, float]] = {}
    latencies: List[float] = []

    for qid, qtext in queries.items():
        t0 = time.time()
        try:
            output = controller.run(qtext)
            latencies.append((time.time() - t0) * 1000)
            results[qid] = {
                doc.id: 1.0 / (rank + 1) for rank, doc in enumerate(output.documents)
            }
        except Exception as e:
            _console.print(f"  [yellow]ERR {qid}: {e}[/yellow]")

    avg_latency = float(pd.Series(latencies).mean()) if latencies else 0.0
    return results, avg_latency


def run_bm25_baseline(
    bm25,
    queries: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Pure BM25 baseline — no reranking."""
    print_step("  Running [BM25 Baseline]...")
    results: Dict[str, Dict[str, float]] = {}
    for qid, qtext in queries.items():
        try:
            res = bm25.transform(pd.DataFrame([{"qid": qid, "query": qtext}]))
            results[qid] = {row["docno"]: row["score"] for _, row in res.iterrows()}
        except:
            pass
    return results


# --- Main ---


def main():
    config.set("retrieval.original_query_depth", CANDIDATES_TOP_K)
    print_header("RAGtune Tool & Skill Retrieval Benchmarks")
    print_step(
        f"Benchmark: {BENCHMARK}  |  Subset: {SUBSET or '(all)'}  |  Queries: {QUERIES_PER_TASK}  |  Top-K: {CANDIDATES_TOP_K}"
    )

    # Determine subsets to run
    if BENCHMARK == "toolret":
        subsets = [SUBSET] if SUBSET else TOOLRET_SUBSETS
    elif BENCHMARK == "skillret":
        subsets = [SUBSET] if SUBSET else ["test"]
    elif BENCHMARK == "sra":
        subsets = [SUBSET] if SUBSET else SRA_BENCH_SUBSETS
    else:
        _console.print(f"[red]Unknown benchmark: {BENCHMARK}[/red]")
        return

    all_rows: List[Dict] = []

    for subset in subsets:
        print_header(f"\n--- {BENCHMARK}/{subset} ---")
        corpus, queries, qrels = load_task(BENCHMARK, subset)
        n_qrels = sum(len(v) for v in qrels.values())
        print_step(
            f"Loaded {len(corpus)} docs, {len(queries)} queries, {n_qrels} qrel pairs"
        )

        retriever, bm25 = build_retriever(corpus, qrels)

        def _record(scenario_name: str, results: Dict, avg_latency: float = 0):
            metrics = score_results(results, qrels)
            all_rows.append(
                {
                    "benchmark": BENCHMARK,
                    "subset": subset,
                    "scenario": scenario_name,
                    **metrics,
                    "Avg Latency (ms)": round(avg_latency, 1),
                }
            )

        bm25_results = run_bm25_baseline(bm25, queries)
        _record("BM25 Baseline", bm25_results)

        for name, controller in build_scenarios(retriever):
            ctrl_results, avg_latency = run_controller_scenario(
                name, controller, queries, qrels
            )
            _record(name, ctrl_results, avg_latency)

    print_header("\nFINAL RESULTS")
    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))
    print_success("\nTool retrieval benchmark complete.")


if __name__ == "__main__":
    main()
