"""
RAGtune Tool & Skill Retrieval Benchmark Runner
================================================
Evaluates RAGtune against BM25 baseline and static reranking on
ToolRet, SkillRet, and SRA-Bench datasets.

Usage:
    python scripts/run_tool_retrieval.py
    BENCHMARK=toolret SUBSET=apibank python scripts/run_tool_retrieval.py
    INDEX_TYPE=dense EMBEDDING_MODEL=all-MiniLM-L6-v2 python scripts/run_tool_retrieval.py
    SCENARIOS='[{"name":"baseline","reranker":"noop"}]' python scripts/run_tool_retrieval.py
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
from ragtune.indexing import IndexFactory
from ragtune.components.scenarios import ScenarioSpec, build_controller
from ragtune.utils.config import config

_console = Console()


def print_header(msg):
    _console.print(f"[bold blue]{msg}[/bold blue]")


def print_step(msg):
    _console.print(f"[dim]{msg}[/dim]")


def print_success(msg):
    _console.print(f"[bold green]{msg}[/bold green]")


# --- Configuration (via environment variables) ---

BENCHMARK: str = os.environ.get("BENCHMARK", "toolret")
SUBSET: str = os.environ.get("SUBSET", "")
QUERIES_PER_TASK: int = int(os.environ.get("QUERIES", "0"))
CANDIDATES_TOP_K: int = int(os.environ.get("TOP_K", "100"))
EVAL_K: int = int(os.environ.get("EVAL_K", "10"))

_evaluator = RetrievalEvaluator(k_values=[EVAL_K])

# --- Data Loading ---


def load_task(
    benchmark: str, subset: str
) -> Tuple[
    Dict[str, Dict[str, str]],  # corpus: {doc_id: {"text": str, "title": str}}
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


def build_retriever(corpus):
    """Build retriever via IndexFactory (configurable: pyterrier/faiss/flex)."""
    index_type = os.environ.get("INDEX_TYPE", "pyterrier")
    print_step(f"Indexing {len(corpus)} documents ({index_type})...")

    # Build index using Mandeep's IndexFactory
    indexer = IndexFactory.create(index_type)
    index_path = os.path.join(tempfile.mkdtemp(), "index")
    indexer.build_from_corpus(corpus, index_path=index_path)

    # Create PyTerrierRetriever from the built index
    from ragtune.adapters.pyterrier import PyTerrierRetriever
    import pyterrier as pt

    if not pt.java.started():
        pt.java.init()
    index_ref = pt.IndexFactory.of(index_path)
    bm25 = pt.terrier.Retriever(
        index_ref,
        wmodel="BM25",
        metadata=["docno", "text"],
        num_results=CANDIDATES_TOP_K,
    )
    return PyTerrierRetriever(bm25), bm25


# --- Evaluation ---


def score_results(results, qrels):
    """Computes NDCG@k, Recall@k."""
    metrics = _evaluator.evaluate(qrels, results)
    return {
        f"NDCG@{EVAL_K}": round(metrics["ndcg"].get(f"NDCG@{EVAL_K}", 0.0), 4),
        f"Recall@{EVAL_K}": round(metrics["recall"].get(f"Recall@{EVAL_K}", 0.0), 4),
    }


# --- Scenario Execution ---


def build_scenarios(retriever) -> List[Tuple[str, object]]:
    """Builds list of (name, controller) from ScenarioSpec."""
    specs = ScenarioSpec.from_env()
    return [(spec.name, build_controller(spec, retriever)) for spec in specs]


def run_controller_scenario(name, controller, queries, qrels):
    """Runs a controller over all queries. Returns (results, avg_latency_ms)."""
    print_step(f"  Running [{name}]...")
    results, latencies = {}, []
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


def run_baseline(retriever, queries):
    """Pure retrieval baseline — no reranking (uses controller with budget_docs=0)."""
    print_step("  Running [Baseline]...")
    baseline_spec = ScenarioSpec(name="baseline", reranker="noop", budget_docs=0)
    controller = build_controller(baseline_spec, retriever)
    results = {}
    for qid, qtext in queries.items():
        try:
            output = controller.run(qtext)
            results[qid] = {
                doc.id: 1.0 / (rank + 1) for rank, doc in enumerate(output.documents)
            }
        except Exception as e:
            _console.print(f"  [yellow]ERR {qid}: {e}[/yellow]")
    return results


# --- Main ---


def main():
    config.set("retrieval.original_query_depth", CANDIDATES_TOP_K)
    print_header("RAGtune Tool & Skill Retrieval Benchmarks")
    print_step(
        f"Benchmark: {BENCHMARK}  |  Subset: {SUBSET or '(all)'}  |  Queries: {QUERIES_PER_TASK}  |  Top-K: {CANDIDATES_TOP_K}"
    )

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

        retriever = build_retriever(corpus)

        def _record(scenario_name, results, avg_latency=0):
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

        baseline_results = run_baseline(retriever, queries)
        _record("Baseline", baseline_results)

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
