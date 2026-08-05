from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ragtune.tuning.evaluator import EvalDataset, TrialEvaluator
from ragtune.tuning.pruners import CostPruner, ParetoPruner, RuntimePruner
from ragtune.tuning.search_space import RAGtuneSearchSpace
from ragtune.tuning.study_config import TuningStudyConfig


def run_study(
    config: TuningStudyConfig,
    fixed_retriever: Any,
    eval_dataset: EvalDataset,
) -> Any:
    """
    Run a Bayesian multi-objective optimization study.

    Returns the completed optuna.Study.  Pareto-optimal configs can be
    extracted with extract_pareto_configs().

    Parameters
    ----------
    config
        Study configuration (trial counts, pruner thresholds, etc.).
    fixed_retriever
        A pre-built BaseRetriever instance shared across all trials.
        Only the reranking / scheduling / estimation pipeline is tuned.
    eval_dataset
        The queries and qrels used to score each trial.
    """
    import optuna
    from optuna.samplers import TPESampler

    sampler = TPESampler(
        multivariate=True,
        constant_liar=True,
        n_startup_trials=config.n_startup_trials,
        seed=config.seed,
    )

    study = optuna.create_study(
        study_name=config.name,
        directions=["maximize", "minimize"],
        sampler=sampler,
        storage=config.storage_url,
        load_if_exists=True,
    )

    search_space = RAGtuneSearchSpace(**config.search_space_overrides)

    pruners = [
        CostPruner(
            max_mean_rerank_docs=config.max_mean_rerank_docs,
            warmup_steps=3,
        ),
        RuntimePruner(
            max_trial_seconds=config.max_trial_seconds,
            warmup_steps=3,
        ),
        ParetoPruner(
            study=study,
            warmup_trials=config.pareto_warmup_trials,
            zscore=1.645,
        ),
    ]

    evaluator = TrialEvaluator(
        dataset=eval_dataset,
        n_eval_queries=config.n_eval_queries,
        pruners=pruners,
    )

    def objective(trial: Any) -> tuple:
        params = search_space.sample(trial)

        try:
            controller = search_space.build_controller(params, fixed_retriever)
        except Exception as exc:
            # Component construction failed (e.g. model not available).
            # Return worst-case values so the trial is completed but ignored.
            trial.set_user_attr("build_error", str(exc))
            return 0.0, float("inf")

        retrieval_overrides = search_space.to_retrieval_overrides(params)
        objectives = evaluator.evaluate(controller, trial, retrieval_overrides)

        trial.set_user_attr("latency_ms", objectives.latency_ms)
        trial.set_user_attr("queries_evaluated", objectives.queries_evaluated)

        return objectives.ndcg_at_10, objectives.rerank_docs

    study.optimize(
        objective,
        n_trials=config.n_trials,
        n_jobs=config.n_parallel_workers,
        catch=(Exception,),
    )

    return study


def extract_pareto_configs(
    study: Any,
    search_space: RAGtuneSearchSpace,
    output_dir: str,
) -> List[str]:
    """
    Write each Pareto-front trial's pipeline config to output_dir as YAML.

    Files are named: pareto_<trial_number>_ndcg<val>_cost<val>.yaml
    Returns a list of written file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pareto_trials = study.best_trials  # non-dominated trials, built-in for multi-objective
    paths = []

    for trial in pareto_trials:
        if trial.values is None:
            continue
        ndcg_val = trial.values[0]
        cost_val = trial.values[1]

        pipeline_dict = _params_to_pipeline_dict(trial.params)

        filename = (
            f"pareto_trial_{trial.number}"
            f"_ndcg{ndcg_val:.3f}"
            f"_cost{cost_val:.0f}"
            ".yaml"
        )
        path = os.path.join(output_dir, filename)

        with open(path, "w") as f:
            yaml.dump({"pipeline": pipeline_dict}, f, sort_keys=False)

        paths.append(path)

    return paths


def _params_to_pipeline_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat Optuna param dict to a PipelineConfig-compatible nested dict."""
    reranker_type = params.get("reranker_type", "noop")
    reranker_params: Dict[str, Any] = {}
    if reranker_type == "cross-encoder":
        reranker_params["model_name"] = params.get("ce_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    elif reranker_type == "monot5":
        reranker_params["model_name"] = params.get("monot5_model", "castorini/monot5-base-msmarco")
        reranker_params["batch_size"] = int(params.get("monot5_batch_size", 16))
    elif reranker_type == "llm":
        reranker_params["model_name"] = params.get("llm_reranker_model", "gpt-4o-mini")

    reformulator_type = params.get("reformulator_type", "identity")
    reformulator_params: Dict[str, Any] = {}
    if reformulator_type == "llm_rewrite":
        reformulator_params["model_name"] = params.get("reformulator_model", "gpt-4o-mini")
    elif reformulator_type == "reformir":
        reformulator_params["model"] = params.get("reformulator_model", "gpt-4o-mini")
        reformulator_params["n_variants"] = params.get("reformulator_n_variants", 5)

    estimator_type = params.get("estimator_type", "baseline")
    estimator_params: Dict[str, Any] = {}
    if estimator_type == "similarity":
        estimator_params["model_name"] = params.get("similarity_model", "all-MiniLM-L6-v2")
    elif estimator_type == "reformir":
        estimator_params["min_reranked_for_regression"] = params.get("min_reranked_for_regression", 3)

    scheduler_type = params.get("scheduler_type", "graceful-degradation")
    scheduler_params: Dict[str, Any] = {"batch_size": params.get("scheduler_batch_size", 5)}
    if scheduler_type == "graceful-degradation":
        scheduler_params["llm_limit"] = params.get("gd_llm_limit", 3)
        scheduler_params["cross_encoder_limit"] = params.get("gd_ce_limit", 10)

    feedback_type = params.get("feedback_type", "none")
    feedback_cfg: Optional[Dict[str, Any]] = None
    if feedback_type != "none":
        feedback_params: Dict[str, Any] = {}
        if feedback_type == "budget-stop":
            feedback_params["token_threshold"] = params.get("budget_stop_token_threshold", 0.9)
        feedback_cfg = {"type": feedback_type, "params": feedback_params}

    pipeline: Dict[str, Any] = {
        "name": "ragtune-pareto-config",
        "components": {
            "retriever": {"type": "pyterrier"},
            "reformulator": {"type": reformulator_type, "params": reformulator_params},
            "reranker": {"type": reranker_type, "params": reranker_params},
            "estimator": {"type": estimator_type, "params": estimator_params},
            "scheduler": {"type": scheduler_type, "params": scheduler_params},
            "assembler": {
                "type": "greedy",
                "params": {"max_docs": params.get("assembler_max_docs", 10)},
            },
        },
        "budget": {
            "limits": {
                "rerank_docs": params.get("budget_rerank_docs", 50),
                "reformulations": params.get("budget_reformulations", 1),
                "retrieval_calls": 20,
                "tokens": 100000,
                "latency_ms": 30000,
            }
        },
    }

    if feedback_cfg:
        pipeline["feedback"] = feedback_cfg

    return pipeline
