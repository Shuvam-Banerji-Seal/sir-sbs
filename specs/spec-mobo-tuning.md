This is the executable specification for **RAGtune MOBO Tuning (v0.3)** — Bayesian multi-objective optimization of the end-to-end RAGtune pipeline. It defines the search space contract, evaluation protocol, pruning interface, and optimizer loop, drawing directly from the syftr architecture (arxiv 2505.20266).

---

## 1. Motivation & Goals

RAGtune's controller has ~20 tunable hyperparameters across six component types. Grid search is intractable. The goal is to discover the **Pareto frontier** of (retrieval quality, compute cost) configurations automatically, so operators can pick a point on that frontier matching their latency/cost budget.

**Primary objectives:**
- Maximize retrieval quality — NDCG@10 on a fixed evaluation dataset
- Minimize compute cost — total rerank documents consumed (`consumed["rerank_docs"]`)

**Secondary deliverables:**
- A saved Pareto front of YAML configs, ready for `ragtune run`
- A `ragtune tune study.yaml` CLI command
- Pruning that kills dominated/over-budget trials early

---

## 2. Directory Structure

```text
src/ragtune/tuning/
├── __init__.py
├── search_space.py      # RAGtuneSearchSpace: distributions + sampling
├── evaluator.py         # TrialEvaluator: run controller on benchmark, compute objectives
├── pruners.py           # CostPruner, RuntimePruner, ParetoPruner
├── optimizer.py         # Optuna study setup, trial loop, Pareto extraction
└── study_config.py      # TuningStudyConfig Pydantic model (YAML-loadable)

src/ragtune/cli/
└── tune.py              # `ragtune tune` command (new Typer sub-command)

tests/unit/tuning/
├── test_search_space.py
├── test_evaluator.py
└── test_pruners.py

tests/integration/tuning/
└── test_optimizer_smoke.py  # 5-trial smoke test on synthetic data
```

---

## 3. Search Space (`search_space.py`)

### 3A. Interface Contract

Every component of the search space implements `SearchSpaceMixin`:

```python
from abc import ABC, abstractmethod
from typing import Dict
from optuna import Trial
from optuna.distributions import BaseDistribution

ParamDict = Dict[str, object]

class SearchSpaceMixin(ABC):
    @abstractmethod
    def build_distributions(self) -> Dict[str, BaseDistribution]:
        """Return all Optuna distributions this component contributes."""

    @abstractmethod
    def get_cardinality(self) -> int:
        """Return upper bound on distinct configurations (product of option counts)."""

    def sample(self, trial: Trial) -> ParamDict:
        """Default implementation: call trial.suggest_* for each distribution."""
        params = {}
        for name, dist in self.build_distributions().items():
            params[name] = trial._suggest(name, dist)
        return params
```

### 3B. Concrete Search Space

`RAGtuneSearchSpace` inherits `SearchSpaceMixin` and covers the full pipeline. Conditional parameters are sampled only when their parent discrete choice is active — this mirrors syftr's `HierarchicalTPESampler` approach and keeps the parameter space clean.

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from optuna.distributions import (
    CategoricalDistribution, IntDistribution, FloatDistribution
)

RETRIEVER_TYPES     = ["bm25", "dpr", "colbert", "hybrid-bm25-dpr"]
REFORMULATOR_TYPES  = ["identity", "llm_rewrite", "reformir"]
RERANKER_TYPES      = ["noop", "cross-encoder", "llm", "monot5", "ollama-listwise"]
ESTIMATOR_TYPES     = ["baseline", "utility", "similarity", "reformir"]
SCHEDULER_TYPES     = ["active-learning", "graceful-degradation"]
FEEDBACK_TYPES      = ["none", "budget-stop", "reformir-convergence"]

REFORMULATOR_MODELS = ["gpt-4o-mini", "gpt-4o", "claude-3-haiku-20240307"]
CE_MODELS           = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/qnli-distilroberta-base",
]
LLM_RERANKER_MODELS = ["gpt-4o-mini", "gpt-4o", "claude-3-haiku-20240307"]
MONOT5_MODELS       = ["castorini/monot5-base-msmarco", "castorini/monot5-large-msmarco"]
SIMILARITY_MODELS   = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]

class RAGtuneSearchSpace(BaseModel, SearchSpaceMixin):
    retriever_types:    List[str] = Field(default=RETRIEVER_TYPES)
    reformulator_types: List[str] = Field(default=REFORMULATOR_TYPES)
    reranker_types:     List[str] = Field(default=RERANKER_TYPES)
    estimator_types:    List[str] = Field(default=ESTIMATOR_TYPES)
    scheduler_types:    List[str] = Field(default=SCHEDULER_TYPES)
    feedback_types:     List[str] = Field(default=FEEDBACK_TYPES)

    # Numerical ranges
    original_query_depth_range:      tuple[int, int] = (5, 100)
    depth_per_reformulation_range:   tuple[int, int] = (1, 30)
    max_pool_size_range:             tuple[int, int] = (10, 300)
    near_duplicate_threshold_range:  tuple[float, float] = (0.5, 0.95)
    scheduler_batch_size_range:      tuple[int, int] = (1, 30)
    assembler_max_docs_range:        tuple[int, int] = (3, 30)
    budget_rerank_docs_range:        tuple[int, int] = (5, 200)
    budget_reformulations_range:     tuple[int, int] = (0, 5)

    def build_distributions(self) -> Dict[str, BaseDistribution]:
        return {
            # --- Discrete component selection ---
            "retriever_type":    CategoricalDistribution(self.retriever_types),
            "reformulator_type": CategoricalDistribution(self.reformulator_types),
            "reranker_type":     CategoricalDistribution(self.reranker_types),
            "estimator_type":    CategoricalDistribution(self.estimator_types),
            "scheduler_type":    CategoricalDistribution(self.scheduler_types),
            "feedback_type":     CategoricalDistribution(self.feedback_types),

            # --- Always-active continuous params ---
            "original_query_depth":     IntDistribution(*self.original_query_depth_range, log=True),
            "depth_per_reformulation":  IntDistribution(*self.depth_per_reformulation_range, log=True),
            "max_pool_size":            IntDistribution(*self.max_pool_size_range, log=True),
            "near_duplicate_threshold": FloatDistribution(*self.near_duplicate_threshold_range),
            "scheduler_batch_size":     IntDistribution(*self.scheduler_batch_size_range, log=True),
            "assembler_max_docs":       IntDistribution(*self.assembler_max_docs_range),
            "budget_rerank_docs":       IntDistribution(*self.budget_rerank_docs_range, log=True),
            "budget_reformulations":    IntDistribution(*self.budget_reformulations_range),

            # --- Conditional: reformulator model (sampled always; ignored when reformulator=identity) ---
            "reformulator_model":  CategoricalDistribution(REFORMULATOR_MODELS),
            "reformulator_n_variants": IntDistribution(1, 10),

            # --- Conditional: reranker sub-params ---
            "ce_model":            CategoricalDistribution(CE_MODELS),
            "llm_reranker_model":  CategoricalDistribution(LLM_RERANKER_MODELS),
            "monot5_model":        CategoricalDistribution(MONOT5_MODELS),
            "monot5_batch_size":   CategoricalDistribution([4, 8, 16, 32]),

            # --- Conditional: estimator sub-params ---
            "similarity_model":    CategoricalDistribution(SIMILARITY_MODELS),
            "min_reranked_for_regression": IntDistribution(1, 10),

            # --- Conditional: scheduler sub-params ---
            "gd_llm_limit":  IntDistribution(1, 10),
            "gd_ce_limit":   IntDistribution(1, 30),

            # --- Conditional: feedback sub-params ---
            "budget_stop_token_threshold":        FloatDistribution(0.7, 0.99),
            "reformir_convergence_threshold":     FloatDistribution(0.001, 0.1, log=True),
        }

    def get_cardinality(self) -> int:
        discrete = (
            len(self.retriever_types)
            * len(self.reformulator_types)
            * len(self.reranker_types)
            * len(self.estimator_types)
            * len(self.scheduler_types)
            * len(self.feedback_types)
        )
        return discrete  # continuous dims add infinite sub-cardinality on top
```

### 3C. Config Assembly

`RAGtuneSearchSpace.to_pipeline_config(params: ParamDict) -> PipelineConfig` converts a flat `ParamDict` (Optuna trial values) into a `PipelineConfig` suitable for `build_controller()`. Conditional fields are applied based on the value of the parent categorical:

```python
def to_pipeline_config(params: ParamDict) -> PipelineConfig:
    reranker_cfg = _build_reranker_config(
        params["reranker_type"],
        ce_model=params["ce_model"],
        llm_model=params["llm_reranker_model"],
        monot5_model=params["monot5_model"],
        monot5_batch=params["monot5_batch_size"],
    )
    reformulator_cfg = _build_reformulator_config(
        params["reformulator_type"],
        model=params["reformulator_model"],
        n_variants=params["reformulator_n_variants"],
    )
    # ... etc for all components
    return PipelineConfig(
        budget=BudgetConfig(limits={
            "rerank_docs":      params["budget_rerank_docs"],
            "reformulations":   params["budget_reformulations"],
        }),
        components=PipelineComponents(
            retriever=ComponentConfig(type=params["retriever_type"]),
            reformulator=reformulator_cfg,
            reranker=reranker_cfg,
            estimator=ComponentConfig(type=params["estimator_type"], params=_estimator_params(params)),
            scheduler=ComponentConfig(type=params["scheduler_type"], params=_scheduler_params(params)),
            assembler=ComponentConfig(type="greedy", params={"max_docs": params["assembler_max_docs"]}),
        ),
        feedback=_build_feedback_config(params),
    )
```

---

## 4. Evaluator (`evaluator.py`)

### 4A. Interface Contract

```python
from dataclasses import dataclass
from typing import Tuple
import optuna

@dataclass
class TrialObjectives:
    ndcg_at_10: float    # objective 1 — maximize
    rerank_docs: float   # objective 2 — minimize (proxy for API cost)
    latency_ms: float    # logged as user attribute, not an objective by default
    queries_evaluated: int

class TrialEvaluator:
    def __init__(
        self,
        dataset: EvalDataset,        # queries + qrels
        qrel_path: str,
        n_eval_queries: int = 200,
        pruners: List["SyftrPruner"] = (),
    ): ...

    def evaluate(
        self,
        controller: RAGtuneController,
        trial: optuna.Trial,
    ) -> TrialObjectives:
        """
        Runs controller on each query in the evaluation set.
        Calls trial.report() after each batch_report_interval queries
        so pruners can abort early.
        Raises optuna.TrialPruned if any pruner fires.
        """
```

### 4B. Evaluation Loop

```python
def evaluate(self, controller, trial) -> TrialObjectives:
    ndcg_scores, rerank_costs, latencies = [], [], []

    for i, (query, qrels) in enumerate(self.dataset.iter_queries()):
        t0 = time.monotonic()
        output = controller.run(query)
        elapsed_ms = (time.monotonic() - t0) * 1000

        ndcg = compute_ndcg(output.documents, qrels, k=10)
        cost = output.final_budget_state.consumed.get("rerank_docs", 0)

        ndcg_scores.append(ndcg)
        rerank_costs.append(cost)
        latencies.append(elapsed_ms)

        # Report intermediate values for pruning
        step = i + 1
        trial.report(float(np.mean(ndcg_scores)), step=step)
        for pruner in self.pruners:
            pruner.report(
                step=step,
                n_total=self.n_eval_queries,
                obj1=float(np.mean(ndcg_scores)),
                obj2=float(np.mean(rerank_costs)),
                elapsed_ms=sum(latencies),
            )
            if pruner.should_prune():
                raise optuna.TrialPruned(f"Pruned by {type(pruner).__name__} at step {step}")

    return TrialObjectives(
        ndcg_at_10=float(np.mean(ndcg_scores)),
        rerank_docs=float(np.mean(rerank_costs)),
        latency_ms=float(np.mean(latencies)),
        queries_evaluated=len(ndcg_scores),
    )
```

### 4C. Dataset Contract

```python
@dataclass
class EvalDataset:
    name: str              # e.g. "trec-covid", "beir/nfcorpus"
    queries: List[str]
    qrels: Dict[str, Dict[str, int]]  # query_id → {doc_id → relevance}

    def iter_queries(self) -> Iterator[Tuple[str, Dict[str, int]]]:
        ...

    @classmethod
    def from_beir(cls, dataset_name: str, split: str = "test") -> "EvalDataset":
        ...

    @classmethod
    def from_pyterrier(cls, dataset_name: str, topics_variant: str) -> "EvalDataset":
        ...
```

NDCG computation reuses PyTerrier's `measures.nDCG@10` already used in `examples/demo_pyterrier_budgets.py`.

---

## 5. Pruners (`pruners.py`)

Three pruners mirror syftr's design. All implement `SyftrPruner`:

```python
class SyftrPruner(ABC):
    @abstractmethod
    def report(self, step: int, n_total: int, obj1: float, obj2: float, elapsed_ms: float) -> None:
        """Called after each query evaluation."""

    @abstractmethod
    def should_prune(self) -> bool:
        """Return True if the current trial should be aborted."""
```

### 5A. CostPruner

Aborts trials where projected total rerank cost exceeds a threshold, based on current average cost × remaining queries:

```python
class CostPruner(SyftrPruner):
    def __init__(self, max_mean_rerank_docs: float, warmup_steps: int = 3):
        self.max_mean_rerank_docs = max_mean_rerank_docs
        self.warmup_steps = warmup_steps
        self._step = 0
        self._obj2_sum = 0.0

    def report(self, step, n_total, obj1, obj2, elapsed_ms):
        self._step = step
        self._obj2_sum += obj2   # obj2 is per-query rerank_docs consumed

    def should_prune(self) -> bool:
        if self._step < self.warmup_steps:
            return False
        mean_cost = self._obj2_sum / self._step
        return mean_cost > self.max_mean_rerank_docs
```

### 5B. RuntimePruner

Projects total evaluation time and aborts trials that would exceed a wall-clock timeout:

```python
class RuntimePruner(SyftrPruner):
    def __init__(self, max_trial_seconds: float, warmup_steps: int = 3):
        self.max_trial_seconds = max_trial_seconds
        self.warmup_steps = warmup_steps
        self._step = 0
        self._n_total = 0
        self._elapsed_ms = 0.0

    def report(self, step, n_total, obj1, obj2, elapsed_ms):
        self._step = step
        self._n_total = n_total
        self._elapsed_ms = elapsed_ms

    def should_prune(self) -> bool:
        if self._step < self.warmup_steps or self._step == 0:
            return False
        projected_s = (self._elapsed_ms / 1000) * (self._n_total / self._step)
        return projected_s > self.max_trial_seconds
```

### 5C. ParetoPruner

The highest-leverage pruner. Maintains the current Pareto front across completed trials and prunes the current trial if its best-case outcome (upper confidence bound on NDCG, lower confidence bound on cost) is dominated by any Pareto-front point:

```python
class ParetoPruner(SyftrPruner):
    def __init__(
        self,
        study: optuna.Study,
        warmup_trials: int = 30,
        zscore: float = 1.645,   # 90% confidence
    ):
        self.study = study
        self.warmup_trials = warmup_trials
        self.zscore = zscore
        self._step = 0
        self._n_total = 0
        self._obj1_values: List[float] = []
        self._obj2_values: List[float] = []
        self._prune = False

    def report(self, step, n_total, obj1, obj2, elapsed_ms):
        self._step = step
        self._n_total = n_total
        self._obj1_values.append(obj1)
        self._obj2_values.append(obj2)

    def should_prune(self) -> bool:
        if self._prune:
            return True
        completed = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < self.warmup_trials or self._step < 5:
            return False

        n = len(self._obj1_values)
        mean_obj1 = float(np.mean(self._obj1_values))
        mean_obj2 = float(np.mean(self._obj2_values))

        # Optimistic best-case: NDCG upper bound, cost lower bound
        obj1_conf = self.zscore * np.sqrt(mean_obj1 * (1 - mean_obj1) / n)
        obj2_conf = self.zscore * (float(np.std(self._obj2_values)) / np.sqrt(n))

        optimistic_obj1 = min(mean_obj1 + obj1_conf, 1.0)
        optimistic_obj2 = max(mean_obj2 - obj2_conf, 0.0)

        # Check if dominated by any Pareto-front trial
        pareto = self._get_pareto_front(completed)
        for (p_obj1, p_obj2) in pareto:
            # dominated: Pareto point is at least as good on both objectives
            if p_obj1 >= optimistic_obj1 and p_obj2 <= optimistic_obj2:
                self._prune = True
                return True
        return False

    def _get_pareto_front(self, trials) -> List[Tuple[float, float]]:
        """Return (obj1, obj2) pairs on the Pareto front (maximize obj1, minimize obj2)."""
        points = [(t.values[0], t.values[1]) for t in trials if t.values]
        pareto = []
        for p in points:
            dominated = any(
                q[0] >= p[0] and q[1] <= p[1] and q != p
                for q in points
            )
            if not dominated:
                pareto.append(p)
        return pareto
```

---

## 6. Optimizer (`optimizer.py`)

```python
import optuna
from optuna.samplers import TPESampler

def run_study(config: "TuningStudyConfig") -> optuna.Study:
    sampler = TPESampler(
        multivariate=True,
        constant_liar=True,             # optimistic for parallel workers
        n_startup_trials=config.n_startup_trials,
        seed=config.seed,
    )
    study = optuna.create_study(
        study_name=config.name,
        directions=["maximize", "minimize"],
        sampler=sampler,
        storage=config.storage_url,     # None → in-memory; sqlite:///path → persistent
        load_if_exists=True,
    )

    search_space = RAGtuneSearchSpace(**config.search_space_overrides)
    evaluator = TrialEvaluator(
        dataset=EvalDataset.from_config(config.dataset),
        n_eval_queries=config.n_eval_queries,
        pruners=[
            CostPruner(max_mean_rerank_docs=config.max_mean_rerank_docs),
            RuntimePruner(max_trial_seconds=config.max_trial_seconds),
            ParetoPruner(study=study, warmup_trials=config.pareto_warmup_trials),
        ],
    )

    def objective(trial: optuna.Trial):
        params = search_space.sample(trial)
        pipeline_config = search_space.to_pipeline_config(params)
        controller = build_controller(pipeline_config)
        objectives = evaluator.evaluate(controller, trial)

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


def extract_pareto_configs(study: optuna.Study, output_dir: str) -> List[str]:
    """
    Write each Pareto-front trial's PipelineConfig to output_dir as a YAML file.
    Returns list of written file paths.
    """
    pareto_trials = study.best_trials  # Optuna returns non-dominated trials
    paths = []
    for trial in pareto_trials:
        config = RAGtuneSearchSpace().to_pipeline_config(trial.params)
        path = f"{output_dir}/pareto_trial_{trial.number}_ndcg{trial.values[0]:.3f}_cost{trial.values[1]:.0f}.yaml"
        write_yaml(config, path)
        paths.append(path)
    return paths
```

---

## 7. Study Config (`study_config.py`)

YAML-loadable via Pydantic Settings:

```python
class TuningStudyConfig(BaseSettings):
    name: str                          # study name (also used as DB key)
    dataset: DatasetConfig             # dataset name + split
    n_trials: int = 200
    n_startup_trials: int = 50         # random exploration before TPE kicks in
    n_parallel_workers: int = 1        # sequential by default
    n_eval_queries: int = 200
    seed: int = 42
    storage_url: Optional[str] = None  # None=in-memory, sqlite:///path=persistent

    # Pruner settings
    max_mean_rerank_docs: float = 50.0
    max_trial_seconds: float = 120.0
    pareto_warmup_trials: int = 30

    # Output
    output_dir: str = "tuning_results"

    # Optional overrides for search space (e.g. restrict reranker_types to subset)
    search_space_overrides: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        env_prefix = "RAGTUNE_TUNE_"

    @classmethod
    def from_yaml(cls, path: str) -> "TuningStudyConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

**Example study YAML:**

```yaml
name: trec-covid-mobo-v1
dataset:
  name: trec-covid
  split: test
n_trials: 300
n_startup_trials: 60
n_eval_queries: 200
n_parallel_workers: 4
storage_url: "sqlite:///tuning_results/trec-covid-mobo-v1.db"
max_mean_rerank_docs: 40.0
max_trial_seconds: 90.0
pareto_warmup_trials: 40
output_dir: tuning_results/trec-covid-mobo-v1
search_space_overrides:
  reranker_types: ["noop", "cross-encoder", "llm"]
  estimator_types: ["baseline", "similarity", "reformir"]
```

---

## 8. CLI Integration (`cli/tune.py`)

New Typer sub-command added to the existing CLI:

```python
import typer
from ragtune.tuning.study_config import TuningStudyConfig
from ragtune.tuning.optimizer import run_study, extract_pareto_configs

tune_app = typer.Typer()

@tune_app.command()
def tune(
    study_yaml: str = typer.Argument(..., help="Path to study YAML config"),
    resume: bool = typer.Option(False, help="Resume existing study from storage_url"),
    dry_run: bool = typer.Option(False, help="Validate config and estimate cardinality, do not run"),
):
    """Run Bayesian multi-objective optimization over the RAGtune pipeline."""
    config = TuningStudyConfig.from_yaml(study_yaml)

    if dry_run:
        ss = RAGtuneSearchSpace(**config.search_space_overrides)
        typer.echo(f"Search space cardinality (discrete): {ss.get_cardinality():,}")
        typer.echo(f"Planned trials: {config.n_trials}")
        typer.echo(f"Dataset: {config.dataset.name} ({config.n_eval_queries} queries)")
        return

    study = run_study(config)
    paths = extract_pareto_configs(study, config.output_dir)

    typer.echo(f"\nPareto front: {len(paths)} configurations")
    for path in paths:
        trial_num = int(path.split("trial_")[1].split("_")[0])
        trial = study.trials[trial_num]
        typer.echo(f"  {path}  (NDCG@10={trial.values[0]:.3f}, rerank_docs={trial.values[1]:.0f})")
```

Register in `src/ragtune/cli/main.py`:

```python
from ragtune.cli.tune import tune_app
app.add_typer(tune_app, name="tune")
```

---

## 9. Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
tuning = [
    "optuna>=3.6",
    "numpy>=1.24",
    "paretoset>=1.2",
    "kneed>=0.8",          # knee-point detection for Pareto curve
    "pyyaml>=6.0",         # already present
]
```

Install with: `pip install -e ".[tuning]"`

---

## 10. Testing Requirements

Per CLAUDE.md conventions:

### Unit Tests (`tests/unit/tuning/`)

| Test | What it verifies |
|---|---|
| `test_search_space.py::test_build_distributions_returns_all_keys` | All expected keys present |
| `test_search_space.py::test_to_pipeline_config_identity_reformulator` | Conditional fields ignored when parent=identity |
| `test_search_space.py::test_to_pipeline_config_noop_reranker` | Reranker model keys ignored when reranker=noop |
| `test_search_space.py::test_cardinality_positive` | Cardinality > 0 |
| `test_evaluator.py::test_evaluate_calls_trial_report` | `trial.report()` called once per query |
| `test_evaluator.py::test_evaluate_raises_trial_pruned_when_pruner_fires` | `TrialPruned` raised when `should_prune()=True` |
| `test_pruners.py::test_cost_pruner_warmup` | No prune before warmup steps |
| `test_pruners.py::test_cost_pruner_fires_when_over_limit` | Prunes when mean cost > max |
| `test_pruners.py::test_runtime_pruner_fires_on_projection` | Prunes when projected time > limit |
| `test_pruners.py::test_pareto_pruner_no_prune_before_warmup` | No prune before warmup_trials |
| `test_pruners.py::test_pareto_pruner_prunes_dominated_trial` | Prunes when dominated by Pareto front |

All tests use `FakeController` / `FakeTrial` fakes — no real API calls.

### Integration Test (`tests/integration/tuning/`)

`test_optimizer_smoke.py`: Runs a 5-trial study (`n_trials=5, n_startup_trials=3`) using an `InMemoryRetriever` with synthetic documents and a fake qrel set. Verifies:
1. Study completes without exception
2. `study.best_trials` returns at least one trial
3. `extract_pareto_configs()` writes YAML files that pass `ragtune validate`

---

## 11. Verification Checklist

1. **Search space completeness**: `RAGtuneSearchSpace().build_distributions()` returns keys for all component types; `to_pipeline_config` round-trips cleanly through `PipelineConfig` validation.
2. **Conditional gating**: When `reranker_type=noop`, `to_pipeline_config` does not pass any reranker model to the component; when `reformulator_type=identity`, no LLM model is used.
3. **Budget as constraint**: `budget_rerank_docs` and `budget_reformulations` are written into `BudgetConfig.limits`; the controller respects them as hard stops.
4. **Pruner fires correctly**: CostPruner fires when mean rerank_docs exceeds `max_mean_rerank_docs`; ParetoPruner does not fire before `warmup_trials` completes.
5. **Pareto output is usable**: Each YAML in `output_dir` passes `ragtune validate` and can be run with `ragtune run`.
6. **CLI dry-run**: `ragtune tune study.yaml --dry-run` prints cardinality and exits without running any trials.
7. **Persistence**: A study interrupted mid-run resumes correctly from `storage_url` (SQLite) when `--resume` is passed.
8. **No real API calls in unit tests**: All unit tests use fakes from `tests/conftest.py`; integration smoke test uses `InMemoryRetriever` only.
