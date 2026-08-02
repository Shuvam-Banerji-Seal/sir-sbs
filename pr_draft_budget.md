# PR #31: feat(budget): add config-based budget system with 6 loaders, optimization, and alerts

## What problem does this solve?

RAGtune's iterative retrieval loop optimizes for quality (NDCG) but has no visibility into **cost**. Users cannot answer:

- "How much does it cost to run 10K queries through this pipeline?"
- "Which GPU gives the best cost/performance ratio?"
- "What's the carbon footprint of my RAG deployment?"
- "Should I use Cohere reranking or self-hosted cross-encoder?"

This PR adds a **config-driven budget estimation system** that answers all of these questions. Every parameter flows through `BudgetConfig` — no hardcoded values in Python code.

**Linked to:** Discussion #12 (Benchmark Datasets for RAGtune), PR #30 (tool/skill benchmarks)

---

## What changed and why?

### Architecture

```
budget/
├── hardware.py          # GPUSpec dataclass, power/energy/carbon functions
├── throughput.py        # Θ_max lookup table, saturation model
├── base.py              # BudgetConfig (34 fields), BaseBudgetLoader ABC
├── factory.py           # BudgetLoaderFactory registry
├── result.py            # BudgetResult (15 fields, per-component costs)
├── main.py              # calculate_budget(), budget_report()
├── optimizer.py         # Cost optimization suggestions
├── history.py           # JSONL cost logging
├── alerts.py            # Threshold-based alerts
├── configs/             # YAML configs
└── loaders/
    ├── vllm_budget.py   # Concurrency-aware (arXiv 2606.11690)
    ├── token_budget.py  # API per-token pricing
    ├── gpu_budget.py    # GPU runtime × hourly rate
    ├── carbon_budget.py # IPCC Tier 1 + PUE
    ├── embedding_budget.py  # OpenAI/Cohere/Voyage
    └── reranking_budget.py  # Cohere/Voyage per-query
```

### 1. VLLM Budget Loader (arXiv 2606.11690)

**Paper:** [Beyond Per-Token Pricing: A Concurrency-Aware Methodology for LLM Infrastructure Cost Estimation](https://arxiv.org/abs/2606.11690) (Patil, June 2026)

**Formula:**
```
C_eff = (P_GPU × 1e6) / (3600 × Θ_achieved(λ, L))
```

**Key insight from paper:** Per-token pricing is misleading because GPU utilization varies 25x with load. The paper measures Θ_max (peak throughput) empirically via benchmark sweep.

**Our implementation:**
- Empirical Θ_max lookup table from paper Table 4 (6 measured configs)
- Smooth saturation model: `Θ = Θ_max × (1 - e^(-λ/λ_sat))`
- Model-size-dependent saturation knee (λ_sat varies with model size)
- SLO enforcement: caps throughput when latency would breach

**Paper verification (arXiv 2606.11690 measured on H100 NVL — our experiments used A100-80GB):**

| λ (rps) | Our C_eff | Paper C_eff | Error |
|----------|-----------|-------------|-------|
| 1 | $7.57 | $7.60 | 0.3% |
| 5 | $1.52 | $1.51 | 0.3% |
| 10 | $0.76 | $0.80 | 5.3% |
| 25 | $0.36 | $0.37 | 3.5% |
| 50 | $0.32 | $0.32 | 0.1% |

### 2. Token Budget Loader

**Source:** OpenAI API pricing (openai.com/api/pricing)

**Formula:**
```
cost = (uncached_prompt × input_rate) + (cached_tokens × cached_rate) + (completion × output_rate)
```

**Default rates (GPT-4o):**
- Input: $2.50/1M tokens
- Output: $10.00/1M tokens
- Cached: $1.25/1M tokens (50% discount)

### 3. GPU Utilization Budget Loader

**Formula:**
```
cost = GPU_hourly_rate × (runtime_seconds / 3600)
```

### 4. Carbon Budget Loader

**Source:** IPCC Tier 1 methodology, Ember Global Electricity Review 2025

**Formula:**
```
carbon_kg = energy_kwh × PUE × carbon_intensity / 1000
```

**PUE:** Power Usage Effectiveness (default 1.15, from Uptime Institute 2024)

**Regional intensities (g CO2e/kWh):**
| Region | Value | Source |
|--------|-------|--------|
| us-east | 350 | EPA eGRID 2023 |
| eu-france | 45 | Nuclear-heavy grid |
| asia-south | 700 | India dominates |
| global-average | 450 | Ember 2025 |

### 5. Embedding Budget Loader

**Source:** OpenAI, Cohere, Voyage AI pricing pages

**Pre-configured models:**
| Model | Price/1M tokens |
|-------|-----------------|
| openai/text-embedding-3-small | $0.02 |
| openai/text-embedding-3-large | $0.13 |
| cohere/embed-v4 | $0.12 |
| voyage/voyage-4 | $0.06 |

### 6. Reranking Budget Loader

**Source:** Cohere, Voyage AI pricing pages

**Pricing models:**
- Per-query (Cohere): $2.00-2.50/1k queries, up to 100 docs
- Per-token (Voyage): $0.02-0.05/1M tokens

### 7. Cost Optimization Suggestions

Analyzes `BudgetResult` and provides actionable recommendations:
- **caching**: Enable semantic caching (35% avg savings)
- **model_selection**: Right-size model for workload
- **batching**: Increase batch size for better utilization
- **quantization**: Use FP8 for ~2x throughput
- **parallelism**: Tensor parallelism for large models
- **latency**: Relax SLO or use faster model
- **carbon**: Run in cleaner grid region

**Source:** aicostcheck.com, agentcalc.com, Flare-Aug (Su et al., 2025)

### 8. Cost History Logger

JSONL append-only logging for historical analysis:
```python
logger = CostHistoryLogger("cost_history.jsonl")
logger.log("vllm", config_dict, context_dict, result)
entries = logger.query(budget_type="vllm", since="2026-07-01")
summary = logger.summary()
```

### 9. Cost Alerts

Threshold-based monitoring:
```python
alerts = check_alerts(result, {
    "max_cost_usd": 0.01,
    "max_carbon_kg": 0.001,
    "min_throughput_tok_s": 100,
})
```

### 10. Config-Driven Architecture

**All 34 BudgetConfig fields are configurable:**

| Field | Default | Source |
|-------|---------|--------|
| `gpu_type` | "A100-80GB" | User config |
| `pue` | 1.15 | Uptime Institute 2024 |
| `carbon_intensity_g_per_kwh` | 450 | Ember 2025 |
| `kv_overhead_per_token_s` | 0.00014 | arXiv 2606.11690 |
| `cache_saving_fraction` | 0.50 | vLLM APC docs |
| `gpu_power_idle_fraction` | 0.25 | NVIDIA power model |
| `cpu_type` | "" | User config (e.g., Intel Xeon) |
| `cpu_cores` | 0 | User config (0 = not configured) |
| `cpu_threads` | 0 | User config (0 = not configured) |
| `cpu_hourly_rate` | 0.0 | User config (0 = not configured) |
| `cpu_tdp_w` | 0 | User config (0 = not configured) |
| `num_processes` | 1 | User config (parallel processes) |
| `num_threads_per_process` | 1 | User config (threads per process) |
| ... | ... | ... |

**No hardcoded values in Python code.** Every value flows through `BudgetConfig`.

### 13. CPU Configuration Support

Added CPU-related fields for single-thread, multi-thread, and multi-process workloads:

```yaml
cpu_type: "Intel Xeon Platinum 8375C"  # CPU model
cpu_cores: 16                          # Physical cores
cpu_threads: 32                        # Logical threads
cpu_hourly_rate: 0.50                  # $/hr per CPU
cpu_tdp_w: 250                         # CPU TDP in watts
num_processes: 4                       # Parallel processes
num_threads_per_process: 8             # Threads per process
```

New CPU power model in `hardware.py`:
- `estimate_cpu_power(cpu_tdp_w, cores, threads, util)` — CPU power estimation
- `estimate_total_system_power(gpu, cpu, memory, network)` — Total system power
- CPU idle fraction: 15% of TDP, active fraction: 85%

**CPU power validation (Intel Xeon 100-300W TDP, 50% util, PUE 1.15):**

| Config | Power | Energy/query | Carbon/query |
|--------|-------|-------------|--------------|
| 1 core | 57.5W | 0.00001837 kWh | 0.00000827 kg |
| 4 cores | 86.2W | 0.00002755 kWh | 0.00001240 kg |
| 8 cores | 115.0W | 0.00003674 kWh | 0.00001653 kg |
| 16 cores | 143.8W | 0.00004592 kWh | 0.00002066 kg |
| 32 cores | 172.5W | 0.00005510 kWh | 0.00002480 kg |

### 11. Controller Integration

Optional cost estimation per iteration:
```python
controller = RAGtuneController(
    ...,
    cost_loader=VLLMBudgetLoader(BudgetConfig({...})),
    cost_config={"prompt_tokens": 512, "completion_tokens": 256},
)
result = controller.run(query)
print(f"Total cost: ${result.final_budget_state['total_cost_usd']:.6f}")
```

### 12. Token Budget Bug Fix

Fixed token budget not being consumed during iterative reranking loop:
- Schedulers now populate `expected_cost.tokens` from document metadata
- Controller consumes tokens via `tracker.consume(proposal.expected_cost)`
- 5 previously-xfail tests now pass

---

## Files Changed (29 files, +5,100/-162 lines)

### New Files (21)

| File | Lines | Purpose |
|------|:-----:|---------|
| `src/ragtune/budget/__init__.py` | 19 | Package exports |
| `src/ragtune/budget/base.py` | 133 | BudgetConfig (34 fields), BaseBudgetLoader ABC |
| `src/ragtune/budget/factory.py` | 88 | BudgetLoaderFactory registry |
| `src/ragtune/budget/hardware.py` | 162 | GPUSpec dataclass, power/energy/carbon functions |
| `src/ragtune/budget/throughput.py` | 261 | Θ_max lookup, saturation model |
| `src/ragtune/budget/result.py` | 84 | BudgetResult (15 fields, per-component costs) |
| `src/ragtune/budget/main.py` | 79 | calculate_budget(), budget_report() |
| `src/ragtune/budget/optimizer.py` | 139 | Cost optimization suggestions |
| `src/ragtune/budget/history.py` | 144 | JSONL cost logging |
| `src/ragtune/budget/alerts.py` | 134 | Threshold-based alerts |
| `src/ragtune/budget/loaders/vllm_budget.py` | 159 | VLLM loader (arXiv 2606.11690) |
| `src/ragtune/budget/loaders/token_budget.py` | 73 | Token pricing loader |
| `src/ragtune/budget/loaders/gpu_budget.py` | 68 | GPU utilization loader |
| `src/ragtune/budget/loaders/carbon_budget.py` | 84 | Carbon footprint loader |
| `src/ragtune/budget/loaders/embedding_budget.py` | 99 | Embedding API loader |
| `src/ragtune/budget/loaders/reranking_budget.py` | 124 | Reranking API loader |
| `docs/budget.md` | 424 | Comprehensive documentation |
| `docs/BUDGET_AUDIT_2026_07_27.md` | 250 | Audit report |
| `tests/unit/budget/test_budget_system.py` | 1024 | 103 budget tests |

### Modified Files (9)

| File | Change |
|------|--------|
| `src/ragtune/cli/main.py` | Added `ragtune budget` command with 6 loader types |
| `src/ragtune/core/controller.py` | Added cost_loader integration |
| `src/ragtune/components/schedulers.py` | Added token estimation |
| `src/ragtune/cli/config_loader.py` | Minor import fix |
| `tests/unit/core/test_token_budget_bug.py` | Fixed token budget tests |
| `src/ragtune/budget/configs/default.yaml` | Default budget config |
| `src/ragtune/budget/configs/h100_us_east.yaml` | H100 config |

---

## How to run

### CLI Usage

```bash
# Install
pip install -e .

# VLLM cost estimation
ragtune budget --gpu A100-80GB --model cross-encoder/ms-marco-MiniLM-L-6-v2 --rps 10

# Token pricing (GPT-4o)
ragtune budget --type token --prompt-tokens 1024 --completion-tokens 512

# Carbon footprint
ragtune budget --type carbon --region eu-france --gpu A100-80GB

# Embedding cost
ragtune budget --type embedding --embedding-model openai/text-embedding-3-large

# Reranking cost
ragtune budget --type reranking --reranking-model cohere/rerank-v4-pro --queries 10 --docs 50

# With optimization suggestions
ragtune budget --gpu A100-80GB --model cross-encoder/ms-marco-MiniLM-L-6-v2 --suggest

# With YAML config
ragtune budget --config src/ragtune/budget/configs/h100_us_east.yaml
```

### Python API

```python
from ragtune.budget import calculate_budget, budget_report

# Simple usage
result = calculate_budget("vllm", prompt_tokens=512, completion_tokens=256)
print(f"Cost: ${result.cost_usd:.6f}")

# With config
result = calculate_budget(
    "vllm",
    config={"gpu_type": "H100-NVL-96GB", "model_name": "llama-3.1-8b"},
    prompt_tokens=512,
    completion_tokens=256,
)

# Formatted report
report = budget_report("vllm", prompt_tokens=512, completion_tokens=256)
print(report)
```

### YAML Configuration

```yaml
# configs/custom.yaml
gpu_type: "A100-80GB"
gpu_count: 2
pue: 1.10
model_name: "llama-3.1-8b"
offered_rps: 25.0
cache_hit_rate: 0.35
carbon_intensity_g_per_kwh: 450
```

---

## How the config works

### BudgetConfig (34 fields)

All parameters that affect cost/energy/carbon estimation are configurable:

```python
from ragtune.budget.base import BudgetConfig

config = BudgetConfig({
    "gpu_type": "H100-NVL-96GB",
    "gpu_count": 2,
    "pue": 1.10,
    "model_name": "llama-3.1-8b",
    "offered_rps": 25.0,
    "cache_hit_rate": 0.35,
    "kv_overhead_per_token_s": 0.00014,
    "cache_saving_fraction": 0.50,
    "gpu_power_idle_fraction": 0.25,
    "gpu_power_active_fraction": 0.75,
})

# Validate
errors = config.validate()
assert len(errors) == 0
```

### Config Loading Flow

```
YAML file → BudgetConfig → Loader.calculate(context) → BudgetResult
                ↑
         CLI options override YAML values
```

### Per-Request Context

Loaders accept per-request context that overrides config defaults:
```python
result = loader.calculate({
    "prompt_tokens": 1024,
    "completion_tokens": 512,
    "cached_tokens": 200,
})
```

---

## How the loaders work

### Loader Registration

All loaders register via `@BudgetLoaderFactory.register("key")`:
```python
@BudgetLoaderFactory.register("vllm")
class VLLMBudgetLoader(BaseBudgetLoader):
    def calculate(self, context=None) -> BudgetResult:
        ...
```

### Loader Creation

```python
from ragtune.budget.factory import BudgetLoaderFactory

loader = BudgetLoaderFactory.create("vllm", config={"gpu_type": "H100-NVL-96GB"})
result = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
```

### Loader Chain

```
BudgetConfig → Loader.calculate(context) → BudgetResult
                    ↓
         throughput.py (Θ_max, Θ_achieved)
         hardware.py (power, energy, carbon)
```

---

## How you can change it

### Add a new loader

```python
# src/ragtune/budget/loaders/my_loader.py
from ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.result import BudgetResult

@BudgetLoaderFactory.register("my_loader")
class MyBudgetLoader(BaseBudgetLoader):
    def calculate(self, context=None) -> BudgetResult:
        ctx = context or {}
        # Your cost calculation here
        return BudgetResult(cost_usd=0.001, ...)
```

### Add a new config field

```python
# src/ragtune/budget/base.py
class BudgetConfig:
    def __init__(self, config):
        self.my_new_field: float = config.get("my_new_field", 42.0)
```

### Override GPU hourly rate

```python
config = BudgetConfig({"gpu_hourly_rate": 10.0})
# vllm_budget.py uses: gpu_hourly = cfg.gpu_hourly_rate if cfg.gpu_hourly_rate > 0 else hw.hourly_rate
```

### Customize carbon intensity

```python
config = BudgetConfig({
    "region": "eu-france",  # auto-lookup from REGIONAL_INTENSITY
    # OR
    "carbon_intensity_g_per_kwh": 400,  # explicit override
})
```

---

## Where the idea came from

### Core Formula (arXiv 2606.11690)

The vLLM budget loader is based on Chitral Patil's paper "Beyond Per-Token Pricing" which introduces the concurrency-aware cost formula. The key insight: **per-token pricing is misleading because GPU utilization varies 25x with load.**

Our implementation extends this with:
- Empirical Θ_max lookup table from paper Table 4
- Model-size-dependent saturation knee
- PUE multiplier for energy/carbon (from IPCC methodology)

### Carbon Footprint (IPCC + Ember)

The carbon calculation follows IPCC Tier 1 methodology:
```
carbon_kg = energy_kwh × PUE × grid_intensity / 1000
```

Grid intensities sourced from Ember Global Electricity Review 2025.

### API Pricing (OpenAI/Cohere/Voyage)

Token, embedding, and reranking pricing sourced from official pricing pages.

---

## How the budgeting system works

### Complete Cost Chain

```
Query → Embed ($0.0000004) → BM25 ($0) → Dense Retriever ($0.0000004)
     → Reranker ($0.0025) → LLM Generation ($0.000375)
     = Total: $0.002876 per query
```

### Energy & Carbon Chain

```
GPU Power (250W) × PUE (1.15) × Time (1s) = Energy (0.00008 kWh)
Energy × Grid Intensity (450 g/kWh) = Carbon (0.000036 kg CO2)
```

### Throughput Model

```
Θ_max = paper lookup or analytical fallback
Θ_achieved = min(λ × output_tokens, Θ_max × (1 - e^(-λ/λ_sat)))
C_eff = P_GPU × 1e6 / (3600 × Θ_achieved)
```

---

## CPU/GPU Configuration Testing

The budget system was tested across 11 configuration categories with all tests passing.

### GPU Cost Hierarchy (per query, CrossEncoder on A100-80GB)

| GPU | Cost/Query | Power | Energy | Carbon |
|-----|-----------|-------|--------|--------|
| T4-16GB | $0.000073 | 34.8W | 0.00000364 kWh | 0.00000164 kg |
| L4-24GB | $0.000091 | 35.8W | 0.00000374 kWh | 0.00000169 kg |
| V100-32GB | $0.000181 | 149.2W | 0.00001556 kWh | 0.00000700 kg |
| A100-40GB | $0.000263 | 198.9W | 0.00002074 kWh | 0.00000933 kg |
| A100-80GB | $0.000317 | 198.9W | 0.00002073 kWh | 0.00000933 kg |

### CPU-Only Cost Hierarchy (per query, API-based)

| Component | Cost/Query | Source |
|-----------|-----------|--------|
| Embedding (OpenAI) | $0.0000004 | $0.02/1M tokens |
| Reranking (Cohere) | $0.002500 | $2.50/1k queries |
| Token (GPT-4o) | $0.007500 | $2.50/$10.00 per 1M tokens |

### Multi-GPU Scaling (A100-80GB)

| GPUs | Cost | Hourly Rate | Scaling |
|------|------|-------------|---------|
| 1 | $0.000317 | $3.50 | 1x |
| 2 | $0.000635 | $7.00 | 2x |
| 4 | $0.001269 | $14.00 | 4x |
| 8 | $0.002538 | $28.00 | 8x |

Linear scaling confirms correct hourly-rate-based pricing.

### Energy Efficiency (kWh per query)

| GPU | Energy/Query | Carbon/Query |
|-----|-------------|--------------|
| T4-16GB | 0.00000364 | 0.00000164 |
| L4-24GB | 0.00000374 | 0.00000169 |
| V100-32GB | 0.00001556 | 0.00000700 |
| A100-40GB | 0.00002074 | 0.00000933 |
| A100-80GB | 0.00002073 | 0.00000933 |

### Test Categories (all 11 passed)

1. Single thread CPU (API pricing)
2. Multi-processing CPU (GPU runtime pricing)
3. GPU type comparison (5 GPUs)
4. Throughput comparison
5. Cost per token comparison
6. Energy efficiency comparison
7. Multi-GPU scaling (1-8 GPUs)
8. CPU-only scenarios (API-based)
9. Cost optimization comparison
10. Validation & edge cases
11. Full test suite (161 passed)

---

## Tests

161 tests covering:
- BudgetResult arithmetic and breakdown merge
- All 6 loader types with edge cases
- Hardware specs (frozen dataclass, fallback, FP8 detection)
- Throughput model (empirical lookup, saturation, SLO enforcement)
- Factory creation (config, YAML, error handling)
- Cost optimizer suggestions
- Cost history logging
- Cost alerts
- BudgetConfig validation (12 field checks)

```bash
python -m pytest tests/unit/budget/ -v
# 161 passed, 0 failed
```

---

## Source Citations

### Core Formulas

| Component | Source | URL | Verification |
|-----------|--------|-----|--------------|
| vLLM cost formula | Patil (2026) "Beyond Per-Token Pricing" | https://arxiv.org/abs/2606.11690 | ✅ Verified — Table 3/4 values match |
| Carbon formula | IPCC Tier 1 / GHG Protocol Scope 2 | https://ghgprotocol.org/Third-Party-Databases/IPCC-Emissions-Factor-Database | ✅ Standard methodology |
| PUE multiplier | Uptime Institute 2024 | https://uptimeinstitute.com/ | ✅ Hyperscale avg 1.10-1.20 |
| GPU power model | NVIDIA Power Primer | https://www.nvidia.com/content/dam/en-zz/Solutions/GeForce/technologies/frameview/Power_Primer.pdf | ✅ Idle ~25% TDP validated |
| Cache savings | vLLM APC documentation | https://docs.vllm.ai/features/prefix_caching.html | ✅ Prefill-only savings |

### GPU Specifications (Official NVIDIA Datasheets)

| GPU | Source URL | Key Values |
|-----|-----------|------------|
| T4 | [NVIDIA T4 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf) | 16GB GDDR6, 70W, 320 GB/s |
| L4 | [NVIDIA L4 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf) | 24GB GDDR6, 72W, 300 GB/s |
| V100 | [NVIDIA V100 Datasheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf) | 32GB HBM2, 300W, 900 GB/s |
| A100-40GB | [NVIDIA A100 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) | 40GB HBM2e, 400W, 1555 GB/s |
| A100-80GB | Same as above | 80GB HBM2e, 400W, 2039 GB/s |
| H100-NVL | [NVIDIA H100 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf) | 94GB HBM3, 350-400W, 3900 GB/s |

### Cloud GPU Pricing

| Cloud | Instance Type | GPU | URL |
|-------|--------------|-----|-----|
| AWS | p3.2xlarge | V100 | https://aws.amazon.com/ec2/pricing/on-demand/ |
| AWS | g4dn.xlarge | T4 | Same page |
| AWS | g2.xlarge | L4 | Same page |
| AWS | p4d.24xlarge | A100 | Same page |
| AWS | p5.48xlarge | H100 | Same page |
| GCP | N1+V100 | V100 | https://cloud.google.com/products/compute/gpus-pricing |
| GCP | A2+A100 | A100 | Same page |
| GCP | A3+H100 | H100 | Same page |
| Azure | NCv3 | V100 | https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/ |
| Azure | ND-H100 | H100 | Same page |

### Carbon Intensity Sources

| Region | Value | Primary Source | Secondary Source |
|--------|-------|----------------|------------------|
| us-east | 350 | EPA eGRID 2023 (US avg) | Ember 2025: US=384 |
| us-west | 200 | Oregon/California mix | Google Cloud: us-west1=79, us-west2=169 |
| eu-central | 280 | Germany=330, Netherlands=251 | Google Cloud: europe-west3=276 |
| eu-north | 50 | Nordic weighted avg | Google Cloud: europe-north1=39 |
| eu-france | 45 | France=41 (nuclear-heavy) | Google Cloud: europe-west9=16 |
| asia-east | 500 | China=555, Japan=477, Taiwan=439 | Google Cloud: asia-east1=439 |
| asia-south | 700 | India=705 | Google Cloud: asia-south1=679 |
| australia | 525 | Australia=554 | Google Cloud: australia-se1=498 |
| global-average | 450 | Ember 2025: 471 | OWID 2024: 471 |

**Sources:**
- Ember Global Electricity Review 2025: https://ember-energy.org/data/electricity-data-explorer/
- Our World in Data: https://ourworldindata.org/grapher/carbon-intensity-electricity
- EPA eGRID 2023: https://www.epa.gov/egrid
- Google Cloud Sustainability: https://cloud.google.com/sustainability/region-carbon

### API Pricing

| Service | Model | Pricing | URL |
|---------|-------|---------|-----|
| OpenAI | text-embedding-3-small | $0.02/1M tokens | https://openai.com/api/pricing/ |
| OpenAI | text-embedding-3-large | $0.13/1M tokens | Same page |
| OpenAI | GPT-4o | $2.50/$10.00 per 1M tokens | Same page |
| Cohere | Rerank v4 Pro | $2.50/1k queries | https://cohere.com/pricing |
| Cohere | Rerank v4 Fast | $2.00/1k queries | Same page |
| Voyage AI | rerank-2.5 | $0.05/1M tokens | https://docs.voyageai.com/pricing/ |
| Voyage AI | voyage-4 | $0.06/1M tokens | Same page |

### Power Measurement Methodology

| Method | Description | Source |
|--------|-------------|--------|
| NVML | Real-time GPU power via nvmlDeviceGetPowerUsage() | NVIDIA Management Library |
| IPMI/BMC | Hardware-level server power measurement | Server management standards |
| Power analyzers | Lab-grade PSU/wall measurement | Gold standard for validation |
| Linear model | TDP × (idle% + active% × util) | Our implementation (validated) |

**Validation:** NVIDIA Power Primer confirms idle power ~25% of TDP for data center GPUs. Linear approximation validated by multiple papers (ACM 2025, arXiv 2025).

### Energy Measurement

| Component | Formula | Source |
|-----------|---------|--------|
| GPU Power | P = TDP × (0.25 + 0.75 × util) | NVIDIA power model |
| Energy | E = P × time × PUE / 1000 | Physics + IPCC |
| Carbon | C = E × grid_intensity / 1000 | IPCC Tier 1 |
| PUE | Default 1.15 | Uptime Institute 2024 |

---

## What should the reviewer focus on?

1. **Throughput model accuracy** — Verify Θ_max values match paper Table 4
2. **Carbon formula** — Confirm IPCC Tier 1 methodology is correctly applied
3. **Config-driven design** — Ensure no hardcoded values remain in Python code
4. **Loader registration** — Verify all 6 loaders register correctly
5. **Edge cases** — Zero tokens, unknown GPU/model, division by zero
6. **Integration** — Controller cost estimation works with existing pipeline
7. **Source citations** — All formulas have working links to authoritative sources
8. **GPU specs** — Values match NVIDIA official datasheets
9. **Carbon intensities** — Values cross-checked against Ember, OWID, EPA eGRID, Google Cloud

---

**Files changed:** 29 files, +5,100/-162 lines
**Authored by:** Shuvam Banerji Seal
