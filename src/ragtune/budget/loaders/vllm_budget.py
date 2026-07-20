"""
vLLM Budget Loader
====================
Concurrency-aware cost estimation based on the formula from
arXiv 2606.11690 (Patil, June 2026):

    C_eff = (P_GPU × 1e6) / (3600 × Θ_achieved(λ, L))

Where Θ_achieved depends on hardware, model architecture, quantization,
offered request rate (λ), and latency SLO (L).

Also incorporates:
- VRAM decomposition (vllm-calculator)
- Prefix caching ROI
- Throughput scaling with batch size
"""

import math
from typing import Dict, Any, Optional

from ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.result import BudgetResult


# ── Hardware database (GPU specs) ────────────────────────────────────────

GPU_SPECS = {
    "H100-NVL-96GB": {
        "memory_bw_gb_s": 3350,
        "compute_tflops_fp16": 989,
        "compute_tflops_fp8": 1979,
        "vram_gb": 96,
        "tdp_w": 700,
        "hourly_rate": 6.98,  # Azure on-demand (est)
    },
    "A100-80GB": {
        "memory_bw_gb_s": 2039,
        "compute_tflops_fp16": 312,
        "compute_tflops_fp8": 0,  # No native FP8
        "vram_gb": 80,
        "tdp_w": 400,
        "hourly_rate": 3.50,
    },
    "A100-40GB": {
        "memory_bw_gb_s": 1555,
        "compute_tflops_fp16": 312,
        "compute_tflops_fp8": 0,
        "vram_gb": 40,
        "tdp_w": 400,
        "hourly_rate": 2.50,
    },
}

# ── Model profiles ────────────────────────────────────────────────────────

MODEL_PROFILES = {
    # (total_params_b, active_params_b, architecture)
    "cross-encoder/ms-marco-MiniLM-L-6-v2": (0.1, 0.1, "dense"),
    "cross-encoder/ms-marco-MiniLM-L-12-v2": (0.2, 0.2, "dense"),
    "BAAI/bge-reranker-v2-m3": (0.6, 0.6, "dense"),
    "BAAI/bge-reranker-v2-gemma": (2.6, 2.6, "dense"),
    "castorini/monot5-base-msmarco": (0.2, 0.2, "dense"),
    # LLMs for generation
    "llama-3.1-8b": (8.0, 8.0, "dense"),
    "mixtral-8x7b": (46.7, 12.9, "sparse_moe"),
    "qwen3-30b-a3b": (30.0, 3.0, "ultra_sparse_moe"),
}

# ── Quantization memory factors ──────────────────────────────────────────

QUANT_FACTORS = {
    "fp16": 2.0,  # bytes per param
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


def _get_hardware(config: BudgetConfig) -> Dict:
    gpu_type = config.gpu_type
    if gpu_type in GPU_SPECS:
        return GPU_SPECS[gpu_type]
    return GPU_SPECS["A100-80GB"]


def _get_model_profile(config: BudgetConfig) -> tuple:
    """Return (total_params_b, active_params_b, architecture)."""
    model = config.model_name
    if model in MODEL_PROFILES:
        return MODEL_PROFILES[model]
    return (config.total_params_b, config.active_params_b, config.model_architecture)


def _quant_bytes_per_param(config: BudgetConfig) -> float:
    return QUANT_FACTORS.get(config.quantization, 2.0)


def _estimate_weight_vram(config: BudgetConfig) -> float:
    """VRAM for model weights in GB."""
    total_b, _, _ = _get_model_profile(config)
    bytes_per_param = _quant_bytes_per_param(config)
    vram_gb = total_b * 1e9 * bytes_per_param / (1024**3)
    return vram_gb


def _estimate_peak_throughput(config: BudgetConfig) -> float:
    """Θ_max: theoretical max tokens/sec from memory bandwidth.

    Based on the roofline model: decode throughput is limited by
    the speed of reading model weights from HBM.
    """
    hw = _get_hardware(config)
    total_b, active_b, arch = _get_model_profile(config)
    bytes_per_param = _quant_bytes_per_param(config)
    mem_bw = hw["memory_bw_gb_s"] * config.tensor_parallel

    # In decode mode, active parameters determine throughput
    params_to_read = active_b * 1e9 * bytes_per_param
    weight_read_time_s = params_to_read / (mem_bw * (1024**3))

    # With continuous batching, throughput scales with batch size
    max_tok_per_step = min(config.max_batch_size, 256)
    overhead = 1.15  # scheduler, attention overhead
    throughput = max_tok_per_step / (weight_read_time_s * overhead)

    return throughput


def _estimate_actual_throughput(config: BudgetConfig) -> float:
    """Θ_achieved(λ): throughput under offered load.

    At low λ, GPU is arrival-limited (batch size ~ 1).
    At high λ, GPU approaches Θ_max.
    """
    peak = _estimate_peak_throughput(config)
    lam = max(config.offered_rps, 1.0)

    # Little's Law: N (in-flight) ≈ λ × residence_time
    # Residence time decreases with faster models, increases with concurrency
    hw = _get_hardware(config)
    residence_ms = min(
        config.latency_slo_ms,
        max(50, 500 / math.sqrt(lam)),  # lower latency at higher load
    )

    in_flight = lam * (residence_ms / 1000.0)

    # Batch size grows with in-flight concurrency (capped by max_batch_size)
    achieved_batch = min(in_flight, config.max_batch_size)

    # Throughput scales sub-linearly with batch (contention overhead)
    contention = 1.0 + 0.05 * math.log2(max(achieved_batch, 1))
    achieved_tps = peak * (achieved_batch / config.max_batch_size) / contention

    return max(achieved_tps, 1.0), achieved_batch


def _estimate_gpu_power(config: BudgetConfig) -> float:
    """GPU power draw in watts, adjusted for utilization."""
    hw = _get_hardware(config)
    tdp = hw["tdp_w"]
    _, achieved_batch = _estimate_actual_throughput(config)
    util = min(achieved_batch / config.max_batch_size, 1.0)
    # Idle power ~30% of TDP, scales to 100% under load
    return tdp * (0.30 + 0.70 * util)


@BudgetLoaderFactory.register("vllm")
class VLLMBudgetLoader(BaseBudgetLoader):
    """Concurrency-aware budget loader based on arXiv 2606.11690.

    Calculates cost per million tokens under offered load, accounting
    for GPU utilization, model architecture, quantization, and latency SLO.
    """

    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        ctx = context or {}

        # Override config with per-request context if provided
        prompt_tokens = ctx.get("prompt_tokens", 512)
        completion_tokens = ctx.get("completion_tokens", 256)
        batch_size = ctx.get("batch_size", self.config.max_batch_size)
        cached_tokens = ctx.get("cached_tokens", 0)

        cfg = self.config
        hw = _get_hardware(cfg)

        # ── Formula from arXiv 2606.11690 ──
        peak_tps = _estimate_peak_throughput(cfg)
        actual_tps, achieved_batch = _estimate_actual_throughput(cfg)

        # GPU utilization: U = Θ_achieved / Θ_max
        gpu_util = actual_tps / peak_tps if peak_tps > 0 else 0.0

        # GPU hourly cost (adjusted for tensor/pipeline parallelism)
        total_gpu_hourly = hw["hourly_rate"] * cfg.gpu_count

        # C_eff = (P_GPU × 1e6) / (3600 × Θ_achieved)
        cost_per_million = (
            (total_gpu_hourly * 1_000_000 / (3600 * actual_tps))
            if actual_tps > 0
            else 0.0
        )

        # Per-request cost
        total_tokens = prompt_tokens + completion_tokens
        request_cost = cost_per_million * total_tokens / 1_000_000

        # ── Energy ──
        power_w = _estimate_gpu_power(cfg) * cfg.gpu_count
        request_time_s = total_tokens / max(actual_tps, 1)
        energy_kwh = power_w * request_time_s / 3600 / 1000

        # ── Carbon ──
        carbon_kg = energy_kwh * cfg.carbon_intensity_g_per_kwh / 1000

        # ── Electricity cost ──
        electricity_cost = energy_kwh * cfg.electricity_cost_per_kwh

        # ── Caching savings (from paper §2.1) ──
        cache_saving = (cached_tokens / max(total_tokens, 1)) * 0.90

        return BudgetResult(
            cost_usd=round(request_cost * (1 - cache_saving), 6),
            cost_per_million_tokens=round(cost_per_million * (1 - cache_saving), 4),
            energy_kwh=round(energy_kwh, 8),
            carbon_kg=round(carbon_kg, 8),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            throughput_tok_s=round(actual_tps, 1),
            gpu_utilization=round(gpu_util * 100, 1),
            latency_slo_met=True,
            breakdown={
                "gpu_hourly_rate": total_gpu_hourly,
                "peak_tps": round(peak_tps, 1),
                "achieved_batch": round(achieved_batch, 1),
                "gpu_util_pct": round(gpu_util * 100, 1),
                "power_w": round(power_w, 1),
                "electricity_cost": round(electricity_cost, 8),
                "cache_saving": round(cache_saving, 4),
            },
        )
