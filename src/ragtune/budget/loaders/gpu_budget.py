"""
GPU Utilization Budget Loader
==============================
Cost estimation based purely on GPU utilization and runtime.

Formula:
    cost = GPU_hourly_rate × (runtime_seconds / 3600)

Useful for self-hosted deployments where you know GPU runtime
but don't need the full vLLM concurrency model.
"""

from typing import Dict, Any, Optional

from ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.result import BudgetResult

GPU_SPECS = {
    "H100-NVL-96GB": {"hourly_rate": 6.98, "tdp_w": 700},
    "A100-80GB": {"hourly_rate": 3.50, "tdp_w": 400},
    "A100-40GB": {"hourly_rate": 2.50, "tdp_w": 400},
    "V100-32GB": {"hourly_rate": 2.00, "tdp_w": 300},
    "T4-16GB": {"hourly_rate": 0.80, "tdp_w": 70},
    "L4-24GB": {"hourly_rate": 1.00, "tdp_w": 72},
}


@BudgetLoaderFactory.register("gpu_util")
class GPUUtilBudgetLoader(BaseBudgetLoader):
    """Budget based on GPU runtime and utilization.

    Simple: cost = GPU_time × hourly_rate
    """

    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        ctx = context or {}
        runtime_s = ctx.get("runtime_s", 1.0)
        prompt_tokens = ctx.get("prompt_tokens", 512)
        completion_tokens = ctx.get("completion_tokens", 256)
        batch_size = ctx.get("batch_size", 1)
        gpu_util_pct = ctx.get("gpu_util_pct", 50.0)

        hw = GPU_SPECS.get(self.config.gpu_type, GPU_SPECS["A100-80GB"])
        total_hourly = hw["hourly_rate"] * self.config.gpu_count

        # Cost: just GPU time × rate
        cost = total_hourly * (runtime_s / 3600)

        # Energy at given utilization
        power_w = (
            hw["tdp_w"] * (0.30 + 0.70 * gpu_util_pct / 100) * self.config.gpu_count
        )
        energy_kwh = power_w * runtime_s / 3600 / 1000
        carbon_kg = energy_kwh * self.config.carbon_intensity_g_per_kwh / 1000

        total_tokens = prompt_tokens + completion_tokens

        return BudgetResult(
            cost_usd=round(cost, 6),
            cost_per_million_tokens=round(cost / max(total_tokens, 1) * 1_000_000, 4),
            energy_kwh=round(energy_kwh, 8),
            carbon_kg=round(carbon_kg, 8),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            throughput_tok_s=round(total_tokens / runtime_s, 1) if runtime_s > 0 else 0,
            gpu_utilization=gpu_util_pct,
            breakdown={
                "gpu_type": self.config.gpu_type,
                "hourly_rate": total_hourly,
                "runtime_s": runtime_s,
                "gpu_util_pct": gpu_util_pct,
                "power_w": round(power_w, 1),
            },
        )
