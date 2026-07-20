"""
Carbon Budget Loader
======================
Estimates carbon footprint of LLM inference based on:
- Energy consumption (kWh)
- Carbon intensity of the regional grid (g CO2/kWh)

Uses data from Electricity Maps / IPCC for regional intensity.
"""

from typing import Dict, Any, Optional

from src.ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from src.ragtune.budget.factory import BudgetLoaderFactory
from src.ragtune.budget.result import BudgetResult

# Regional carbon intensity (g CO2e/kWh) — Electricity Maps 2025 averages
REGIONAL_INTENSITY = {
    "us-east": 350,
    "us-west": 200,
    "eu-central": 250,
    "eu-north": 50,
    "eu-france": 60,  # Nuclear-heavy
    "asia-east": 600,
    "asia-south": 700,
    "australia": 500,
    "global-average": 475,
}


@BudgetLoaderFactory.register("carbon")
class CarbonBudgetLoader(BaseBudgetLoader):
    """Carbon footprint estimation for LLM inference.

    Formula: carbon_kg = energy_kwh × carbon_intensity_g_per_kwh / 1000

    Set region in config to automatically pick carbon intensity.
    """

    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        ctx = context or {}
        prompt_tokens = ctx.get("prompt_tokens", 512)
        completion_tokens = ctx.get("completion_tokens", 256)
        runtime_s = ctx.get("runtime_s", 1.0)
        gpu_util_pct = ctx.get("gpu_util_pct", 50.0)

        # Carbon intensity from config or region lookup
        intensity = self.config.carbon_intensity_g_per_kwh
        if intensity == 400:  # default, try region lookup
            intensity = REGIONAL_INTENSITY.get(
                self.config.region, REGIONAL_INTENSITY["global-average"]
            )

        # Estimate energy from GPU TDP
        tdp_w = {"A100-80GB": 400, "H100-NVL-96GB": 700, "A100-40GB": 400}.get(
            self.config.gpu_type, 400
        )
        power_w = tdp_w * (0.30 + 0.70 * gpu_util_pct / 100) * self.config.gpu_count
        energy_kwh = power_w * runtime_s / 3600 / 1000
        carbon_kg = energy_kwh * intensity / 1000

        total_tokens = prompt_tokens + completion_tokens
        cost_usd = 0.0  # Carbon-only, no monetary cost

        return BudgetResult(
            cost_usd=cost_usd,
            energy_kwh=round(energy_kwh, 8),
            carbon_kg=round(carbon_kg, 8),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            breakdown={
                "region": self.config.region,
                "carbon_intensity": intensity,
                "gpu_util_pct": gpu_util_pct,
                "power_w": round(power_w, 1),
                "runtime_s": runtime_s,
            },
        )
