"""
Carbon Budget Loader
======================
Estimates carbon footprint of LLM inference based on:
- Energy consumption (kWh)
- Carbon intensity of the regional grid (g CO2/kWh)

Uses data from Electricity Maps / IPCC for regional intensity.
"""

from typing import Dict, Any, Optional

from ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.hardware import get_gpu_spec
from ragtune.budget.result import BudgetResult

# Regional carbon intensity (g CO2e/kWh) — 2024 data
# Sources: Our World in Data, Ember Global Electricity Review 2024,
#          IEA Electricity 2025, EPA eGRID 2023, Google Cloud Sustainability
REGIONAL_INTENSITY = {
    "us-east": 350,  # EPA eGRID 2023: US national avg ≈ 350
    "us-west": 200,  # Oregon=79, California=195, weighted ≈ 200
    "eu-central": 280,  # Germany=336, Netherlands=251, weighted ≈ 280
    "eu-north": 50,  # Nordic weighted avg (Norway=31, Sweden=35, Finland=67)
    "eu-france": 45,  # France grid: 41-52 g CO2/kWh (nuclear-heavy)
    "asia-east": 500,  # China=555, Japan=483, Korea=416, weighted ≈ 500
    "asia-south": 700,  # India dominates: 670-705
    "australia": 525,  # Australia: 498-554
    "global-average": 450,  # IEA 2024: 442-471 (down from 475 in 2023)
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
        # Use explicit flag instead of magic-number sentinel
        intensity = self.config.carbon_intensity_g_per_kwh
        if not self.config._carbon_intensity_set and self.config.region:
            # intensity not explicitly set — use region lookup
            intensity = REGIONAL_INTENSITY.get(
                self.config.region, REGIONAL_INTENSITY["global-average"]
            )

        # Estimate energy from GPU TDP (source: NVIDIA datasheets)
        hw = get_gpu_spec(self.config.gpu_type)
        power_w = hw.tdp_w * (0.25 + 0.75 * gpu_util_pct / 100) * self.config.gpu_count
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
                "carbon_intensity": intensity,
                "gpu_util_pct": gpu_util_pct,
                "power_w": round(power_w, 1),
                "runtime_s": runtime_s,
            },
        )
