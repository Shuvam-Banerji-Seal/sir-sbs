"""
Budget Result
===============
Unified output from any budget loader.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class BudgetResult:
    """Standardized cost output from any budget loader.

    All loaders produce this same structure — the user sees dollars,
    carbon, electricity, and tokens regardless of which backend formula
    is selected.
    """

    # Monetary cost (USD)
    cost_usd: float = 0.0
    cost_per_million_tokens: float = 0.0

    # Energy / environment
    energy_kwh: float = 0.0
    carbon_kg: float = 0.0

    # Token accounting
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    # Performance indicators
    throughput_tok_s: float = 0.0
    gpu_utilization: float = 0.0
    latency_slo_met: bool = True

    # Raw breakdown (loader-specific)
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __add__(self, other: "BudgetResult") -> "BudgetResult":
        """Combine two budget results (for multi-step pipelines)."""
        return BudgetResult(
            cost_usd=self.cost_usd + other.cost_usd,
            cost_per_million_tokens=(
                (self.cost_usd + other.cost_usd)
                / max(self.total_tokens + other.total_tokens, 1)
                * 1_000_000
            ),
            energy_kwh=self.energy_kwh + other.energy_kwh,
            carbon_kg=self.carbon_kg + other.carbon_kg,
            total_tokens=self.total_tokens + other.total_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            throughput_tok_s=(
                (self.total_tokens + other.total_tokens)
                / max(
                    (self.total_tokens / max(self.throughput_tok_s, 1))
                    + (other.total_tokens / max(other.throughput_tok_s, 1)),
                    1,
                )
            ),
            gpu_utilization=max(self.gpu_utilization, other.gpu_utilization),
            latency_slo_met=self.latency_slo_met and other.latency_slo_met,
        )
