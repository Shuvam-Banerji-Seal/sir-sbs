"""
Base Budget Loader
==================
Abstract base class for all budget calculation backends.

Every loader produces a BudgetResult with cost in USD, carbon, kWh, and
tokens. The specific formula depends on the loader — vLLM concurrency-aware,
simple token counting, GPU utilization, etc.

Usage:
    class MyBudgetLoader(BaseBudgetLoader):
        def calculate(self, context) -> BudgetResult:
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ragtune.budget.result import BudgetResult


class BudgetConfig:
    """Configuration for a budget calculation.

    This is populated from YAML and passed to loaders.
    """

    def __init__(self, config: Dict[str, Any]):
        self.gpu_type: str = config.get("gpu_type", "A100-80GB")
        self.gpu_count: int = config.get("gpu_count", 1)
        self.gpu_hourly_rate: float = config.get("gpu_hourly_rate", 0.0)
        self.region: str = config.get("region", "us-east")
        self.electricity_cost_per_kwh: float = config.get(
            "electricity_cost_per_kwh", 0.12
        )
        self.carbon_intensity_g_per_kwh: float = config.get(
            "carbon_intensity_g_per_kwh", 400.0
        )
        self._carbon_intensity_set: bool = "carbon_intensity_g_per_kwh" in config
        self.model_name: str = config.get(
            "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.model_architecture: str = config.get("model_architecture", "dense")
        self.active_params_b: float = config.get("active_params_b", 0.1)
        self.total_params_b: float = config.get("total_params_b", 0.1)
        self.quantization: str = config.get("quantization", "fp16")
        self.max_batch_size: int = config.get("max_batch_size", 256)
        self.latency_slo_ms: int = config.get("latency_slo_ms", 500)
        self.offered_rps: float = config.get("offered_rps", 10.0)
        self.tensor_parallel: int = config.get("tensor_parallel", 1)
        self.pipeline_parallel: int = config.get("pipeline_parallel", 1)
        self.extra: Dict[str, Any] = config.get("extra", {})

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def validate(self) -> List[str]:
        """Validate configuration values. Returns list of error messages."""
        errors = []
        if self.gpu_count < 1:
            errors.append(f"gpu_count must be >= 1, got {self.gpu_count}")
        if self.latency_slo_ms < 0:
            errors.append(f"latency_slo_ms must be >= 0, got {self.latency_slo_ms}")
        if self.offered_rps < 0:
            errors.append(f"offered_rps must be >= 0, got {self.offered_rps}")
        if self.max_batch_size < 1:
            errors.append(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.tensor_parallel < 1:
            errors.append(f"tensor_parallel must be >= 1, got {self.tensor_parallel}")
        if self.electricity_cost_per_kwh < 0:
            errors.append(
                f"electricity_cost_per_kwh must be >= 0, got {self.electricity_cost_per_kwh}"
            )
        if self.carbon_intensity_g_per_kwh < 0:
            errors.append(
                f"carbon_intensity_g_per_kwh must be >= 0, got {self.carbon_intensity_g_per_kwh}"
            )
        return errors


class BaseBudgetLoader(ABC):
    """Abstract base class for all budget calculation backends.

    Subclasses implement `calculate()` which takes a BudgetConfig and
    optional per-request context, and returns a BudgetResult.

    The class-level `key` attribute identifies this loader in the
    registry (e.g., "vllm", "token", "gpu_util").
    """

    key: str = ""

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig({})

    @abstractmethod
    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        """Calculate cost for the given operation context.

        Args:
            context: Optional per-request/per-batch context containing
                things like token counts, batch size, latency, etc.

        Returns:
            BudgetResult with cost, carbon, energy, tokens.
        """
        ...
