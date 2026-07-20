"""
Unit tests for the config-based budgeting system.
"""

import os
import pytest
from src.ragtune.budget import BudgetResult, BudgetConfig, BudgetLoaderFactory
from src.ragtune.budget.main import calculate_budget, budget_report


class TestBudgetResult:
    def test_default_creation(self):
        r = BudgetResult()
        assert r.cost_usd == 0.0
        assert r.total_tokens == 0
        assert r.carbon_kg == 0.0

    def test_addition(self):
        a = BudgetResult(
            cost_usd=1.0,
            total_tokens=100,
            carbon_kg=0.01,
            throughput_tok_s=50.0,
            gpu_utilization=60.0,
        )
        b = BudgetResult(
            cost_usd=2.0,
            total_tokens=200,
            carbon_kg=0.02,
            throughput_tok_s=100.0,
            gpu_utilization=80.0,
        )
        c = a + b
        assert c.cost_usd == 3.0
        assert c.total_tokens == 300
        assert c.carbon_kg == 0.03
        assert c.gpu_utilization == 80.0  # max


class TestBudgetConfig:
    def test_default_config(self):
        cfg = BudgetConfig({})
        assert cfg.gpu_type == "A100-80GB"
        assert cfg.gpu_hourly_rate == 3.50

    def test_custom_config(self):
        cfg = BudgetConfig({"gpu_type": "H100-NVL-96GB", "gpu_hourly_rate": 10.00})
        assert cfg.gpu_type == "H100-NVL-96GB"
        assert cfg.gpu_hourly_rate == 10.00

    def test_to_dict(self):
        cfg = BudgetConfig({"gpu_type": "test"})
        d = cfg.to_dict()
        assert d["gpu_type"] == "test"


class TestBudgetLoaderFactory:
    def test_create_vllm(self):
        loader = BudgetLoaderFactory.create("vllm")
        assert loader.key == "vllm"
        assert type(loader).__name__ == "VLLMBudgetLoader"

    def test_create_token(self):
        loader = BudgetLoaderFactory.create("token")
        assert loader.key == "token"

    def test_create_gpu_util(self):
        loader = BudgetLoaderFactory.create("gpu_util")
        assert loader.key == "gpu_util"

    def test_create_carbon(self):
        loader = BudgetLoaderFactory.create("carbon")
        assert loader.key == "carbon"

    def test_create_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown budget type"):
            BudgetLoaderFactory.create("nonexistent")

    def test_list_types(self):
        types = BudgetLoaderFactory.list_types()
        assert "vllm" in types
        assert "token" in types
        assert "gpu_util" in types
        assert "carbon" in types
        assert len(types) >= 4

    def test_create_with_config(self):
        loader = BudgetLoaderFactory.create(
            "vllm",
            config={"gpu_type": "H100-NVL-96GB", "gpu_hourly_rate": 6.98},
        )
        assert loader.config.gpu_type == "H100-NVL-96GB"
        assert loader.config.gpu_hourly_rate == 6.98

    def test_create_with_yaml(self):
        path = "src/ragtune/budget/configs/h100_us_east.yaml"
        loader = BudgetLoaderFactory.create("vllm", config_path=path)
        assert loader.config.gpu_type == "H100-NVL-96GB"


class TestVLLMBudgetLoader:
    def test_basic_calculation(self):
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        assert r.cost_usd > 0
        assert r.cost_per_million_tokens > 0
        assert r.total_tokens == 768
        assert r.prompt_tokens == 512
        assert r.completion_tokens == 256
        assert r.throughput_tok_s > 0
        assert r.gpu_utilization >= 0

    def test_large_batch(self):
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate(
            {
                "prompt_tokens": 4096,
                "completion_tokens": 2048,
                "batch_size": 128,
            }
        )
        assert r.cost_usd > 0
        assert r.total_tokens == 6144

    def test_cached_tokens(self):
        loader = BudgetLoaderFactory.create("vllm")
        r_no_cache = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            }
        )
        r_cached = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "cached_tokens": 800,
            }
        )
        # Cached should be cheaper
        assert r_cached.cost_usd <= r_no_cache.cost_usd

    def test_h100_config(self):
        loader = BudgetLoaderFactory.create(
            "vllm",
            config_path="src/ragtune/budget/configs/h100_us_east.yaml",
        )
        r = loader.calculate({"prompt_tokens": 1024, "completion_tokens": 512})
        assert r.cost_usd > 0
        assert r.breakdown.get("gpu_hourly_rate", 0) > 6


class TestTokenBudgetLoader:
    def test_basic(self):
        loader = BudgetLoaderFactory.create("token")
        r = loader.calculate({"prompt_tokens": 1000, "completion_tokens": 500})
        assert r.cost_usd > 0
        assert r.total_tokens == 1500

    def test_cached_tokens_discount(self):
        loader = BudgetLoaderFactory.create("token")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "cached_tokens": 800,
            }
        )
        assert r.cost_usd > 0


class TestGPUUtilBudgetLoader:
    def test_basic(self):
        loader = BudgetLoaderFactory.create("gpu_util")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "runtime_s": 5.0,
                "gpu_util_pct": 80.0,
            }
        )
        assert r.cost_usd > 0
        assert r.total_tokens == 1500

    def test_short_runtime(self):
        loader = BudgetLoaderFactory.create("gpu_util")
        r = loader.calculate(
            {
                "prompt_tokens": 512,
                "completion_tokens": 256,
                "runtime_s": 0.5,
                "gpu_util_pct": 50.0,
            }
        )
        assert r.cost_usd > 0


class TestCarbonBudgetLoader:
    def test_basic(self):
        loader = BudgetLoaderFactory.create("carbon")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "runtime_s": 10.0,
                "gpu_util_pct": 60.0,
            }
        )
        assert r.carbon_kg > 0
        assert r.energy_kwh > 0

    def test_regional_intensity(self):
        loader = BudgetLoaderFactory.create("carbon", config={"region": "eu-france"})
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "runtime_s": 10.0,
                "gpu_util_pct": 60.0,
            }
        )
        assert r.carbon_kg > 0


class TestBudgetMain:
    def test_calculate_budget(self):
        r = calculate_budget("vllm", prompt_tokens=512, completion_tokens=256)
        assert isinstance(r, BudgetResult)
        assert r.cost_usd > 0

    def test_budget_report(self):
        report = budget_report("vllm", prompt_tokens=512, completion_tokens=256)
        assert "Cost:" in report
        assert "$/M tokens:" in report
        assert "Carbon:" in report
        assert "GPU util:" in report

    def test_result_addition(self):
        a = calculate_budget("vllm", prompt_tokens=512, completion_tokens=256)
        b = calculate_budget("vllm", prompt_tokens=512, completion_tokens=256)
        combined = a + b
        assert combined.cost_usd > a.cost_usd
        assert combined.total_tokens == a.total_tokens + b.total_tokens
