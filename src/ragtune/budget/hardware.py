"""
GPU Hardware Specifications
============================
Single source of truth for GPU specs used by all budget loaders.

Source: NVIDIA official datasheets (nvidia.com/en-us/data-center/h100/,
        nvidia.com/en-us/data-center/a100/)
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class GPUSpec:
    """Immutable GPU hardware specification."""

    name: str
    memory_bw_gb_s: int  # Memory bandwidth (GB/s)
    compute_tflops_fp16: int  # FP16 Tensor Core TFLOPS (dense)
    compute_tflops_fp8: int  # FP8 Tensor Core TFLOPS (dense, 0 if unsupported)
    vram_gb: int  # VRAM (GB)
    tdp_w: int  # TDP (Watts)
    hourly_rate: float  # Cloud on-demand hourly rate ($)

    @property
    def has_fp8(self) -> bool:
        return self.compute_tflops_fp8 > 0


# ── Hardware database ──────────────────────────────────────────────────────
# H100 NVL specs differ from H100 SXM — key variant distinction.

GPU_SPECS: Dict[str, GPUSpec] = {
    "H100-NVL-96GB": GPUSpec(
        name="H100-NVL-96GB",
        memory_bw_gb_s=3900,  # H100 NVL: 3.9 TB/s
        compute_tflops_fp16=835,  # H100 NVL dense FP16
        compute_tflops_fp8=1670,  # H100 NVL dense FP8
        vram_gb=94,  # H100 NVL: 94 GB HBM3
        tdp_w=400,  # H100 NVL: 350-400W
        hourly_rate=4.50,  # Azure on-demand estimate
    ),
    "A100-80GB": GPUSpec(
        name="A100-80GB",
        memory_bw_gb_s=2039,  # A100 SXM: 2.039 TB/s
        compute_tflops_fp16=312,  # A100 dense FP16
        compute_tflops_fp8=0,  # No native FP8 on A100 (Ampere)
        vram_gb=80,  # A100: 80 GB HBM2e
        tdp_w=400,  # A100 SXM: 400W
        hourly_rate=3.50,  # Azure on-demand estimate
    ),
    "A100-40GB": GPUSpec(
        name="A100-40GB",
        memory_bw_gb_s=1555,  # A100-40GB: 1.555 TB/s
        compute_tflops_fp16=312,  # Same compute as 80GB variant
        compute_tflops_fp8=0,  # No native FP8
        vram_gb=40,  # A100-40GB: 40 GB HBM2e
        tdp_w=400,  # A100-40GB SXM: 400W
        hourly_rate=2.90,  # Azure on-demand estimate
    ),
    "V100-32GB": GPUSpec(
        name="V100-32GB",
        memory_bw_gb_s=900,  # V100 SXM2: 900 GB/s
        compute_tflops_fp16=125,  # V100 dense FP16
        compute_tflops_fp8=0,  # No FP8 on V100 (Volta)
        vram_gb=32,  # V100: 32 GB HBM2
        tdp_w=300,  # V100 SXM2: 300W
        hourly_rate=2.00,  # Cloud estimate
    ),
    "T4-16GB": GPUSpec(
        name="T4-16GB",
        memory_bw_gb_s=320,  # T4: 320 GB/s
        compute_tflops_fp16=65,  # T4 dense FP16
        compute_tflops_fp8=0,  # No FP8 on T4 (Turing)
        vram_gb=16,  # T4: 16 GB GDDR6
        tdp_w=70,  # T4: 70W
        hourly_rate=0.80,  # Cloud estimate
    ),
    "L4-24GB": GPUSpec(
        name="L4-24GB",
        memory_bw_gb_s=300,  # L4: 300 GB/s
        compute_tflops_fp16=242,  # L4 dense FP16
        compute_tflops_fp8=485,  # L4 dense FP8
        vram_gb=24,  # L4: 24 GB GDDR6
        tdp_w=72,  # L4: 72W
        hourly_rate=1.00,  # Cloud estimate
    ),
}

# Default GPU when type not found
DEFAULT_GPU = "A100-80GB"


def get_gpu_spec(gpu_type: str) -> GPUSpec:
    """Get GPU spec by type, falling back to default."""
    return GPU_SPECS.get(gpu_type, GPU_SPECS[DEFAULT_GPU])


def list_gpu_types():
    """List all available GPU types."""
    return list(GPU_SPECS.keys())
