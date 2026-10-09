"""ComfyUI node surface (DESIGN.md §5.7). Category "DGX Monarch".

Keys are stable inside a major version; a major release may remove one, and the
CHANGELOG names the removed class.
"""
from . import routes as _routes
from .fleet import DGXMonarchFleetKSampler
from .gate import DGXMonarchIdentityGate
from .guiders import DGXMonarchBasicGuider, DGXMonarchCFGGuider, DGXMonarchDualModelGuider
from .init import DGXMonarchInit
from .loaders import DGXMonarchLoraLoader, DGXMonarchUncondUNETLoader, DGXMonarchUNETLoader
from .model_sampling import DGXMonarchModelSamplingSD3
from .ops import DGXMonarchClearVRAM, DGXMonarchStatus
from .qwen_image21_cache import DGXMonarchQwenImage21Cache
from .samplers import (
    DGXMonarchBasicScheduler,
    DGXMonarchKSampler,
    DGXMonarchKSamplerAdvanced,
    DGXMonarchKSamplerPipeline,
    DGXMonarchSamplerCustom,
)

# Comfy's node registry contract: built at import and read-only after import.
NODE_CLASS_MAPPINGS = {
    "DGXMonarchInit": DGXMonarchInit,
    "DGXMonarchUNETLoader": DGXMonarchUNETLoader,
    "DGXMonarchUncondUNETLoader": DGXMonarchUncondUNETLoader,
    "DGXMonarchLoraLoader": DGXMonarchLoraLoader,
    "DGXMonarchModelSamplingSD3": DGXMonarchModelSamplingSD3,
    "DGXMonarchQwenImage21Cache": DGXMonarchQwenImage21Cache,
    "DGXMonarchKSampler": DGXMonarchKSampler,
    "DGXMonarchKSamplerAdvanced": DGXMonarchKSamplerAdvanced,
    "DGXMonarchKSamplerPipeline": DGXMonarchKSamplerPipeline,
    "DGXMonarchSamplerCustom": DGXMonarchSamplerCustom,
    "DGXMonarchBasicScheduler": DGXMonarchBasicScheduler,
    "DGXMonarchBasicGuider": DGXMonarchBasicGuider,
    "DGXMonarchCFGGuider": DGXMonarchCFGGuider,
    "DGXMonarchDualModelGuider": DGXMonarchDualModelGuider,
    "DGXMonarchIdentityGate": DGXMonarchIdentityGate,
    "DGXMonarchFleetKSampler": DGXMonarchFleetKSampler,
    "DGXMonarchStatus": DGXMonarchStatus,
    "DGXMonarchClearVRAM": DGXMonarchClearVRAM,
}

# Same contract, read-only after import.
NODE_DISPLAY_NAME_MAPPINGS = {
    "DGXMonarchInit": "DGX Monarch Init",
    "DGXMonarchUNETLoader": "Load Diffusion Model (DGX Monarch)",
    "DGXMonarchUncondUNETLoader": "Load Unconditional Diffusion Model (DGX Monarch)",
    "DGXMonarchLoraLoader": "Load LoRA (DGX Monarch)",
    "DGXMonarchModelSamplingSD3": "Model Sampling SD3 (DGX Monarch)",
    "DGXMonarchQwenImage21Cache": "Qwen Image 2.1 Cache (DGX Monarch)",
    "DGXMonarchKSampler": "KSampler (DGX Monarch)",
    "DGXMonarchKSamplerAdvanced": "KSampler Advanced (DGX Monarch)",
    "DGXMonarchKSamplerPipeline": "KSampler Pipeline (DGX Monarch)",
    "DGXMonarchSamplerCustom": "Sampler Custom (DGX Monarch)",
    "DGXMonarchBasicScheduler": "Basic Scheduler (DGX Monarch)",
    "DGXMonarchBasicGuider": "Basic Guider (DGX Monarch)",
    "DGXMonarchCFGGuider": "CFG Guider (DGX Monarch)",
    "DGXMonarchDualModelGuider": "Dual Model Guider (DGX Monarch)",
    "DGXMonarchIdentityGate": "Identity Gate (DGX Monarch)",
    "DGXMonarchFleetKSampler": "Fleet KSampler (DGX Monarch)",
    "DGXMonarchStatus": "Cluster Status (DGX Monarch)",
    "DGXMonarchClearVRAM": "Clear VRAM (DGX Monarch)",
}

_routes.register()  # on ComfyUI's server: dgxm top telemetry, metrics, mesh reset, consent cards, graph advice

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
