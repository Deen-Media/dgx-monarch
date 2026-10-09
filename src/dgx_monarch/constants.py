"""Shared keys and identifiers used across runtime layers."""

# These are ModelPatcher wrapper keys. Each installer reads its key before it
# adds: comfy appends a second add under one key, so the wrappers would stack.
CFG_WRAPPER_KEY = "dgx_monarch_cfg_parallel"
# The per-cond dispatch wraps CALC_COND_BATCH, a seam the cfg split never uses.
# Comfy files wrappers by seam type, then key, so the two installers never read
# each other's list. The second key is not needed for correctness; it gives
# each wrapper its own name.
CFG_DISPATCH_WRAPPER_KEY = "dgx_monarch_cfg_dispatch"

NODE_CATEGORY = "DGX Monarch"

# These graph type names are semver-governed API.
MESH_TYPE = "DGXM_MESH"
MODEL_TYPE = "DGXM_MODEL"
GUIDER_TYPE = "DGXM_GUIDER"

DEFAULT_WORKER_PORT = 26600
DEFAULT_NCCL_MASTER_PORT = 29777

# One-sided return must amortize memory-registration and setup costs.
DEFAULT_RDMA_MIN_BYTES = 8 * 2**20  # 8 MiB

# Store transition names are operator-visible diagnostics.
TRANSITION_REUSE = "reuse"
TRANSITION_HOT_SWAP = "hot-swap"
TRANSITION_LOAD = "load"
