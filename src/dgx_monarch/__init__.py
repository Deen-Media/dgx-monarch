"""dgx-monarch: distributed ComfyUI on PyTorch Monarch.

Control plane: Monarch actors, with one GPUWorker per GPU and one Worker
service per host. Data plane: NCCL and torch.distributed through xfuser. See
docs/DESIGN.md for the architecture contract.
"""

import os

# Hyperactor reads this variable once, at its first import, so the default is
# set before any actor import; setdefault keeps an operator's export.
# docs/TROUBLESHOOTING.md #1 quotes it (tests/test_doctrine_hygiene.py checks).
ATTACH_CONFIG_TIMEOUT = "60s"
os.environ.setdefault("HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT", ATTACH_CONFIG_TIMEOUT)

__version__ = "1.0.0"

# pyproject.toml holds the torchmonarch pin; `dgxm doctor` compares this mirror
# with the installed package.
TORCHMONARCH_PIN = "0.6.0"
