"""Actor-side runtime: the GPUWorker actor and its comfy bridge.

The worker code runs inside Monarch worker procs, one per GPU. The driver (the
ComfyUI process) imports parts of this package too, such as the actor class,
capacity quote rows, artifact identity helpers and typed errors, but must not
import comfy state from here.
"""
from .worker import GPUWorker

__all__ = ["GPUWorker"]
