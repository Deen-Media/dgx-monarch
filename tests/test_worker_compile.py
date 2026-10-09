"""The opt-in DiT compile warns when it finds nothing to compile, skips a
slab-resident model, and does nothing when the switch is off."""
from __future__ import annotations

import types

from dgx_monarch.actor import worker_compile


class _Log:
    def __init__(self):
        self.warnings, self.infos = [], []

    def warning(self, message, *args):
        self.warnings.append(message % args)

    def info(self, message, *args):
        self.infos.append(message % args)


class _Block:
    def __init__(self):
        self.compiled = []

    def compile(self, **kwargs):
        self.compiled.append(kwargs)


def test_a_model_without_a_blocks_list_warns_and_compiles_nothing(monkeypatch):
    log = _Log()
    monkeypatch.setattr(worker_compile, "log", log)
    monkeypatch.setenv("DGXM_COMPILE_DIT", "1")
    flux_like = types.SimpleNamespace(double_blocks=[_Block()], single_blocks=[_Block()])

    worker_compile.maybe_compile_dit(flux_like)

    assert len(log.warnings) == 1 and "nothing was compiled" in log.warnings[0]
    assert not flux_like.double_blocks[0].compiled and not flux_like.single_blocks[0].compiled


def test_slab_marker_blocks_the_worker_compile_seam():
    patcher = types.SimpleNamespace(_dgxm_slab_resident=True)
    assert not worker_compile.compile_dit_allowed(patcher)


def test_a_model_with_a_blocks_list_compiles_without_the_warning(monkeypatch):
    log = _Log()
    monkeypatch.setattr(worker_compile, "log", log)
    monkeypatch.setenv("DGXM_COMPILE_DIT", "1")
    # The compile path sets this process-wide when unset; setenv restores it.
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    model = types.SimpleNamespace(blocks=[_Block(), _Block()])

    worker_compile.maybe_compile_dit(model)

    assert not log.warnings
    assert all(block.compiled for block in model.blocks)
    assert log.infos == ["DiT: 2 blocks compiled (max-autotune-no-cudagraphs)"]


def test_the_lever_off_touches_nothing(monkeypatch):
    log = _Log()
    monkeypatch.setattr(worker_compile, "log", log)
    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)

    worker_compile.maybe_compile_dit(types.SimpleNamespace())

    assert not log.warnings and not log.infos
