"""Slab routing through ModelStore.ensure() and the slab teardown it owns."""
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import model_store as ms
from dgx_monarch.constants import TRANSITION_LOAD, TRANSITION_REUSE
from dgx_monarch.safetensors_header import (
    SafetensorsHeaderError,
    UnsupportedSafetensorsDtypeError,
)
from slab_lifetime_helpers import reset_slab_lifetime
from store_ensure_helpers import STACK
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.


def test_full_load_slab_branch(rig):
    store, calls = rig
    store.slab_weights = True
    _active, transition = store.ensure("m.safetensors", None, STACK)
    assert transition == TRANSITION_LOAD
    slab = calls.slabs[0]
    assert store.current.slab is slab
    assert slab.closed == 0
    # offload pinned to the GPU so detach's model.to() is a no-op
    assert store.current.base_patcher.offload_device == "cuda:0"
    # comfy's bake and cast strays moved back into the slab, then the allocator cache emptied
    assert slab.reabsorbed == ["DIT_MODULE"]
    assert calls.soft_empty == 1


def test_auto_first_stock_discovery_reloads_exact_request_into_slab(
        rig, monkeypatch):
    store, calls = rig
    store.slab_weights = "auto"
    memo_reads = []
    memo_writes = []
    monkeypatch.setattr(
        ms, "memoized_family",
        lambda path: memo_reads.append(path),
    )
    monkeypatch.setattr(
        ms, "memoize_family",
        lambda path, family: memo_writes.append((path, family)),
    )

    first, first_transition = store.ensure("m.safetensors", None, STACK)
    assert first_transition == TRANSITION_LOAD
    assert first is store.current.active_patcher
    assert store.current.slab is None
    assert store.current.slab_auto_retry is True

    second, second_transition = store.ensure("m.safetensors", None, STACK)
    assert second_transition == TRANSITION_LOAD
    assert second is store.current.active_patcher
    assert store.current.slab is calls.slabs[0]
    assert store.current.slab_auto_retry is False
    assert calls.load == 2
    assert calls.unload_all == 1

    _third, third_transition = store.ensure("m.safetensors", None, STACK)
    assert third_transition == TRANSITION_REUSE
    assert calls.load == 2
    assert memo_reads == ["/fake/m.safetensors"]
    assert memo_writes == [
        ("/fake/m.safetensors", "krea2"),
        ("/fake/m.safetensors", "krea2"),
    ]


def test_compile_noop_auto_retries_only_after_flux2_live_family_discovery(
        rig, monkeypatch):
    """Under compile the first auto load stays stock, since the family is unknown before load;
    flux2, where compile does nothing, then reloads the same request into a slab."""
    store, calls = rig
    store.slab_weights = "auto"
    monkeypatch.setenv("DGXM_COMPILE_DIT", "1")
    monkeypatch.setattr(ms, "_detect_family", lambda _base: "flux2")
    memo = {}
    monkeypatch.setattr(ms, "memoized_family", lambda path: memo.get(path))
    monkeypatch.setattr(ms, "memoize_family", lambda path, family: memo.__setitem__(path, family))

    store.ensure("m.safetensors", None, STACK)
    assert store.current.slab is None
    assert store.current.slab_auto_retry is True
    assert memo == {"/fake/m.safetensors": "flux2"}

    store.ensure("m.safetensors", None, STACK)
    assert store.current.slab is calls.slabs[0]
    assert store.current.slab_auto_retry is False


def test_compile_noop_explicit_slab_retries_after_flux2_live_family_discovery(
        rig, monkeypatch):
    store, calls = rig
    store.slab_weights = True
    monkeypatch.setenv("DGXM_COMPILE_DIT", "1")
    monkeypatch.setattr(ms, "_detect_family", lambda _base: "flux2")
    memo = {}
    monkeypatch.setattr(ms, "memoized_family", lambda path: memo.get(path))
    monkeypatch.setattr(ms, "memoize_family", lambda path, family: memo.__setitem__(path, family))

    store.ensure("m.safetensors", None, STACK)
    assert store.current.slab is None
    assert store.current.slab_auto_retry is True

    store.ensure("m.safetensors", None, STACK)
    assert store.current.slab is calls.slabs[0]
    assert store.current.slab_auto_retry is False


@pytest.mark.parametrize("override,unet_name", [
    ("flux2", "m.safetensors"),
    (None, "legacy.ckpt"),
])
def test_compile_noop_explicit_slab_does_not_retry_outside_discovery_guards(
        rig, monkeypatch, override, unet_name):
    store, calls = rig
    store.slab_weights = True
    store.family_override = override
    monkeypatch.setenv("DGXM_COMPILE_DIT", "1")
    monkeypatch.setattr(ms, "_detect_family", lambda _base: "flux2")

    store.ensure(unet_name, None, STACK)

    assert store.current.slab is None
    assert store.current.slab_auto_retry is False
    assert calls.load == 1


def test_compile_blocks_auto_krea2_stays_stock_without_a_retry(rig, monkeypatch):
    store, calls = rig
    store.slab_weights = "auto"
    monkeypatch.setenv("DGXM_COMPILE_DIT", "1")

    store.ensure("m.safetensors", None, STACK)

    assert store.current.slab is None
    assert store.current.slab_auto_retry is False
    assert calls.load == 1


@pytest.mark.parametrize("unet_name", ["legacy.ckpt", "legacy.bin"])
def test_auto_unsupported_suffix_stock_loads_then_reuses(
        rig, monkeypatch, caplog, unet_name):
    store, calls = rig
    store.slab_weights = "auto"
    monkeypatch.setattr(ms, "memoized_family", lambda _path: None)

    _active, first_transition = store.ensure(unet_name, None, STACK)
    assert first_transition == TRANSITION_LOAD
    assert store.current.slab is None
    assert store.current.slab_auto_retry is False
    assert calls.slabs == []
    assert "the next load of the same file slab-loads" not in caplog.text

    _active, second_transition = store.ensure(unet_name, None, STACK)
    assert second_transition == TRANSITION_REUSE
    assert calls.load == 1


@pytest.mark.parametrize("unet_name", ["m.safetensors", "m.sft", "m.SAFETENSORS"])
def test_slab_capable_suffixes_use_the_slab_loader(rig, unet_name):
    store, calls = rig
    store.slab_weights = True

    store.ensure(unet_name, None, STACK)

    assert store.current.slab is calls.slabs[0]


@pytest.mark.parametrize("slab_mode", [True, "auto"])
def test_unsupported_slab_dtype_logs_and_falls_back_to_stock(
        rig, monkeypatch, caplog, slab_mode):
    store, calls = rig
    store.slab_weights = slab_mode
    if slab_mode == "auto":
        monkeypatch.setattr(ms, "memoized_family", lambda _path: "krea2")
    from dgx_monarch.actor import comfy_bridge

    class UnsupportedSlab:
        def __enter__(self):
            raise UnsupportedSafetensorsDtypeError(
                "model.safetensors: dtype 'FUTURE' is not supported by this torch build"
            )

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        comfy_bridge,
        "slab_load",
        lambda _path, *, handoff=None: UnsupportedSlab(),
    )

    _active, transition = store.ensure("m.safetensors", None, STACK)

    assert transition == TRANSITION_LOAD
    assert calls.load == 1
    assert store.current.slab is None
    assert store.current.base_patcher.offload_device == "cpu"
    assert "falling back to stock non-slab loading" in caplog.text
    assert "the next load of the same file slab-loads" not in caplog.text

    _active, repeat_transition = store.ensure("m.safetensors", None, STACK)
    assert repeat_transition == TRANSITION_REUSE
    assert calls.load == 1


def test_auto_discovery_dtype_fallback_consumes_its_single_retry(
        rig, monkeypatch):
    store, calls = rig
    store.slab_weights = "auto"
    monkeypatch.setattr(ms, "memoized_family", lambda _path: None)
    monkeypatch.setattr(ms, "memoize_family", lambda _path, _family: None)
    from dgx_monarch.actor import comfy_bridge

    class UnsupportedSlab:
        def __enter__(self):
            raise UnsupportedSafetensorsDtypeError(
                "model.safetensors: dtype 'FUTURE' is not supported by this torch build"
            )

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        comfy_bridge,
        "slab_load",
        lambda _path, *, handoff=None: UnsupportedSlab(),
    )

    _active, first_transition = store.ensure("m.safetensors", None, STACK)
    assert first_transition == TRANSITION_LOAD
    assert store.current.slab_auto_retry is True

    _active, retry_transition = store.ensure("m.safetensors", None, STACK)
    assert retry_transition == TRANSITION_LOAD
    assert store.current.slab is None
    assert store.current.slab_auto_retry is False
    assert calls.load == 2

    _active, reuse_transition = store.ensure("m.safetensors", None, STACK)
    assert reuse_transition == TRANSITION_REUSE
    assert calls.load == 2


def test_malformed_slab_header_is_not_masked_by_stock_fallback(rig, monkeypatch):
    store, calls = rig
    store.slab_weights = True
    from dgx_monarch.actor import comfy_bridge

    class MalformedSlab:
        def __enter__(self):
            raise SafetensorsHeaderError("malformed tensor descriptor")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        comfy_bridge,
        "slab_load",
        lambda _path, *, handoff=None: MalformedSlab(),
    )

    with pytest.raises(SafetensorsHeaderError, match="malformed tensor descriptor"):
        store.ensure("m.safetensors", None, STACK)

    assert calls.load == 0
    assert store.current is None


def test_full_load_slab_skips_non_safetensors(rig):
    store, calls = rig
    store.slab_weights = True
    store.ensure("legacy.ckpt", None, STACK)
    assert calls.slabs == []                 # extension gate: no slab


def test_error_after_load_closes_slab(rig):
    store, calls = rig
    store.slab_weights = True

    def boom(base, quant, stack, precision):
        raise RuntimeError("adapter typed refusal")

    with pytest.raises(RuntimeError, match="typed refusal"):
        store.ensure("m.safetensors", None, STACK, on_base_loaded=boom)
    slab = calls.slabs[0]
    assert slab.closed == 1                  # nothing strands the memfd
    assert calls.unload_all == 1             # model released before close
    assert store.current is None


def test_drop_closes_slab_after_model_release(rig):
    store, calls = rig
    store.slab_weights = True
    store.ensure("m.safetensors", None, STACK)
    slab = calls.slabs[0]
    store._drop("cond")
    assert slab.closed == 1
    assert calls.unload_all >= 1             # release ordering: model first
    assert store.current is None


def test_drop_slab_close_failure_retains_owner_and_blocks_reuse(rig, monkeypatch):
    from dgx_monarch.actor import slab_lifetime

    reset_slab_lifetime(monkeypatch)
    store, calls = rig
    store.slab_weights = True
    store.ensure("m.safetensors", None, STACK)
    slab = calls.slabs[0]
    close_attempts = 0

    def fail_close_once():
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts == 1:
            raise RuntimeError("slab close interrupted")

    calls.close_guard = fail_close_once
    with pytest.raises(RuntimeError, match="slab close interrupted"):
        store._drop("cond")

    assert store.current is None
    assert slab.closed == 0
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime.retained_count() == 1
    with pytest.raises(RuntimeError, match="previous model load"):
        store.ensure("m.safetensors", None, STACK)

    calls.close_guard = None
    store.unload_all()
    assert slab.closed == 1
    assert not slab_lifetime.cleanup_pending()


def test_failed_global_unload_retains_slab_and_blocks_reuse(rig, monkeypatch):
    store, calls = rig
    store.slab_weights = True
    store.ensure("m.safetensors", None, STACK)
    original = store.current
    slab = calls.slabs[0]
    mm = sys.modules["comfy.model_management"]
    healthy_unload = mm.unload_all_models
    monkeypatch.setattr(
        mm, "unload_all_models",
        lambda: (_ for _ in ()).throw(RuntimeError("comfy unload failed")),
    )

    with pytest.raises(RuntimeError, match="keeps ownership of the resident"):
        store._drop("cond")

    assert store.current is original
    assert slab.closed == 0
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    with pytest.raises(RuntimeError, match="previous unload failed"):
        store.ensure("m.safetensors", None, STACK)

    monkeypatch.setattr(mm, "unload_all_models", healthy_unload)
    store._drop("cond")
    assert store.current is None
    assert slab.closed == 1
    assert store.snapshot()["cleanup_failed_slots"] == []


@pytest.mark.parametrize("abort_type", [KeyboardInterrupt, SystemExit])
def test_interrupted_global_unload_preserves_identity_and_ownership(
    rig,
    monkeypatch,
    abort_type,
):
    store, _calls = rig
    store.ensure("m.safetensors", None, STACK)
    original = store.current
    mm = sys.modules["comfy.model_management"]
    healthy_unload = mm.unload_all_models
    failure = abort_type("global unload interrupted")
    monkeypatch.setattr(mm, "unload_all_models", lambda: (_ for _ in ()).throw(failure))

    with pytest.raises(abort_type) as caught:
        store._drop("cond")

    assert caught.value is failure
    assert store.current is original
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]

    monkeypatch.setattr(mm, "unload_all_models", healthy_unload)
    store._drop("cond")
    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == []


def test_retained_resource_close_interruption_preserves_exact_identity(
    rig,
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime

    class CleanupAbort(BaseException):
        pass

    failure = CleanupAbort("retained resource close interrupted")

    class Resource:
        def __init__(self):
            self.attempts = 0
            self.fail = True

        def close(self):
            self.attempts += 1
            if self.fail:
                raise failure

    resource = Resource()
    monkeypatch.setattr(slab_lifetime, "_RETAINED_FAILED_LOAD_SLABS", [])
    monkeypatch.setattr(
        slab_lifetime,
        "_RETAINED_FAILED_LOAD_RESOURCES",
        [(resource, None)],
    )
    monkeypatch.setattr(slab_lifetime, "_FAILED_LOAD_CLEANUP_POISONED", True)
    store, _calls = rig

    with pytest.raises(CleanupAbort) as caught:
        store.unload_all()

    assert caught.value is failure
    assert resource.attempts == 1
    assert slab_lifetime.retained_resource_count() == 1
    assert slab_lifetime.cleanup_pending()
    assert store.snapshot()["retained_failed_load_resources"] == 1

    resource.fail = False
    store.unload_all()
    assert resource.attempts == 2
    assert slab_lifetime.retained_resource_count() == 0
    assert not slab_lifetime.cleanup_pending()
    assert store.snapshot()["retained_failed_load_resources"] == 0


def test_drop_frees_sharded_weights_in_place_before_the_global_unload(
        rig, monkeypatch):
    """The store frees DTensor parameters in place before comfy's global unload, which
    would otherwise stage each rank's shards to host. That took minutes in a refused
    ceremony's abort arm on 2026-08-26 (adapters/fsdp_shard_build.drop_sharded_weights_in_place)."""
    store, _calls = rig
    _active, transition = store.ensure("m.safetensors", None, STACK)
    assert transition == TRANSITION_LOAD

    events = []
    monkeypatch.setattr(
        ms.fsdp_shard_build, "drop_sharded_weights_in_place",
        lambda patchers, origin="": (events.append(("drop", origin)), (2, 64))[1])
    mm = sys.modules["comfy.model_management"]
    healthy_unload = mm.unload_all_models

    def unload_all_models():
        events.append(("unload", None))
        healthy_unload()

    monkeypatch.setattr(mm, "unload_all_models", unload_all_models)
    store.unload_all()
    assert events[0] == ("drop", "model store [cond]")
    assert ("unload", None) in events
    assert events.index(("unload", None)) > 0


def _short_slab(size=62 * 2**30, avail=47 * 2**30):
    from dgx_monarch.capacity_fit import SlabFit

    return SlabFit(False, True, size, avail, 4 * 2**30)


def _fitting_slab():
    from dgx_monarch.capacity_fit import SlabFit

    return SlabFit(True, True, 20 * 2**30, 90 * 2**30, 4 * 2**30)


def test_a_slab_load_that_does_not_fit_never_reaches_the_bridge(rig, monkeypatch):
    """The refusal happens before ComfyUI is entered, so nothing was loaded."""
    from dgx_monarch import capacity_fit
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    store, calls = rig
    store.slab_weights = True
    monkeypatch.setattr(capacity_fit, "slab_load_fit",
                        lambda _path, _options, **_kw: _short_slab())

    with pytest.raises(StockLoadCapacityError, match="slab residency cannot load"):
        store.ensure("m.safetensors", None, STACK)

    assert calls.slabs == []
    assert calls.load == 0
    assert store.current is None


def test_the_second_wall_catches_a_memory_drop_after_the_price(rig, monkeypatch):
    """Memory can drop between the rung's price and the load.

    ``preload_capacity_check`` runs on the slab branch as well as the classic
    one, at the last monarch-owned moment before ComfyUI is entered, so a box
    that lost the memory in between refuses instead of allocating.
    """
    from dgx_monarch import capacity_fit
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    store, calls = rig
    store.slab_weights = True
    prices = [_fitting_slab(), _short_slab()]
    monkeypatch.setattr(capacity_fit, "slab_load_fit",
                        lambda _path, _options, **_kw: prices.pop(0))

    with pytest.raises(StockLoadCapacityError, match="slab residency cannot load"):
        store.ensure("m.safetensors", None, STACK)

    assert prices == []          # the rung priced, then the load path re-priced
    assert calls.slabs == []
    assert calls.load == 0


def test_a_fitting_slab_load_still_runs_through_both_walls(rig, monkeypatch):
    from dgx_monarch import capacity_fit
    from dgx_monarch.constants import TRANSITION_LOAD

    store, calls = rig
    store.slab_weights = True
    seen = []

    # Name reserve_bytes explicitly so dropping the keyword fails this test;
    # **_kw would silently accept the missing reserve.
    def probe(path, _options, *, reserve_bytes=None, **_kw):
        seen.append((path, reserve_bytes))
        return _fitting_slab()

    monkeypatch.setattr(capacity_fit, "slab_load_fit", probe)
    _active, transition = store.ensure("m.safetensors", None, STACK)

    assert transition == TRANSITION_LOAD
    assert store.current.slab is calls.slabs[0]
    assert seen == [("/fake/m.safetensors", 0), ("/fake/m.safetensors", 0)]


def test_the_reserve_the_worker_set_is_what_both_slab_walls_charge(rig, monkeypatch, isolated_environ):
    """The operator's headroom is charged where the load happens.

    ``uma_reserve_gb`` is co-residency headroom the driver already charges on
    the head, so a slab wall that charges the default floor instead admits on
    the rank what the driver refuses. Both walls of one load take the figure
    from the store, and the worker sets it there.
    """
    from dgx_monarch import capacity_fit
    from dgx_monarch.actor import comfy_bridge, worker_capacity, worker_env

    store, _calls = rig
    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None, raising=False)
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _wa: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", dict)
    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)
    worker = SimpleNamespace(
        store=store,
        topology={"dp": 1, "ulysses": 2, "ring": 1, "cfg": 1, "fsdp": False},
        uma_reserve_gb=0.0,
        _slab_mode_effective=worker_env.slab_mode_effective,
        _active_worker_args={},
    )
    worker_env.apply_worker_args_impl(
        worker,
        {"slab_weights": True, "lora_low_rss": True, "uma_reserve_gb": 10.0},
    )
    seen = []

    def probe(_path, _options, *, reserve_bytes=None, **_kw):
        seen.append(reserve_bytes)
        return _fitting_slab()

    monkeypatch.setattr(capacity_fit, "slab_load_fit", probe)
    store.ensure("m.safetensors", None, STACK)

    assert seen == [10 * 2 ** 30, 10 * 2 ** 30]
    # The same number the driver's own quote charges for this worker.
    assert worker_capacity._reserve_bytes(worker) == seen[0]



def _unsupported_dtype_slab(monkeypatch):
    from dgx_monarch.actor import comfy_bridge

    class UnsupportedSlab:
        def __enter__(self):
            raise UnsupportedSafetensorsDtypeError(
                "model.safetensors: dtype 'FUTURE' is not supported by this torch build")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(comfy_bridge, "slab_load",
                        lambda _path, *, handoff=None: UnsupportedSlab())


def test_the_dtype_fallback_takes_the_ladder_price_not_the_bare_file_size(
        rig, monkeypatch):
    """A stored dtype the slab cannot wrap falls back to a stock load at the ladder's price.

    A wall that charged the file alone would pass every load between 1.0x and
    2.1x the file, a band that includes both recorded kills (1.27x on 2026-08-12
    and 1.694x on 2026-09-02; docs/VALIDATION.md, the 2026-09-02 stock-load section).
    """
    from dgx_monarch import capacity_fit
    from dgx_monarch.capacity_fit import StockFit
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    store, calls = rig
    store.slab_weights = True
    _unsupported_dtype_slab(monkeypatch)
    short = StockFit(False, True, 62 * 2 ** 30, 47 * 2 ** 30)
    monkeypatch.setattr(capacity_fit, "stock_load_fit", lambda _p, _o: short)

    with pytest.raises(StockLoadCapacityError) as raised:
        store.ensure("m.safetensors", None, STACK)

    message = str(raised.value)
    assert (f"the load needs {short.required_gib} GiB (a {short.size_gib} GiB "
            "file, the host copy") in message
    assert "Slab residency cannot help" in message
    assert "guard=stock_load_preflight waivable=0" in message
    assert calls.load == 0
    assert store.current is None


def test_the_dtype_fallback_still_loads_where_the_ladder_price_fits(rig, monkeypatch):
    from dgx_monarch import capacity_fit
    from dgx_monarch.capacity_fit import StockFit

    store, calls = rig
    store.slab_weights = True
    _unsupported_dtype_slab(monkeypatch)
    monkeypatch.setattr(capacity_fit, "stock_load_fit",
                        lambda _p, _o: StockFit(True, True, 20 * 2 ** 30, 90 * 2 ** 30))

    _active, transition = store.ensure("m.safetensors", None, STACK)

    assert transition == TRANSITION_LOAD
    assert calls.load == 1
    assert store.current.slab is None
