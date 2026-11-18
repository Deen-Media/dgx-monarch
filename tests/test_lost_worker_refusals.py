"""A refusal the driver raises while it handles a lost Worker stays exactly that refusal.

After a memory refusal the driver unloads what ComfyUI holds and asks the
actors to release their caches before the refusal stands
(driver_unload._after_refusal, reached from the render preflight and the spend
price). When a Worker is gone, that trim's call fails with Monarch's error,
which the receive point words (worker_box), and the refusal is raised
``from None`` inside the handler. The node layer (nodes/refusal_lead) leaves it
as raised: its own type, its tag first, no lost-Worker words. An error of a type
the pack declares, or with a tag, raised ``from`` the loss stays as it is too;
an error raised while the loss was handled, with no ``from``, was not caused by
it. A generic error of Python's own type raised ``from`` the loss still opens
with the loss's cause, as a ``WorkerLostError``.

These run the real raise sites (render_memory_price, the spend price), the real
check_with_unload, trim_actors, call_all and call_all_bound, the real worker_box
naming and lead_with_refusal, with Monarch's own SupervisionError (built from a
text captured on CPU) as the trim's failure.
"""
from __future__ import annotations

import contextlib
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

import test_worker_loss as captured
from dgx_monarch import driver_footprint, driver_unload, mesh_rpc, mesh_session, mesh_setup, vae_decode_relief
from dgx_monarch.delivery_faults import WorkerDeliveryError
from dgx_monarch.driver_spend_price import DriverSpendCapacityError
from dgx_monarch.native_return_latch import NativeReturnLatchedError
from dgx_monarch.nodes import refusal_lead
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag, refusal
from dgx_monarch.render_memory_price import RenderMemoryPriceError
from dgx_monarch.studio.job_observations import observation
from dgx_monarch.worker_box import LOST_WORKER
from dgx_monarch.worker_loss import WorkerLostError
from test_driver_spend_price import _guide
from test_driver_spend_price import rig as spend_rig  # noqa: F401  # a fixture: the spend price on stand-ins
from test_lost_worker_words import PAIR_CONFIG, _supervision_error
from test_render_memory_price import _arm, _shape_only_checkpoint

WORDS = "The Worker on the sibling was stopped by signal 9 (SIGKILL)"


def _dying_fleet(monkeypatch, error) -> SimpleNamespace:
    """A set-up fleet whose every call fails with ``error``, through the real call path."""
    monkeypatch.setattr(driver_footprint, "host_is_local", lambda name: name == "box-a")
    monkeypatch.setattr(mesh_setup, "dispatch_endpoint", lambda _h, _n, send, setup_token=None: send())
    monkeypatch.setattr(mesh_setup, "AMBIGUOUS_MUTATION_ENDPOINTS", frozenset())
    monkeypatch.setattr(mesh_session, "mutation_render_session", lambda *_a, **_k: contextlib.nullcontext())
    monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: None)  # ComfyUI holds nothing
    monkeypatch.setattr(driver_unload, "busy_reason", lambda handle: "")
    handle = SimpleNamespace(
        world=2, gpus_per_host=1, owns_hosts=True, config=PAIR_CONFIG, active_worker_args={},
        setup_key=object(), defunct=False, teardown_complete=False, setup_cleanup_state=None,
        lock=threading.RLock(), setup_generation=1)
    handle.workers = SimpleNamespace(**{driver_unload.TRIM_ENDPOINT: SimpleNamespace(call=lambda *_a, **_k: object())})

    def await_or_evict(_future, _timeout):
        raise error

    handle._await_or_evict = await_or_evict
    handle._call_all_bound = lambda name, *a, **k: mesh_rpc.call_all_bound(handle, name, *a, **k)
    handle.call_all = lambda name, *a, **k: mesh_rpc.call_all(handle, name, *a, **k)
    return handle


def _assert_the_refusal_stands(raised: BaseException, error, refusal_type: type, guard: str) -> None:
    text = str(raised)
    assert type(raised) is refusal_type, (type(raised), text[:300])
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.guard == guard and tag.refusal_class is RefusalClass.CAPACITY
    # The trim ran and failed on the lost Worker, whose words were recorded and not used.
    assert "that request failed (SupervisionError)" in text
    assert getattr(error, LOST_WORKER, "").startswith(WORDS)
    assert "signal 9" not in text and "WorkerLostError" not in repr(raised)


def test_a_render_memory_refusal_after_a_lost_workers_trim_stays_the_refusal(monkeypatch, tmp_path):
    from dgx_monarch.nodes import render_preflight

    error = _supervision_error(captured.KILL_ERROR)
    handle = _dying_fleet(monkeypatch, error)
    checkpoint = _shape_only_checkpoint(str(tmp_path / "big.safetensors"))
    folder_paths = ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: checkpoint
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint", lambda _path: ("zimage", "bf16"))
    _arm(monkeypatch, bound=137 * 2**30, avail=93 * 2**30)

    class _Latent:
        shape = (1, 16, 1, 128, 128)

    model = SimpleNamespace(unet_name="big.safetensors",
                            mesh=SimpleNamespace(world=2, worker_args={}, handle=handle))

    @refusal_lead.lead_with_refusal
    def sample():  # a render node's method: the preflight runs first (nodes/render_preflight.py)
        render_preflight.activation_footprint_preflight_for_request(model, {}, {"samples": _Latent()})

    with pytest.raises(Exception) as caught:
        sample()
    _assert_the_refusal_stands(caught.value, error, RenderMemoryPriceError, "activation_footprint_preflight")


@pytest.mark.parametrize("node_layer", [False, True], ids=["as-comfyui-runs-it", "under-lead_with_refusal"])
def test_a_spend_refusal_after_a_lost_workers_trim_stays_the_refusal(spend_rig, monkeypatch, node_layer):  # noqa: F811
    error = _supervision_error(captured.KILL_ERROR)
    handle = _dying_fleet(monkeypatch, error)
    monkeypatch.setattr(vae_decode_relief, "_serving", lambda: (handle, ""))
    spend_rig.box.value = 8 * 2**30  # the spend and the floor do not fit
    run = refusal_lead.lead_with_refusal(lambda: _guide(1)) if node_layer else (lambda: _guide(1))
    with pytest.raises(Exception) as caught:
        run()
    _assert_the_refusal_stands(caught.value, error, DriverSpendCapacityError, "driver_footprint_preflight")


def _node_method(handle, handler):
    @refusal_lead.lead_with_refusal
    def method():
        try:
            return mesh_rpc.call_all_bound(handle, driver_unload.TRIM_ENDPOINT, "soft")
        except Exception as exc:
            handler(exc)

    return method


def _tagged(exc):
    raise WorkerDeliveryError(refusal(RefusalClass.CAPACITY, "a delivery refusal chained from the loss",
                                      guard="cross_rank_capacity", troubleshooting=2)) from exc


def _typed(exc):
    raise NativeReturnLatchedError("a lifecycle fault of the pack's own type, untagged") from exc


def _tagged_builtin(exc):
    raise RuntimeError(refusal(RefusalClass.CAPACITY, "a tagged refusal of Python's own type",
                               guard="cross_rank_capacity", troubleshooting=2)) from exc


def _while_handling(exc):
    raise ValueError("an unrelated driver error raised while handling the loss")


@pytest.mark.parametrize("handler, raised_type, phrase", [
    (_tagged, WorkerDeliveryError, "a delivery refusal chained from the loss"),
    (_typed, NativeReturnLatchedError, "a lifecycle fault of the pack's own type"),
    (_tagged_builtin, RuntimeError, "a tagged refusal of Python's own type"),
    (_while_handling, ValueError, "an unrelated driver error raised while handling the loss"),
], ids=["tagged-pack-type", "untagged-pack-type", "tagged-builtin", "no-from"])
def test_an_error_the_driver_raised_around_a_loss_is_never_reworded(monkeypatch, handler, raised_type, phrase):
    error = _supervision_error(captured.KILL_ERROR)
    method = _node_method(_dying_fleet(monkeypatch, error), handler)
    with pytest.raises(Exception) as caught:
        method()
    assert type(caught.value) is raised_type, str(caught.value)[:300]
    assert phrase in str(caught.value) and WORDS not in str(caught.value)
    assert getattr(error, LOST_WORKER, "").startswith(WORDS)  # the words were there to take


def test_a_generic_error_raised_from_the_loss_opens_with_its_cause(monkeypatch):
    error = _supervision_error(captured.KILL_ERROR)

    def wrap(exc):
        raise RuntimeError("the cache release failed") from exc

    method = _node_method(_dying_fleet(monkeypatch, error), wrap)
    with pytest.raises(WorkerLostError) as caught:
        method()
    assert str(caught.value).startswith(WORDS)
    assert str(caught.value).endswith("\nthe cache release failed")


def test_comfyui_and_studio_name_the_lost_workers_type(monkeypatch):
    """ComfyUI's execution_error carries ``full_type_name(type(ex))`` (execution.py); Studio's code is that string."""
    error = _supervision_error(captured.KILL_ERROR)
    method = _node_method(_dying_fleet(monkeypatch, error), lambda exc: (_ for _ in ()).throw(exc))
    with pytest.raises(WorkerLostError) as caught:
        method()
    klass = type(caught.value)
    full = klass.__qualname__ if klass.__module__ == "builtins" else f"{klass.__module__}.{klass.__qualname__}"
    assert full == "dgx_monarch.worker_loss.WorkerLostError"
    assert isinstance(caught.value, RuntimeError) and caught.value.__cause__ is error
    _event, updates = observation({"type": "execution_error", "data": {
        "prompt_id": "p", "exception_type": full, "exception_message": str(caught.value)}})
    assert updates["error"]["code"] == "dgx_monarch.worker_loss.WorkerLostError"
    before = type(error)  # what 1.4.0 showed: Monarch's own error reached ComfyUI as it was
    assert f"{before.__module__}.{before.__qualname__}" == (
        "monarch._rust_bindings.monarch_hyperactor.supervision.SupervisionError")
