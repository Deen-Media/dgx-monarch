"""``[worker_args] fp8_matrix_mult`` and the ``FP8 compute:`` build line.

ComfyUI's ``--fast`` flags never reach the actors, so the actors take the FP8
matmul path through the loader option ComfyUI reads beside the flag
(src/dgx_monarch/fp8_compute.py). These cases hold the knob's validation, how
each mode rewrites an actor build's comfy options, that the options reach
comfy's load call, that a changed mode drops the residents, the doctor row and
status line, and the one INFO line per build that names the class ComfyUI
picked. Comfy itself is a stand-in: the wrapper never imports it.
"""
from __future__ import annotations

import ast
import enum
import inspect
import sys
import textwrap
import types
from types import SimpleNamespace

import pytest

from dgx_monarch import fp8_compute, monarch_policy_io
from dgx_monarch.actor import worker_env
from dgx_monarch.cli import fp8_matmul_row, json_report, mesh_health_row
from dgx_monarch.config import ClusterConfig
from dgx_monarch.config_schema import ClusterConfigError, validate_worker_args
from store_ensure_helpers import STACK
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.

pytestmark = pytest.mark.usefixtures("isolated_environ")


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Each case starts from a process that has configured nothing."""
    monkeypatch.setattr(fp8_compute, "_STATE", fp8_compute._State())


@pytest.fixture
def lines(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(fp8_compute.log, "info", lambda fmt, *args: seen.append(fmt % args))
    return seen


@pytest.mark.parametrize("mode", ["auto", "on", "off"])
def test_the_config_accepts_the_three_modes(mode):
    assert validate_worker_args({"fp8_matrix_mult": mode}) == {"fp8_matrix_mult": mode}


@pytest.mark.parametrize("value", [True, "ON", "fast", 1, None])
def test_the_config_refuses_anything_else(value):
    with pytest.raises(ClusterConfigError, match='fp8_matrix_mult must be "auto", "on" or "off"'):
        validate_worker_args({"fp8_matrix_mult": value})


def test_the_setting_changes_math_so_it_changes_the_trust_fingerprint():
    """A Gate PASS earned under one setting must not authorize the other."""
    fingerprint = monarch_policy_io.trust_config_fingerprint
    auto = fingerprint(ClusterConfig(), "fallback")
    on = fingerprint(ClusterConfig(worker_args={"fp8_matrix_mult": "on"}), "fallback")
    off = fingerprint(ClusterConfig(worker_args={"fp8_matrix_mult": "off"}), "fallback")
    assert len({auto, on, off}) == 3


@pytest.mark.parametrize(("mode", "given", "expected"), [
    ("auto", {"dtype": "fp8", "fp8_optimizations": True}, {"dtype": "fp8", "fp8_optimizations": True}),
    ("auto", {}, {}),
    ("on", {}, {"fp8_optimizations": True}),
    ("on", {"dtype": "fp8"}, {"dtype": "fp8", "fp8_optimizations": True}),
    ("off", {"dtype": "fp8", "fp8_optimizations": True}, {"dtype": "fp8"}),
    ("off", {}, {}),
])
def test_each_mode_rewrites_a_build_options(mode, given, expected):
    fp8_compute.configure_from_worker_args({"fp8_matrix_mult": mode})
    assert fp8_compute.load_options(given) == expected
    assert fp8_compute.load_options(given) is not given


def test_configure_reports_a_change_once(lines):
    assert fp8_compute.configure_from_worker_args({}) is False
    assert fp8_compute.configure_from_worker_args({"fp8_matrix_mult": "on"}) is True
    assert fp8_compute.configure_from_worker_args({"fp8_matrix_mult": "on"}) is False
    assert fp8_compute.configure_from_worker_args(None) is True
    assert fp8_compute.current_mode() == "auto"
    assert lines == [
        "FP8 compute: this actor builds with fp8_matrix_mult=on from [worker_args] "
        "fp8_matrix_mult; the next build of each model takes it",
        "FP8 compute: this actor builds with fp8_matrix_mult=auto from the default; "
        "the next build of each model takes it",
    ]


@pytest.mark.parametrize(("mode", "weight_dtype", "expected"), [
    ("auto", "fp8_e4m3fn_fast", {"dtype": "float8_e4m3fn", "fp8_optimizations": True}),
    ("auto", "default", {}),
    ("on", "default", {"fp8_optimizations": True}),
    ("on", "fp8_e4m3fn", {"dtype": "float8_e4m3fn", "fp8_optimizations": True}),
    ("off", "fp8_e4m3fn_fast", {"dtype": "float8_e4m3fn"}),
])
def test_the_option_reaches_the_actor_load_call(rig, mode, weight_dtype, expected):
    """What comfy.sd.load_diffusion_model receives in the actor, per mode and loader dtype."""
    import torch

    store, calls = rig
    fp8_compute.configure_from_worker_args({"fp8_matrix_mult": mode})
    store.ensure("m.safetensors", {"weight_dtype": weight_dtype}, STACK)
    want = {key: getattr(torch, value) if key == "dtype" else value for key, value in expected.items()}
    assert calls.model_options == [want]


class _PolicyStore:
    def __init__(self) -> None:
        self.lora_low_rss = True
        self.slab_weights: bool | str = "auto"
        self.swap_verify = 2
        self.family_override: str | None = None
        self.events: list[str] = []

    def set_lora_mode(self, low_rss: bool) -> None:
        self.lora_low_rss = bool(low_rss)

    def set_slab_mode(self, slab_weights, reason: str = "") -> None:
        self.slab_weights = slab_weights

    def unload_all(self) -> None:
        self.events.append("unload_all")

    def release_retained_cleanup(self) -> None:
        self.events.append("release")


def test_a_changed_mode_drops_every_resident(monkeypatch):
    """The operations class binds when comfy builds the model, so a resident
    built under the previous mode cannot serve the new one."""
    from dgx_monarch.actor import comfy_bridge

    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None, raising=False)
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _wa: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", dict)
    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)
    store = _PolicyStore()
    worker = SimpleNamespace(store=store, topology={"ulysses": 2}, uma_reserve_gb=0.0,
                             _slab_mode_effective=worker_env.slab_mode_effective,
                             _active_worker_args={})

    for args in ({"fp8_matrix_mult": "on"}, {"fp8_matrix_mult": "on"}, {}, {"fp8_matrix_mult": "auto"}):
        worker_env.apply_worker_args_impl(worker, args)
        worker._active_worker_args = dict(args)
    assert store.events == ["unload_all", "release", "unload_all", "release"]
    assert fp8_compute.current_mode() == "auto"


def _calls_in(function) -> list[tuple[int, str]]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return sorted((node.lineno, ast.unparse(node.func)) for node in ast.walk(tree)
                  if isinstance(node, ast.Call))


def test_setup_takes_the_mode_after_bootstrap_and_before_nccl():
    calls = _calls_in(worker_env.setup_impl)
    order = [name for _line, name in calls]
    configure = order.index("fp8_compute.configure_from_worker_args")
    assert order.index("ensure_comfy") < configure < order.index("dist.init_process_group")
    source = inspect.getsource(worker_env.setup_impl)
    assert "fp8_compute.configure_from_worker_args(_wa_eff)" in source


def _fake_comfy(monkeypatch, *, flag: bool, fp8: bool = True):
    class Feature(enum.Enum):
        Fp8MatrixMultiplication = "fp8_matrix_mult"
        AutoTune = "autotune"

    picks: list[dict] = []

    class fp8_ops:  # comfy.ops spells its classes in lower case
        pass

    class manual_cast:
        pass

    def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False,
                        fp8_optimizations=False, model_config=None):
        picks.append({"weight_dtype": weight_dtype, "fp8_optimizations": fp8_optimizations})
        if model_config is not None and getattr(model_config, "quant_config", None):
            def mixed_precision_ops():
                class MixedPrecisionOps:
                    pass
                return MixedPrecisionOps
            return mixed_precision_ops()
        if fp8 and (fp8_optimizations or flag):
            return fp8_ops
        return manual_cast

    ops = types.ModuleType("comfy.ops")
    ops.pick_operations = pick_operations
    cli_args = types.ModuleType("comfy.cli_args")
    cli_args.args = SimpleNamespace(fast={Feature.Fp8MatrixMultiplication} if flag else set())
    mm = types.ModuleType("comfy.model_management")
    mm.supports_fp8_compute = lambda device=None: fp8
    for name, module in (("comfy.ops", ops), ("comfy.cli_args", cli_args), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, module)
    return ops, picks


def test_install_needs_comfy_already_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "comfy.ops", raising=False)
    assert fp8_compute.install("the driver") is False
    assert "comfy.ops" not in sys.modules


def test_the_driver_logs_one_line_per_pick_and_returns_comfys_result(monkeypatch, lines):
    ops, picks = _fake_comfy(monkeypatch, flag=True)
    original = ops.pick_operations
    assert fp8_compute.install_driver() is True
    assert fp8_compute.install_driver() is True
    wrapped = ops.pick_operations
    assert wrapped is not original and getattr(wrapped, "__wrapped__", None) is original

    result = wrapped("torch.float8_e4m3fn", "torch.bfloat16", load_device="cuda:0")
    assert result.__qualname__.startswith("_fake_comfy.<locals>.fp8_ops") or result.__name__ == "fp8_ops"
    assert picks == [{"weight_dtype": "torch.float8_e4m3fn", "fp8_optimizations": False}]
    assert lines == [
        "FP8 compute: the driver picked fp8_ops for weights stored as float8_e4m3fn, compute dtype "
        "bfloat16 (fp8_optimizations=False, --fast fp8_matrix_mult on, FP8 compute supported); "
        "linears stored as float8_e4m3fn run torch._scaled_mm on FP8 tensor cores"]


def test_operations_names_cover_comfys_classes():
    def mixed_precision_ops():
        class MixedPrecisionOps:
            pass
        return MixedPrecisionOps

    assert fp8_compute.operations_name(mixed_precision_ops()) == "mixed_precision_ops"
    named = type("fp8_ops", (), {})
    named.__qualname__ = "fp8_ops"
    assert fp8_compute.operations_name(named) == "fp8_ops"
    other = type("CustomOps", (), {})
    other.__qualname__ = "somepack.CustomOps"
    assert fp8_compute.operations_name(other) == "somepack.CustomOps"


def test_an_actor_line_names_the_mode_and_the_dequantizing_path(monkeypatch, lines):
    ops, _picks = _fake_comfy(monkeypatch, flag=False)
    fp8_compute.configure_from_worker_args({"fp8_matrix_mult": "off"})
    lines.clear()
    assert fp8_compute.install("this actor") is True
    ops.pick_operations("torch.float8_e4m3fn", "torch.bfloat16")
    assert lines == [
        "FP8 compute: this actor picked manual_cast for weights stored as float8_e4m3fn, compute dtype "
        "bfloat16 (fp8_optimizations=False, --fast fp8_matrix_mult off, FP8 compute supported, "
        "fp8_matrix_mult=off from [worker_args] fp8_matrix_mult); FP8 weights are dequantized to "
        "the compute dtype for every matmul"]


def test_kitchen_files_and_full_precision_fp8_ops_say_what_runs(monkeypatch, lines):
    ops, _picks = _fake_comfy(monkeypatch, flag=True)
    fp8_compute.install("the driver")
    ops.pick_operations(None, "torch.bfloat16", model_config=SimpleNamespace(quant_config={"mixed_ops": True}))
    ops.pick_operations("torch.bfloat16", None)
    assert "picked mixed_precision_ops for comfy-kitchen quantized weights" in lines[0]
    assert lines[0].endswith("comfy-kitchen's kernels run the quantized linears whatever the flags say")
    assert "picked fp8_ops for weights stored as bfloat16" in lines[1]
    assert lines[1].endswith("so these keep the cast path")


def test_a_failing_pick_raises_as_comfys_own_and_a_failing_line_is_silent(monkeypatch, lines):
    ops, _picks = _fake_comfy(monkeypatch, flag=False)

    def broken(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False,
               fp8_optimizations=False, model_config=None):
        raise RuntimeError("comfy failed")

    ops.pick_operations = broken
    fp8_compute.install("the driver")
    with pytest.raises(RuntimeError, match="comfy failed"):
        ops.pick_operations("torch.bfloat16", None)
    monkeypatch.setattr(fp8_compute, "describe", lambda *_args: 1 / 0)
    ops2, _ = _fake_comfy(monkeypatch, flag=False)
    fp8_compute.install("the driver")
    assert ops2.pick_operations("torch.bfloat16", None).__name__ == "manual_cast"
    assert lines == []


@pytest.mark.parametrize(("args", "value", "source"), [
    ({}, "auto", "the default"),
    ({"fp8_matrix_mult": "on"}, "on", "[worker_args] fp8_matrix_mult"),
    ({"fp8_matrix_mult": "off"}, "off", "[worker_args] fp8_matrix_mult"),
])
def test_doctor_and_status_show_the_setting(args, value, source):
    config = ClusterConfig(worker_args=args)
    status, name, detail = fp8_matmul_row.doctor_row(config)
    assert (status, name) == ("ok", "FP8 matmul (actors)")
    assert detail.startswith(f"fp8_matrix_mult={value} from {source}: ")
    assert "`FP8 compute:`" in detail and "docs/TROUBLESHOOTING.md #141" in detail
    assert fp8_matmul_row.status_fields(config) == {"value": value, "source": source}
    line = fp8_matmul_row.status_line(config)
    # Status keeps its rows as they were for a file that sets nothing.
    assert (line is None) if not args else line.startswith(f"actors fp8_matrix_mult={value} ({source};")
    assert fp8_matmul_row.doctor_row(None)[2].startswith("fp8_matrix_mult=auto from the default")


def test_status_prints_the_line_and_carries_the_field(monkeypatch, capsys):
    import json

    rows = [{"host": "worker", "loop": "running", "port": "open", "address": "tcp://192.0.2.2:26600",
             "mode": "systemd"}]
    monkeypatch.setattr(mesh_health_row.lifecycle, "status", lambda _config: rows)
    monkeypatch.setattr(mesh_health_row, "fetch_mesh_block", lambda: None)
    config = ClusterConfig(worker_args={"fp8_matrix_mult": "on"})
    mesh_health_row.status_report(config)
    assert "  actors fp8_matrix_mult=on ([worker_args] fp8_matrix_mult;" in capsys.readouterr().out
    mesh_health_row.status_report(config, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["fp8_matrix_mult"] == {"value": "on", "source": "[worker_args] fp8_matrix_mult"}
    assert "fp8_matrix_mult" not in json_report.status_payload(rows, "ok", "ok")
    # A file that sets nothing prints no line, and the JSON still says auto.
    mesh_health_row.status_report(ClusterConfig())
    assert "fp8_matrix_mult" not in capsys.readouterr().out
    mesh_health_row.status_report(ClusterConfig(), as_json=True)
    assert json.loads(capsys.readouterr().out)["fp8_matrix_mult"] == {"value": "auto", "source": "the default"}
