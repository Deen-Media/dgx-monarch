"""The int8_gemm option's rails, on CPU with stub comfy-kitchen modules.

The option (src/dgx_monarch/kernels/int8_gemm.py) wraps comfy-kitchen's CUDA
extension so its INT8 GEMM runs the vendored Triton kernel. These tests pin
what must hold without a GPU: the proxy's decline rules hand every call they
exclude to CUTLASS untouched, the knob's values and sources, the version
guard, one WARNING and stock CUTLASS on every failure, the in-place wrap of an
imported comfy-kitchen, the import hook for one imported later, and the order
in which actors and the driver install it. The real kernel's equality with
CUTLASS is tests/test_int8_gemm_triton_equality.py, on a GPU.
"""
from __future__ import annotations

import inspect
import logging
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

from dgx_monarch.config_schema import ClusterConfigError, validate_worker_args
from dgx_monarch.kernels import int8_gemm
from dgx_monarch.kernels import int8_gemm_triton as vendored

REPO = Path(__file__).resolve().parents[1]


def _kitchen_module(name: str) -> bool:
    return name == "comfy_kitchen" or name.startswith("comfy_kitchen.")


class _Lines(logging.Handler):
    """Every record the option's logger emits, as (level, message)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelname, record.getMessage()))


class _Gemm:
    """The Triton kernel's stand-in: records each call's tensors."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[tuple] = []
        self.error = error

    def __call__(self, *tensors):
        self.calls.append(tensors)
        if self.error is not None:
            raise self.error


@pytest.fixture
def lines():
    handler = _Lines()
    int8_gemm.log.addHandler(handler)
    try:
        yield handler.records
    finally:
        int8_gemm.log.removeHandler(handler)


@pytest.fixture
def option(monkeypatch):
    """A fresh option state with stubbed prerequisites and no comfy-kitchen loaded.

    The import hook goes into sys.meta_path and comfy-kitchen modules into
    sys.modules, both process-wide, so both are put back afterwards. The suite
    process may hold the real comfy-kitchen and the hook the repository-root
    __init__.py registered; neither is visible while a test runs.
    """
    monkeypatch.setattr(int8_gemm, "_STATE", int8_gemm._State())
    monkeypatch.setattr(vendored, "kitchen_version", lambda: "0.2.40")
    monkeypatch.setattr(int8_gemm, "python_headers", lambda: (True, "/include"))
    gemm = _Gemm()
    monkeypatch.setattr(vendored, "build_triton_gemm", lambda: gemm)
    saved_path = list(sys.meta_path)
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, vendored.Finder)]
    saved_modules = {name: module for name, module in sys.modules.items() if _kitchen_module(name)}
    for name in saved_modules:
        del sys.modules[name]
    try:
        yield gemm
    finally:
        for name in [name for name in sys.modules if _kitchen_module(name)]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.meta_path[:] = saved_path


def _stub_c():
    calls: list[tuple] = []

    def cutlass_int8_dequant(*args):
        calls.append(args)
        return "cutlass"

    def cutlass_int8_dequant_residual(*args):
        return "residual"

    real = types.SimpleNamespace(cutlass_int8_dequant=cutlass_int8_dequant,
                                 cutlass_int8_dequant_residual=cutlass_int8_dequant_residual,
                                 HAS_CUBLASLT=True, _private=1)
    return real, calls


def _stub_kitchen():
    """An imported comfy-kitchen: the package, its CUDA backend and the dtype table."""
    real, calls = _stub_c()
    modules = {name: types.ModuleType(name) for name in (
        "comfy_kitchen", "comfy_kitchen.backends", "comfy_kitchen.backends.cuda",
        "comfy_kitchen.backends.eager", "comfy_kitchen.backends.eager.quantization")}
    modules["comfy_kitchen.backends.eager.quantization"].DTYPE_CODE_TO_DTYPE = {
        0: torch.float32, 1: torch.float16, 2: torch.bfloat16}
    modules["comfy_kitchen.backends.cuda"]._C = real
    linear_calls: list[tuple] = []

    def int8_linear(*args):
        linear_calls.append(args)
        return "eager"

    modules["comfy_kitchen"].int8_linear = int8_linear
    sys.modules.update(modules)
    return modules["comfy_kitchen"], modules["comfy_kitchen.backends.cuda"], real, calls, linear_calls


# --- the proxy's decline rules -------------------------------------------------


def _operands(M=128, N=16, K=32, out_dtype=torch.bfloat16, bias=0):
    x = torch.zeros(M, K, dtype=torch.int8)
    w = torch.zeros(N, K, dtype=torch.int8)
    xs = torch.ones(M, 1, dtype=torch.float32)
    ws = torch.ones(N, dtype=torch.float32)
    b = torch.zeros(bias, dtype=out_dtype)
    out = torch.zeros(M, N, dtype=out_dtype)
    return [x, w, xs, ws, b, out]


def _call(proxy, tensors):
    capsules = [tensor.__dlpack__(stream=-1) for tensor in tensors]
    return proxy.cutlass_int8_dequant(*capsules, 2, 0)


def _same_storage(got, tensors):
    return [torch.from_dlpack(t).data_ptr() for t in got[:6]] == [t.data_ptr() for t in tensors]


def test_an_eligible_call_runs_the_triton_kernel_on_the_callers_own_tensors():
    real, calls = _stub_c()
    gemm = _Gemm()
    proxy = vendored.CProxy(real, gemm, lambda exc: None)
    tensors = _operands()
    assert _call(proxy, tensors) is True
    assert calls == []
    assert len(gemm.calls) == 1
    assert [t.data_ptr() for t in gemm.calls[0]] == [t.data_ptr() for t in tensors]


def _strided(tensors):
    wide = torch.zeros(tensors[0].shape[0], tensors[0].shape[1] * 2, dtype=torch.int8)
    return [wide[:, :tensors[0].shape[1]], *tensors[1:]]


DECLINES = {
    "K not a multiple of 16": lambda: _operands(K=40),
    "rows under MIN_M": lambda: _operands(M=vendored.MIN_M - 1),
    "a strided activation": lambda: _strided(_operands()),
    "a strided output": lambda: [*_operands()[:5], torch.zeros(16, 128, dtype=torch.bfloat16).t()],
    "a float activation": lambda: [torch.zeros(128, 32), *_operands()[1:]],
    "an int32 output": lambda: [*_operands()[:5], torch.zeros(128, 16, dtype=torch.int32)],
    "a row-scale count off M": lambda: [*_operands()[:2], torch.ones(127, dtype=torch.float32), *_operands()[3:]],
    "a weight-scale count off N and 1": lambda: [*_operands()[:3], torch.ones(15, dtype=torch.float32), *_operands()[4:]],
    "a bias in another dtype": lambda: [*_operands()[:4], torch.zeros(16, dtype=torch.float16), _operands()[5]],
    "an output of the wrong shape": lambda: [*_operands()[:5], torch.zeros(128, 8, dtype=torch.bfloat16)],
}


@pytest.mark.parametrize("label", sorted(DECLINES))
def test_a_declined_call_reaches_cutlass_unchanged(label):
    real, calls = _stub_c()
    gemm = _Gemm()
    proxy = vendored.CProxy(real, gemm, lambda exc: None)
    tensors = DECLINES[label]()
    assert _call(proxy, tensors) == "cutlass"
    assert gemm.calls == []
    assert len(calls) == 1
    assert _same_storage(calls[0], tensors), "CUTLASS got other tensors than the caller's"
    assert calls[0][6:] == (2, 0), "the dtype code and stream must pass through"


def test_fused_residual_calls_and_other_attributes_stay_on_the_real_extension():
    real, _calls = _stub_c()
    proxy = vendored.CProxy(real, _Gemm(), lambda exc: None)
    assert proxy.cutlass_int8_dequant_residual is real.cutlass_int8_dequant_residual
    assert proxy.HAS_CUBLASLT is True
    with pytest.raises(AttributeError):
        proxy._private  # noqa: B018 - private names never fall through


def test_a_kernel_failure_hands_that_call_and_every_later_one_to_cutlass():
    real, calls = _stub_c()
    gemm = _Gemm(RuntimeError("no launcher"))
    failures: list[BaseException] = []
    proxy = vendored.CProxy(real, gemm, failures.append)
    first, second = _operands(), _operands()
    assert _call(proxy, first) == "cutlass"
    assert _call(proxy, second) == "cutlass"
    assert len(gemm.calls) == 1, "a broken kernel must not be tried again"
    assert [type(exc) for exc in failures] == [RuntimeError]
    assert _same_storage(calls[0], first) and _same_storage(calls[1], second)


def test_a_raising_failure_callback_never_reaches_the_render():
    real, _calls = _stub_c()

    def boom(exc):
        raise ValueError("callback")

    proxy = vendored.CProxy(real, _Gemm(RuntimeError("no launcher")), boom)
    assert _call(proxy, _operands()) == "cutlass"


# --- the knob: values, sources and precedence ------------------------------------


@pytest.mark.parametrize("value, mode", [
    (None, "triton"), ("triton", "triton"), ("cutlass", "cutlass"), (" Cutlass ", "cutlass"),
    ("TRITON", "triton"), ("cublas", None), ("", None), ("off", None), (True, None), (0, None),
])
def test_parse_names_a_mode_or_none(value, mode):
    assert int8_gemm.parse(value) == mode


def test_the_schema_admits_exactly_the_two_modes():
    for value in ("triton", "cutlass"):
        assert validate_worker_args({"int8_gemm": value})["int8_gemm"] == value
    for value in ("TRITON", "cublas", "", True, 1, None):
        with pytest.raises(ClusterConfigError, match=r"worker_args\.int8_gemm must be \"triton\" or \"cutlass\""):
            validate_worker_args({"int8_gemm": value})


def test_cluster_toml_carries_the_key_to_the_worker_args(tmp_path):
    from dgx_monarch.config import load_cluster_config

    path = tmp_path / "cluster.toml"
    path.write_text(textwrap.dedent('''
        [cluster]
        fabric_profile = "single-node"
        client_bind = "tcp://127.0.0.1:0"
        [[hosts]]
        name = "127.0.0.1"
        [worker_args]
        int8_gemm = "cutlass"
    '''), encoding="utf-8")
    assert load_cluster_config(path).worker_args == {"int8_gemm": "cutlass"}


def test_the_worker_never_writes_the_default_into_its_args():
    """Writing the key changes every identity-gate capability context (CLUSTER.md)."""
    from dgx_monarch.actor import comfy_bridge

    for integrated in (True, False):
        assert "int8_gemm" not in comfy_bridge._uma_memory_defaults({}, integrated=integrated)


def test_the_driver_reads_the_environment_and_names_its_source(option, lines):
    _stub_kitchen()
    assert int8_gemm.install_driver({}) == "triton"
    assert lines[-1][1].endswith("(int8_gemm=triton, from the default)")
    assert int8_gemm.install_driver({"DGXM_INT8_GEMM": "  "}) == "triton"
    assert int8_gemm.install_driver({"DGXM_INT8_GEMM": "cutlass"}) == "cutlass"
    assert lines[-1][1].endswith("(int8_gemm=cutlass, from DGXM_INT8_GEMM)")


def test_actors_read_worker_args_and_never_the_driver_variable(option, lines, monkeypatch):
    _stub_kitchen()
    monkeypatch.setenv("DGXM_INT8_GEMM", "cutlass")
    assert int8_gemm.configure_from_worker_args({}) == "triton"
    assert int8_gemm.configure_from_worker_args(None) == "triton"
    assert int8_gemm.configure_from_worker_args({"int8_gemm": "cutlass"}) == "cutlass"
    assert lines[-1][1].endswith("(int8_gemm=cutlass, from [worker_args] int8_gemm)")


def test_an_unknown_value_runs_stock_cutlass_with_one_warning(option, lines):
    _, cuda, real, _calls, _ = _stub_kitchen()
    for _ in range(3):
        assert int8_gemm.install_driver({"DGXM_INT8_GEMM": "cutlas"}) == "cutlass"
    assert cuda._C is real
    assert lines == [("WARNING", "INT8 GEMM: DGXM_INT8_GEMM is 'cutlas', which names neither "
                                 "\"triton\" nor \"cutlass\"; stock CUTLASS for this process")]


# --- the version guard and the fallbacks ---------------------------------------------


def test_the_validated_set_is_the_recorded_one():
    assert vendored.VALIDATED_KITCHEN == ("0.2.37", "0.2.40")
    assert int8_gemm.VALIDATED_KITCHEN is vendored.VALIDATED_KITCHEN


@pytest.mark.parametrize("version", ["0.2.41", "0.2.36", None])
def test_an_unvalidated_comfy_kitchen_keeps_stock_cutlass_with_one_warning(option, lines, monkeypatch, version):
    monkeypatch.setattr(vendored, "kitchen_version", lambda: version)
    top, cuda, real, _calls, _ = _stub_kitchen()
    for _ in range(3):
        assert int8_gemm.configure_from_worker_args({}) == "cutlass"
    assert cuda._C is real
    assert not getattr(top.int8_linear, "_ck_compile_safe", False)
    assert option.calls == []
    shown = version or "without version metadata"
    assert lines == [("WARNING", f"INT8 GEMM: comfy-kitchen {shown} is not in the validated set "
                                 "(0.2.37, 0.2.40); stock CUTLASS for this process")]


def test_missing_python_headers_keep_stock_cutlass_with_one_warning(option, lines, monkeypatch):
    monkeypatch.setattr(int8_gemm, "python_headers", lambda: (False, "/no/include"))
    _, cuda, real, _calls, _ = _stub_kitchen()
    for _ in range(2):
        assert int8_gemm.install_driver({}) == "cutlass"
    assert cuda._C is real
    assert lines == [("WARNING", "INT8 GEMM: Python.h is missing for this interpreter (Triton builds "
                                 "its launcher against /no/include); stock CUTLASS for this process")]


def test_triton_failing_to_load_keeps_stock_cutlass_with_one_warning(option, lines, monkeypatch):
    def no_triton():
        raise ModuleNotFoundError("No module named 'triton'")

    monkeypatch.setattr(vendored, "build_triton_gemm", no_triton)
    _, cuda, real, _calls, _ = _stub_kitchen()
    for _ in range(2):
        assert int8_gemm.install_driver({}) == "cutlass"
    assert cuda._C is real
    assert [level for level, _ in lines] == ["WARNING"]
    assert lines[0][1].startswith("INT8 GEMM: Triton could not be loaded (ModuleNotFoundError(")
    assert lines[0][1].endswith("; stock CUTLASS for this process")


def test_a_kernel_failure_at_a_call_logs_one_warning_and_unwraps(option, lines):
    option.error = RuntimeError("Python.h: No such file or directory")
    top, cuda, real, calls, _ = _stub_kitchen()
    eager = top.int8_linear
    assert int8_gemm.install_driver({}) == "triton"
    assert top.int8_linear.__wrapped__ is eager
    proxy = cuda._C
    assert _call(proxy, _operands()) == "cutlass"
    assert cuda._C is real, "a failed kernel must leave stock CUTLASS in place"
    assert top.int8_linear is eager, "and comfy-kitchen's own int8_linear, for compiled calls too"
    assert int8_gemm.active_mode() == "cutlass"
    assert int8_gemm.install_driver({}) == "cutlass", "the failure holds for the process"
    assert cuda._C is real and top.int8_linear is eager
    assert len(calls) == 1
    assert [level for level, _ in lines] == ["INFO", "WARNING"]
    assert lines[1][1].startswith("INT8 GEMM: the Triton kernel failed to compile or run (RuntimeError(")
    assert lines[1][1].endswith("; stock CUTLASS for the rest of this process")


def _surprise(*_args, **_kwargs):
    raise RuntimeError("surprise")


def test_nothing_the_option_does_raises(option, lines, monkeypatch):
    """An unexpected error anywhere in the option ends in stock CUTLASS, never a raise."""
    top, cuda, real, _calls, _ = _stub_kitchen()
    eager = top.int8_linear
    monkeypatch.setattr(int8_gemm, "_apply_backend", _surprise)
    assert int8_gemm.install_driver({}) == "cutlass"
    assert int8_gemm.configure_from_worker_args({"int8_gemm": "triton"}) == "cutlass"
    assert cuda._C is real and top.int8_linear is eager
    assert [level for level, _ in lines] == ["WARNING"]
    assert "the option could not be applied (RuntimeError('surprise'))" in lines[0][1]


def test_an_error_after_the_wrap_restores_both_replacements_for_good(option, lines, monkeypatch):
    """The fallback puts back _C and int8_linear, and a later configure keeps both stock."""
    top, cuda, real, _calls, _ = _stub_kitchen()
    eager = top.int8_linear
    assert int8_gemm.install_driver({}) == "triton"
    assert isinstance(cuda._C, vendored.CProxy) and top.int8_linear.__wrapped__ is eager
    apply_top = int8_gemm._apply_top
    monkeypatch.setattr(int8_gemm, "_apply_top", _surprise)
    assert int8_gemm.install_driver({}) == "cutlass"
    assert cuda._C is real and top.int8_linear is eager
    monkeypatch.setattr(int8_gemm, "_apply_top", apply_top)
    assert int8_gemm.install_driver({}) == "cutlass", "the WARNING said stock for this process"
    assert cuda._C is real and top.int8_linear is eager
    assert [level for level, _ in lines] == ["INFO", "WARNING"]
    assert lines[1][1] == ("INT8 GEMM: the option could not be applied (RuntimeError('surprise')); "
                           "stock CUTLASS for this process")


def test_a_backend_without_its_cuda_extension_has_nothing_to_replace(option, lines):
    _, cuda, _real, _calls, _ = _stub_kitchen()
    cuda._C = None
    assert int8_gemm.install_driver({}) == "cutlass"
    assert cuda._C is None
    assert lines == [("INFO", "INT8 GEMM: comfy-kitchen 0.2.40 has no CUDA extension in this process, "
                              "so there is no CUTLASS INT8 GEMM to replace")]


# --- in place, and through the import hook ----------------------------------------------


def test_an_imported_comfy_kitchen_is_wrapped_in_place_and_unwrapped_on_cutlass(option, lines):
    top, cuda, real, calls, linear_calls = _stub_kitchen()
    eager = top.int8_linear
    assert int8_gemm.install_driver({}) == "triton"
    assert isinstance(cuda._C, vendored.CProxy) and cuda._C._real is real
    assert top.int8_linear._ck_compile_safe and top.int8_linear.__wrapped__ is eager
    # Eager calls reach the original with every argument in place.
    assert top.int8_linear("x", "w", "s") == "eager"
    assert linear_calls == [("x", "w", "s", None, None, False, 256, None, None, 0.0, None, None)]
    assert int8_gemm.install_driver({}) == "triton"
    assert len(lines) == 1, "an unchanged mode logs nothing more"
    assert lines[0] == ("INFO", "INT8 GEMM: Triton kernel on for comfy-kitchen 0.2.40 INT8 linears "
                                "(int8_gemm=triton, from the default)")

    assert int8_gemm.install_driver({"DGXM_INT8_GEMM": "cutlass"}) == "cutlass"
    assert cuda._C is real and top.int8_linear is eager
    assert lines[-1] == ("INFO", "INT8 GEMM: stock comfy-kitchen 0.2.40 CUTLASS kernel "
                                 "(int8_gemm=cutlass, from DGXM_INT8_GEMM)")
    assert int8_gemm.install_driver({}) == "triton"
    assert isinstance(cuda._C, vendored.CProxy) and cuda._C._real is real
    assert calls == []


def _fake_kitchen_on_disk(root: Path) -> None:
    """A comfy-kitchen package the import system has to find and execute."""
    files = {
        "comfy_kitchen/__init__.py": """
            from .backends import cuda as _cuda_backend

            def int8_linear(*args):
                return "eager"
        """,
        "comfy_kitchen/backends/__init__.py": "",
        "comfy_kitchen/backends/cuda/__init__.py": """
            import types

            def _dequant(*args):
                return "cutlass"

            _C = types.SimpleNamespace(cutlass_int8_dequant=_dequant)
        """,
        "comfy_kitchen/backends/eager/__init__.py": "",
        "comfy_kitchen/backends/eager/quantization.py": """
            import torch

            DTYPE_CODE_TO_DTYPE = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16}
        """,
    }
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.mark.parametrize("mode", ["triton", "cutlass"])
def test_the_import_hook_applies_the_mode_set_before_comfy_kitchen_loads(option, lines, tmp_path,
                                                                          monkeypatch, mode):
    _fake_kitchen_on_disk(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    int8_gemm.install_actor()
    assert any(isinstance(finder, vendored.Finder) for finder in sys.meta_path)
    assert int8_gemm.configure_from_worker_args({"int8_gemm": mode}) == "cutlass", "nothing loaded yet"
    assert lines == []

    import comfy_kitchen
    from comfy_kitchen.backends import cuda

    assert Path(comfy_kitchen.__file__).is_relative_to(tmp_path)
    assert isinstance(cuda._C, vendored.CProxy) is (mode == "triton")
    assert bool(getattr(comfy_kitchen.int8_linear, "_ck_compile_safe", False)) is (mode == "triton")
    assert int8_gemm.active_mode() == mode
    assert [level for level, _ in lines] == ["INFO"]
    assert f"(int8_gemm={mode}, from [worker_args] int8_gemm)" in lines[0][1]


def test_installing_twice_registers_one_hook(option):
    for _ in range(3):
        int8_gemm.install_actor()
        int8_gemm.install_driver({})
    assert sum(isinstance(finder, vendored.Finder) for finder in sys.meta_path) == 1


# --- where each process installs it ---------------------------------------------------


def test_the_option_module_imports_only_the_standard_library():
    """bootstrap_proc imports it in every actor before CUDA is set up."""
    probe = ("import sys\n"
             "from dgx_monarch.kernels import int8_gemm\n"
             "int8_gemm.install_actor()\n"
             "heavy = sorted(name for name in ('torch', 'triton', 'comfy_kitchen', 'comfy') if name in sys.modules)\n"
             "print(heavy)\n")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120,
                            env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_bootstrap_proc_registers_the_hook_after_the_source_path(monkeypatch):
    from dgx_monarch import (
        actor_lifetime,
        mesh_runtime,
        monarch_capabilities,
        monarch_diagnostics,
        monarch_policy,
        runtime_provenance,
    )

    source = inspect.getsource(mesh_runtime.bootstrap_proc)
    assert source.index("sys.path.insert") < source.index("int8_gemm.install_actor()")
    monkeypatch.setenv(monarch_policy.POLICY_ENV, monarch_policy.encode_policy(monarch_policy.MonarchPolicy()))
    monkeypatch.setattr(monarch_capabilities, "assert_supported_runtime", lambda: None)
    monkeypatch.setattr(monarch_policy, "apply_policy", lambda *_a, **_kw: None)
    monkeypatch.setattr(monarch_policy, "activate_native_policy", lambda: None)
    monkeypatch.setattr(monarch_diagnostics, "configure_policy", lambda *_a: None)
    monkeypatch.setattr(runtime_provenance, "enforce_source_only_imports", lambda: None)
    monkeypatch.setattr(actor_lifetime, "install", lambda: None)
    installed: list[bool] = []
    monkeypatch.setattr(int8_gemm, "install_actor", lambda: installed.append(True))
    mesh_runtime.bootstrap_proc()
    assert installed == [True]


def test_ensure_comfy_sets_the_mode_before_comfy_imports_comfy_kitchen(tmp_path, monkeypatch, isolated_environ):
    from dgx_monarch.actor import comfy_bridge

    events: list[str] = []
    probe = types.ModuleType("dgxm_int8_order_probe")
    probe.events = events
    monkeypatch.setitem(sys.modules, "dgxm_int8_order_probe", probe)
    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / "comfy").mkdir(parents=True)
    (comfy_dir / "comfy" / "sd.py").write_text("", encoding="utf-8")
    (comfy_dir / "comfy" / "model_management.py").write_text(
        "import dgxm_int8_order_probe\ndgxm_int8_order_probe.events.append('comfy.model_management')\n",
        encoding="utf-8")
    comfy = types.ModuleType("comfy")
    comfy.__path__ = [str(comfy_dir / "comfy")]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    # Absent before the import, and removed again afterwards.
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    del sys.modules["comfy.model_management"]
    monkeypatch.setattr(comfy_bridge, "_BOOTSTRAPPED", None)
    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None)
    monkeypatch.setattr(comfy_bridge, "_patch_uma_free_memory", lambda *_args: None)
    monkeypatch.setattr(comfy_bridge, "_instrument_gpu_load", lambda: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", lambda args: dict(args))
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _args: None)
    monkeypatch.setattr(comfy_bridge, "_load_custom_node_modules", lambda: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(int8_gemm, "configure_from_worker_args",
                        lambda args: events.append(f"int8_gemm={dict(args or {}).get('int8_gemm')}"))

    comfy_bridge.ensure_comfy(str(comfy_dir), {"int8_gemm": "cutlass"}, gpus_per_host=1)
    assert events == ["int8_gemm=cutlass", "comfy.model_management"]
    # A bootstrapped proc takes a changed mode in place, without a new import.
    comfy_bridge.ensure_comfy(str(comfy_dir), {"int8_gemm": "triton"}, gpus_per_host=1)
    assert events[-1] == "int8_gemm=triton"


def test_apply_worker_args_switches_a_live_proc(monkeypatch):
    from dgx_monarch.actor import comfy_bridge

    fake = types.ModuleType("comfy.model_management")
    fake.EXTRA_RESERVED_VRAM = 0
    fake.NUM_STREAMS = 4
    fake.DISABLE_SMART_MEMORY = False
    fake.MAX_PINNED_MEMORY = 1024
    comfy = types.ModuleType("comfy")
    comfy.model_management = fake
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake)
    monkeypatch.setattr(comfy_bridge, "_MEMORY_BASELINE_MODULE", None)
    seen: list[object] = []
    monkeypatch.setattr(int8_gemm, "configure_from_worker_args", lambda args: seen.append(args.get("int8_gemm")))
    comfy_bridge._apply_worker_args({"int8_gemm": "cutlass"})
    comfy_bridge._apply_worker_args({})
    assert seen == ["cutlass", None]


def test_the_entrypoint_installs_the_driver_side_after_its_imports():
    source = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert "from dgx_monarch.kernels import int8_gemm" in source
    assert source.index("int8_gemm.install_driver()") > source.index("from dgx_monarch.nodes import")
