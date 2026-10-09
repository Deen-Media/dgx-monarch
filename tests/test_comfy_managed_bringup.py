"""Policy forcing, the two-stage bring-up and its latch, the worker status
fields, the RDMA guard, and the source pins that hold the bring-up in place."""
from __future__ import annotations

import ast
import inspect

import pytest

from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    SRC,
    _func,
    _isolated_process_state,
    _tag,
    _tree,
)
from comfy_managed_helpers import rung_on as rung_on  # a fixture tests ask for by name.
from dgx_monarch import residency_mode
from dgx_monarch.actor import comfy_dynamic, store_residency
from dgx_monarch.refusal import RefusalClass


def test_apply_policy_forces_every_lever_this_rung_owns():
    out = comfy_dynamic.apply_policy({"comfy_managed": True})
    assert out["lora_low_rss"] is False        # sentinel-trap doctrine: no swap path
    assert out["slab_weights"] is False        # a raw host mapping aimdo cannot see
    assert out["comfy_managed"] is True


def test_apply_policy_leaves_pinned_memory_to_the_hosts_own_posture():
    """The rung owns placement, not pinning.

    On unified memory, forcing pinning on is a bug (measured 2026-08-12):
    ComfyUI sizes DynamicVRAM's staging buffer at twice the model whenever a
    pinned budget exists, then fills it with a second copy of every weight in
    the same pool the device computes from. Leaving the key alone lets the
    worker's unified-memory posture turn pinning off, which is what makes the
    staging zero-copy.
    """
    out = comfy_dynamic.apply_policy({"comfy_managed": True})
    assert "disable_pinned_memory" not in out


def test_apply_policy_overrides_an_operator_who_asked_for_both():
    out = comfy_dynamic.apply_policy(
        {"comfy_managed": True, "lora_low_rss": True, "slab_weights": "auto",
         "disable_pinned_memory": True})
    assert (out["lora_low_rss"], out["slab_weights"]) == (False, False)
    assert out["disable_pinned_memory"] is True


def test_an_operator_can_still_ask_for_pinned_staging_under_this_rung():
    """Availability is the operator's call; the capacity wall prices it."""
    out = comfy_dynamic.apply_policy(
        {"comfy_managed": True, "disable_pinned_memory": False})
    assert out["disable_pinned_memory"] is False


def test_apply_policy_is_a_no_op_when_the_rung_is_off():
    values = {"lora_low_rss": True, "slab_weights": "auto", "reserve_vram_gb": 8.0}
    assert comfy_dynamic.apply_policy(dict(values)) == values


def test_apply_policy_does_not_mutate_its_input():
    values = {"comfy_managed": True, "lora_low_rss": True}
    comfy_dynamic.apply_policy(values)
    assert values == {"comfy_managed": True, "lora_low_rss": True}


def test_the_bootstrap_latch_refuses_a_policy_flip():
    comfy_dynamic.apply_policy({"comfy_managed": True})
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        comfy_dynamic.apply_policy({})
    text = str(excinfo.value)
    assert _tag(text).refusal_class is RefusalClass.PHYSICS
    assert "reset the attached mesh" in text.lower()
    assert "docs/TROUBLESHOOTING.md #62" in text
    assert "instead" in text


def test_the_latch_refuses_the_other_direction_too():
    comfy_dynamic.apply_policy({})
    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        comfy_dynamic.apply_policy({"comfy_managed": True})


def test_the_latch_accepts_the_same_answer_twice():
    for _ in range(3):
        comfy_dynamic.apply_policy({"comfy_managed": True})
    assert comfy_dynamic.apply_policy({"comfy_managed": True})["comfy_managed"] is True


def test_stage_a_with_the_rung_off_arms_the_latch_and_imports_nothing(isolated_environ):
    assert comfy_dynamic.stage_a({}) is False
    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        comfy_dynamic.apply_policy({"comfy_managed": True})


def test_stage_a_pops_a_stale_environment_claim(monkeypatch):
    """A leaked or inherited DGXM_COMFY_MANAGED=1 must not claim the rung.

    Claiming it without a bring-up is the worst case: the ladder takes rung 0,
    the stock capacity wall is skipped, and no aimdo is running to place
    anything. Only this process's own successful stage_b sets the variable.
    """
    monkeypatch.setenv(residency_mode.ENV_ACTIVE, "1")
    assert comfy_dynamic.stage_a({}) is False
    assert residency_mode.active() is False
    assert store_residency.comfy_managed_active() is False


def test_stage_a_refuses_when_comfy_aimdo_is_not_installed(monkeypatch, isolated_environ):
    import builtins

    real_import = builtins.__import__

    def _missing(name, *args, **kwargs):
        if name.startswith("comfy_aimdo"):
            raise ImportError("No module named 'comfy_aimdo'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing)
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        comfy_dynamic.stage_a({"comfy_managed": True})
    text = str(excinfo.value)
    assert _tag(text).refusal_class is RefusalClass.PHYSICS
    assert "instead" in text


def test_stage_b_with_the_rung_off_clears_the_environment_claim(monkeypatch):
    monkeypatch.setenv(residency_mode.ENV_ACTIVE, "1")
    assert comfy_dynamic.stage_b(False, reserve_vram_gb=8.0) is False
    assert residency_mode.active() is False


def test_a_failed_stage_b_refuses_every_later_policy_application(monkeypatch):
    """A failed stage B must not fall back to classic residency in silence.

    ``ensure_comfy`` marks the proc bootstrapped before stage B, so a stage B
    raise leaves a process that every later ensure_comfy call early-returns
    through. Without this latch the retry succeeds, the worker runs classic
    residency, and its _active_worker_args and every gate capability context
    still say comfy_managed. That binds a durable ledger PASS to a capability
    context the worker never honored.
    """
    monkeypatch.setattr(comfy_dynamic, "_REQUESTED", True, raising=False)
    monkeypatch.setattr(comfy_dynamic, "_STAGE_B_DONE", False, raising=False)
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        comfy_dynamic.apply_policy({"comfy_managed": True})
    text = str(excinfo.value)
    assert _tag(text).refusal_class is RefusalClass.PHYSICS
    assert "docs/TROUBLESHOOTING.md #62" in text
    assert "instead" in text
    assert "reset the attached mesh" in text.lower()


def test_stage_b_arms_the_failure_latch_before_it_can_raise():
    """The latch is set before stage_b's imports, so a raise from an import or
    from aimdo marks the proc failed, not never-attempted."""
    body = _func(_tree("actor/comfy_dynamic.py"), "stage_b").body
    assignments = [
        index for index, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_STAGE_B_DONE"
                for target in node.targets)
    ]
    imports = [index for index, node in enumerate(body) if isinstance(node, ast.Import)]
    assert assignments, "stage_b no longer records its own outcome"
    assert assignments[0] < min(imports), (
        "the failure latch must be armed above stage_b's comfy/aimdo imports"
    )


def test_the_bring_up_path_itself_is_not_refused_by_its_own_latch():
    """None, not False, until stage B has been reached: the first bootstrap
    calls apply_policy before stage_b runs and must pass."""
    assert comfy_dynamic._STAGE_B_DONE is None
    assert comfy_dynamic.apply_policy({"comfy_managed": True})["comfy_managed"] is True
    assert comfy_dynamic.apply_policy({"comfy_managed": True})["comfy_managed"] is True


def test_a_completed_bring_up_leaves_the_latch_alone(monkeypatch):
    monkeypatch.setattr(comfy_dynamic, "_REQUESTED", True, raising=False)
    monkeypatch.setattr(comfy_dynamic, "_STAGE_B_DONE", True, raising=False)
    assert comfy_dynamic.apply_policy({"comfy_managed": True})["comfy_managed"] is True


def test_a_classic_worker_is_untouched_by_the_latch(monkeypatch):
    """_REQUESTED False means the proc never asked for the rung. A False
    stage-B latch beside it cannot happen, and pinning that it would not
    refuse is cheap."""
    monkeypatch.setattr(comfy_dynamic, "_REQUESTED", False, raising=False)
    monkeypatch.setattr(comfy_dynamic, "_STAGE_B_DONE", False, raising=False)
    assert comfy_dynamic.apply_policy({"lora_low_rss": True}) == {"lora_low_rss": True}


def test_the_worker_policy_endpoint_runs_the_same_gate():
    """apply_worker_args_impl is the other way a live worker takes a policy, so
    a failed bring-up has to refuse there too, not only through ensure_comfy."""
    source = inspect.getsource(
        __import__("dgx_monarch.actor.worker_env", fromlist=["worker_env"])
        .apply_worker_args_impl)
    assert "_uma_memory_defaults" in source
    defaults = inspect.getsource(
        __import__("dgx_monarch.actor.comfy_bridge", fromlist=["comfy_bridge"])
        ._uma_memory_defaults)
    assert "comfy_dynamic.apply_policy" in defaults


def test_snapshot_reports_the_two_worker_status_fields():
    snapshot = comfy_dynamic.snapshot()
    assert snapshot["comfy_managed"] is False
    assert "comfy_aimdo" in snapshot


def test_assert_rdma_compatible_is_inert_until_the_rung_is_up():
    assert comfy_dynamic.assert_rdma_compatible() is None


def test_assert_rdma_compatible_refuses_inside_a_comfy_managed_worker(rung_on):
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        comfy_dynamic.assert_rdma_compatible()
    text = str(excinfo.value)
    assert _tag(text).refusal_class is RefusalClass.PHYSICS
    assert "RDMA" in text
    assert "docs/TROUBLESHOOTING.md #62" in text
    assert "instead" in text


def test_the_rdma_guard_sits_at_the_single_constructor_of_that_path(rung_on):
    """Dormant while rdma_latent_return defaults off. It makes turning RDMA on
    later a refusal instead of a collision between aimdo's host registration
    and the multi-NIC scatter."""
    from dgx_monarch.transfer import LatentReturn

    assert LatentReturn(mode="message") is not None
    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        LatentReturn(mode="rdma")


def test_the_rdma_guard_is_inert_on_a_classic_worker():
    from dgx_monarch.transfer import LatentReturn

    assert LatentReturn(mode="rdma").mode == "rdma"


def test_the_bring_up_brackets_the_comfy_import():
    """Stage A dlopens aimdo before comfy is imported; stage B installs the
    hooks, claims the device and rebinds the patcher after the reserve is
    resolved. Asserted on statement position inside ensure_comfy, not on a
    substring index, so a reformat cannot fake it."""
    ensure = _func(_tree("actor/comfy_bridge.py"), "ensure_comfy")
    stage_a = stage_b = comfy_import = None
    for node in ast.walk(ensure):
        if isinstance(node, ast.Import) and any(
                alias.name == "comfy.model_management" for alias in node.names):
            comfy_import = min(comfy_import or node.lineno, node.lineno)
        if isinstance(node, ast.Attribute) and node.attr in ("stage_a", "stage_b"):
            if node.attr == "stage_a":
                stage_a = node.lineno
            else:
                stage_b = node.lineno
    assert stage_a is not None and stage_b is not None and comfy_import is not None
    assert stage_a < comfy_import < stage_b


def test_the_policy_forcing_runs_above_the_discrete_early_return():
    """A policy flip must be caught on discrete hardware too, and the hot
    re-apply path reaches the latch through the same call."""
    defaults = _func(_tree("actor/comfy_bridge.py"), "_uma_memory_defaults")
    apply_line = next(
        node.lineno for node in ast.walk(defaults)
        if isinstance(node, ast.Attribute) and node.attr == "apply_policy")
    early_return = next(
        node.lineno for node in ast.walk(defaults)
        if isinstance(node, ast.If) and "not integrated" in ast.unparse(node.test))
    assert apply_line < early_return


def test_stage_b_owns_both_globals_and_stage_a_owns_neither():
    """A refactor that moves one without the other leaves comfy half-rebound."""
    tree = _tree("actor/comfy_dynamic.py")
    a_source = ast.unparse(_func(tree, "stage_a"))
    b_source = ast.unparse(_func(tree, "stage_b"))
    assert "CoreModelPatcher" in b_source
    assert "aimdo_enabled" in b_source
    assert "CoreModelPatcher" not in a_source
    assert "aimdo_enabled" not in a_source


def test_stage_b_creates_the_cuda_context_before_claiming_the_device():
    """The one real ordering constraint: init_devices installs the six funchook
    inline patches through plat_init, and they dlsym into an already-loaded
    libcuda. The R1 solo leg (docs/VALIDATION.md, 2026-08-04) exercised that
    layer."""
    body = ast.unparse(_func(_tree("actor/comfy_dynamic.py"), "stage_b"))
    assert body.index("cuda.init") < body.index("init_devices")
    assert body.index("set_device") < body.index("init_devices")


def test_the_submodule_rewire_names_every_capturing_submodule():
    """comfy_aimdo submodules bind `lib = control.lib` at import time against a
    None, and comfy imports them before a library-style bootstrap can run its
    init. Still required at 0.4.13."""
    assert set(comfy_dynamic._SUBMODULES) == {
        "host_buffer", "vram_buffer", "model_vbar", "model_mmap"}
    # comfy_aimdo.torch has no module-level `lib`, so naming it would only
    # create a bogus attribute.
    assert "torch" not in comfy_dynamic._SUBMODULES
    source = ast.unparse(_func(_tree("actor/comfy_dynamic.py"), "_rewire_submodule_lib"))
    assert "_SUBMODULES" in source
    assert "lib" in source


def test_the_bring_up_lands_between_the_device_narrowing_and_nccl():
    """The bring-up runs after the device narrowing and before NCCL.

    ``CUDA_VISIBLE_DEVICES`` is narrowed to this worker's GPU before the
    bring-up, which is why stage_b claims device index 0 of this process's own
    view rather than main.py's full device list. The bring-up then completes
    before ``init_process_group("nccl")``, so the six inline libcuda patches are
    in place when the communicator is built.

    The R1 research leg (2026-08-04) did not test that last part: its two
    world-2 legs failed with ncclInvalidUsage before a single collective, so no
    communicator was built, and docs/VALIDATION.md still marks hooks beside a
    live NCCL communicator as untested. The stable-source sweep (2026-09-12 to
    2026-10-02) ran world-2 comfy-managed cells. Among them docs/VALIDATION.md
    records two Omnigen2 cells whose first-use Gate recorded FAIL (largest
    latent difference 0.141 on Torch Flash, 2.08 on SAGE_AUTO) and names no
    cause; their cuDNN twin passed at 0.0 and rendered.
    """
    setup = _func(_tree("actor/worker_env.py"), "setup_impl")
    lines = {}
    for node in ast.walk(setup):
        if isinstance(node, ast.Constant) and node.value == "CUDA_VISIBLE_DEVICES":
            lines.setdefault("devices", node.lineno)
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", "")
            if name == "ensure_comfy":
                lines.setdefault("bringup", node.lineno)
            if name == "init_process_group":
                lines.setdefault("nccl", node.lineno)
            if name == "LatentReturn":
                lines.setdefault("latent_return", node.lineno)
    assert lines["devices"] < lines["bringup"] < lines["nccl"]
    # The RDMA guard lives in LatentReturn.__init__, so the bring-up must have
    # already published its environment claim by the time that runs.
    assert lines["bringup"] < lines["latent_return"]


def test_the_disproven_spike_monkeypatches_never_come_back():
    """The 2026-07-08 dynamic-VRAM spike replaced torch.cuda.ipc_collect and
    empty_cache with no-ops as suspects and recorded "death unchanged". Never
    bring those monkeypatches back as a fix."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in (
                        "ipc_collect", "empty_cache"):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"disproven spike monkeypatches reappeared: {offenders}"


def test_the_spike_dyntrace_hook_is_gone_for_good():
    hits = [path.relative_to(SRC).as_posix() for path in SRC.rglob("*.py")
            if "DGXM_DYNTRACE" in path.read_text()]
    assert hits == []


def test_the_swap_machinery_never_learns_about_this_rung():
    """The sentinel-trap doctrine, as a source pin.

    The rung refuses before the swap machinery runs instead of teaching that
    machinery about itself. Nothing in bake, un-bake, the slab or the
    certificate may mention it: if one of these files needs to know, the yield
    sits in the wrong place.
    """
    forbidden = ("comfy_managed", "aimdo", "ModelPatcherDynamic", "residency_mode")
    offenders = {}
    for relative in ("actor/store_bake.py", "actor/unbake.py", "actor/unbake_file.py",
                     "actor/slab.py", "actor/slab_certificate.py"):
        text = (SRC / relative).read_text()
        found = [token for token in forbidden if token in text]
        if found:
            offenders[relative] = found
    assert not offenders, f"the swap machinery learned about the rung: {offenders}"
