"""Driver and loader footprint pricing for the rung, the inertness pins on the
slab ceremony, and the source scan for lever-forcing sites."""
from __future__ import annotations

import ast
import inspect
import os
from types import SimpleNamespace

from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    GIB,
    SRC,
    _isolated_process_state,
)
from dgx_monarch import capacity_fit, driver_footprint, residency_mode
from line_limit_helpers import ceiling_for


def test_the_rung_is_priced_like_slab_on_the_driver():
    """The stock placement term is charged only when slab_weights is False,
    which is exactly the value the rung forces on the worker. Left alone,
    every comfy-managed render would pay a placement cost that does not exist
    and could be refused for it."""
    profile = driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"]
    managed = driver_footprint.estimate_driver_footprint(
        profile=profile, mem_available=100 * GIB, weight_bytes=20 * GIB,
        slab_weights=residency_mode.MODE_COMFY_MANAGED)
    assert managed.arena_bytes == 0
    assert managed.weight_bytes == 20 * GIB
    assert "comfy-managed" in managed.notes["arena"]


def test_explicit_slab_off_still_pays_the_stock_placement_price():
    profile = driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"]
    legacy = driver_footprint.estimate_driver_footprint(
        profile=profile, mem_available=100 * GIB, weight_bytes=20 * GIB,
        slab_weights=False)
    assert legacy.arena_bytes == int(capacity_fit.STOCK_PLACEMENT_RATIO * 20 * GIB)
    assert "stock load host copy" in legacy.notes["arena"]


def test_the_render_site_prices_through_the_shared_predicate():
    from dgx_monarch.nodes import render_preflight

    source = inspect.getsource(render_preflight)
    assert "residency_mode.pricing_value" in source, (
        "the render site and the loader site must price the same render "
        "identically; that is what residency_mode.pricing_value is for"
    )


def test_the_loader_site_resolves_the_rung_before_the_vouched_set():
    from dgx_monarch.nodes.loader_preflight import _effective_slab

    assert _effective_slab(
        {"comfy_managed": True, "slab_weights": False}, "krea2",
    ) == residency_mode.MODE_COMFY_MANAGED
    assert _effective_slab({"slab_weights": True}, "anything") is True
    assert _effective_slab({"slab_weights": False}, "krea2") is False
    assert _effective_slab({}, "krea2") is True
    assert _effective_slab({}, "minimax_h3") is False


def test_the_loader_windows_carry_the_value_through_without_a_bool_cast():
    """A bool() cast would give the right number and the wrong note: the
    transient window would revert to the slab wording and never name the rung."""
    from dgx_monarch.nodes.loader_preflight import _Shape, _windows

    shape = _Shape(
        profile=driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"],
        mem_available=200 * GIB, weight_bytes=20 * GIB, weights_resident=False,
        co_resident=True, floor_reserve=4 * GIB, stack_bytes=10 * GIB)
    transient, settled = _windows(shape, slab=residency_mode.MODE_COMFY_MANAGED)
    assert transient.arena_bytes == 0
    assert "comfy-managed" in transient.notes["arena"]
    slab_transient, slab_settled = _windows(shape, slab=True)
    assert settled.projected == slab_settled.projected
    assert transient.projected == slab_transient.projected


def test_the_loader_site_gates_its_rescue_on_the_same_blocker():
    """The end-to-end refusal (no card, no rescue sentence, names the widget)
    is pinned against the real loader rig in
    tests/test_loader_site_preflight.py::
    test_the_loader_site_offers_no_rescue_under_comfy_managed_residency.

    This test pins the wiring: the loader site must pre-qualify its
    rescue through the same blocker the projection uses, or the two disagree
    and a card appears for a projection that will be refused.
    """
    from dgx_monarch.nodes import loader_preflight
    from dgx_monarch.nodes.consent_projection import projection_blocker

    source = inspect.getsource(loader_preflight.preflight_loader_footprint)
    assert "consent_projection.projection_blocker(worker_args)" in source
    assert "and not blocked else None" in source
    assert projection_blocker({"comfy_managed": True, "slab_weights": False})


def test_the_slab_proof_is_inert_under_the_rung():
    """The ceremony is two proof renders under this rung, not three or four.

    Exercised, not re-derived: slab_weights is False so `slab_expected` is
    False, and `policy_mismatch` needs a rank reporting slab_active, which none
    does. A predicate that let comfy_managed make slab expected would send this
    through the retry branch: an extra unload and proof render, carrying the
    stock placement price. A test that transcribed the predicate into its own
    body would not notice.
    """
    from dgx_monarch.nodes import slab_proof

    rendered: list = []

    def _render(*args, **kwargs):
        rendered.append(args)
        raise AssertionError(
            "the slab proof rendered an extra proof image under the rung")

    def _call_all(*args, **_kwargs):
        raise AssertionError(f"the slab proof reached the worker mesh: {args[:1]}")

    handle = SimpleNamespace(world=1, call_all=_call_all)
    proof = slab_proof.establish(
        None, {"unet_name": "model.safetensors"}, {}, {}, {}, {}, 1.0, 1,
        handle, {"comfy_managed": True, "slab_weights": False}, {"latent": "b"},
        [{"family": "krea2", "conclusive": True, "slab_active": False}], _render)
    assert proof.expected is False
    assert proof.active is False
    assert proof.error is None
    assert proof.b == {"latent": "b"}
    assert rendered == []
    # krea2 is in the vouched set, so `expected` is False because of the
    # policy, not because the fixture picked an unvouched family.
    from dgx_monarch.capacity_fit import SLAB_VOUCHED_FAMILIES

    assert "krea2" in SLAB_VOUCHED_FAMILIES


def test_the_cross_residency_leg_does_not_run_under_the_rung():
    """No third proof render, and therefore no stock placement price from one.

    Every argument below but the proof is a placeholder on purpose: the guard
    has to return before it touches any of them, so passing anything usable
    would weaken the pin.
    """
    from dgx_monarch.nodes import gate_cross_mode

    proof = SimpleNamespace(error=None, expected=False, active=False)
    assert gate_cross_mode.run_cross_residency_reference(
        runtime={}, ceremony_model=None, handle=None, original_worker_args={},
        slab_proof=proof, frozen_request={}, frozen_latent={},
        artifact_binding={}, cfg_value=1.0, steps_hint=1, stock_latent={},
        transaction_render=None, bind_request=None,
    ) == (None, None)


def test_no_sibling_slab_context_is_stamped_under_the_rung():
    """There is no deterministic auto-to-explicit relationship for this rung,
    so a comfy-managed PASS covers exactly the context it ran under."""
    from dgx_monarch.nodes.gate_identity import equivalent_slab_mode_contexts

    proof = SimpleNamespace(error=None, expected=False, active=False)
    assert equivalent_slab_mode_contexts(
        None, None, proof, "PASS", None,
        {"comfy_managed": True, "slab_weights": False}) == []


def test_no_capacity_certificate_row_exists_on_this_path():
    """Nothing is byte-verified under the rung, so there is no evidence a
    CAPACITY_CERTIFIED row could carry. The vocabulary stays slab-only."""
    from dgx_monarch import gate_audit
    from dgx_monarch.gate_audit_vocab import RESIDENCY_KINDS

    assert RESIDENCY_KINDS == frozenset({"slab"})
    assert "comfy_managed" not in inspect.getsource(gate_audit.build_capacity_row)


def test_the_consented_rescue_stock_check_never_sees_the_third_value():
    """``_every_rank_took_stock`` compares describe()'s residency to the literal
    "stock", so the third value reads as not-stock and would raise the class P
    uncertified-slab refusal. Unreachable today because a projection is refused
    under the rung; pinned so a later relaxation fails here."""
    from dgx_monarch.nodes.consent_projection import projection_blocker
    from dgx_monarch.nodes.consent_rescue import _every_rank_took_stock

    assert _every_rank_took_stock(
        [{"rank": 0, "cond": {"residency": residency_mode.MODE_COMFY_MANAGED}}],
        1, "cond") is False
    # The control. The same complete cohort reporting literal stock passes, so
    # the refusal above comes from the third value, not from the cohort guard.
    assert _every_rank_took_stock(
        [{"rank": 0, "cond": {"residency": "stock"}}], 1, "cond") is True
    assert projection_blocker({"comfy_managed": True, "slab_weights": False})


def test_the_capability_context_changes_only_when_the_rung_is_on():
    """The key stays absent on a classic policy, so at the token a comfy-managed
    combination re-proves once and every classic combination's canonical string
    stays byte-identical."""
    from dgx_monarch.nodes.gate_identity import gate_verdict_token

    base = {"worker_args": {"slab_weights": "auto", "lora_low_rss": True},
            "mesh_mode": "local", "world": 1}
    off = {**base, "worker_args": dict(base["worker_args"])}
    on = {**base, "worker_args": {**base["worker_args"], "comfy_managed": True}}
    assert gate_verdict_token("k", "a", "c", off) == gate_verdict_token("k", "a", "c", base)
    assert gate_verdict_token("k", "a", "c", on) != gate_verdict_token("k", "a", "c", base)


def _subscript_forces_lora_low_rss_off(text: str) -> bool:
    """`values["lora_low_rss"] = False`, which neither string pattern matches.

    That spelling is the one the rung's own policy forcing uses, so a pin that
    only grepped for the dict-literal and keyword forms would miss a new site
    written the same way as the site it was added for.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:  # this pin reads forcing sites only; it skips a file that does not parse
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and node.value.value is False):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "lora_low_rss"):
                return True
    return False


def test_only_the_known_sites_force_lora_low_rss_off():
    """A new silent forcing site would bypass the sentinel-trap doctrine."""
    allowed = {
        "actor/comfy_dynamic.py",      # the rung's own policy forcing
        "residency_mode.py",           # the one quarantine lever table
        "nodes/render_quarantine.py",  # the persisted quarantine
        "nodes/fleet_policy.py",       # the fleet fail-closed stamp
        "nodes/gate_identity.py",      # the ceremony's stock stamps
        "nodes/gate_verdict.py",       # the cross-mode temporary policy
        "nodes/consent_rescue.py",     # the rescue's own residency projection
        "nodes/init.py",               # the widget's own forcing, beside the schema's
        "operator_profiles.py",        # reviewed safe/balanced setup profile output
    }
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative in allowed:
            continue
        text = path.read_text()
        if ('lora_low_rss": False' in text or "lora_low_rss=False" in text
                or _subscript_forces_lora_low_rss_off(text)):
            offenders.append(relative)
    assert not offenders, (
        f"a new site forces lora_low_rss off: {offenders}. Every forcing site "
        "must be reviewed against the sentinel-trap doctrine."
    )


def test_the_forcing_site_pin_catches_the_spelling_the_rung_itself_uses():
    """Detect the subscript assignment used by the managed residency path."""
    sneaky = 'def _sneaky(values):\n    values["lora_low_rss"] = False\n'
    assert _subscript_forces_lora_low_rss_off(sneaky)
    assert 'lora_low_rss": False' not in sneaky
    assert "lora_low_rss=False" not in sneaky
    assert _subscript_forces_lora_low_rss_off(
        (SRC / "actor/comfy_dynamic.py").read_text())
    assert not _subscript_forces_lora_low_rss_off(
        'values["lora_low_rss"] = True\nother["slab_weights"] = False\n')


def test_the_signed_adoption_evidence_is_deliberately_unchanged():
    """The resident-adoption row reports the storage backing, and this rung
    changes the manager, not the storage, so the row keeps its two-value
    vocabulary and the signed schema does not move. docs/TROUBLESHOOTING.md #62
    states this, so that nobody reads a comfy-managed row as proof of a
    classic load. Adding a third value is a reviewed edit with its own
    evidence-schema argument.
    """
    text = (SRC / "adoption_evidence.py").read_text()
    assert "comfy_managed" not in text
    assert "residency_mode." not in text          # no import of the leaf
    assert '{"cudaMalloc", "slab"}' in text


def test_the_two_new_leaves_carry_no_line_limit_ledger_row():
    """The ledger collects files over 500 lines and asserts set equality, so a
    leaf that grows past 500 fails, and so does pre-registering it."""
    for relative in ("residency_mode.py", "actor/comfy_dynamic.py"):
        lines = len((SRC / relative).read_text().splitlines())
        assert lines <= 500, (
            f"{relative} is {lines} lines. New leaves stay under the default "
            "cap; they have no ledger row and must not need one."
        )


def test_other_fixed_ceiling_files_did_not_grow():
    """Each file's ceiling comes from the one ledger, never a copy of it: its
    row where it has one, else the default cap. Raising any of them is one
    edit, in the ledger."""
    for relative in ("actor/model_store.py", "actor/worker_env.py",
                     "nodes/gate_identity.py"):
        ceiling = ceiling_for(relative)
        lines = len((SRC / relative).read_text().splitlines())
        assert lines <= ceiling, f"{relative} grew to {lines}, ceiling {ceiling}"


def test_the_environment_variable_is_the_only_process_seam():
    """One name, read in one shape, in both the pure ladder and the leaf."""
    ladder = (SRC / "actor/store_residency.py").read_text()
    assert residency_mode.ENV_ACTIVE not in ladder, (
        "the ladder reads residency_mode.ENV_ACTIVE through the leaf, never by "
        "spelling the variable name a second time"
    )
    assert os.environ.get(residency_mode.ENV_ACTIVE) is None
