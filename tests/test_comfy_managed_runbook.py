"""Keep docs/TROUBLESHOOTING.md #62, the shipped refusal texts and the prose pages
that describe the comfy-managed rung in sync."""
from __future__ import annotations

import ast

from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    REPO,
    SRC,
    _isolated_process_state,
    _tree,
)
from dgx_monarch import config_schema, residency_mode
from dgx_monarch.actor import comfy_dynamic


def _entry_62() -> str:
    text = (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    start = text.index("\n## 62. ")
    return text[start:text.index("\n## ", start + 1)]


def test_the_runbook_entry_exists_at_the_number_every_refusal_names():
    """Every comfy-managed class P refusal points at docs/TROUBLESHOOTING.md #62;
    a renumbered entry would send operators to the wrong page."""
    entry = _entry_62()
    assert "comfy_managed" in entry
    assert residency_mode.TROUBLESHOOTING == 62


# Every refusal this rung ships, by an opening that appears verbatim in both the
# source and the runbook. Keep each opening inside one source string literal: one
# that spans an implicit-concatenation boundary fails the source check for a
# message that never changed.
SHIPPED_REFUSAL_OPENINGS = (
    # render time
    "comfy-managed residency is on for this worker and this render carries",
    "comfy-managed residency is on for this worker and FSDP is active for this",
    "comfy-managed residency and the RDMA latent return cannot run in the same",
    "Fleet cannot run the identity ceremony itself",
    "comfy_managed is a bootstrap policy and this worker process already",
    "comfy-managed residency is on for this worker and the driver could",
    "comfy-managed residency is on for this worker and the first-use identity",
    # capacity
    "comfy-managed residency cannot load",
    # bring-up
    "comfy-managed residency was requested but comfy-aimdo is not installed",
    "comfy-managed residency was requested but comfy_aimdo.control.init()",
    "comfy-managed residency was requested but this host does not meet",
    "comfy_aimdo.control.init_devices() reported no working install for",
    "comfy-managed residency was requested and this worker process already tried",
)

REFUSAL_SOURCE_FILES = (
    "actor/store_fsdp.py", "actor/comfy_dynamic.py",
    "actor/store_residency.py", "nodes/fleet_policy.py", "mesh_residency.py",
    # The FSDP refusal is bound once in residency_mode.py and raised at a driver
    # site (nodes/loader_preflight.py) and a worker site (actor/store_fsdp.py).
    "residency_mode.py", "nodes/loader_preflight.py",
    # residency_mode.py also binds the known-wrong ceremony sentence, which the
    # driver's residency check in nodes/consent_waiver.py raises.
    "nodes/consent_waiver.py",
)


def _shared_refusal_strings() -> dict[str, dict[str, str]]:
    """Module-level string constants, keyed by the name a raise site writes.

    A message two modules raise is bound once. Without this the AST check below
    would see a bare attribute at one of the sites and read the rung as having
    lost a refusal it still ships.
    """
    table: dict[str, dict[str, str]] = {}
    for relative in REFUSAL_SOURCE_FILES:
        stem = relative.rsplit("/", 1)[-1].removesuffix(".py")
        for node in _tree(relative).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        table.setdefault(stem, {})[target.id] = node.value.value
    return table


def test_the_runbook_lists_the_first_line_of_every_shipped_refusal():
    """An operator finds the entry by matching the refusal they got, so every
    opening, the bring-up ones included, must appear in it verbatim."""
    entry = _entry_62()
    for opening in SHIPPED_REFUSAL_OPENINGS:
        assert opening in entry, f"docs/TROUBLESHOOTING.md #62 does not carry: {opening}"


def test_the_runbook_carries_the_shipped_refusal_texts_verbatim():
    """Every opening must still appear in the source files, or a reworded message
    leaves docs/TROUBLESHOOTING.md #62 telling operators to match a string
    nothing raises."""
    shipped = "\n".join(
        (SRC / relative).read_text() for relative in REFUSAL_SOURCE_FILES)
    missing = [opening for opening in SHIPPED_REFUSAL_OPENINGS
               if opening not in shipped]
    assert not missing, f"the runbook names refusals no source raises: {missing}"


def test_every_shipped_refusal_is_actually_a_refusal_call():
    """A fragment that stopped being raised, or moved out of refusal(), would
    still satisfy a plain substring search of the same files."""
    shared = _shared_refusal_strings()
    texts: list[str] = []

    def literal(sub: ast.AST) -> str | None:
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            return sub.value
        if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                and sub.attr in shared.get(sub.value.id, {})):
            return shared[sub.value.id][sub.attr]
        return None

    for relative in REFUSAL_SOURCE_FILES:
        for node in ast.walk(_tree(relative)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "refusal" and len(node.args) > 1):
                texts.append(" ".join(
                    found for sub in ast.walk(node.args[1])
                    if (found := literal(sub)) is not None))
    joined = "\n".join(texts)
    missing = [opening for opening in SHIPPED_REFUSAL_OPENINGS
               if opening not in joined]
    assert not missing, f"not raised through refusal(): {missing}"


def test_the_bring_up_refusals_share_one_documented_tail():
    """Four setup-time refusals bind the same "what works instead" sentence, so
    the runbook states it once rather than four times."""
    entry = " ".join(_entry_62().split())   # the runbook hard-wraps its prose
    assert "bring-up refusals" in entry.lower()
    for promise in ("remove `comfy_managed` from `cluster.toml`",
                    "reset the attached mesh"):
        assert promise in entry.lower()
    source = (SRC / "actor/comfy_dynamic.py").read_text()
    assert source.count("_BRINGUP_TAIL") >= 5   # one definition, four uses


def test_the_runbook_names_the_forced_levers_and_the_recycle():
    entry = _entry_62()
    for promise in ("lora_low_rss", "slab_weights", "disable_pinned_memory",
                    "**Reset attached mesh**", "no `auto`", "off by default"):
        assert promise in entry


def test_the_runbook_says_pinning_is_not_forced_and_says_what_that_buys():
    """The entry must say the rung leaves pinning to the host, not that it forces
    pinning on. ComfyUI processes on this class of box run DynamicVRAM with
    pinning off, so a zero-capacity staging buffer is the working configuration,
    not a degraded one."""
    entry = " ".join(_entry_62().split())
    assert "the host's configuration governs" in entry
    assert "zero-copy" in entry
    assert "twice the model" in entry
    assert "2.3x" in entry


def test_the_runbook_states_the_status_fields_the_worker_actually_reports():
    """A reader following "how to check it took" must find the real keys."""
    entry = _entry_62()
    for field in comfy_dynamic.snapshot():
        assert field in entry
    for field in ("aimdo_enabled", "weight_residency", "patcher_class"):
        assert field in entry
    assert "comfy-managed residency ACTIVE" in entry
    # The entry names main.py's line only to say a worker never prints it.
    assert "DynamicVRAM support detected and enabled" in entry


def test_the_runbook_leads_with_what_the_operator_gets():
    entry = _entry_62()
    for promise in ("comfy_managed", "LoRA", "FSDP", "RDMA", "Fleet",
                    "**Reset attached mesh**", "dual-model"):
        assert promise in entry


def test_the_front_page_no_longer_carries_the_falsified_claim():
    """README must not say DynamicVRAM works only inside stock's single process.
    docs/VALIDATION.md (UMA section) records aimdo running inside a worker-shaped
    actor on a one-rank leg with no NCCL on 2026-08-04."""
    readme = (REPO / "README.md").read_text()
    assert "only\nworks inside stock's single process" not in readme
    assert "incompatible with the\nNCCL/actor runtime" not in readme
    assert "comfy_managed" in readme
    # The short front page links to the full residency limits.
    assert "docs/TROUBLESHOOTING.md#62-comfy-managed-residency" in readme
    guide = " ".join(_entry_62().split())
    for measured in ("Hard Clear VRAM emptied model stores", "two small Chroma FP8-mixed tests",
                     "reloading reproduced the first output exactly", "actor RSS stayed about 8.6 GiB",
                     "1.2 GiB baseline", "stock retained about 8.4 GiB",
                     "retire the actors", "persistent Worker services stay available"):
        assert measured in guide
    assert "**Reset attached mesh**" in guide
    assert "sustained multi-prompt soak" in guide
    assert "VALIDATION.md#memory-calibration" in guide
    assert "slab_weights" in guide


def test_the_cluster_reference_documents_the_key_and_its_trap():
    """config_schema accepts comfy_managed in [worker_args] with two cross-key
    rules, and CLUSTER.md presents its list of allowed keys as complete.

    The trap: [worker_args] is merged under the graph's own and rides verbatim in
    every capability context, so writing the key with either value re-proves
    every deployed PASS on the box.
    """
    cluster = (REPO / "docs" / "CLUSTER.md").read_text()
    assert "comfy_managed" in cluster
    assert "comfy_managed" in config_schema._BOOL_KEYS
    assert "Never write `comfy_managed = false`" in cluster
    assert "docs/TROUBLESHOOTING.md #62" in cluster


def test_the_concepts_page_describes_three_residencies_not_two():
    concepts = (REPO / "docs" / "CONCEPTS.md").read_text()
    assert "**Comfy-managed (`comfy_managed`).**" in concepts
    assert "A PASS cannot migrate\nfrom stock to slab residency" not in concepts
    assert "between any two of the three residencies" in concepts


def test_the_public_skill_mentions_the_widget():
    skill = (REPO / "skills" / "dgx-monarch" / "SKILL.md").read_text()
    assert "comfy_managed" in skill
    assert "docs/TROUBLESHOOTING.md #62" in skill


def test_the_runbook_names_the_cluster_toml_trap_too():
    entry = _entry_62()
    assert "never write `comfy_managed = false`" in " ".join(entry.lower().split())


def test_the_falsified_dynamic_vram_claim_is_withdrawn():
    """docs/TROUBLESHOOTING.md must not blame aimdo's background threads for
    failing inside monarch workers: measurement disproved that on 2026-08-04."""
    text = (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    assert "its background threads are incompatible" not in text
    assert "withdrawn" in text
    # The paragraph's advice stays.
    assert "Never disable DynamicVRAM in stock ComfyUI on a Spark" in text
    assert "accidental lowvram mode" in text
