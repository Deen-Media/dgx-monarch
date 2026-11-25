"""Every sharded forward keeps its pad rows out of the shared NVFP4 scale, or says why not.

A forward that shards a stream with ``shard_seq`` can pad it, and a pad row in
an NVFP4 Linear's input can set the scale the whole group quantizes against
(adapters/scale_rows.py). Read from the source: each adapter module that
calls ``shard_seq`` names its streams' real rows with ``scale_rows.real_rows``
and has a test that runs its Linears under the shared scale with NaN pad rows,
or is listed here with the reason it needs neither.
"""
from __future__ import annotations

import ast
from pathlib import Path

ADAPTERS = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch" / "adapters"
TESTS = Path(__file__).resolve().parent

# Modules that call shard_seq and set no scope of their own, and why.
NO_SCOPE = {
    "flux_shared.py": "a helper; its callers in flux_family.py set the scope",
    "flux3_packing.py": "a helper; its caller flux3.py sets the scope",
    "omnigen2.py": "shards with allow_padding=False, so it never pads",
    "zimage.py": "refuses a stream the degree does not divide (_assert_concat_divisible)",
    "pixeldit_comfy.py": "sp_validated_quants admits bf16 only, so NVFP4 never runs sharded",
    "moss_image_seams.py": "sp_validated_quants admits bf16 and int8 only, so NVFP4 never "
                           "runs sharded",
    "mage_flow.py": "keeps its own row scope (adapters/mage_nvfp4_scale.py)",
    "wan_variants.py": "scopes through wan_family.named_pads; test_wan_pad_exclusion.py "
                       "runs SCAIL, SCAIL-2 and WanDancer under the shared scale",
}

# The test that runs each scoped module's Linears under the shared scale.
SCALE_TESTS = {
    "anima.py": ("test_anima_pad_exclusion.py", "test_the_shared_nvfp4_scale_counts_only_real_rows"),
    "boogu.py": ("test_scale_rows_families.py", "test_boogu"),
    "cogvideo.py": ("test_cogvideo_pad_exclusion.py",
                    "test_the_shared_nvfp4_scale_counts_only_real_rows"),
    "ernie.py": ("test_scale_rows_families.py", "test_ernie"),
    "flux3.py": ("test_scale_rows_families.py", "test_flux3"),
    "flux_family.py": ("test_scale_rows_families.py", "test_flux"),
    "hunyuan.py": ("test_scale_rows_families.py", "test_hunyuan"),
    "kandinsky5.py": ("test_kandinsky5_pad_exclusion.py",
                      "test_the_shared_nvfp4_scale_counts_only_real_rows"),
    "krea2.py": ("test_scale_rows_families.py", "test_krea2"),
    "lens.py": ("test_scale_rows_families.py", "test_lens"),
    "ltx.py": ("test_scale_rows_families.py", "test_ltx"),
    "minimax_h3.py": ("test_scale_rows_families.py", "test_minimax_h3"),
    "minimax_h3_image.py": ("test_scale_rows_families.py", "test_minimax_h3_image"),
    "pixeldit.py": ("test_scale_rows_families.py", "test_ideogram4"),
    "qwen_image.py": ("test_scale_rows_families.py", "test_qwen_image"),
    "qwen_image21.py": ("test_scale_rows_families.py", "test_qwen_image21"),
    # The same test runs SCAIL, SCAIL 2 and WanDancer (its CASES), whose scope
    # is wan_family.named_pads.
    "wan_family.py": ("test_wan_pad_exclusion.py", "test_the_shared_nvfp4_scale_counts_only_real_rows"),
}


def _calls(path: Path, name: str) -> bool:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called == name:
                return True
    return False


def _defines(path: Path, name: str) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == name
               for node in ast.walk(ast.parse(path.read_text())))


def test_every_sharding_adapter_scopes_the_scale_or_says_why_not():
    sharding = {path.name for path in ADAPTERS.glob("*.py") if _calls(path, "shard_seq")}
    scoped = {path.name for path in ADAPTERS.glob("*.py") if _calls(path, "real_rows")}
    assert sharding, "found no shard_seq caller"
    unaccounted = sorted(sharding - scoped - set(NO_SCOPE))
    assert not unaccounted, f"shard without a scale scope or a reason: {unaccounted}"
    stale = sorted(name for name in NO_SCOPE if name not in sharding)
    assert not stale, f"listed but no longer sharding: {stale}"
    assert not sorted(scoped & set(NO_SCOPE)), "listed as unscoped but sets a scope"


def test_every_scoped_adapter_has_a_scale_test():
    scoped = {path.name for path in ADAPTERS.glob("*.py")
              if _calls(path, "real_rows") and path.name != "scale_rows.py"}
    assert scoped == set(SCALE_TESTS), sorted(scoped ^ set(SCALE_TESTS))
    for module, (test_file, test_name) in SCALE_TESTS.items():
        assert _defines(TESTS / test_file, test_name), (module, test_file, test_name)
