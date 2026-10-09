"""The adapter ``matches()`` trio, as one table.

Each ``_Row`` checks that an adapter accepts its own exact comfy.model_base
type(s), declines an unrelated model without raising, and either raises a typed
UnsupportedModelError on a subclass it has not vetted or, when its matches()
never raises, admits that subclass. The rows are written by hand;
``test_each_row_accepts_every_type_its_adapter_declares`` checks them against
each adapter's ``model_base_classes`` / ``exact_model_base_classes``. A new
adapter whose matches() has this shape needs one ``_Row``, not a new test file.

``test_every_registered_adapter_has_a_matches_row_or_is_named_bespoke`` walks
the registry (``dgx_monarch.adapters.ADAPTERS``) and fails on an adapter that
is neither tabled here nor in ``_BESPOKE_CLASSES``. The adapters in
``_BESPOKE_CLASSES`` have another shape and are tested in their own files:

  * FluxAdapter / Flux2Adapter / LongCatAdapter: their matches() tests are
    about registry order (Chroma, Flux2 and LongCat must precede the root Flux
    adapter on one real model_base.Flux subclass tree), not one adapter alone.
    tests/test_flux_adapter.py.
  * HunyuanAdapter: two broad roots (HunyuanImage21 / HunyuanVideo) whose
    exact set admits real subclasses (the refiner, the SR-distilled variant)
    and whose typed reject names a whole excluded model family, not an
    exotic probe. tests/test_hunyuan_adapter.py.
  * ZImageAdapter: never raises; its gate reads
    `diffusion_model.pad_tokens_multiple` to tell a latent Z-Image from a
    real, unsupported Lumina 2.0 sharing the same comfy type.
    tests/test_zimage_adapter.py.

These tests pin a real sibling, parent or registry-order fact, not the generic
shape, so they stay in their files; keep family-specific detail there rather
than flatten it into a row:

  * tests/test_boogu_adapter.py::test_matches_ignores_omnigen2_parent and
    ::test_registry_order_omnigen2_raises_on_boogu
  * tests/test_omnigen2_adapter.py::test_matches_rejects_boogu_subclass_typed
  * tests/test_anima_adapter.py::test_matches_excludes_sibling_cosmos_predict2
  * tests/test_wan_variants.py::test_scail2_must_precede_scail_in_registry
  * tests/test_ltx_adapter.py's registry-dispatch tests and
    ::test_supported_exact_matches_declared_model_base_classes

Declared-contract tests (family, cfg_cond_padding, model_base_classes) stay in
their adapter files; this module covers ``matches()`` only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest

from dgx_monarch.adapters import ADAPTERS
from dgx_monarch.adapters.anima import AnimaAdapter
from dgx_monarch.adapters.base import Adapter, UnsupportedModelError
from dgx_monarch.adapters.boogu import BooguAdapter
from dgx_monarch.adapters.cogvideo import CogVideoXAdapter
from dgx_monarch.adapters.ernie import ErnieAdapter
from dgx_monarch.adapters.flux_family import ChromaAdapter, Flux2Adapter, FluxAdapter, LongCatAdapter
from dgx_monarch.adapters.hunyuan import HunyuanAdapter
from dgx_monarch.adapters.kandinsky5 import Kandinsky5Adapter
from dgx_monarch.adapters.krea2 import Krea2Adapter
from dgx_monarch.adapters.lens import LensAdapter
from dgx_monarch.adapters.ltx import LTXAdapter
from dgx_monarch.adapters.mage_flow import MageFlowAdapter
from dgx_monarch.adapters.minimax_h3 import MiniMaxH3Adapter
from dgx_monarch.adapters.omnigen2 import Omnigen2Adapter
from dgx_monarch.adapters.pixeldit import Ideogram4Adapter
from dgx_monarch.adapters.pixeldit_comfy import PixelDiTAdapter
from dgx_monarch.adapters.qwen_image import QwenImageAdapter
from dgx_monarch.adapters.qwen_image21 import QwenImage21Adapter
from dgx_monarch.adapters.wan_family import WanAdapter
from dgx_monarch.adapters.wan_variants import SCAIL2Adapter, SCAILAdapter, WanAnimate2Adapter, WanDancerAdapter
from dgx_monarch.adapters.zimage import ZImageAdapter
from fake_model_base_helpers import install_fake_model_base

# A name no row's tree defines, present in every tree this module builds:
# the universal decline case. Separate from the real-sibling names a few rows
# list in `extra_declines`.
_FOREIGN = "_UnrelatedModel"


@dataclass(frozen=True)
class _Row:
    adapter: Adapter
    # name -> parent name (or None for a root), fed to install_fake_model_base.
    tree: dict
    # tree names this adapter's matches() must accept (return True for).
    accepts: tuple
    # tree names, beyond the generic `_FOREIGN` probe, this adapter's
    # matches() must decline (return False, never raise): a real sibling or
    # parent in this family's own tree, not a made-up unrelated class.
    extra_declines: tuple = ()
    # The `match=` text of the typed reject this adapter raises on a subclass
    # of `accepts[0]` that is not itself in `accepts`, or None if its
    # matches() never raises (the inherited base.Adapter.matches admits any
    # isinstance subclass).
    reject_fragment: str | None = None


# Each row's exact set and raise text are copied by hand from the adapter's
# source. Many adapters name their exact type as a literal inside matches() or
# under another name (`_EXACT`, `exact_model_base`, a module constant), so no
# one attribute holds the exact set for every adapter.
_ROWS = (
    _Row(
        AnimaAdapter(), {"Anima": None}, ("Anima",),
        reject_fragment="anima variant",
    ),
    _Row(
        BooguAdapter(), {"Boogu": None}, ("Boogu",),
        reject_fragment="boogu variant",
    ),
    _Row(
        CogVideoXAdapter(), {"CogVideoX": None}, ("CogVideoX",),
        reject_fragment="cogvideo variant",
    ),
    _Row(
        ErnieAdapter(), {"ErnieImage": None}, ("ErnieImage",),
        reject_fragment="ernie variant",
    ),
    _Row(
        Kandinsky5Adapter(),
        {"Kandinsky5": None, "Kandinsky5Image": "Kandinsky5"},
        ("Kandinsky5", "Kandinsky5Image"),
        reject_fragment="kandinsky5 variant",
    ),
    _Row(
        LTXAdapter(), {"LTXV": None, "LTXAV": None}, ("LTXV", "LTXAV"),
        reject_fragment="ltx variant",
    ),
    _Row(
        MiniMaxH3Adapter(), {"MiniMaxH3": None}, ("MiniMaxH3",),
        reject_fragment="minimax_h3 variant",
    ),
    _Row(
        # test_matches_rejects_boogu_subclass_typed (test_omnigen2_adapter.py)
        # keeps the real Boogu-subclass case and its "omnigen2 variant Boogu"
        # message; this row's own reject probe is a made-up exotic subclass.
        Omnigen2Adapter(), {"Omnigen2": None}, ("Omnigen2",),
        reject_fragment="omnigen2 variant",
    ),
    _Row(
        PixelDiTAdapter(),
        {"PixelDiTT2I": None, "PiD": "PixelDiTT2I"},
        ("PixelDiTT2I", "PiD"),
        reject_fragment="pixeldit variant",
    ),
    _Row(
        MageFlowAdapter(), {"MageFlow": None}, ("MageFlow",),
        reject_fragment="mage_flow variant",
    ),
    _Row(
        QwenImage21Adapter(), {"QwenImage21": None}, ("QwenImage21",),
        reject_fragment="qwen_image21 variant",
    ),
    _Row(
        QwenImageAdapter(), {"QwenImage": None}, ("QwenImage",),
        reject_fragment="qwen_image variant",
    ),
    _Row(
        LensAdapter(), {"Lens": None}, ("Lens",),
        reject_fragment="lens variant",
    ),
    # This row is WanAdapter's only direct matches() test.
    _Row(
        WanAdapter(),
        {"WAN21": None, "WAN22": "WAN21", "WAN21_FlowRVS": "WAN21"},
        ("WAN21", "WAN22", "WAN21_FlowRVS"),
        reject_fragment="wan variant",
    ),
    # Each wan variant also declines its own real parent or sibling.
    _Row(
        SCAILAdapter(),
        {"WAN21": None, "WAN21_SCAIL": "WAN21"},
        ("WAN21_SCAIL",),
        extra_declines=("WAN21",),
        reject_fragment="wan-scail variant",
    ),
    _Row(
        SCAIL2Adapter(),
        {"WAN21": None, "WAN21_SCAIL": "WAN21", "WAN21_SCAIL2": "WAN21_SCAIL"},
        ("WAN21_SCAIL2",),
        extra_declines=("WAN21_SCAIL",),
        reject_fragment="wan-scail variant",
    ),
    _Row(
        WanDancerAdapter(),
        {"WAN21": None, "WAN21_SCAIL": "WAN21", "WAN22_WanDancer": "WAN21"},
        ("WAN22_WanDancer",),
        extra_declines=("WAN21_SCAIL",),
        reject_fragment="wandancer variant",
    ),
    _Row(
        WanAnimate2Adapter(), {"WAN21": None, "WAN_Animate2": "WAN21"},
        ("WAN_Animate2",),
        extra_declines=("WAN21",),
        reject_fragment="wan-animate2 variant",
    ),
    # Krea2Adapter, Ideogram4Adapter and ChromaAdapter do not override
    # matches(): the inherited isinstance check admits any subclass of their
    # one model_base_classes entry (flux_family.py: ChromaAdapter admits the
    # real ChromaRadiance subclass on purpose). Their reject_fragment is None;
    # test_matches_admits_an_unlisted_subclass covers them instead.
    _Row(Krea2Adapter(), {"Krea2": None}, ("Krea2",)),
    _Row(Ideogram4Adapter(), {"Ideogram4": None}, ("Ideogram4",)),
    _Row(
        ChromaAdapter(), {"Chroma": None, "ChromaRadiance": "Chroma"},
        ("Chroma", "ChromaRadiance"),
    ),
)

# Adapters whose matches() has another shape; the module docstring names the
# file that tests each.
_BESPOKE_CLASSES = frozenset({FluxAdapter, Flux2Adapter, LongCatAdapter, HunyuanAdapter, ZImageAdapter})
_TABLED_CLASSES = frozenset(type(row.adapter) for row in _ROWS)


def _full_tree(row: _Row) -> dict:
    tree = dict(row.tree)
    tree.setdefault(_FOREIGN, None)
    return tree


# SCAIL2Adapter inherits SCAILAdapter's family "wan_scail", so where two rows
# share a family the row id adds the class name.
_FAMILY_ROW_COUNT = Counter(row.adapter.family for row in _ROWS)


def _row_id(row: _Row) -> str:
    if _FAMILY_ROW_COUNT[row.adapter.family] > 1:
        return f"{row.adapter.family}-{type(row.adapter).__name__}"
    return row.adapter.family


def _accept_cases():
    for row in _ROWS:
        for name in row.accepts:
            yield pytest.param(row, name, id=f"{_row_id(row)}-{name}")


def _decline_cases():
    for row in _ROWS:
        for name in (*row.extra_declines, _FOREIGN):
            yield pytest.param(row, name, id=f"{_row_id(row)}-{name}")


def _reject_cases():
    for row in _ROWS:
        if row.reject_fragment is not None:
            yield pytest.param(row, id=_row_id(row))


def _admits_subclass_cases():
    for row in _ROWS:
        if row.reject_fragment is None:
            yield pytest.param(row, id=_row_id(row))


@pytest.mark.parametrize("row,name", list(_accept_cases()))
def test_matches_accepts_its_registered_types(monkeypatch, row: _Row, name: str):
    mb = install_fake_model_base(monkeypatch, _full_tree(row))
    assert row.adapter.matches(getattr(mb, name)()) is True


@pytest.mark.parametrize("row,name", list(_decline_cases()))
def test_matches_declines_without_raising(monkeypatch, row: _Row, name: str):
    mb = install_fake_model_base(monkeypatch, _full_tree(row))
    assert row.adapter.matches(getattr(mb, name)()) is False


@pytest.mark.parametrize("row", list(_reject_cases()))
def test_matches_rejects_an_unvetted_subclass_typed(monkeypatch, row: _Row):
    tree = _full_tree(row)
    probe = f"_{row.adapter.family}_unvetted_probe"
    tree[probe] = row.accepts[0]
    mb = install_fake_model_base(monkeypatch, tree)
    with pytest.raises(UnsupportedModelError, match=row.reject_fragment):
        row.adapter.matches(getattr(mb, probe)())


@pytest.mark.parametrize("row", list(_admits_subclass_cases()))
def test_matches_admits_an_unlisted_subclass(monkeypatch, row: _Row):
    # An adapter that never raises admits a subclass of its registered type
    # (see the comment above the Krea2 row in _ROWS).
    tree = _full_tree(row)
    probe = f"_{row.adapter.family}_unlisted_probe"
    tree[probe] = row.accepts[0]
    mb = install_fake_model_base(monkeypatch, tree)
    assert row.adapter.matches(getattr(mb, probe)()) is True


def test_every_registered_adapter_has_a_matches_row_or_is_named_bespoke():
    for adapter in ADAPTERS:
        cls = type(adapter)
        assert cls in _TABLED_CLASSES or cls in _BESPOKE_CLASSES, (
            f"{cls.__name__} is new to dgx_monarch.adapters.ADAPTERS and this module "
            "does not cover its matches() contract. If its matches() follows the "
            "common accept/decline/(typed-reject-or-admits-a-subclass) shape, add a "
            "_Row for it above. If it does not, add the class to _BESPOKE_CLASSES "
            "with a comment naming the test file that covers it on its own."
        )


def _descends_from(tree: dict, name: str, roots: set) -> bool:
    parent = tree.get(name)
    while parent is not None:
        if parent in roots:
            return True
        parent = tree.get(parent)
    return False


@pytest.mark.parametrize("row", [pytest.param(row, id=_row_id(row)) for row in _ROWS])
def test_each_row_accepts_every_type_its_adapter_declares(row: _Row):
    # Without this check, a type an existing adapter adds to its declared tuple
    # would get no accept case and the registry walk above would still pass. A
    # name beyond the declared set is allowed only as a real subclass a
    # non-raising adapter admits on purpose (ChromaRadiance under Chroma).
    declared = set(row.adapter.exact_model_base_classes or row.adapter.model_base_classes)
    accepted = set(row.accepts)
    assert declared <= accepted, (
        f"{type(row.adapter).__name__} declares {sorted(declared - accepted)} "
        "but its _Row does not accept them; add them to the row's tree and accepts.")
    for name in accepted - declared:
        assert row.reject_fragment is None and _descends_from(row.tree, name, declared), (
            f"{type(row.adapter).__name__}'s _Row accepts {name}, which the adapter "
            "does not declare and which is not a subclass it admits on purpose.")
