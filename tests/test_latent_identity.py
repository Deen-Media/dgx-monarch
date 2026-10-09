"""Fail-closed identity comparison for plain and packed Comfy latents."""
from __future__ import annotations

import sys
import types

import pytest
import torch

from dgx_monarch.nodes.latent_identity import compare_latents


class LyingTensor(torch.Tensor):
    """Tensor subclass whose ``__torch_function__`` says every ``torch.equal`` is True."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.equal:
            return True
        return super().__torch_function__(func, types, args, kwargs or {})


def _lying_tensor(value: float) -> LyingTensor:
    return torch.tensor([value]).as_subclass(LyingTensor)


@pytest.fixture
def nested_tensor_type(monkeypatch):
    class NestedTensor:
        def __init__(self, tensors, *, unbind_error: Exception | None = None):
            self.tensors = list(tensors)
            self.is_nested = True
            self._unbind_error = unbind_error

        @property
        def shape(self):
            raise AssertionError(
                "packed identity must inspect modalities, not wrapper shape")

        def unbind(self):
            if self._unbind_error is not None:
                raise self._unbind_error
            return self.tensors

    NestedTensor.__module__ = "comfy.nested_tensor"
    comfy = types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = NestedTensor
    comfy.nested_tensor = nested
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested)
    return NestedTensor


def _packed(nested_tensor_type, *, video=None, audio=None):
    return nested_tensor_type((
        torch.zeros(1, 4, 2, 3) if video is None else video,
        torch.zeros(1, 3, 2) if audio is None else audio,
    ))


def test_plain_latents_compare_exactly_and_report_max_diff():
    left = torch.tensor([0.0, 2.0])

    assert compare_latents(left, left.clone()) == (True, 0.0)
    assert compare_latents(left, torch.tensor([0.0, 5.0])) == (False, 3.0)
    assert compare_latents(left, torch.zeros(1, 2)) == (False, None)


def test_tensor_subclass_cannot_override_flat_or_packed_identity(
    nested_tensor_type,
):
    left = _lying_tensor(1.0)
    right = _lying_tensor(9.0)

    assert torch.equal(left, right) is True
    assert compare_latents(left, right) == (False, None)
    assert compare_latents(
        nested_tensor_type((left, torch.zeros(1))),
        nested_tensor_type((right, torch.zeros(1))),
    ) == (False, None)


def test_nested_latents_compare_every_modality_and_aggregate_diff(
    nested_tensor_type,
):
    left = _packed(nested_tensor_type)
    video = left.tensors[0].clone()
    audio = left.tensors[1].clone()
    video.flatten()[0] = 2.5
    audio.flatten()[0] = 4.0

    assert compare_latents(left, _packed(nested_tensor_type)) == (True, 0.0)
    assert compare_latents(
        left, _packed(nested_tensor_type, video=video, audio=audio)
    ) == (False, 4.0)


@pytest.mark.parametrize("changed_modality", ["video", "audio"])
def test_nested_latent_mutation_in_either_modality_denies_identity(
    nested_tensor_type, changed_modality,
):
    left = _packed(nested_tensor_type)
    video, audio = (part.clone() for part in left.unbind())
    (video if changed_modality == "video" else audio).flatten()[0] = 2.5

    assert compare_latents(
        left, _packed(nested_tensor_type, video=video, audio=audio)
    ) == (False, 2.5)


def test_nested_comparison_requires_the_exact_direct_wrapper_type(
    nested_tensor_type,
):
    class DerivedNestedTensor(nested_tensor_type):
        pass

    direct = _packed(nested_tensor_type)
    derived = DerivedNestedTensor(tuple(part.clone() for part in direct.unbind()))

    assert compare_latents(direct, derived) == (False, None)
    assert compare_latents(derived, derived) == (False, None)
    assert compare_latents(torch.zeros(1), nested_tensor_type((torch.zeros(1),))) == (
        False,
        None,
    )


def test_same_named_non_comfy_wrapper_is_unsupported(nested_tensor_type):
    Impostor = type(
        "NestedTensor",
        (),
        {"unbind": lambda self: (torch.zeros(1),)},
    )

    assert compare_latents(Impostor(), Impostor()) == (False, None)


def test_nested_modality_count_and_payload_mismatches_fail_closed(
    nested_tensor_type,
):
    left = _packed(nested_tensor_type)
    wrong_payload = nested_tensor_type((torch.zeros(1),))
    wrong_payload.tensors = torch.zeros(2)
    wrong_kind = nested_tensor_type((torch.zeros(1),))
    wrong_kind.is_nested = False
    malformed = (
        nested_tensor_type(()),
        nested_tensor_type((torch.zeros(1),)),
        nested_tensor_type((torch.zeros(1), torch.zeros(1), torch.zeros(1))),
        nested_tensor_type((torch.zeros(1), "not-a-tensor")),
        nested_tensor_type(
            (torch.zeros(1),), unbind_error=RuntimeError("broken wrapper")),
        wrong_payload,
        wrong_kind,
        object(),
    )

    for right in malformed:
        assert compare_latents(left, right) == (False, None)


@pytest.mark.parametrize(
    "right_factory",
    [
        lambda nested: _packed(
            nested, video=torch.zeros(1, 4, 2, 4)),
        lambda nested: _packed(
            nested, audio=torch.zeros(1, 3, 2, dtype=torch.float64)),
    ],
)
def test_nested_shape_or_dtype_mismatch_fails_closed(
    nested_tensor_type, right_factory,
):
    assert compare_latents(
        _packed(nested_tensor_type), right_factory(nested_tensor_type)
    ) == (False, None)


def test_plain_layout_device_and_exact_type_mismatches_fail_closed():
    with torch.sparse.check_sparse_tensor_invariants():
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0]]), torch.tensor([0.0]), size=(2,))

    assert compare_latents(torch.zeros(2), sparse) == (False, None)
    assert compare_latents(torch.zeros(2), torch.empty(2, device="meta")) == (
        False,
        None,
    )
    assert compare_latents(torch.zeros(2), torch.nn.Parameter(torch.zeros(2))) == (
        False,
        None,
    )


def test_empty_tensor_modalities_are_exact(nested_tensor_type):
    left = nested_tensor_type((
        torch.empty(0),
        torch.empty(1, 0, dtype=torch.float64),
    ))
    right = nested_tensor_type((
        torch.empty(0),
        torch.empty(1, 0, dtype=torch.float64),
    ))

    assert compare_latents(left, right) == (True, 0.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_plain_or_nested_latents_never_grant_identity(
    nested_tensor_type, value,
):
    plain = torch.tensor([value])
    packed = _packed(nested_tensor_type, audio=torch.tensor([value]))

    assert compare_latents(plain, plain.clone()) == (False, None)
    assert compare_latents(
        packed,
        _packed(nested_tensor_type, audio=torch.tensor([value])),
    ) == (False, None)


def test_finite_exact_float64_extremes_remain_pass_eligible():
    latent = torch.tensor([torch.finfo(torch.float64).max], dtype=torch.float64)

    assert compare_latents(latent, latent.clone()) == (True, 0.0)
