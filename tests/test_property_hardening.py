"""Bounded properties for pure distributed, gate, and conditioning math."""

from __future__ import annotations

import math
import string
import sys
import types
from unittest.mock import patch

import torch
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from dgx_monarch.actor.sampling import equalize_cond_lengths
from dgx_monarch.adapters import base
from dgx_monarch.adapters.detect import _has_segment
from dgx_monarch.adapters.flux_family import _control_span_overlap
from dgx_monarch.gate_ledger import GateLedger, combo_key

_IDENTIFIER = st.text(alphabet=string.ascii_lowercase + string.digits + "_", min_size=1, max_size=10)
_OPTION_VALUE = st.one_of(st.booleans(), st.integers(-100, 100), _IDENTIFIER)
_OPTIONS = st.dictionaries(_IDENTIFIER, _OPTION_VALUE, max_size=6)
_LORA_NAMES = st.lists(_IDENTIFIER, unique=True, max_size=6)

_JSON_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-1000, 1000),
    st.text(alphabet=string.ascii_letters + string.digits + "_-", max_size=12),
)
_JSON_VALUE = st.recursive(
    _JSON_SCALAR,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(_IDENTIFIER, children, max_size=5),
    ),
    max_leaves=20,
)
_CONTEXT = st.dictionaries(_IDENTIFIER, _JSON_VALUE, max_size=6)


@st.composite
def _tensor_case(draw):
    rank = draw(st.integers(1, 4))
    axis = draw(st.integers(0, rank - 1))
    dim = draw(st.sampled_from((axis, axis - rank)))
    seq_len = draw(st.integers(0, 64))
    world = draw(st.integers(1, 8))
    shape = [draw(st.integers(1, 4)) for _ in range(rank)]
    shape[axis] = seq_len
    return shape, axis, dim, seq_len, world


def _distributed_modules(group):
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_sp_group = lambda: group
    core = types.ModuleType("xfuser.core")
    core.distributed = distributed
    xfuser = types.ModuleType("xfuser")
    xfuser.core = core
    return {
        "xfuser": xfuser,
        "xfuser.core": core,
        "xfuser.core.distributed": distributed,
    }


@given(case=_tensor_case())
def test_shard_seq_and_real_sp_gather_round_trip(case):
    shape, axis, dim, seq_len, world = case
    values = torch.arange(1, math.prod(shape) + 1, dtype=torch.int64).reshape(shape)
    chunks = []
    with patch.object(base, "sp_world", return_value=world):
        for rank in range(world):
            with patch.object(base, "sp_rank", return_value=rank):
                local, orig_len = base.shard_seq(values, dim=dim)
            assert local is not None
            assert orig_len == seq_len
            assert local.shape[axis] == math.ceil(seq_len / world)
            chunks.append(local)

    padded = torch.cat(chunks, dim=dim)
    assert torch.equal(padded.narrow(dim, 0, seq_len), values)
    assert not bool(padded.narrow(dim, seq_len, padded.shape[axis] - seq_len).any())

    class _Group:
        expected: torch.Tensor

        def all_gather(self, tensor, dim):
            assert dim == case[2]
            assert tensor.is_contiguous()
            assert torch.equal(tensor, self.expected.contiguous())
            return torch.cat(chunks, dim=dim)

    group = _Group()
    with patch.dict(sys.modules, _distributed_modules(group)):
        for local in chunks:
            group.expected = local
            gathered = base.sp_gather(local, seq_len, dim=dim)
            assert torch.equal(gathered, values)


@given(
    world=st.integers(1, 8),
    lengths=st.lists(st.integers(0, 32), min_size=1, max_size=4),
)
@example(world=3, lengths=[1, 1])
def test_padded_row_indices_match_the_rank_interleaved_marker_layout(world, lengths):
    rank_parts: list[list[torch.Tensor]] = [[] for _ in range(world)]
    metadata: list[tuple[int, int]] = []
    marker = 1
    with patch.object(base, "sp_world", return_value=world):
        for length in lengths:
            segment = torch.arange(marker, marker + length, dtype=torch.int64).reshape(1, length, 1)
            marker += max(length, 1)
            local_len = None
            for rank in range(world):
                with patch.object(base, "sp_rank", return_value=rank):
                    local, orig_len = base.shard_seq(segment)
                assert local is not None and orig_len == length
                local_len = local.shape[1]
                rank_parts[rank].append(local)
            assert local_len is not None
            metadata.append((length, local_len))

        layout = torch.cat(
            [torch.cat(parts, dim=1) for parts in rank_parts],
            dim=1,
        )
        actual = base.padded_row_indices(metadata)

    expected = torch.where(layout[0, :, 0] == 0)[0].tolist()
    assert actual == expected
    assert actual == sorted(set(actual))
    assert all(0 <= row < layout.shape[1] for row in actual)


@given(
    seq_len=st.integers(1, 64),
    world=st.integers(1, 8),
    span_start=st.integers(0, 64),
    requested_len=st.integers(0, 64),
)
@example(seq_len=16, world=4, span_start=0, requested_len=1)
def test_control_span_overlap_matches_global_add_then_shard(
    seq_len, world, span_start, requested_len
):
    span_start = min(span_start, seq_len)
    span_len = min(requested_len, seq_len - span_start)
    stream = torch.arange(1, seq_len + 1, dtype=torch.int64).reshape(1, seq_len, 1)
    add = torch.arange(101, 101 + span_len, dtype=torch.int64).reshape(1, span_len, 1)
    expected_global = stream.clone()
    expected_global[:, span_start:span_start + span_len] += add
    rows = math.ceil(seq_len / world)

    with patch.object(base, "sp_world", return_value=world):
        for rank in range(world):
            with patch.object(base, "sp_rank", return_value=rank):
                local, _ = base.shard_seq(stream)
                expected_local, _ = base.shard_seq(expected_global)
            assert local is not None and expected_local is not None
            actual_local = local.clone()
            row0 = rank * rows
            overlap = _control_span_overlap(row0, rows, span_start, span_len)
            disjoint = max(row0, span_start) >= min(row0 + rows, span_start + span_len)
            assert (overlap is None) == disjoint
            if overlap is None:
                assert torch.equal(actual_local, local)
            else:
                local_lo, local_hi, add_lo, add_hi = overlap
                assert local_hi - local_lo == add_hi - add_lo
                assert row0 + local_lo == span_start + add_lo
                assert row0 + local_hi == span_start + add_hi
                actual_local[:, local_lo:local_hi] += add[:, add_lo:add_hi]
            assert torch.equal(actual_local, expected_local)


@given(unet_name=_IDENTIFIER, options=_OPTIONS, lora_names=_LORA_NAMES)
def test_combo_key_is_stable_under_mapping_and_lora_order(unet_name, options, lora_names):
    reversed_options = dict(reversed(list(options.items())))
    assert combo_key(unet_name, options, lora_names) == combo_key(
        unet_name, reversed_options, list(reversed(lora_names))
    )


def _reverse_mappings(value):
    if isinstance(value, dict):
        return {
            key: _reverse_mappings(child)
            for key, child in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mappings(child) for child in value]
    return value


@given(context=_CONTEXT)
def test_capability_context_is_stable_under_recursive_mapping_order(context):
    assert GateLedger._context_value(context) == GateLedger._context_value(
        _reverse_mappings(context)
    )


@settings(max_examples=200)
@given(
    unet_a=_IDENTIFIER,
    options_a=_OPTIONS,
    loras_a=_LORA_NAMES,
    unet_b=_IDENTIFIER,
    options_b=_OPTIONS,
    loras_b=_LORA_NAMES,
)
def test_distinct_canonical_combos_do_not_collide_in_bounded_sample(
    unet_a, options_a, loras_a, unet_b, options_b, loras_b
):
    canonical_a = (unet_a, tuple(sorted(options_a.items())), tuple(sorted(loras_a)))
    canonical_b = (unet_b, tuple(sorted(options_b.items())), tuple(sorted(loras_b)))
    assume(canonical_a != canonical_b)
    key_a = combo_key(unet_a, options_a, loras_a)
    key_b = combo_key(unet_b, options_b, loras_b)
    assert len(key_a) == len(key_b) == 24
    assert key_a != key_b


@given(
    needle=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10),
    left=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6),
    right=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6),
)
def test_dotless_segment_needles_reject_mid_name_supersets(needle, left, right):
    negatives = [
        left + needle,
        needle + right,
        left + needle + right,
        f"0.{left}{needle}{right}.1",
    ]
    assert not _has_segment(negatives, needle)
    positives = [needle, f"{needle}.weight", f"root.{needle}", f"root.{needle}.weight"]
    assert _has_segment(positives, needle)


class _PaddingAdapter:
    def __init__(self, rule: str):
        self.cfg_cond_padding = rule


def _conditioning_entries(lengths, batch, width, start):
    entries = []
    originals = []
    offset = start
    for length in lengths:
        tensor = torch.arange(
            offset,
            offset + batch * length * width,
            dtype=torch.float32,
        ).reshape(batch, length, width)
        offset += max(batch * length * width, 1)
        entries.append([tensor, {"source": offset}])
        originals.append(tensor.clone())
    return entries, originals


@given(
    positive_lengths=st.lists(st.integers(0, 16), min_size=1, max_size=3),
    negative_lengths=st.lists(st.integers(0, 16), min_size=1, max_size=3),
    batch=st.integers(1, 3),
    width=st.integers(1, 8),
    rule=st.sampled_from(("pad", "pad+mask", "pad+text-mask")),
    none_in_positive=st.booleans(),
)
@example(
    positive_lengths=[3], negative_lengths=[3], batch=1, width=2,
    rule="pad", none_in_positive=True,
)
@example(
    positive_lengths=[0], negative_lengths=[4], batch=1, width=2,
    rule="pad", none_in_positive=False,
)
def test_equalize_cond_lengths_preserves_prefixes_zero_pads_and_keeps_none(
    positive_lengths,
    negative_lengths,
    batch,
    width,
    rule,
    none_in_positive,
):
    positive, original_positive = _conditioning_entries(
        positive_lengths, batch, width, 1
    )
    negative, original_negative = _conditioning_entries(
        negative_lengths, batch, width, 1001
    )
    none_entry = [None, {"marker": "none"}]
    (positive if none_in_positive else negative).append(none_entry)
    positive_before = list(positive)
    negative_before = list(negative)
    lengths_all = [*positive_lengths, *negative_lengths]
    longest = max(lengths_all)
    if rule == "pad+mask" and len(set(lengths_all)) > 1:
        # pad+mask aligns the joint text+image key extent to a multiple of 8;
        # the (.., 9, 11) latent below has a 5x6 = 30 token grid.
        longest = -(-(longest + 30) // 8) * 8 - 30
    elif rule == "pad+text-mask" and len(set(lengths_all)) > 1:
        # Hunyuan owns its joint mask, so only the text extent is aligned.
        longest = -(-longest // 8) * 8

    out_positive, out_negative = equalize_cond_lengths(
        _PaddingAdapter(rule),
        positive,
        negative,
        torch.zeros(batch, 4, 9, 11),
    )

    input_tensors = [*original_positive, *original_negative]
    output_tensors = [
        entry[0]
        for entry in [*out_positive, *out_negative]
        if torch.is_tensor(entry[0])
    ]
    lengths = [*positive_lengths, *negative_lengths]
    assert len(output_tensors) == len(input_tensors) == len(lengths)
    for output, original, length in zip(output_tensors, input_tensors, lengths, strict=True):
        assert output.shape[1] == longest
        assert torch.equal(output[:, :length], original)
        assert not bool(output[:, length:].any())

    output_none = [entry for entry in [*out_positive, *out_negative] if entry[0] is None]
    assert output_none == [[None, {"marker": "none"}]]
    assert "attention_mask" not in output_none[0][1]
    for entry, original in zip(positive_before, [*original_positive, None], strict=False):
        if original is not None and torch.is_tensor(entry[0]):
            assert torch.equal(entry[0], original)
    for entry, original in zip(negative_before, [*original_negative, None], strict=False):
        if original is not None and torch.is_tensor(entry[0]):
            assert torch.equal(entry[0], original)

    if len(set(lengths)) == 1:
        assert out_positive is positive
        assert out_negative is negative
