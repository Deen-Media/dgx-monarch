"""adapters/fsdp_lora_admission.py: the checkpoint-property refusal itself.

tests/test_detect.py and tests/test_fsdp_capacity.py exercise header and live
admission through their real call sites. This file pins the shared raiser's
contract: it carries class P, names no guard (no bypass exists), and leads the
message with its tag, so a worker raise reads as a deliberate, CONSUMED
refusal, not an abandoned lease (nodes/pending.py ``typed_worker_refusal``).
"""
from __future__ import annotations

import pytest

from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.fsdp_lora_admission import (
    live_admission_property,
    refuse_fsdp_lora_checkpoint_property,
    refuse_unless_fsdp_lora_admits,
)
from dgx_monarch.refusal import RefusalClass, RefusalTag, parse_leading_refusal_tag


def test_refuse_fsdp_lora_checkpoint_property_carries_a_leading_class_p_tag():
    with pytest.raises(UnsupportedModelError) as raised:
        refuse_fsdp_lora_checkpoint_property("quantized_shards")
    message = str(raised.value)
    tag = parse_leading_refusal_tag(message)
    assert tag == RefusalTag(refusal_class=RefusalClass.PHYSICS, guard=None, waivable=False)
    assert "not admitted for this checkpoint" in message
    # lora_low_rss is never named as the fix: this card must not be mistaken
    # for the lever check's own card.
    assert "needs lora_low_rss on" not in message


def test_refuse_fsdp_lora_checkpoint_property_names_the_cause():
    with pytest.raises(UnsupportedModelError, match="comfy-kitchen quantized"):
        refuse_fsdp_lora_checkpoint_property("quantized_shards")


def test_refuse_fsdp_lora_checkpoint_property_rejects_an_unknown_property():
    with pytest.raises(ValueError, match="unknown FSDP LoRA admission property"):
        refuse_fsdp_lora_checkpoint_property("something_else")
    # fp32_islands must stay unrecognized here: actor/fsdp_lora.py's in-bake
    # check owns that case.
    with pytest.raises(ValueError, match="unknown FSDP LoRA admission property"):
        refuse_fsdp_lora_checkpoint_property("fp32_islands")


class _Evidence:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_live_admission_property_reads_quantized_shards_off_the_evidence():
    assert live_admission_property(
        _Evidence(quantized_shards=True, live_dtype_profile="uniform_or_quantized_fp8"),
    ) == "quantized_shards"


def test_live_admission_property_does_not_flag_fp32_islands():
    """A fp32-islands live profile is not a disqualifying property here. No
    coarse live signal (or header signal, adapters/detect.py) can tell a
    checkpoint whose island stays live fp32 (safe to bake) from one that casts
    to the core dtype live (a mismatch actor/fsdp_lora.py's in-bake check
    refuses); adapters/fsdp_lora_admission.py's module docstring names the two
    real checkpoints that disagree on it."""
    from dgx_monarch.adapters.fsdp_islands import FP32_ISLANDS_PROFILE

    assert live_admission_property(
        _Evidence(quantized_shards=False, live_dtype_profile=FP32_ISLANDS_PROFILE),
    ) is None


def test_live_admission_property_admits_a_plain_uniform_profile():
    assert live_admission_property(
        _Evidence(quantized_shards=False, live_dtype_profile="all_bf16"),
    ) is None


def test_refuse_unless_fsdp_lora_admits_is_a_noop_when_admitted():
    refuse_unless_fsdp_lora_admits(
        _Evidence(quantized_shards=False, live_dtype_profile="all_bf16"))  # no raise
