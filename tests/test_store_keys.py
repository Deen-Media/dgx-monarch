"""Model-store keys (docs/DESIGN.md section 5.5), tested as pure functions without comfy."""
from dgx_monarch.actor.model_store import ModelStore, lora_signature, normalize_options


def test_normalize_options_stable():
    a = normalize_options({"weight_dtype": "fp8_e4m3fn", "x": 1})
    b = normalize_options({"x": 1, "weight_dtype": "fp8_e4m3fn"})
    assert a == b
    assert normalize_options(None) == ()
    assert normalize_options({}) == ()


def test_lora_signature_order_matters():
    stack1 = [{"name": "a.safetensors", "strength": 1.0}, {"name": "b.safetensors", "strength": 0.5}]
    stack2 = list(reversed(stack1))
    assert lora_signature(stack1) != lora_signature(stack2)
    assert lora_signature(None) == ()


def test_keys_reuse_vs_hotswap_vs_load():
    base1, req1 = ModelStore.make_keys("m.safetensors", {"weight_dtype": "fp8_e4m3fn"}, [], "fp8")
    base2, req2 = ModelStore.make_keys(
        "m.safetensors", {"weight_dtype": "fp8_e4m3fn"},
        [{"name": "l.safetensors", "strength": 1.0}], "fp8")
    base3, _req3 = ModelStore.make_keys("other.safetensors", {}, [], "bf16")

    assert base1 == base2          # same base: a LoRA hot-swap
    assert req1 != req2            # different request: not a pure reuse
    assert base1 != base3          # different base: a full load


def test_fp16_is_distinct_request_identity_but_the_same_base():
    bf16_base, bf16_request = ModelStore.make_keys("m.safetensors", {}, [], "bf16")
    fp16_base, fp16_request = ModelStore.make_keys("m.safetensors", {}, [], "fp16")

    assert fp16_base == bf16_base
    assert fp16_request != bf16_request
    assert fp16_request[-1] == "fp16"


def test_strength_rounding_stable():
    sig_a = lora_signature([{"name": "l", "strength": 0.30000000001}])
    sig_b = lora_signature([{"name": "l", "strength": 0.3}])
    assert sig_a == sig_b
