"""Sampler paths: DP inputs follow the latent batch partition, every rank checks the
sampler output, and the result leader returns detached CPU copies."""
import sys
import types

import pytest
import torch

from dgx_monarch.actor import sampling


class LyingTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.equal:
            return True
        return super().__torch_function__(func, types, args, kwargs or {})


def _cond(batch: int):
    rows = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1)
    pooled = torch.arange(batch, dtype=torch.float32).view(batch, 1)
    return [[rows, {"pooled_output": pooled, "broadcast": torch.ones(1, 2)}]]


class NestedTensor:
    """Stand-in for Comfy's NestedTensor; each keyword can break one contract the checks enforce."""

    def __init__(self, tensors, *, unbind_error=None, marker=True, direct_list=True):
        self.tensors = list(tensors) if direct_list else tuple(tensors)
        self.is_nested = marker
        self._unbind_error = unbind_error

    def unbind(self):
        if self._unbind_error is not None:
            raise self._unbind_error
        return self.tensors

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def dtype(self):
        return self.tensors[0].dtype

    @property
    def layout(self):
        return self.tensors[0].layout

    def size(self):
        return self.tensors[0].size()


@pytest.fixture
def dp_rank1(monkeypatch):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (1, 2))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: False)


@pytest.fixture
def comfy_sampling_stubs(monkeypatch):
    captured = {}
    sample_mod = types.ModuleType("comfy.sample")
    sample_mod.fix_empty_latent_channels = lambda _model, value, *_size_tags: value
    sample_mod.prepare_noise = lambda value, seed, batch_index=None: torch.zeros_like(value)

    def sample(_model, noise, _steps, _cfg, _sampler_name, _scheduler,
               positive, negative, latent_image, **kwargs):
        captured["positive"] = positive
        captured["negative"] = negative
        captured["latent"] = latent_image
        captured["noise"] = noise
        return captured.get("sample_result", latent_image)

    sample_mod.sample = sample
    samplers_mod = types.ModuleType("comfy.samplers")
    samplers_mod.KSampler = types.SimpleNamespace(SAMPLERS=("euler",), SCHEDULERS=("normal",))
    utils_mod = types.ModuleType("comfy.utils")
    utils_mod.repeat_to_batch_size = lambda value, batch: value.repeat(
        (batch + value.shape[0] - 1) // value.shape[0], *([1] * (value.ndim - 1)))[:batch]

    def unpack_latents(combined, shapes):
        parts = []
        offset = 0
        for shape in shapes:
            width = int(torch.tensor(shape[1:]).prod().item())
            parts.append(combined[:, :, offset:offset + width].reshape(shape))
            offset += width
        return parts

    utils_mod.unpack_latents = unpack_latents
    nested_mod = types.ModuleType("comfy.nested_tensor")
    nested_mod.NestedTensor = NestedTensor
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.__path__ = []
    comfy_mod.sample = sample_mod
    comfy_mod.samplers = samplers_mod
    comfy_mod.utils = utils_mod
    comfy_mod.nested_tensor = nested_mod
    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample_mod)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers_mod)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested_mod)
    return captured


def test_ksampler_slices_conditioning_with_latent(dp_rank1, comfy_sampling_stubs):
    request = {
        "latent": {"samples": torch.arange(4.0).view(4, 1, 1, 1)},
        "positive": _cond(4),
        "negative": _cond(4),
        "sampler_name": "euler",
        "scheduler": "normal",
        "steps": 1,
        "cfg": 1.0,
        "noise_seed": 1,
        "advanced": {"add_noise": False},
    }
    samples, out = sampling.run_ksampler(object(), request)
    captured = comfy_sampling_stubs
    assert out is None
    assert torch.equal(samples[:, 0, 0, 0], torch.tensor([2.0, 3.0]))
    assert torch.equal(captured["positive"][0][0][:, 0, 0], torch.tensor([2.0, 3.0]))
    assert torch.equal(captured["positive"][0][1]["pooled_output"][:, 0], torch.tensor([2.0, 3.0]))
    assert captured["positive"][0][1]["broadcast"].shape[0] == 1


def test_custom_guider_slices_conditioning_with_latent(
    monkeypatch, dp_rank1, comfy_sampling_stubs,
):
    captured = comfy_sampling_stubs

    class FakeGuider:
        def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
            captured["custom_latent"] = latent_image
            return latent_image

    def build_guider(_model, spec, _uncond=None):
        captured["guider_spec"] = spec
        return FakeGuider()

    monkeypatch.setattr(sampling, "build_guider", build_guider)
    request = {
        "latent": {"samples": torch.arange(4.0).view(4, 1, 1, 1)},
        "guider": {"kind": "basic", "positive": _cond(4)},
        "sampler_object": object(),
        "sigmas": torch.tensor([1.0, 0.0]),
        "noise_seed": 1,
        "add_noise": False,
    }
    samples, out = sampling.run_custom(object(), request)
    assert out is None
    assert torch.equal(samples[:, 0, 0, 0], torch.tensor([2.0, 3.0]))
    positive = captured["guider_spec"]["positive"]
    assert torch.equal(positive[0][0][:, 0, 0], torch.tensor([2.0, 3.0]))
    assert torch.equal(positive[0][1]["pooled_output"][:, 0], torch.tensor([2.0, 3.0]))


def _leader_ksampler_request(latent_samples):
    return {
        "latent": {"samples": latent_samples},
        "positive": [],
        "negative": [],
        "sampler_name": "euler",
        "scheduler": "normal",
        "steps": 1,
        "cfg": 1.0,
        "noise_seed": 1,
        "advanced": {"add_noise": False},
    }


def _custom_request(latent_samples, sigmas, **extra):
    return {
        "latent": {"samples": latent_samples},
        "guider": {"kind": "basic", "positive": []},
        "sampler_object": object(),
        "sigmas": sigmas,
        "noise_seed": 1,
        "add_noise": False,
        **extra,
    }


def _packed_av(*, requires_grad=False, mixed_dtype=False):
    video = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)[:, :, ::2]
    audio_dtype = torch.float64 if mixed_dtype else torch.float32
    audio = torch.arange(6, dtype=audio_dtype).reshape(1, 3, 2)[:, :, 0]
    if requires_grad:
        video.requires_grad_()
        audio.requires_grad_()
    return NestedTensor((video, audio))


def _storage_id(value):
    return value.untyped_storage().data_ptr()


def test_ksampler_leader_normalizes_plain_output_without_alias(
    monkeypatch, comfy_sampling_stubs,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    latent = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    raw = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)[:, :, ::2]
    raw.requires_grad_()
    comfy_sampling_stubs["sample_result"] = raw

    samples, out = sampling.run_ksampler(
        object(), _leader_ksampler_request(latent)
    )

    assert samples is raw
    normalized = out["samples"]
    assert normalized.shape == raw.shape
    assert normalized.dtype == torch.float64
    assert normalized.device.type == "cpu"
    assert normalized.is_contiguous() and not normalized.requires_grad
    assert torch.equal(normalized, raw)
    assert _storage_id(normalized) != _storage_id(raw)


def test_ksampler_leader_normalizes_packed_output_and_zeroes_each_modality(
    monkeypatch, comfy_sampling_stubs,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    packed = _packed_av(requires_grad=True, mixed_dtype=True)
    comfy_sampling_stubs["sample_result"] = packed

    samples, out = sampling.run_ksampler(
        object(), _leader_ksampler_request(packed)
    )

    assert samples is packed
    normalized = out["samples"]
    assert type(normalized) is NestedTensor
    assert [tuple(part.shape) for part in normalized.unbind()] == [(1, 2, 2), (1, 3)]
    assert [part.dtype for part in normalized.unbind()] == [torch.float32, torch.float64]
    for original, result in zip(packed.unbind(), normalized.unbind(), strict=True):
        assert torch.equal(result, original)
        assert result.device.type == "cpu"
        assert result.is_contiguous() and not result.requires_grad
        assert _storage_id(result) != _storage_id(original)

    noise = comfy_sampling_stubs["noise"]
    assert type(noise) is NestedTensor
    assert [tuple(part.shape) for part in noise.unbind()] == [(1, 2, 2), (1, 3)]
    assert [part.dtype for part in noise.unbind()] == [torch.float32, torch.float64]
    assert all(part.is_contiguous() and not torch.count_nonzero(part) for part in noise.unbind())


def test_custom_leader_reconstructs_packed_denoised_in_modality_order(
    monkeypatch, comfy_sampling_stubs,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    packed = _packed_av(requires_grad=True)
    x0 = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
    observed = {}

    class Model:
        @staticmethod
        def process_latent_out(value):
            observed["x0"] = value
            return value + 10

    class Guider:
        model_patcher = types.SimpleNamespace(model=Model())

        @staticmethod
        def sample(noise, latent_image, sampler, sigmas, **kwargs):
            observed["noise"] = noise
            kwargs["callback"](0, x0, latent_image, 1)
            return packed

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    samples, out = sampling.run_custom(
        object(), _custom_request(packed, torch.tensor([1.0, 0.0]))
    )

    assert samples is packed
    assert observed["x0"].device.type == "cpu"
    assert type(observed["noise"]) is NestedTensor
    assert all(not torch.count_nonzero(part) for part in observed["noise"].unbind())
    normalized = out["samples"]
    denoised = out["denoised"]
    assert type(normalized) is NestedTensor
    assert type(denoised) is NestedTensor
    assert [tuple(part.shape) for part in denoised.unbind()] == [(1, 2, 2), (1, 3)]
    assert torch.equal(denoised.unbind()[0].reshape(-1), torch.arange(10.0, 14.0))
    assert torch.equal(denoised.unbind()[1].reshape(-1), torch.arange(14.0, 17.0))
    for original, primary, preview in zip(
        packed.unbind(), normalized.unbind(), denoised.unbind(), strict=True
    ):
        assert primary.dtype == preview.dtype == original.dtype
        assert primary.device.type == preview.device.type == "cpu"
        assert primary.is_contiguous() and preview.is_contiguous()
        assert not primary.requires_grad and not preview.requires_grad
        assert len({_storage_id(original), _storage_id(primary), _storage_id(preview)}) == 3


def test_custom_leader_rebuilds_the_denoised_output_from_a_nested_x0(
    monkeypatch, comfy_sampling_stubs,
):
    """A packed custom render must not die on its last step (docs/TROUBLESHOOTING.md #64).

    run_custom feeds the nested x0 to process_latent_out as stock does; that
    entry and actor/latent_outputs._denoised_from_nested_x0 say why no packed
    render avoids this path.
    """
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    packed = _packed_av(requires_grad=True)
    x0 = NestedTensor([torch.full((1, 2, 2), 3.0), torch.full((1, 3), 5.0)])
    observed = {}

    class Model:
        @staticmethod
        def process_latent_out(value):
            observed["x0"] = value
            return NestedTensor([part / 0.5 for part in value.unbind()])

    class Guider:
        model_patcher = types.SimpleNamespace(model=Model())

        @staticmethod
        def sample(_noise, latent_image, _sampler, _sigmas, **kwargs):
            kwargs["callback"](0, x0, latent_image, 1)
            return packed

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    _samples, out = sampling.run_custom(
        object(), _custom_request(packed, torch.tensor([1.0, 0.0]))
    )

    # Stock's `x0.cpu()`, done per modality: process_latent_out gets the wrapper,
    # which unbinds to plain CPU tensors.
    assert type(observed["x0"]) is NestedTensor
    assert all(part.device.type == "cpu" for part in observed["x0"].unbind())
    denoised = out["denoised"]
    assert type(denoised) is NestedTensor
    assert torch.equal(denoised.unbind()[0], torch.full((1, 2, 2), 6.0))
    assert torch.equal(denoised.unbind()[1], torch.full((1, 3), 10.0))
    for preview, original in zip(denoised.unbind(), packed.unbind(), strict=True):
        assert preview.dtype == original.dtype
        assert preview.device.type == "cpu" and preview.is_contiguous()
        assert not preview.requires_grad
        assert _storage_id(preview) != _storage_id(original)


@pytest.mark.parametrize("nested_x0", [False, True], ids=["flat-x0", "nested-x0"])
@pytest.mark.parametrize(
    "video_shape",
    [(1, 128, 16, 11, 20), (1, 128, 16, 22, 40)],
    ids=["low-stage", "final-stage"],
)
def test_custom_denoised_reconstruction_matches_ltx_av_shapes(
    video_shape, nested_x0, comfy_sampling_stubs,
):
    """Both x0 shapes rebuild to the LTX audio/video stage shapes; stock SamplerCustom accepts both.

    ComfyUI's callback wrapper reads the modality count, not the family, so LTX
    audio/video, packed the same way as MiniMax H3, gets the nested x0 too.
    """
    audio_shape = (1, 8, 126, 16)
    packed = NestedTensor((
        torch.zeros(video_shape, dtype=torch.float32),
        torch.zeros(audio_shape, dtype=torch.float32),
    ))
    if nested_x0:
        x0 = NestedTensor([torch.ones_like(part) for part in packed.unbind()])
    else:
        packed_width = sum(part.numel() for part in packed.unbind())
        x0 = torch.ones((1, 1, packed_width), dtype=torch.float32)
    model = types.SimpleNamespace(process_latent_out=lambda value: value)
    guider = types.SimpleNamespace(
        model_patcher=types.SimpleNamespace(model=model)
    )

    denoised = sampling._custom_denoised_output(guider, packed, x0)

    assert [tuple(part.shape) for part in denoised.unbind()] == [
        video_shape,
        audio_shape,
    ]
    assert all(part.dtype == torch.float32 for part in denoised.unbind())


@pytest.mark.parametrize(
    ("rebuild", "match"),
    [
        (lambda parts: NestedTensor(parts[:1]), "wrong number of NestedTensor"),
        (
            lambda parts: NestedTensor([parts[0].reshape(1, 4), parts[1]]),
            "malformed NestedTensor modality shapes",
        ),
        (
            lambda parts: NestedTensor([parts[0].double(), parts[1]]),
            "modalities with changed dtype",
        ),
    ],
    ids=["dropped", "reshaped", "recast"],
)
def test_custom_denoised_rejects_a_nested_x0_processed_off_the_sampled_shapes(
    rebuild, match, comfy_sampling_stubs,
):
    packed = _packed_av()
    x0 = NestedTensor([part.clone() for part in packed.unbind()])
    guider = types.SimpleNamespace(
        model_patcher=types.SimpleNamespace(
            model=types.SimpleNamespace(
                process_latent_out=lambda value: rebuild(value.unbind())
            )
        )
    )

    with pytest.raises(RuntimeError, match=match):
        sampling._custom_denoised_output(guider, packed, x0)


@pytest.mark.parametrize("width_delta", [-1, 1])
def test_custom_denoised_rejects_short_or_trailing_packed_x0(
    width_delta, comfy_sampling_stubs,
):
    packed = _packed_av()
    x0 = torch.zeros((1, 1, 7 + width_delta), dtype=torch.float32)
    model = types.SimpleNamespace(process_latent_out=lambda value: value)
    guider = types.SimpleNamespace(
        model_patcher=types.SimpleNamespace(model=model)
    )

    with pytest.raises(RuntimeError, match="malformed packed denoised shape"):
        sampling._custom_denoised_output(guider, packed, x0)


@pytest.mark.parametrize("packed", [False, True], ids=["flat", "packed"])
def test_custom_denoised_rejects_nonfinite_output(packed, comfy_sampling_stubs):
    samples = _packed_av() if packed else torch.zeros(1, 1, 2, 2)
    width = 7 if packed else 4
    x0 = torch.zeros((1, 1, width), dtype=torch.float32)

    def process(value):
        result = value.clone()
        result.reshape(-1)[-1] = float("nan")
        return result

    guider = types.SimpleNamespace(
        model_patcher=types.SimpleNamespace(
            model=types.SimpleNamespace(process_latent_out=process)
        )
    )
    with pytest.raises(ValueError, match="non-finite values"):
        sampling._custom_denoised_output(guider, samples, x0)


def test_custom_denoised_rejects_tuple_unpack_payload(
    monkeypatch, comfy_sampling_stubs,
):
    packed = _packed_av()
    x0 = torch.zeros((1, 1, 7), dtype=torch.float32)
    utils = sys.modules["comfy.utils"]
    unpack = utils.unpack_latents
    monkeypatch.setattr(utils, "unpack_latents", lambda value, shapes: tuple(unpack(value, shapes)))
    guider = types.SimpleNamespace(
        model_patcher=types.SimpleNamespace(
            model=types.SimpleNamespace(process_latent_out=lambda value: value)
        )
    )
    with pytest.raises(RuntimeError, match="direct list payload"):
        sampling._custom_denoised_output(guider, packed, x0)


@pytest.mark.parametrize("kind", ["ksampler", "custom"])
@pytest.mark.parametrize(
    ("bad_output", "match"),
    [
        (NestedTensor(()), "at least one modality"),
        (NestedTensor((torch.zeros(1), "not-a-tensor")), "must all be Tensors"),
        (NestedTensor((NestedTensor((torch.zeros(1),)),)), "must all be Tensors"),
        (NestedTensor((torch.zeros(()),)), "must include a batch dimension"),
        (
            NestedTensor((torch.zeros(1, 2), torch.zeros(2, 2))),
            "matching batches",
        ),
        (NestedTensor((torch.zeros(1),), marker=False), "nested-kind marker"),
        (
            NestedTensor((torch.zeros(1),), direct_list=False),
            "direct list payload",
        ),
        (
            NestedTensor((torch.zeros(1),), unbind_error=RuntimeError("broken")),
            "unbind failed",
        ),
    ],
)
def test_sampler_leader_paths_reject_malformed_nested_output(
    monkeypatch, comfy_sampling_stubs, kind, bad_output, match,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    latent = torch.zeros(1, 1, 1, 1)
    if kind == "ksampler":
        comfy_sampling_stubs["sample_result"] = bad_output
        with pytest.raises((TypeError, ValueError), match=match):
            sampling.run_ksampler(object(), _leader_ksampler_request(latent))
        return

    class Guider:
        @staticmethod
        def sample(*_args, **_kwargs):
            return bad_output

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    with pytest.raises((TypeError, ValueError), match=match):
        sampling.run_custom(
            object(), _custom_request(latent, torch.tensor([1.0, 0.0]))
        )


@pytest.mark.parametrize("kind", ["ksampler", "custom"])
@pytest.mark.parametrize("packed", [False, True], ids=["flat", "packed"])
def test_sampler_leader_paths_reject_nonfinite_output_modality(
    monkeypatch, comfy_sampling_stubs, kind, packed,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    bad_output = (
        NestedTensor((torch.zeros(1, 2), torch.tensor([[float("inf")]])))
        if packed
        else torch.tensor([[[float("nan")]]])
    )
    latent = torch.zeros(1, 1, 1, 1)
    if kind == "ksampler":
        comfy_sampling_stubs["sample_result"] = bad_output
        with pytest.raises(ValueError, match="non-finite values"):
            sampling.run_ksampler(object(), _leader_ksampler_request(latent))
        return

    class Guider:
        @staticmethod
        def sample(*_args, **_kwargs):
            return bad_output

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    with pytest.raises(ValueError, match="non-finite values"):
        sampling.run_custom(
            object(), _custom_request(latent, torch.tensor([1.0, 0.0]))
        )


@pytest.mark.parametrize("kind", ["ksampler", "custom"])
@pytest.mark.parametrize("packed", [False, True], ids=["flat", "packed"])
def test_sampler_leader_paths_reject_tensor_subclass_outputs(
    monkeypatch, comfy_sampling_stubs, kind, packed,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    lying = torch.zeros(1, 2).as_subclass(LyingTensor)
    bad_output = NestedTensor((lying, torch.zeros(1, 3))) if packed else lying
    latent = torch.zeros(1, 1, 1, 1)

    if kind == "ksampler":
        comfy_sampling_stubs["sample_result"] = bad_output
        with pytest.raises(TypeError, match=r"subclasses|exact Tensors"):
            sampling.run_ksampler(object(), _leader_ksampler_request(latent))
        return

    class Guider:
        @staticmethod
        def sample(*_args, **_kwargs):
            return bad_output

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    with pytest.raises(TypeError, match=r"subclasses|exact Tensors"):
        sampling.run_custom(
            object(), _custom_request(latent, torch.tensor([1.0, 0.0]))
        )


@pytest.mark.parametrize("kind", ["ksampler", "custom"])
@pytest.mark.parametrize(
    "case",
    ["same-name-spoof", "subclass", "marker", "tuple", "flat-nan", "packed-inf"],
)
def test_sampler_nonleader_rejects_malformed_or_nonfinite_result(
    monkeypatch, comfy_sampling_stubs, kind, case,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: False)
    if case == "same-name-spoof":
        Spoof = type(
            "NestedTensor",
            (),
            {
                "is_nested": True,
                "unbind": lambda self: [torch.zeros(1, 2)],
            },
        )
        bad_output = Spoof()
    elif case == "subclass":
        class NestedSubclass(NestedTensor):
            pass

        bad_output = NestedSubclass((torch.zeros(1, 2),))
    elif case == "marker":
        bad_output = NestedTensor((torch.zeros(1, 2),), marker=False)
    elif case == "tuple":
        bad_output = NestedTensor((torch.zeros(1, 2),), direct_list=False)
    elif case == "flat-nan":
        bad_output = torch.tensor([[[float("nan")]]])
    else:
        bad_output = NestedTensor((
            torch.zeros(1, 2),
            torch.tensor([[float("inf")]]),
        ))

    latent = torch.zeros(1, 1, 1, 1)
    if kind == "ksampler":
        comfy_sampling_stubs["sample_result"] = bad_output
        with pytest.raises((TypeError, ValueError)):
            sampling.run_ksampler(object(), _leader_ksampler_request(latent))
        return

    class Guider:
        @staticmethod
        def sample(*_args, **_kwargs):
            return bad_output

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    with pytest.raises((TypeError, ValueError)):
        sampling.run_custom(
            object(), _custom_request(latent, torch.tensor([1.0, 0.0]))
        )


@pytest.mark.parametrize("kind", ["ksampler", "custom"])
def test_sampler_nonleader_valid_result_is_not_cpu_normalized(
    monkeypatch, comfy_sampling_stubs, kind,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: False)
    monkeypatch.setattr(
        sampling.latent_outputs,
        "normalize_tensor_tree",
        lambda *_args, **_kwargs: pytest.fail("non-leader result was CPU-normalized"),
    )
    latent = torch.zeros(1, 1, 1, 1)
    if kind == "ksampler":
        samples, out = sampling.run_ksampler(
            object(),
            {
                **_leader_ksampler_request(latent),
                "advanced": {"add_noise": True},
            },
        )
    else:
        class Guider:
            @staticmethod
            def sample(_noise, latent_image, *_args, **_kwargs):
                return latent_image

        monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
        samples, out = sampling.run_custom(
            object(),
            _custom_request(
                latent,
                torch.tensor([1.0, 0.0]),
                add_noise=True,
            ),
        )
    assert samples is latent
    assert out is None


def test_normalizer_rejects_non_comfy_wrapper_even_with_unbind():
    class PackedImpostor:
        @staticmethod
        def unbind():
            return (torch.zeros(1, 2),)

    with pytest.raises(TypeError, match="exact Comfy NestedTensor"):
        sampling._normalize_sample_output(PackedImpostor())


def test_normalizer_rejects_same_named_spoof_and_subclass(comfy_sampling_stubs):
    Spoof = type(
        "NestedTensor",
        (),
        {"unbind": lambda self: (torch.zeros(1, 2),)},
    )

    class NestedSubclass(NestedTensor):
        pass

    cases = (
        (Spoof(), "exact Comfy NestedTensor"),
        (NestedSubclass((torch.zeros(1, 2),)), "subclasses"),
    )
    for value, match in cases:
        with pytest.raises(TypeError, match=match):
            sampling._normalize_sample_output(value)


def test_normalizer_requires_class_level_direct_unbind(
    monkeypatch, comfy_sampling_stubs,
):
    value = NestedTensor((torch.zeros(1, 2),))
    monkeypatch.setattr(NestedTensor, "unbind", None)
    with pytest.raises(TypeError, match="no direct unbind method"):
        sampling._normalize_sample_output(value)


def test_custom_packed_output_fails_closed_without_nonzero_x0(
    monkeypatch, comfy_sampling_stubs,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    packed = _packed_av()

    class Guider:
        @staticmethod
        def sample(*_args, **_kwargs):
            return packed

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    with pytest.raises(RuntimeError, match="without a denoised x0"):
        sampling.run_custom(
            object(), _custom_request(packed, torch.tensor([1.0, 0.0]))
        )


def test_custom_packed_output_fails_closed_for_malformed_nonzero_x0(
    monkeypatch, comfy_sampling_stubs,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    packed = _packed_av()

    class Guider:
        model_patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(process_latent_out=lambda value: value)
        )

        @staticmethod
        def sample(_noise, latent_image, _sampler, _sigmas, **kwargs):
            kwargs["callback"](0, torch.zeros(1, 1, 6), latent_image, 1)
            return packed

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    with pytest.raises(RuntimeError, match="could not reconstruct packed denoised"):
        sampling.run_custom(
            object(), _custom_request(packed, torch.tensor([1.0, 0.0]))
        )


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_custom_zero_step_returns_input_and_two_independent_values(
    monkeypatch, comfy_sampling_stubs, packed,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    latent = _packed_av() if packed else torch.arange(4.0).reshape(1, 1, 2, 2)

    class Guider:
        @staticmethod
        def sample(_noise, latent_image, _sampler, _sigmas, **_kwargs):
            return latent_image

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    returned, out = sampling.run_custom(
        object(), _custom_request(latent, torch.tensor([0.0]))
    )
    primary = out["samples"]
    denoised = out["denoised"]
    original_parts = latent.unbind() if packed else (latent,)
    returned_parts = returned.unbind() if packed else (returned,)
    primary_parts = primary.unbind() if packed else (primary,)
    denoised_parts = denoised.unbind() if packed else (denoised,)
    for original, rank_value, result, preview in zip(
        original_parts, returned_parts, primary_parts, denoised_parts, strict=True
    ):
        assert torch.equal(rank_value, original)
        assert torch.equal(result, original)
        assert torch.equal(preview, original)
        assert len({
            _storage_id(original),
            _storage_id(rank_value),
            _storage_id(result),
            _storage_id(preview),
        }) == 4


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_nonzero_singleton_short_circuits_noise_guider_and_progress(
    monkeypatch, comfy_sampling_stubs, packed,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    latent = _packed_av() if packed else torch.arange(4.0).reshape(1, 1, 2, 2)

    class PoisonNoise:
        @staticmethod
        def generate_noise(_latent):
            pytest.fail("singleton schedule generated noise")

    monkeypatch.setattr(
        sampling,
        "build_guider",
        lambda *_args: pytest.fail("singleton schedule built a guider"),
    )
    progress = types.SimpleNamespace(
        send=lambda _message: pytest.fail("singleton schedule emitted progress")
    )
    returned, out = sampling.run_custom(
        object(),
        _custom_request(
            latent,
            torch.tensor([1.0]),
            add_noise=True,
            noise_object=PoisonNoise(),
        ),
        progress_port=progress,
    )

    primary = out["samples"]
    denoised = out["denoised"]
    original_parts = latent.unbind() if packed else (latent,)
    returned_parts = returned.unbind() if packed else (returned,)
    primary_parts = primary.unbind() if packed else (primary,)
    denoised_parts = denoised.unbind() if packed else (denoised,)
    for original, rank_value, result, preview in zip(
        original_parts, returned_parts, primary_parts, denoised_parts, strict=True
    ):
        assert torch.equal(rank_value, original)
        assert torch.equal(result, original)
        assert torch.equal(preview, original)
        assert len({
            _storage_id(original),
            _storage_id(rank_value),
            _storage_id(result),
            _storage_id(preview),
        }) == 4


@pytest.mark.parametrize(
    "sigmas",
    [
        None,
        torch.tensor([]),
        torch.tensor(0.0),
        torch.zeros(1, 1),
        torch.tensor([0]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
    ],
    ids=["not-tensor", "empty", "scalar", "matrix", "integral", "nan", "inf"],
)
def test_custom_rejects_malformed_schedule_before_guider(
    monkeypatch, comfy_sampling_stubs, sigmas,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(
        sampling,
        "build_guider",
        lambda *_args: pytest.fail("malformed schedule reached guider construction"),
    )
    with pytest.raises(ValueError, match="finite, non-empty, 1-D floating Tensor"):
        sampling.run_custom(
            object(), {"latent": {"samples": _packed_av()}, "sigmas": sigmas}
        )


def test_custom_cancellation_still_finishes_progress_channel(
    monkeypatch, comfy_sampling_stubs,
):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    latent = torch.zeros(1, 1, 1, 1)
    messages = []
    port = types.SimpleNamespace(send=messages.append)
    cancelled = types.SimpleNamespace(is_set=lambda: True)

    class Guider:
        @staticmethod
        def sample(_noise, latent_image, _sampler, _sigmas, **kwargs):
            kwargs["callback"](0, latent_image, latent_image, 1)
            return latent_image

    monkeypatch.setattr(sampling, "build_guider", lambda *_args: Guider())
    with pytest.raises(sampling.RenderCancelledError):
        sampling.run_custom(
            object(),
            _custom_request(latent, torch.tensor([1.0, 0.0])),
            progress_port=port,
            cancel_event=cancelled,
        )
    assert messages == [{"done": True}]


def test_condition_batch_cycles_globally_before_dp_slice(dp_rank1, comfy_sampling_stubs):
    out = sampling.split_conditioning_for_dp(_cond(3), 4, "positive")
    # Stock global repeat is [0, 1, 2, 0]; rank 1 owns rows [2, 0].
    assert torch.equal(out[0][0][:, 0, 0], torch.tensor([2.0, 0.0]))
    assert torch.equal(out[0][1]["pooled_output"][:, 0], torch.tensor([2.0, 0.0]))


class _FakeControl:
    def __init__(self, hint, previous=None):
        self.cond_hint_original = hint
        self.previous_controlnet = previous
        self.extra_concat_orig = [hint + 10]
        self.extra_args = {"guide": hint + 20}
        self.cond_hint = hint + 30
        self.extra_concat = [hint + 40]

    def copy(self):
        return _FakeControl(self.cond_hint_original, self.previous_controlnet)


def test_control_chain_is_cloned_and_sliced(dp_rank1):
    hint = torch.arange(4.0).view(4, 1, 1, 1)
    control = _FakeControl(hint, _FakeControl(torch.ones(1, 1, 1, 1)))
    cond = [[torch.ones(1, 2, 3), {"control": control}]]
    out = sampling.split_conditioning_for_dp(cond, 4, "positive")
    local = out[0][1]["control"]
    assert local is not control
    assert torch.equal(local.cond_hint_original[:, 0, 0, 0], torch.tensor([2.0, 3.0]))
    assert torch.equal(local.extra_concat_orig[0][:, 0, 0, 0], torch.tensor([12.0, 13.0]))
    assert torch.equal(local.extra_args["guide"][:, 0, 0, 0], torch.tensor([22.0, 23.0]))
    assert local.previous_controlnet.cond_hint_original.shape[0] == 1
    assert local.cond_hint is None and local.extra_concat is None
    assert control.cond_hint_original.shape[0] == 4


def test_opaque_control_rejects(dp_rank1):
    cond = [[torch.ones(1, 2, 3), {"control": object()}]]
    with pytest.raises(RuntimeError, match="opaque control object"):
        sampling.split_conditioning_for_dp(cond, 4, "positive")


# Anima-shaped conditioning: a batch-1 cross-attention tensor beside two T5
# sequences of shape (num_t5_tokens,), with no batch axis, in the extras dict.
_ANIMA_EXEMPT_KEYS = frozenset({"t5xxl_ids", "t5xxl_weights"})


def _anima_cond(t5_len: int = 7):
    cross_attn = torch.arange(6.0).view(1, 3, 2)
    return [[cross_attn, {
        "t5xxl_ids": torch.arange(t5_len, dtype=torch.int64),
        "t5xxl_weights": torch.ones(t5_len),
    }]]


def test_exempt_keys_ride_whole_on_every_dp_rank(monkeypatch, comfy_sampling_stubs):
    cond = _anima_cond(7)
    for rank in (0, 1):
        monkeypatch.setattr(sampling, "_dp_info", lambda rank=rank: (rank, 2))
        out = sampling.split_conditioning_for_dp(
            cond, latent_batch=2, name="positive", exempt_keys=_ANIMA_EXEMPT_KEYS
        )
        assert torch.equal(out[0][1]["t5xxl_ids"], cond[0][1]["t5xxl_ids"])
        assert torch.equal(out[0][1]["t5xxl_weights"], cond[0][1]["t5xxl_weights"])
        # The batch-1 cross-attention tensor stays whole through the ordinary slice path.
        assert torch.equal(out[0][0], cond[0][0])


def test_without_exempt_keys_reproduces_the_pre_fix_mangling(monkeypatch, comfy_sampling_stubs):
    # Negative control: without exempt keys the slicer treats the T5 tokens as a
    # batch and cuts them, so the test above exercises the exemption, not a no-op.
    cond = _anima_cond(5)
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 2))
    out = sampling.split_conditioning_for_dp(cond, 4, "positive")
    assert out[0][1]["t5xxl_ids"].numel() != 5


def test_custom_guider_threads_exempt_keys_through_split_guider_for_dp(
    monkeypatch, comfy_sampling_stubs,
):
    cond = _anima_cond(7)
    spec = {"kind": "basic", "positive": cond}
    for rank in (0, 1):
        monkeypatch.setattr(sampling, "_dp_info", lambda rank=rank: (rank, 2))
        out = sampling.split_guider_for_dp(
            spec, latent_batch=2, exempt_keys=frozenset({"t5xxl_ids"})
        )
        assert torch.equal(out["positive"][0][1]["t5xxl_ids"], cond[0][1]["t5xxl_ids"])


def test_ksampler_threads_dp_cond_exempt_keys_end_to_end(dp_rank1, comfy_sampling_stubs):
    cond = _anima_cond(7)
    request = {
        "latent": {"samples": torch.arange(4.0).view(4, 1, 1, 1)},
        "positive": cond,
        "negative": _cond(4),
        "sampler_name": "euler",
        "scheduler": "normal",
        "steps": 1,
        "cfg": 1.0,
        "noise_seed": 1,
        "advanced": {"add_noise": False},
    }
    sampling.run_ksampler(
        object(), request, dp_cond_exempt_keys=frozenset({"t5xxl_ids"})
    )
    captured = comfy_sampling_stubs
    assert torch.equal(captured["positive"][0][1]["t5xxl_ids"], cond[0][1]["t5xxl_ids"])


def test_custom_threads_dp_cond_exempt_keys_end_to_end(
    monkeypatch, dp_rank1, comfy_sampling_stubs,
):
    captured = comfy_sampling_stubs
    cond = _anima_cond(7)

    class FakeGuider:
        def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
            return latent_image

    def build_guider(_model, spec, _uncond=None):
        captured["guider_spec"] = spec
        return FakeGuider()

    monkeypatch.setattr(sampling, "build_guider", build_guider)
    request = {
        "latent": {"samples": torch.arange(4.0).view(4, 1, 1, 1)},
        "guider": {"kind": "cfg", "positive": cond, "negative": _cond(4), "cfg": 1.0},
        "sampler_object": object(),
        "sigmas": torch.tensor([1.0, 0.0]),
        "noise_seed": 1,
        "add_noise": False,
    }
    sampling.run_custom(
        object(), request, dp_cond_exempt_keys=frozenset({"t5xxl_ids"})
    )
    positive = captured["guider_spec"]["positive"]
    assert torch.equal(positive[0][1]["t5xxl_ids"], cond[0][1]["t5xxl_ids"])


def test_dp_cond_exempt_keys_registry_invariant():
    # Only Anima declares exempt keys; every other adapter keeps the empty default.
    from dgx_monarch.adapters import ADAPTERS
    from dgx_monarch.adapters.anima import AnimaAdapter

    for adapter in ADAPTERS:
        keys = adapter.dp_cond_exempt_keys
        assert isinstance(keys, frozenset)
        assert all(isinstance(key, str) for key in keys)
        if isinstance(adapter, AnimaAdapter):
            assert keys == _ANIMA_EXEMPT_KEYS
        else:
            assert keys == frozenset()
