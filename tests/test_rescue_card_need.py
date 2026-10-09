"""The slab rescue card must print the priced need, not the file size.

The first test rebuilds the case fixed on 2026-09-28: a qwen_image 20.4 GB fp8
checkpoint refused with 39.2 GiB available while the card read "needs 19.0 GiB",
which looks like a refusal that should have fit.
"""
from dgx_monarch.actor.rescue_offer import build_descriptor
from dgx_monarch.capacity_fit import StockFit


def test_rescue_card_states_the_priced_need_beside_the_file_size():
    fit = StockFit(fits=False, applies=True, size_bytes=20_430_679_144, avail_bytes=42_089_672_704)
    card = build_descriptor(
        unet_name="qwen_image_2512_fp8_e4m3fn.safetensors", path="/nonexistent/model.safetensors",
        fit=fit, model_options={}, file_identity=lambda _path: "identity", family_hint=None,
        context=lambda *_a: {}, host=lambda: "host",
    )
    reason = card.human_reason
    assert fit.required_gib > fit.avail_gib > fit.size_gib
    assert f"needs {fit.required_gib} GiB" in reason
    assert f"a {fit.size_gib} GiB file" in reason
    assert f"only {fit.avail_gib} GiB is available" in reason


def test_sweep_warns_when_audio_cells_cannot_be_compared(monkeypatch):
    """Without soundfile an audio cell that reaches a reference compare scores CHECK at best.

    In the stable-source sweep, 11 AV cells scored CHECK that way (docs/VALIDATION.md). The runner's
    session-start warning reads audio_compare_available, which must report False, and compare_audio
    must return an error.
    """
    import builtins

    from benchmark.sweep import compare

    real_import = builtins.__import__

    def no_soundfile(name, *args, **kwargs):
        if name == "soundfile":
            raise ImportError("no soundfile")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_soundfile)
    assert compare.audio_compare_available() is False
    assert compare.compare_audio([__file__], [__file__]) == {"error": "soundfile is not installed on this rig"}
