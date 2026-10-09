"""Compare sweep output pixels and audio envelopes.

Compare decoded PNG arrays: ComfyUI embeds prompt graphs in PNG metadata, so
file hashes can differ even when pixels match. Compare every video frame in
filename order. Unequal frame counts produce an error and an unproven
comparison (CHECK), not a fidelity finding.
"""
from __future__ import annotations

from pathlib import Path

ENVELOPE_WINDOW = 512  # samples per envelope point


def frames_for(output_dir: Path, prefix: str) -> list[Path]:
    """Every PNG a leg saved, in the order comfy numbered them."""
    return sorted(Path(output_dir).glob(f"{prefix}_*.png"))


def audio_for(output_dir: Path, prefix: str) -> list[Path]:
    return sorted(path for suffix in ("flac", "wav", "mp3", "opus")
                  for path in Path(output_dir).glob(f"{prefix}_*.{suffix}"))


def compare_frames(candidate: list[Path], reference: list[Path]) -> dict:
    """Frame-set difference as NRMS and max_abs over int16 pixel differences."""
    import numpy as np
    from PIL import Image

    if not candidate or not reference:
        return {"error": f"missing frames ({len(candidate)} vs {len(reference)})"}
    if len(candidate) != len(reference):
        return {"error": f"frame count {len(candidate)} != reference {len(reference)}",
                "frames": len(candidate), "reference_frames": len(reference)}
    total = reference_energy = 0.0
    samples = max_abs = 0
    for left, right in zip(candidate, reference, strict=True):
        first = np.asarray(Image.open(left), dtype=np.int16)
        second = np.asarray(Image.open(right), dtype=np.int16)
        if first.shape != second.shape:
            return {"error": f"shape {first.shape} != reference {second.shape}"}
        diff = (first - second).astype(np.int32)
        total += float((diff.astype(np.float64) ** 2).sum())
        reference_energy += float((second.astype(np.float64) ** 2).sum())
        samples += diff.size
        max_abs = max(max_abs, int(np.abs(diff).max()))
    rms = (total / samples) ** 0.5
    reference_rms = (reference_energy / samples) ** 0.5 or 1.0
    return {"frames": len(candidate), "nrms": round(rms / reference_rms, 6),
            "max_abs": max_abs, "identical": max_abs == 0,
            "candidate_file": candidate[0].name, "reference_file": reference[0].name}


def _envelope(path: Path):
    import numpy as np
    import soundfile

    data, _rate = soundfile.read(str(path), always_2d=True)
    mono = np.abs(np.asarray(data, dtype=np.float64).mean(axis=1))
    usable = (mono.size // ENVELOPE_WINDOW) * ENVELOPE_WINDOW
    return mono if usable == 0 else mono[:usable].reshape(-1, ENVELOPE_WINDOW).mean(axis=1)


def audio_compare_available() -> bool:
    """Whether this interpreter can score audio legs (soundfile is a dev extra)."""
    try:
        import soundfile  # noqa: F401
    except ImportError:
        return False
    return True


def compare_audio(candidate: list[Path], reference: list[Path]) -> dict:
    """Envelope correlation between the first audio file of each leg."""
    if not candidate or not reference:
        return {"error": f"missing audio ({len(candidate)} vs {len(reference)})"}
    try:
        import numpy as np

        left, right = _envelope(candidate[0]), _envelope(reference[0])
    except ImportError:
        return {"error": "soundfile is not installed on this rig"}
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    span = min(left.size, right.size)
    left, right = left[:span], right[:span]
    if span < 2 or left.std() == 0 or right.std() == 0:
        return {"error": "audio too short or too quiet to correlate"}
    return {"envelope_correlation": round(float(np.corrcoef(left, right)[0, 1]), 6),
            "envelope_points": int(span),
            "candidate_file": candidate[0].name, "reference_file": reference[0].name}


def compare_outputs(output_dir: Path, candidate: str, reference: str,
                    audio: bool = False) -> dict:
    result = compare_frames(frames_for(output_dir, candidate), frames_for(output_dir, reference))
    if audio:
        result["audio"] = compare_audio(audio_for(output_dir, candidate),
                                        audio_for(output_dir, reference))
    return result
