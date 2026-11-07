"""Compare sweep output pixels and audio envelopes.

Compare decoded PNG arrays: ComfyUI embeds prompt graphs in PNG metadata, so
file hashes can differ even when pixels match. Compare every video frame in
filename order. Unequal frame counts produce an error and an unproven
comparison (CHECK), not a fidelity finding. Audio follows the same rule: every
file is read, and a different file count, sample rate, channel count or length
is an error, not a correlation over what the two legs share.
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


def _read_audio(path: Path):
    """One file's samples as frames by channels, and the shape the compare keeps."""
    import numpy as np
    import soundfile

    data, rate = soundfile.read(str(path), always_2d=True)
    samples = np.asarray(data, dtype=np.float64)
    frames, channels = samples.shape
    return samples, {"file": path.name, "rate": int(rate), "channels": int(channels),
                     "frames": int(frames), "seconds": round(frames / rate, 3) if rate else 0.0}


def _envelope(channel):
    """Mean absolute amplitude over each ENVELOPE_WINDOW samples of one channel."""
    magnitude = abs(channel)
    usable = (magnitude.size // ENVELOPE_WINDOW) * ENVELOPE_WINDOW
    return magnitude if usable == 0 else magnitude[:usable].reshape(-1, ENVELOPE_WINDOW).mean(axis=1)


def _shape_mismatch(mine: dict, theirs: dict) -> str:
    """The first shape two paired files differ in, or ""."""
    for key, name in (("rate", "sample rate"), ("channels", "channels"), ("frames", "samples")):
        if mine[key] != theirs[key]:
            return (f"{mine['file']} {name} {mine[key]} != reference "
                    f"{theirs['file']} {theirs[key]}")
    return ""


def audio_compare_available() -> bool:
    """Whether this interpreter can score audio legs (soundfile is a dev extra)."""
    try:
        import soundfile  # noqa: F401
    except ImportError:
        return False
    return True


def compare_audio(candidate: list[Path], reference: list[Path]) -> dict:
    """Every audio file of both legs: its shape, and envelope correlation per channel.

    Files pair in the order comfy numbered them. Each keeps its rate, channel
    count and sample frames. A different file count, or a pair whose rate,
    channels or length differ, is an error: a shorter exact prefix, the same
    samples at another rate or a channel mix each correlate at 1.0 against
    audio that differs. On equal shapes each channel of each pair is
    correlated on its own and the lowest is the number; ``identical`` says
    every sample matched. A channel constant in both legs and equal adds no
    correlation; one constant in only one leg, or no channel to correlate at
    all, is "too short or too quiet". The correlation is not a calibrated
    fidelity threshold (docs/SWEEP.md section 5), so nothing scores on it.
    """
    if not candidate or not reference:
        return {"error": f"missing audio ({len(candidate)} vs {len(reference)})"}
    try:
        import numpy as np

        mine = [_read_audio(path) for path in candidate]
        theirs = [_read_audio(path) for path in reference]
    except ImportError:
        return {"error": "soundfile is not installed on this rig"}
    except (OSError, ValueError, RuntimeError) as exc:
        # soundfile raises LibsndfileError, a RuntimeError, on a file it cannot decode.
        return {"error": f"{type(exc).__name__}: {exc}"}
    result: dict = {"files": len(mine), "reference_files": len(theirs),
                    "candidate": [shape for _, shape in mine],
                    "reference": [shape for _, shape in theirs]}
    if len(mine) != len(theirs):
        return {**result, "error": f"audio file count {len(mine)} != reference {len(theirs)}"}
    pairs = [(left, right) for (left, _), (right, _) in zip(mine, theirs, strict=True)]
    for (_, left), (_, right) in zip(mine, theirs, strict=True):
        if why := _shape_mismatch(left, right):
            return {**result, "error": why}
    result["identical"] = all(np.array_equal(left, right) for left, right in pairs)
    correlations: list[float] = []
    points: list[int] = []
    for left, right in pairs:
        for channel in range(left.shape[1]):
            mine_column, their_column = left[:, channel], right[:, channel]
            flat = mine_column.size == 0 or mine_column.min() == mine_column.max()
            if flat and np.array_equal(mine_column, their_column):
                continue
            first, second = _envelope(mine_column), _envelope(their_column)
            if first.size < 2 or first.std() == 0 or second.std() == 0:
                return {**result, "error": "audio too short or too quiet to correlate"}
            correlations.append(float(np.corrcoef(first, second)[0, 1]))
            points.append(int(first.size))
    if not correlations:
        return {**result, "error": "audio too short or too quiet to correlate"}
    return {**result, "envelope_correlation": round(min(correlations), 6),
            "envelope_points": min(points), "channels_correlated": len(correlations)}


def audio_line(audio: dict) -> str:
    """One line for an audio compare, for the record's notes and the report."""
    if audio.get("error"):
        return f"not compared: {audio['error']}"
    if "candidate" not in audio:  # written before the compare kept each file's shape
        return (f"envelope correlation {audio.get('envelope_correlation')} "
                f"({audio.get('candidate_file')}); shape not recorded")
    first = (audio.get("candidate") or [{}])[0]
    shape = (f"{audio.get('files')} file(s), the first {first.get('channels')} channel(s) "
             f"at {first.get('rate')} Hz for {first.get('seconds')} s")
    if audio.get("identical"):
        return f"identical samples; {shape}"
    return (f"lowest channel envelope correlation {audio.get('envelope_correlation')} over "
            f"{audio.get('channels_correlated')} channel(s); {shape}")


def compare_outputs(output_dir: Path, candidate: str, reference: str,
                    audio: bool = False) -> dict:
    result = compare_frames(frames_for(output_dir, candidate), frames_for(output_dir, reference))
    if audio:
        result["audio"] = compare_audio(audio_for(output_dir, candidate),
                                        audio_for(output_dir, reference))
    return result
