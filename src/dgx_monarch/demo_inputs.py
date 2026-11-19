"""Place the public templates' demo inputs in ComfyUI's input folder.

Each Load Image, Load Image Mask, Load Video and Load Audio node in a shipped
template names a file. ComfyUI ships only ``input/example.png``; the pack
ships every other name in ``example_workflows/input`` (tools/gen_demo_inputs.py
draws them). At pack import this copies each one the input folder lacks, so a
template opens and runs on a fresh install. It never creates the folder, and
never overwrites, replaces or follows a name already there, a file, a link or
a folder. It removes a copy it could not finish, logs every name it placed,
and never raises: a folder it cannot write gets one WARNING and the templates
name their files as before. Set ``DGXM_DEMO_INPUTS=off`` to place nothing.
"""
from __future__ import annotations

import os
from pathlib import Path

from .log import get_logger

log = get_logger(__name__)

SOURCE = Path(__file__).resolve().parents[2] / "example_workflows" / "input"
OFF_VALUES = frozenset({"0", "off", "false", "no"})


def shipped() -> list[Path]:
    """The demo inputs this checkout ships, sorted by name.

    A link in the folder is left out: its target could be any file on the box,
    and ComfyUI serves its input folder over ``/view``.
    """
    if not SOURCE.is_dir():
        return []
    return sorted(path for path in SOURCE.iterdir()
                  if path.is_file() and not path.is_symlink() and not path.name.startswith("."))


def _comfy_input_dir() -> Path | None:
    try:
        import folder_paths  # ComfyUI's own module; absent outside a ComfyUI process
    except ImportError:
        return None
    return Path(folder_paths.get_input_directory())


def place_demo_inputs(input_dir: Path | None = None) -> list[str]:
    """Copy every shipped demo input ``input_dir`` lacks; return the names placed."""
    if os.environ.get("DGXM_DEMO_INPUTS", "").strip().lower() in OFF_VALUES:
        return []
    placed: list[str] = []
    target = input_dir
    try:  # nothing here may raise into ComfyUI's startup
        if target is None:
            target = _comfy_input_dir()
        if target is None:
            return []
        if not target.is_dir():  # ComfyUI makes its own input folder; a share not yet mounted stays empty
            raise NotADirectoryError("it is not a folder, and the pack never creates one")
        for source in shipped():
            destination = target / source.name
            if os.path.lexists(destination):
                continue
            data = source.read_bytes()
            try:
                handle = destination.open("xb")  # exclusive: a racing writer wins
            except FileExistsError:
                continue
            try:
                with handle:  # the close flushes the last bytes, so it sits inside the guard
                    handle.write(data)
            except BaseException:
                destination.unlink(missing_ok=True)  # never leave a truncated copy
                raise
            placed.append(source.name)
    except Exception as exc:
        where = target if target is not None else "ComfyUI's input folder"
        if placed:
            log.warning("demo inputs: placed %d template demo file(s) in %s (%s), then stopped (%s); "
                        "the templates still name the rest, so supply your own",
                        len(placed), where, ", ".join(placed), exc)
        else:
            log.warning("demo inputs: could not place the templates' demo files in %s (%s); "
                        "the templates still name them, so supply your own", where, exc)
        return placed
    if placed:
        log.info("demo inputs: placed %d template demo file(s) in %s that it lacked "
                 "(files already there are never replaced): %s", len(placed), target, ", ".join(placed))
    return placed
