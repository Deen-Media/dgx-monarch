"""The xFuser builds this release runs on, and the words for any other.

Two builds run: stock xFuser 0.8.0 from the package index, and the patched
``0.7.0+dgxm.npuimport1`` that an install of 1.4.0 built. The package requirement
cannot say that by itself: PEP 440 ignores a local label when it matches, so
no single specifier refuses stock 0.7.0 while it admits the patched build. The
requirement (``REQUIREMENT``, repeated in pyproject.toml and requirements.txt)
is therefore the narrowest range that admits both, and this module holds the
exact list. Doctor checks it on the driver and on every Worker host, and
Worker setup refuses any other build before it imports xFuser.

Stock 0.7.0 imports an optional NPU ring helper while it builds CUDA
attention, and released yunchang lacks that helper, so a Worker on it cannot
set up. Stock 0.8.0 guards that import, so the patched build is no longer
needed; it still runs, and doctor names the one-line move to 0.8.0.

Leaf module: no dgx_monarch import. Versions compare by PEP 440 through
``packaging``, which transformers and accelerate bring wherever xFuser is
installed, or pip's own copy of it; with neither, as exact strings.
"""
from __future__ import annotations

from typing import Any

STOCK = "0.8.0"
PATCHED = "0.7.0+dgxm.npuimport1"
ALLOWED = (STOCK, PATCHED)
# The dependency string pyproject.toml and requirements.txt carry (the tests pin all three equal).
REQUIREMENT = "xfuser>=0.7.0,<=0.8.0"
MISSING = "missing"
ENTRY = "docs/TROUBLESHOOTING.md #104"

OK, WARN, FAIL = "ok", "warn", "fail"


# The build this actor process checked at setup and then imported, once it has.
_PINNED: str | None = None


def pin_for_process(version: str) -> None:
    """Hold this process to the build its setup checked; a later pip install cannot move it."""
    global _PINNED
    _PINNED = version


def installed_version() -> str:
    """The xFuser distribution version this process runs, or ``MISSING``.

    An actor answers with the build its setup checked before it imported
    xFuser (``pin_for_process``): a pip install while it runs changes the disk,
    not the module it holds. Any other process reads package metadata on every
    call, so it neither imports xFuser nor registers its torch operators, and
    after a box moves to 0.8.0 the driver's Gate identity follows its disk
    without a ComfyUI restart, as its freshly started actors do.
    """
    if _PINNED is not None:
        return _PINNED
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        return str(distribution("xfuser").version)
    except PackageNotFoundError:
        return MISSING


RESTART = ("then restart that box's Worker (in local mode, where the actors run inside ComfyUI's "
           "own mesh, reset the attached mesh or restart ComfyUI)")
# Pip still reads the requirement of release 1.4.0, `xfuser==0.7.0+dgxm.npuimport1`, from the
# driver's editable install metadata a verified update did not rewrite, and names it as a conflict.
METADATA = ("After a verified update from 1.4.0, first refresh the driver's dgx-monarch metadata "
            "(docs/INSTALL.md, Upgrading from 1.4.0, item 30), or pip reports the move as a conflict.")


def _pep440() -> Any:
    """``(version, specifiers)`` from ``packaging`` or pip's copy, or None when neither imports."""
    try:
        from packaging import specifiers, version
    except ImportError:
        try:  # pip's own copy, which the update checks also use
            from pip._vendor.packaging import specifiers, version  # type: ignore[no-redef]
        except ImportError:
            return None
    return version, specifiers


def _parse(text: str) -> Any:
    """The PEP 440 version ``text`` names, or None when it names none (``MISSING`` among them)."""
    modules = _pep440()
    if modules is None:
        return None
    version, _ = modules
    try:
        return version.Version(text)
    except version.InvalidVersion:
        return None


def _is(text: str, build: str) -> bool:
    """Whether ``text`` names ``build``: equal by PEP 440, local label included.

    With no PEP 440 parser importable the comparison is exact strings, so no
    other spelling is ever admitted.
    """
    if _pep440() is None:
        return text == build
    found = _parse(text)
    return found is not None and found == _parse(build)


def is_stock(text: str) -> bool:
    return _is(text, STOCK)


def is_allowed(text: str) -> bool:
    """Whether doctor and Worker setup admit ``text``: PEP 440 equal to a build in ``ALLOWED``.

    Only admission and the move line compare this way. The identities that name
    a build (doctor's match row, the Gate runtime identity, the actor's pin)
    compare exact strings on purpose, so a Gate grant never equates two
    spellings: boxes that spell one version two ways refuse a risky render.
    """
    return any(_is(text, build) for build in ALLOWED)


def move_command(python: str, installed: str = MISSING) -> str:
    """The one line that puts stock 0.8.0 in ``python`` in place of ``installed``.

    Pip reads ``==0.8.0`` as met by a local build of 0.8.0 (``0.8.0+<label>``) and
    would answer "already satisfied", so for such a build the line reinstalls
    xFuser alone, leaving its dependencies, torch among them, as they are.
    """
    modules = _pep440()
    if modules is None:  # no parser: the exact local-label form alone
        local = installed != STOCK and installed.split("+", 1)[0] == STOCK
    else:
        found = _parse(installed)
        local = (found is not None and not is_stock(installed)
                 and modules[1].Specifier(f"=={STOCK}").contains(found, prereleases=True))
    if local:
        return f"{python} -m pip install --force-reinstall --no-deps xfuser=={STOCK}"
    return f"{python} -m pip install --upgrade-strategy only-if-needed xfuser=={STOCK}"


def _why(version: str) -> str:
    if version == MISSING:
        return "xFuser is not installed there"
    if _is(version, "0.7.0"):
        return ("stock xFuser 0.7.0 imports an NPU ring helper that released yunchang lacks, so "
                "its CUDA attention cannot start")
    return f"no render has been checked on xFuser {version}"


def verdict(version: str, python: str, *, driver: bool = True) -> tuple[str, str]:
    """``(OK|WARN|FAIL, detail)`` for one interpreter's installed build.

    Only the driver's row names the metadata refresh: INSTALL gives it for the
    driver's editable install alone.
    """
    if is_stock(version):
        return OK, f"xFuser {version}"
    if _is(version, PATCHED):
        refresh = f" {METADATA}" if driver else ""
        return WARN, (f"xFuser {version}, the patched build 1.4.0 installed. It still runs; stock "
                      f"{STOCK} replaces it. While no render runs: "
                      f"`{move_command(python, version)}`, {RESTART}.{refresh} ({ENTRY})")
    return FAIL, (f"{'xFuser is not installed' if version == MISSING else f'xFuser {version}'}: this "
                  f"release runs on xFuser {STOCK}, or on the patched {PATCHED} that 1.4.0 "
                  f"installed, because {_why(version)}. Install {STOCK}: "
                  f"`{move_command(python, version)}`, {RESTART} ({ENTRY})")


def setup_refusal(version: str, python: str) -> str | None:
    """Why Worker setup refuses this build, or None when it runs.

    The raise site (actor/worker_env.py) adds the fix, so its message reads as
    one literal sentence to the refusal-class walker.
    """
    if is_allowed(version):
        return None
    found = "has no xFuser installed" if version == MISSING else f"has xFuser {version}"
    return (f"this Worker's Python ({python}) {found}, and Worker setup refuses it because "
            f"{_why(version)}. dgx-monarch runs on xFuser {STOCK}, or on the patched {PATCHED} "
            f"that 1.4.0 installed.")
