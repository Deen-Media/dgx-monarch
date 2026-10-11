"""Validate the built wheel's own requirements before any no-deps installation."""
from __future__ import annotations

import re
import zipfile
from collections.abc import Mapping
from email.parser import BytesParser
from pathlib import Path


class WheelRequirementsError(ValueError):
    """Bounded diagnostics; arbitrary metadata text is never returned."""


def validate_wheel_requirements(path: Path, versions: Mapping[str, str]) -> None:
    """Require complete, applicable base dependencies in the projected environment.

    No extras are requested by this installer. Inactive markers are skipped;
    active requested extras and URL requirements require provenance/closure that
    a version map cannot attest, so they refuse instead of being assumed met.
    """
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.version import Version

    with zipfile.ZipFile(path) as archive:
        entries = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(entries) != 1:
            raise WheelRequirementsError("staged wheel requirement metadata ambiguous")
        metadata = BytesParser().parsebytes(archive.read(entries[0]))
    python_requirement = metadata.get("Requires-Python")
    if python_requirement:
        import platform

        try:
            compatible = SpecifierSet(python_requirement).contains(platform.python_version(), prereleases=True)
        except Exception:
            raise WheelRequirementsError("staged wheel Python requirement unreadable") from None
        if not compatible:
            raise WheelRequirementsError("staged wheel Python requirement unsatisfied")
    for raw in metadata.get_all("Requires-Dist", []):
        try:
            requirement = Requirement(raw)
            applicable = not requirement.marker or requirement.marker.evaluate({"extra": ""})
        except Exception:
            raise WheelRequirementsError("staged wheel dependency requirement unreadable") from None
        if not applicable:
            continue
        name = re.sub(r"[-_.]+", "-", requirement.name).lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name):
            raise WheelRequirementsError("staged wheel dependency name invalid")
        if requirement.url:
            raise WheelRequirementsError(f"staged wheel URL dependency {name} cannot be attested")
        if requirement.extras:
            raise WheelRequirementsError(f"staged wheel dependency extras for {name} cannot be attested")
        value = versions.get(name)
        if value is None:
            raise WheelRequirementsError(f"staged wheel requires missing dependency {name}")
        try:
            compatible = requirement.specifier.contains(Version(value), prereleases=True)
        except Exception:
            raise WheelRequirementsError(f"staged wheel dependency version for {name} unreadable") from None
        if not compatible:
            raise WheelRequirementsError(f"staged wheel dependency version for {name} unsatisfied")
