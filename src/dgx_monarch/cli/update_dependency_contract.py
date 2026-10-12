"""Read-only dependency closure checks executed in each target interpreter."""
from __future__ import annotations

import inspect
import re
import zipfile
from collections.abc import Callable
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import PurePosixPath
from typing import Any


def check_contract(contract: dict, roots: list[str]) -> list[str]:
    """Validate only the target runtime's closure; never import its packages."""
    import importlib.metadata as metadata
    import platform

    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version

    errors: list[str] = []
    try:
        if (not isinstance(contract, dict) or set(contract) != {
                "version", "requires_python", "requirements", "provides_extras"}
                or not isinstance(contract["version"], str)
                or not isinstance(contract["requires_python"], str)
                or not isinstance(contract["requirements"], list)
                or not isinstance(contract["provides_extras"], list)
                or not isinstance(roots, list)):
            return ["invalid dependency contract"]
        if len(roots) + len(contract["requirements"]) > 4096:
            return ["dependency closure exceeds inspection limit"]
        target_version = Version(contract["version"])
        python_version = Version(platform.python_version())
        if python_version not in SpecifierSet(contract["requires_python"]):
            errors.append("torchmonarch: incompatible Python version")
        environment = {str(key): str(value) for key, value in default_environment().items()}
        pending: list[tuple[str, tuple[str, ...]]] = [(raw, ("",)) for raw in roots]
        pending.extend((raw, ("",)) for raw in contract["requirements"])
        visited: set[tuple[str, tuple[str, ...]]] = set()
        checked = 0
        while pending:
            checked += 1
            if checked > 4096 or len(visited) > 512 or len(pending) > 4096:
                return ["dependency closure exceeds inspection limit"]
            raw, parent_extras = pending.pop()
            if not isinstance(raw, str) or len(raw) > 4096:
                return ["invalid dependency requirement"]
            requirement = Requirement(raw)
            if requirement.marker and not any(
                requirement.marker.evaluate({**environment, "extra": extra})
                for extra in parent_extras
            ):
                continue
            name = canonicalize_name(requirement.name)
            if requirement.url:
                errors.append(f"{name}: direct URL requirements cannot be verified")
                continue
            extras = tuple(sorted({"", *map(canonicalize_name, requirement.extras)}))
            if name == "torchmonarch":
                installed = target_version
                children = contract["requirements"]
                requires_python = contract["requires_python"]
                provided_extras: list[str] = contract["provides_extras"]
            else:
                try:
                    distribution = metadata.distribution(name)
                except metadata.PackageNotFoundError:
                    errors.append(f"{name}: missing")
                    continue
                installed = Version(distribution.version)
                children = distribution.requires or []
                requires_python = distribution.metadata.get("Requires-Python", "")
                provided_extras = list(distribution.metadata.get_all("Provides-Extra") or ())
            if installed not in requirement.specifier:
                errors.append(f"{name}: installed {installed}, requires {requirement.specifier}")
            if python_version not in SpecifierSet(requires_python):
                errors.append(f"{name}: incompatible Python version")
            if any(extra and extra not in {canonicalize_name(item) for item in provided_extras}
                   for extra in extras):
                errors.append(f"{name}: requested extra is unavailable")
            key = name, extras
            if key not in visited:
                visited.add(key)
                pending.extend((child, extras) for child in children)
        return sorted(set(errors))
    except Exception:
        # Metadata and parser errors may include URLs or private local paths.
        return ["dependency metadata could not be verified"]


def probe_source() -> str:
    """Ship the same checker to hosts without importing the target release."""
    return (
        inspect.getsource(check_contract)
        + "\nimport json, sys\n"
        + "payload = json.loads(sys.argv[1])\n"
        + "errors = check_contract(payload['contract'], payload['roots'])\n"
        + "print(json.dumps(errors))\n"
        + "raise SystemExit(1 if errors else 0)\n"
    )


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def parse_wheel_metadata(payload: bytes, error_type: type[Exception]) -> tuple[str, str]:
    try:
        message = BytesParser(policy=email_policy).parsebytes(payload, headersonly=True)
        name = str(message["Name"])
        version = str(message["Version"])
    except Exception as exc:
        raise error_type("wheel METADATA is unreadable") from exc
    if not name or not version or name == "None" or version == "None":
        raise error_type("wheel METADATA lacks distribution identity")
    return name, version


def dependency_contract(payload: bytes, pin: str, error_type: type[Exception]) -> dict:
    message = BytesParser(policy=email_policy).parsebytes(payload)
    python_fields = message.get_all("Requires-Python", [])
    if len(python_fields) > 1:
        raise error_type("target wheel has ambiguous Python requirements")
    return {
        "version": pin,
        "requires_python": str(python_fields[0]) if python_fields else "",
        "requirements": [str(value) for value in message.get_all("Requires-Dist", [])],
        "provides_extras": [str(value) for value in message.get_all("Provides-Extra", [])],
    }


def staged_wheel_contract(target: Any, inspect_wheel: Callable, hash_file: Callable,
                          error_type: type[Exception]) -> dict:
    """Bind candidate dependency metadata to the staged wheel before and after reading."""
    if target is None:
        raise error_type("target wheel is not staged")
    wheel_hash, _files = inspect_wheel(target.wheel, target.pin)
    if wheel_hash != target.wheel_sha256:
        raise error_type("target wheel changed after staging")
    with zipfile.ZipFile(target.wheel) as archive:
        entries = [name for name in archive.namelist()
                   if len(PurePosixPath(name).parts) == 2
                   and re.fullmatch(r".+\.dist-info", PurePosixPath(name).parts[0], re.IGNORECASE)
                   and PurePosixPath(name).name == "METADATA"]
        payload = archive.read(entries[0])
    if hash_file(target.wheel) != target.wheel_sha256:
        raise error_type("target wheel changed during dependency inspection")
    return dependency_contract(payload, target.pin, error_type)


def require_candidate_dependencies(config: Any, roots: tuple[str, ...] | None,
                                   pins: Any, run: Callable, checker: Callable) -> None:
    """Refuse before service mutation when candidate prerequisites have drifted."""
    from .update_transaction import UpdatePreconditionError

    if roots is None or not checker(config, roots, pins.target_dependency_contract, run):
        raise UpdatePreconditionError(
            "target_dependencies_unsatisfied",
            "candidate wheel dependency closure is not satisfied on every interpreter; "
            "review its Requires-Dist and Requires-Python before updating prerequisites",
        )
