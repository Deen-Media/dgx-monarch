"""The touchpoint record, its three constructors, and signature comparison.

``_signature_record`` checks one observed object against one record and puts
the reason for any mismatch in the returned ``detail``.
Like ``monarch_surface``, this module needs only the standard library.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from .monarch_semantics import (
    type_name as _type_name,
)


@dataclass(frozen=True)
class MonarchTouchpoint:
    """One required upstream attribute and the call shape production uses."""

    module: str
    attribute: str
    positional: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    callable: bool = False
    require_class: bool = False
    var_positional: bool = False
    var_keyword: bool = False

    @property
    def path(self) -> str:
        return f"{self.module}.{self.attribute}"


def _call(
    module: str,
    attribute: str,
    *keywords: str,
    positional: tuple[str, ...] = (),
    require_class: bool = False,
    var_positional: bool = False,
    var_keyword: bool = False,
) -> MonarchTouchpoint:
    return MonarchTouchpoint(
        module,
        attribute,
        positional,
        keywords,
        callable=True,
        require_class=require_class,
        var_positional=var_positional,
        var_keyword=var_keyword,
    )


def _attr(module: str, attribute: str) -> MonarchTouchpoint:
    return MonarchTouchpoint(module, attribute)


def _class(module: str, attribute: str) -> MonarchTouchpoint:
    return MonarchTouchpoint(module, attribute, require_class=True)


def _resolve(root: object, attribute: str) -> object:
    value = root
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def _signature_record(value: object, touchpoint: MonarchTouchpoint) -> dict[str, Any]:
    record: dict[str, Any] = {
        "status": "ok",
        "kind": _type_name(value),
        "signature": None,
        "detail": None,
    }
    if touchpoint.require_class and not inspect.isclass(value):
        record.update(status="incompatible", detail="not a class")
        return record
    if not touchpoint.callable:
        return record
    if not callable(value):
        record.update(status="incompatible", detail="not callable")
        return record
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as exc:
        record.update(
            status="incompatible",
            detail=f"signature unavailable: {type(exc).__name__}: {exc}",
        )
        return record
    record["signature"] = str(signature)
    positional = tuple(
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )[: len(touchpoint.positional)]
    if positional != touchpoint.positional:
        record.update(
            status="incompatible",
            detail=(
                "positional parameter order changed: "
                f"expected {touchpoint.positional!r}, observed {positional!r}"
            ),
        )
        return record
    for name in touchpoint.keywords:
        parameter = signature.parameters.get(name)
        if parameter is None:
            record.update(
                status="incompatible",
                detail=f"signature missing keyword parameter {name!r}",
            )
            return record
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            record.update(
                status="incompatible",
                detail=f"keyword parameter {name!r} became positional-only",
            )
            return record
    kinds = {parameter.kind for parameter in signature.parameters.values()}
    if touchpoint.var_positional and inspect.Parameter.VAR_POSITIONAL not in kinds:
        record.update(status="incompatible", detail="signature no longer accepts *args")
        return record
    if touchpoint.var_keyword and inspect.Parameter.VAR_KEYWORD not in kinds:
        record.update(status="incompatible", detail="signature no longer accepts **kwargs")
        return record
    try:
        signature.bind(
            *([object()] * len(touchpoint.positional)),
            **dict.fromkeys(touchpoint.keywords, object()),
        )
    except TypeError as exc:
        record.update(
            status="incompatible",
            detail=f"production call shape rejected: {exc}",
        )
    return record
