"""Exact signature comparison for adapter-rebound Comfy methods."""

from __future__ import annotations

import inspect
from collections.abc import Collection

from .comfy_rebound_signatures import (
    BROAD_ADAPTER_SUBCLASS_CONTRACTS,
    ReboundMethodContract,
)


def rebound_signature_incompatibility(
    signature: inspect.Signature,
    contract: ReboundMethodContract,
) -> str | None:
    """Return a precise incompatibility, or ``None`` for a declared shape."""
    declared = (
        *((name, inspect.Parameter.POSITIONAL_ONLY) for name in contract.positional_only),
        *((name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
          for name in contract.positional_or_keyword),
    )
    keyword_only = tuple(
        (name, inspect.Parameter.KEYWORD_ONLY) for name in contract.keyword_only
    )
    trailing = tuple(
        (name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in contract.optional_trailing
    )
    infix = tuple(
        (name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in contract.optional_infix
    )
    # A closed set: the declared shape, plus, where a contract declares one,
    # that shape with the whole trailing suffix or with the whole infix just
    # before the last declared name. Any other list, a partial suffix or infix
    # included, is drift and reports the same detail.
    admitted = [declared + keyword_only]
    if trailing:
        admitted.append(declared + trailing + keyword_only)
    if infix:
        admitted.append(declared[:-1] + infix + declared[-1:] + keyword_only)
    observed_explicit = tuple(
        (parameter.name, parameter.kind)
        for parameter in signature.parameters.values()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )
    if observed_explicit not in admitted:
        return "exact explicit parameter list changed"

    positional = contract.positional_only + contract.positional_or_keyword
    expected_required = (
        *positional[:contract.required_positional],
        *contract.required_keyword_only,
    )
    observed_required = tuple(
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    )
    if observed_required != expected_required:
        return "exact parameter defaults changed"

    observed_variadics = tuple(
        (parameter.name, parameter.kind.name)
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )
    expected_variadics: list[tuple[str, str]] = []
    if contract.variadic_positional is not None:
        expected_variadics.append(
            (contract.variadic_positional, inspect.Parameter.VAR_POSITIONAL.name)
        )
    if contract.variadic_keyword is not None:
        expected_variadics.append(
            (contract.variadic_keyword, inspect.Parameter.VAR_KEYWORD.name)
        )
    if observed_variadics != tuple(expected_variadics):
        return "exact variadic parameters changed"
    return None


def broad_adapter_subclass_incompatibility(
    model_base: object,
    enabled_families: Collection[str],
) -> tuple[str, str] | None:
    """Compare real Comfy subclasses with pinned broad-adapter branches."""
    namespace = vars(model_base)
    for contract in BROAD_ADAPTER_SUBCLASS_CONTRACTS:
        if contract.family not in enabled_families:
            continue
        root = namespace.get(contract.root_model_base)
        if not inspect.isclass(root):
            continue  # The ordinary model-base class check reports this first.
        expected = {branch.model_base for branch in contract.branches}
        actual = {
            name
            for name, value in namespace.items()
            if inspect.isclass(value) and issubclass(value, root)
        }
        if actual != expected:
            return (
                f"comfy.model_base.{contract.root_model_base}",
                "accepted subclass set changed: "
                f"added={sorted(actual - expected)}, removed={sorted(expected - actual)}",
            )
    return None
