"""Proof-cohort source attestation for the identity Gate."""
from __future__ import annotations

from typing import Any

from .. import mesh_setup, runtime_provenance


class GateProvenanceError(RuntimeError):
    """A provenance verdict this attestor decided from evidence in hand.

    Every raise below reads a value the driver already holds, so the refusal
    repeats on the same inputs. The ``provenance_baseline`` call keeps raising
    whatever the transport raises, which is how a caller tells a mismatch from
    a call that never came back.
    """


class ProofCohortAttestor:
    """Bracket one ceremony with exact all-rank package-source evidence."""

    def __init__(self, observer: Any = None) -> None:
        self._observer = observer
        self._setup_token: mesh_setup.SetupToken | None = None
        self._world: int | None = None
        self._source: str | None = None

    def __call__(
        self,
        boundary: str,
        handle: Any,
        setup_token: mesh_setup.SetupToken | None,
    ) -> None:
        if boundary not in {"pre", "post"}:
            raise GateProvenanceError(f"invalid Gate provenance boundary {boundary!r}")
        if not isinstance(setup_token, mesh_setup.SetupToken):
            raise GateProvenanceError("Gate provenance requires an exact setup token")
        token = setup_token
        world = getattr(handle, "world", None)
        if type(world) is not int or world < 1:
            raise GateProvenanceError("Gate provenance requires a positive worker world")
        source = runtime_provenance.cached_dgx_source_manifest_sha256()
        if (
            not isinstance(source, str)
            or len(source) != 64
            or any(character not in "0123456789abcdef" for character in source)
        ):
            raise GateProvenanceError("Gate provenance requires a canonical driver source digest")

        identity = (token, world, source)
        expected = (self._setup_token, self._world, self._source)
        if boundary == "pre":
            if self._setup_token is not None:
                raise GateProvenanceError("Gate provenance pre-attestation was repeated")
        elif self._setup_token is None:
            raise GateProvenanceError("Gate provenance post-attestation lacks a pre-attestation")
        elif identity != expected:
            raise GateProvenanceError("Gate provenance setup or driver source changed during proof")

        rows = handle.call_all(
            "provenance_baseline",
            token.generation,
            timeout_s=600,
            **mesh_setup.token_kwargs(token),
        )
        self._validate_rows(rows, world, token.generation, source, boundary)
        if boundary == "pre":
            self._setup_token, self._world, self._source = identity
        if self._observer is not None:
            self._observer(boundary, handle)

    @staticmethod
    def _validate_rows(
        rows: object,
        world: int,
        generation: int,
        source: str,
        boundary: str,
    ) -> None:
        if not isinstance(rows, list) or len(rows) != world:
            raise GateProvenanceError(
                f"Gate {boundary} provenance lacks complete world coverage"
            )
        ranks: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise GateProvenanceError(f"Gate {boundary} provenance row is malformed")
            rank = row.get("rank")
            if type(rank) is not int or not 0 <= rank < world or rank in ranks:
                raise GateProvenanceError(f"Gate {boundary} provenance rank coverage is invalid")
            if type(row.get("world")) is not int or row["world"] != world:
                raise GateProvenanceError(f"Gate {boundary} provenance world is inconsistent")
            if (
                type(row.get("setup_generation")) is not int
                or row["setup_generation"] != generation
            ):
                raise GateProvenanceError(
                    f"Gate {boundary} provenance setup generation is inconsistent"
                )
            if row.get("source_manifest_sha256") != source:
                raise GateProvenanceError(
                    f"Gate {boundary} provenance package source does not match driver"
                )
            ranks.add(rank)
        if ranks != set(range(world)):
            raise GateProvenanceError(f"Gate {boundary} provenance rank coverage is incomplete")
