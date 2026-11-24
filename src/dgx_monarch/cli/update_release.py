"""Private, atomically switched release slots for verified cluster updates."""
from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from ..config import ClusterConfig, HostConfig
from ..runtime_provenance import dgx_source_manifest_sha256
from ..studio_resource_validation import validate_studio_resources
from . import lifecycle
from .lifecycle_host import remote_colocation_error
from .update_cancellation import stronger_cancellation
from .update_driver_pin import site_payload_digest
from .update_installation import (
    capture_installation,
    change_installation,
    installation_script,
    settle_installation,
    start_prior_installation,
    stop_installation,
)
from .update_pin_diagnostic import PinDiagnostic
from .update_release_types import (  # noqa: F401  # Public update_release API.
    _DIGEST,
    _REMOTE_BASE,
    _SHA,
    _TOKEN,
    CommandRunner,
    HostRunner,
    ReleaseLayoutError,
    ReleaseMetadata,
    ReleaseSlot,
    _default_command_runner,
    _default_host_runner,
    _private_directory,
    _result,
    _validate_metadata,
)
from .update_transaction import Certainty, OperationResult
from .update_worker_attestation import (
    WorkerReleaseIdentity,
    release_attestation_script,
    remote_release_check,
    remote_release_matches,
    site_tree_digest,
    verifier_tree_digest,
)


class ReleaseManager:
    """Stage, switch, compensate, and retire one exact worker release."""

    def __init__(
        self,
        config: ClusterConfig,
        *,
        command_runner: CommandRunner = _default_command_runner,
        host_runner: HostRunner = _default_host_runner,
        state_root: Path | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.config = config
        self._run = command_runner
        self._host_run = host_runner
        self._state_root = state_root or Path.home() / ".local" / "state" / "dgx-monarch" / "update-staging"
        self._token_factory = token_factory
        self.slot: ReleaseSlot | None = None
        # (host name, ownership refusal or None) for each host whose prior
        # attestation did not match; the update names them beside prior_worker_drift.
        self.prior_refusals: list[tuple[str, PinDiagnostic | None]] = []

    def ensure_supported(self) -> None:
        error = remote_colocation_error(self.config.hosts, lifecycle._is_local)
        if error is not None:
            raise ReleaseLayoutError(error)

    def prepare(self, checkout: Path, metadata: ReleaseMetadata) -> ReleaseSlot:
        """Install the package and exact pin into a private slot, then copy and verify it on every host."""
        # Metadata and token refusals need no host, so they precede the layout
        # check, which resolves every worker host.
        _validate_metadata(metadata)
        token = (
            self._token_factory() if self._token_factory is not None
            else f"u-{metadata.target_sha[:12]}-{secrets.token_hex(8)}"
        )
        if _TOKEN.fullmatch(token) is None:
            raise ValueError("release token is invalid")
        self.ensure_supported()
        root = _private_directory(_private_directory(self._state_root) / token)
        site = root / "site"
        site.mkdir(mode=0o700)
        slot = ReleaseSlot(token=token, root=root, site=site, metadata=metadata)
        self.slot = slot
        try:
            self._pip_stage(site, checkout, metadata.torchmonarch_pin)
            if dgx_source_manifest_sha256(site / "dgx_monarch") != metadata.source_manifest:
                raise ReleaseLayoutError("private driver release source manifest mismatched")
            validate_studio_resources(str(site / "dgx_monarch"))
            slot.pin_payload_digest = site_payload_digest(site, metadata.torchmonarch_pin)
            slot.site_manifest = site_tree_digest(site)
            shutil.copytree(
                Path(__file__).resolve().parents[1], root / "verifier" / "dgx_monarch",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            slot.verifier_digest = verifier_tree_digest(root / "verifier" / "dgx_monarch")
            (root / "release.json").write_text(
                json.dumps(metadata.__dict__, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            (root / "release.json").chmod(0o600)
            (root / ".reservation").write_text(slot.reservation, encoding="ascii")
            (root / ".reservation").chmod(0o600)
            for host in self.config.hosts:
                # Count attempted reservations too: a copy or verification error
                # after mkdir still owns a remote path that must be removed.
                slot.remote_staged += 1
                self._stage_host(host, slot)
            return slot
        except BaseException as primary:
            try:
                self._cleanup_unactivated(slot)
            except BaseException as cleanup_error:
                raise stronger_cancellation(primary, cleanup_error) from None
            raise

    def attest_prior(self, metadata: ReleaseMetadata, pin_payload_digest: str) -> bool:
        """Before any live change, record the prior release if every host runs it exactly; activation needs it."""
        slot = self._require_slot()
        slot.prior_metadata = None
        slot.prior_pin_payload_digest = ""
        slot.installations = {}
        self.prior_refusals = []
        _validate_metadata(metadata)
        identity = WorkerReleaseIdentity(
            metadata.version, metadata.torchmonarch_pin,
            metadata.source_manifest, pin_payload_digest,
        )
        identity.validate()
        tools = self._remote(slot.token, "verifier")
        snapshots = {host.name: capture_installation(self.config, host, self._host_run) for host in self.config.hosts}
        if any(value is None for value in snapshots.values()):
            self.prior_refusals = [(name, None) for name, value in snapshots.items() if value is None]
            return False
        slot.installations = {name: value for name, value in snapshots.items() if value is not None}
        matched = True
        for host in self.config.hosts:  # every host, so each refusal is named
            host_matched, refusal = remote_release_check(
                self.config, host, site=slot.installations[host.name]["site"], tools=tools, identity=identity,
                captured_site=True,
                host_runner=self._host_run, verifier_digest=slot.verifier_digest,
            )
            if not host_matched:
                self.prior_refusals.append((host.name, refusal))
            matched = matched and host_matched
        if matched:
            slot.prior_metadata = metadata
            slot.prior_pin_payload_digest = pin_payload_digest
        return matched

    def _prior_attestation_script(self, host: HostConfig, slot: ReleaseSlot) -> str:
        meta = slot.prior_metadata
        if meta is None:
            raise ReleaseLayoutError("prior release was not attested")
        return release_attestation_script(
            self.config, site=slot.installations[host.name]["site"],
            tools=self._remote(slot.token, "verifier"),
            identity=WorkerReleaseIdentity(meta.version, meta.torchmonarch_pin, meta.source_manifest, slot.prior_pin_payload_digest),
            verifier_digest=slot.verifier_digest, captured_site=True,
        )

    def _pip_stage(self, site: Path, checkout: Path, pin: str) -> None:
        base = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--target", str(site)]
        for requirement in (f"torchmonarch=={pin}", str(checkout)):
            result = self._run([*base, requirement], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise ReleaseLayoutError("private no-deps release staging failed")

    def _remote(self, token: str, leaf: str = "") -> str:
        if _TOKEN.fullmatch(token) is None:
            raise ValueError("release token is invalid")
        base = f"$HOME/{_REMOTE_BASE}/releases/{token}"
        return f"{base}/{leaf}" if leaf else base

    def _stage_host(self, host: HostConfig, slot: ReleaseSlot) -> None:
        create = self._host_run(
            self.config,
            host,
            installation_script({"operation": "reserve", "token": slot.token, "reservation": slot.reservation}),
            30,
        )
        if create.returncode != 0:
            raise ReleaseLayoutError("remote private release reservation failed")
        if lifecycle._is_local(host) is True:
            destination = Path.home() / _REMOTE_BASE / "releases" / slot.token
            shutil.copytree(slot.root, destination, dirs_exist_ok=True)
            if not self._host_release_matches(host, slot):
                raise ReleaseLayoutError("local private release validation failed")
            return
        ssh_cmd = lifecycle._rsync_remote_shell(self.config, host)
        target = lifecycle._rsync_target(host)
        copied = self._run(
            [
                "rsync", "-a", "--delete", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                "-e", ssh_cmd, "--", f"{slot.root}/", f"{target}:~/{_REMOTE_BASE}/releases/{slot.token}/",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if copied.returncode != 0 or not self._host_release_matches(host, slot):
            raise ReleaseLayoutError("remote private release copy or validation failed")

    def _host_release_matches(self, host: HostConfig, slot: ReleaseSlot) -> bool:
        meta = slot.metadata
        site = self._remote(slot.token, "site")
        return remote_release_matches(
            self.config, host, site=site, tools=self._remote(slot.token, "verifier"),
            identity=WorkerReleaseIdentity(
                meta.version, meta.torchmonarch_pin,
                meta.source_manifest, slot.pin_payload_digest, slot.site_manifest,
            ),
            host_runner=self._host_run, verifier_digest=slot.verifier_digest,
        )

    def stop_prior_workers(self) -> OperationResult:
        """Stop only the still-idle systemd identities captured before confirmation."""
        slot = self._require_slot()
        if not slot.installations or slot.prior_metadata is None:
            return _result(Certainty.UNKNOWN)
        for host in self.config.hosts:
            certainty = stop_installation(self.config, host, slot.installations[host.name], self._host_run, self._prior_attestation_script(host, slot))
            if certainty != Certainty.SUCCEEDED:
                return _result(certainty)
        return _result(Certainty.SUCCEEDED)

    def start_prior_workers(self, generation: str) -> OperationResult:
        """Restart only a definitely compensated prior installation.

        A host after the one where activation stopped was never sent an
        activation, so its prior start expects no journal of this transaction.
        """
        slot = self._require_slot()
        if slot.activation_ambiguous or slot.remote_activated or not slot.installations:
            return _result(Certainty.UNKNOWN)
        for index, host in enumerate(self.config.hosts):
            certainty = start_prior_installation(
                self.config, host, slot.token, slot.installations[host.name], self._host_run,
                self._prior_attestation_script(host, slot), generation,
                attempted=index < slot.activation_attempted,
            )
            if certainty != Certainty.SUCCEEDED:
                return _result(certainty)
        return _result(Certainty.SUCCEEDED)

    def activate_workers(self) -> OperationResult:
        """Switch one host at a time and stop at the first switch that does not confirm.

        No host is sent an activation before the host ahead of it read back as
        switched. A switch whose reply was lost, or that never ran, is read back
        under the same locks first (``settle_installation``), which also cancels
        a switch that never began. Only a host that still cannot be read leaves
        the result unknown; the later hosts then still run the prior release.
        After a definite failure compensation restores only the hosts the
        update reached.
        """
        slot = self._require_slot()
        if (
            slot.prior_metadata is None
            or _DIGEST.fullmatch(slot.prior_pin_payload_digest) is None
            or not slot.installations
        ):
            return _result(Certainty.UNKNOWN)
        slot.activation_attempted = 0
        certainty = Certainty.SUCCEEDED
        try:
            for host in self.config.hosts:
                slot.activation_attempted += 1
                certainty = change_installation(
                    self.config, host, slot.token, slot.installations[host.name], self._host_run,
                    preflight_script=self._prior_attestation_script(host, slot),
                )
                if certainty == Certainty.UNKNOWN:
                    certainty = settle_installation(
                        self.config, host, slot.token, slot.installations[host.name], self._host_run,
                    )
                if certainty != Certainty.SUCCEEDED:
                    break
        except BaseException:
            slot.activation_ambiguous = True
            raise
        if certainty == Certainty.SUCCEEDED:
            slot.remote_activated = True
            return _result(Certainty.SUCCEEDED)
        self._report_activation(slot, certainty)
        if certainty == Certainty.UNKNOWN:
            slot.activation_ambiguous = True
            return _result(Certainty.UNKNOWN)
        return _result(Certainty.FAILED)

    def _report_activation(self, slot: ReleaseSlot, certainty: Certainty) -> None:
        """Name each host's release after a switch that stopped short, on stderr only."""
        hosts = self.config.hosts
        stopped_at = slot.activation_attempted - 1
        host = hosts[stopped_at].name
        switched = ", ".join(item.name for item in hosts[:stopped_at])
        untouched = ", ".join(item.name for item in hosts[stopped_at + 1:])
        if certainty == Certainty.UNKNOWN:
            print(f"activation on {host} is unknown: its source link and unit can name either release", file=sys.stderr)
        else:
            print(f"activation on {host} failed and left a state the update can restore", file=sys.stderr)
        if switched:
            print(f"switched to the new release before it: {switched}", file=sys.stderr)
        if untouched:
            print(f"never sent an activation, still on the old release: {untouched}", file=sys.stderr)
        if certainty == Certainty.UNKNOWN:
            print(
                "the hosts may now run different releases, all stopped. Read each host's link "
                "(readlink -f ~/.local/share/dgx-monarch/src): when every link names the old "
                "release, `dgxm up` starts the Workers again on it; when any names the new one, "
                "follow the forward repair in TROUBLESHOOTING entry 118",
                file=sys.stderr,
            )

    def compensate_workers(self) -> OperationResult:
        """Restore every host activation reached; the hosts after them were never changed."""
        slot = self._require_slot()
        if slot.activation_ambiguous or not slot.installations:
            return _result(Certainty.UNKNOWN)
        certainties: list[Certainty] = []
        pending: BaseException | None = None
        for host in self.config.hosts[:slot.activation_attempted]:
            try:
                certainties.append(change_installation(
                    self.config, host, slot.token, slot.installations[host.name], self._host_run,
                    compensate=True,
                ))
            except BaseException as error:
                certainties.append(Certainty.UNKNOWN)
                pending = stronger_cancellation(pending, error)
        if pending is not None:
            slot.activation_ambiguous = True
            raise pending
        try:
            prior_matches = not all(value == Certainty.SUCCEEDED for value in certainties) or self._prior_hosts_match(slot)
        except BaseException:
            slot.activation_ambiguous = True
            raise
        if all(value == Certainty.SUCCEEDED for value in certainties):
            if not prior_matches:
                slot.activation_ambiguous = True
                return _result(Certainty.UNKNOWN)
            slot.remote_activated = False
            return _result(Certainty.SUCCEEDED)
        if any(value == Certainty.UNKNOWN for value in certainties):
            slot.activation_ambiguous = True
            return _result(Certainty.UNKNOWN)
        slot.activation_ambiguous = True
        return _result(Certainty.FAILED)

    def _prior_hosts_match(self, slot: ReleaseSlot) -> bool:
        meta = slot.prior_metadata
        if meta is None or not slot.installations:
            return False
        identity = WorkerReleaseIdentity(
            meta.version, meta.torchmonarch_pin,
            meta.source_manifest, slot.prior_pin_payload_digest,
        )
        return all(
            remote_release_matches(
                self.config, host, site=slot.installations[host.name]["site"], captured_site=True,
                tools=self._remote(slot.token, "verifier"), identity=identity,
                host_runner=self._host_run, verifier_digest=slot.verifier_digest,
            )
            for host in self.config.hosts
        )

    def prior_hosts_match(self) -> bool:
        return self._prior_hosts_match(self._require_slot())

    def finalize(self) -> OperationResult:
        """Finalize every host, after the driver commit, even when an earlier one fails."""
        slot = self._require_slot()
        if slot.activation_ambiguous or not slot.remote_activated or not slot.installations:
            return _result(Certainty.UNKNOWN)
        certainties: list[Certainty] = []
        pending: BaseException | None = None
        for host in self.config.hosts:
            try:
                certainties.append(change_installation(
                    self.config, host, slot.token, slot.installations[host.name], self._host_run,
                    finalize=True,
                ))
            except BaseException as error:
                certainties.append(Certainty.UNKNOWN)
                pending = stronger_cancellation(pending, error)
        if pending is not None:
            raise pending
        certainty = Certainty.SUCCEEDED if all(value == Certainty.SUCCEEDED for value in certainties) else Certainty.UNKNOWN
        slot.finalized = certainty == Certainty.SUCCEEDED
        return _result(certainty)

    def cleanup(self) -> OperationResult:
        slot = self._require_slot()
        if slot.activation_ambiguous:
            return _result(Certainty.UNKNOWN)
        if slot.remote_activated and not slot.finalized:
            return _result(Certainty.UNKNOWN)
        if not slot.remote_activated:
            if not self._cleanup_unactivated(slot):
                return _result(Certainty.UNKNOWN)
        try:
            self._assert_owned_local(slot)
            shutil.rmtree(slot.root)
        except FileNotFoundError:
            pass
        except OSError:
            return _result(Certainty.UNKNOWN)
        return _result(Certainty.SUCCEEDED)

    def _cleanup_unactivated(self, slot: ReleaseSlot) -> bool:
        ok = True
        pending: BaseException | None = None
        for host in self.config.hosts[:slot.remote_staged]:
            script = installation_script({"operation": "cleanup", "token": slot.token, "reservation": slot.reservation})
            try:
                result = self._host_run(self.config, host, script, 60)
            except (OSError, subprocess.TimeoutExpired):
                ok = False
                continue
            except BaseException as error:
                ok = False
                pending = stronger_cancellation(pending, error)
                continue
            ok &= result.returncode == 0 and "CLEANED" in result.stdout
        if ok:
            try:
                self._assert_owned_local(slot)
                shutil.rmtree(slot.root)
            except FileNotFoundError:
                pass
            except OSError:
                ok = False
        if pending is not None:
            raise pending
        return ok

    def readback_hosts(self) -> bool:
        slot = self._require_slot()
        meta = slot.metadata
        target = self._remote(slot.token, "site")
        identity = WorkerReleaseIdentity(
            meta.version, meta.torchmonarch_pin,
            meta.source_manifest, slot.pin_payload_digest, slot.site_manifest,
        )
        return site_tree_digest(slot.site) == slot.site_manifest and all(
            remote_release_matches(
                self.config, host, site=f"$HOME/{_REMOTE_BASE}/src",
                tools=self._remote(slot.token, "verifier"), identity=identity,
                host_runner=self._host_run, verifier_digest=slot.verifier_digest,
                link_target=target,
            )
            for host in self.config.hosts
        )

    def _require_slot(self) -> ReleaseSlot:
        if self.slot is None or _TOKEN.fullmatch(self.slot.token) is None:
            raise ReleaseLayoutError("release slot was not prepared")
        return self.slot

    def _assert_owned_local(self, slot: ReleaseSlot) -> None:
        expected = self._state_root / slot.token
        active = Path.home() / _REMOTE_BASE / "src"
        if active.is_symlink() and active.resolve().is_relative_to(slot.root.resolve()):
            raise ReleaseLayoutError("release cleanup target is active")
        if Path(__file__).resolve().is_relative_to(slot.root.resolve()):
            raise ReleaseLayoutError("release cleanup target contains the running updater")
        if slot.root != expected or slot.root.is_symlink():
            raise ReleaseLayoutError("release cleanup target escaped its private root")
