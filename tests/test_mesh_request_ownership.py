"""Driver request-ownership regressions for mesh dispatch helpers."""
from types import SimpleNamespace

from dgx_monarch import mesh_setup


def test_fleet_dispatch_marks_only_its_owned_request_copy(monkeypatch):
    sent = []
    verified = []

    class SampleEndpoint:
        @staticmethod
        def call_one(request, *, progress_port):
            sent.append((request, progress_port))
            return "future"

    worker = SimpleNamespace(sample=SampleEndpoint())
    workers = SimpleNamespace(
        extent=SimpleNamespace(labels=("gpus",)),
        slice=lambda **coords: worker,
    )
    handle = SimpleNamespace(
        setup_cleanup_state=None,
        gpus_per_host=2,
        workers=workers,
        _verify_request_artifacts=lambda request, worker_index: verified.append(
            (request, worker_index)),
    )
    monkeypatch.setattr(
        mesh_setup,
        "dispatch_sample",
        lambda _handle, send, **_kwargs: send(),
    )
    nested = {"strength": 0.5}
    caller_request = {"model": {"options": nested}}

    result = mesh_setup.dispatch_sample_to_actor(
        handle, 3, caller_request, "progress", None)

    assert result == "future"
    assert "_dgxm_fleet_job" not in caller_request
    assert len(verified) == len(sent) == 1
    owned_request = sent[0][0]
    assert verified == [(owned_request, 3)]
    assert owned_request is not caller_request
    assert owned_request["_dgxm_fleet_job"] is True
    assert owned_request["model"] is caller_request["model"]
    assert sent[0][1] == "progress"
