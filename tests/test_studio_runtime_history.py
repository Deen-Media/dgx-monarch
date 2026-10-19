import pytest

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.runtime_history import page_runtime
from dgx_monarch.studio.store import Store


def test_runtime_pages_preserve_all_render_identities_and_bind_cursor(tmp_path):
    store = Store(tmp_path / "state")
    store.create("jobs", {"id": "j1"})
    store.create("jobs", {"id": "j2"})
    for index in range(140):
        store.create("runtime_renders", {"id": str(index), "job_id": "j1", "created_at": index,
                                         "render_id": str(index), "status": "completed"})
    rows, cursor = [], None
    for _ in range(4):
        page = page_runtime(store, "j1", {"limit": "40", **({"cursor": cursor} if cursor else {})})
        rows.extend(page["items"])
        cursor = page["next_cursor"]
    assert len(rows) == 140 and len({row["render_id"] for row in rows}) == 140
    first = page_runtime(store, "j1", {"limit": "1"})
    with pytest.raises(StudioError, match="cursor"):
        page_runtime(store, "j2", {"cursor": first["next_cursor"]})
    assert page_runtime(store, "j2", {})["items"] == []
    store.close()
