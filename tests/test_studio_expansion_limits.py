"""Expanded drafts must stay bounded before allocation and persistence."""

import pytest

from dgx_monarch.studio.contracts import MAX_BODY_BYTES, StudioError, canonical
from dgx_monarch.studio.drafts import DraftError, apply_styles
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store


class UnallocatedTemplate(str):
    def replace(self, *args, **kwargs):
        pytest.fail("Oversized replacement was allocated before checking its budget")


def test_style_checks_expansion_before_replacing():
    template = UnallocatedTemplate("{prompt}" * 4096)
    with pytest.raises(DraftError, match="4 MiB"):
        apply_styles("x" * 2048, "", [{"id": "repeat", "prompt": template}])


@pytest.mark.parametrize("field", ["prompt", "negative_prompt"])
def test_style_budget_counts_json_escaped_unicode_and_append(field):
    # A non-BMP character occupies twelve bytes in the ASCII JSON wire form.
    text = "\U0001f600" * (MAX_BODY_BYTES // 12)
    with pytest.raises(DraftError, match="4 MiB"):
        apply_styles(text if field == "prompt" else "", text if field == "negative_prompt" else "",
                     [{"id": "append", field: "\U0001f600"}])


def test_expansion_rejection_is_typed_and_does_not_persist(tmp_path):
    store = Store(tmp_path / "state")
    try:
        jobs = Jobs(store, None)
        jobs.schema = {"Text": {"input": {"required": {"text": ["STRING", {}]}},
                                "output": [], "output_node": True}}
        request = {"workflow": {"1": {"class_type": "Text", "inputs": {"text": "x" * 2048}}},
                   "styles": [{"id": str(index), "prompt": "{prompt}{prompt}"} for index in range(12)],
                   "style_bindings": {"prompt": "root:1:text"}}
        assert len(canonical(request)) < 4096
        with pytest.raises(StudioError, match="4 MiB"):
            jobs.build_preview(request)
        assert store.all("previews") == []

        # Even without styles, the complete immutable provenance can exceed
        # the wire budget while the submitted workflow itself fits easily.
        request = {"workflow": {"1": {"class_type": "Text", "inputs": {"text": "x" * (MAX_BODY_BYTES // 4)}}}}
        assert len(canonical(request)) < MAX_BODY_BYTES
        with pytest.raises(StudioError) as caught:
            jobs.build_preview(request)
        assert caught.value.code == "preview_limit" and caught.value.status == 413
        assert store.all("previews") == []
    finally:
        store.close()
