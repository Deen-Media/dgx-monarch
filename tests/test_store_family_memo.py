"""Test the on-disk family memo and its header-digest index.

Covers round trips, staleness, the 1 MiB floor, per-family eviction and one row
per file after metadata-only changes.
"""
import json
import os
import struct
import threading
from types import SimpleNamespace

# Large enough to pass the 1 MiB checkpoint floor.
BIG = b"A" * (2 << 20)
BIGGER = b"B" * (3 << 20)

# The container two store test modules wrote 164 times into the operator's
# cache, found 2026-09-03: 8 bytes of header length, a 67-byte BF16 header and
# 2 bytes of tensor.
FIXTURE_BYTES = 77


def test_family_memo_roundtrip_and_stale_invalidation(tmp_path, monkeypatch):
    from dgx_monarch.actor import model_store as ms2

    memo_path = tmp_path / "family_memo.json"
    monkeypatch.setattr(ms2, "_FAMILY_MEMO_PATH", str(memo_path))
    f = tmp_path / "model.safetensors"
    f.write_bytes(BIG)
    assert ms2.memoized_family(str(f)) is None
    ms2.memoize_family(str(f), "krea2")
    assert ms2.memoized_family(str(f)) == "krea2"
    f.write_bytes(BIGGER)  # new bytes, so a new identity the memo does not know
    assert ms2.memoized_family(str(f)) is None


def test_family_memo_cap_charges_the_crowded_family(tmp_path):
    """One busy family must not evict every other family's rows: a small family
    that loses them loses its slab fast path until a stock load memoizes them again.

    The rule answered a memo found at its cap with 251 rows of one family beside
    5 across two others (the 2026-08-12 cap change). A 2026-09-03 replay of the
    head's memo found its crowd was test fixtures, 164 rows labelled with a
    vouched family, which the 1 MiB floor drops; the rule holds.
    """
    import threading

    from dgx_monarch.actor import store_family

    memo_path = tmp_path / "family_memo.json"
    lock = threading.Lock()
    logger = SimpleNamespace(warning=lambda *a, **k: None)

    def remember(key: str, family: str) -> None:
        store_family.memoize_family(
            key, family,
            identity=lambda path: path,
            memo_path=str(memo_path),
            lock=lock,
            limit=4,
            logger=logger,
        )

    for index in range(4):
        remember(f"busy-{index}", "krea2")
    remember("rare-0", "ltx")

    memo = json.loads(memo_path.read_text())
    assert len(memo) == 4
    assert memo.get("rare-0") == "ltx"      # the family with one row keeps it
    assert "busy-0" not in memo             # the crowd paid, oldest row first
    assert memo.get("busy-1") == "krea2"

    for index in range(4, 12):
        remember(f"busy-{index}", "krea2")

    memo = json.loads(memo_path.read_text())
    assert len(memo) == 4
    assert memo.get("rare-0") == "ltx"
    assert memo.get("busy-11") == "krea2"   # the newest row is never the victim


def test_family_memo_counts_the_row_it_just_wrote(tmp_path):
    """The row just written counts toward which family fills the cap.

    Without that row in the counts, the write that puts a family in the lead
    looks tied with the family it just overtook, and the tie charges the wrong one.
    """
    import threading

    from dgx_monarch.actor import store_family

    memo_path = tmp_path / "family_memo.json"
    lock = threading.Lock()

    def remember(key: str, family: str) -> None:
        store_family.memoize_family(
            key, family,
            identity=lambda path: path,
            memo_path=str(memo_path),
            lock=lock,
            limit=4,
            logger=SimpleNamespace(warning=lambda *a, **k: None),
        )

    remember("ltx-0", "ltx")        # oldest row overall: plain oldest-first would evict it
    remember("ltx-1", "ltx")
    remember("krea2-0", "krea2")
    remember("krea2-1", "krea2")
    remember("krea2-2", "krea2")    # krea2 is now 3 rows against ltx's 2

    memo = json.loads(memo_path.read_text())
    assert len(memo) == 4
    assert memo.get("ltx-0") == "ltx"     # ltx is not the crowd and keeps both
    assert memo.get("ltx-1") == "ltx"
    assert "krea2-0" not in memo          # the crowd paid, oldest of its own
    assert memo.get("krea2-2") == "krea2"


def test_family_memo_falls_back_to_oldest_first_on_a_corrupt_memo(tmp_path):
    import threading

    from dgx_monarch.actor import store_family

    memo_path = tmp_path / "family_memo.json"
    memo_path.write_text(json.dumps({"a": 1, "b": 2, "c": 3}))
    store_family.memoize_family(
        "d", "krea2",
        identity=lambda path: path,
        memo_path=str(memo_path),
        lock=threading.Lock(),
        limit=3,
        logger=SimpleNamespace(warning=lambda *a, **k: None),
    )

    memo = json.loads(memo_path.read_text())
    assert len(memo) == 3
    assert "a" not in memo
    assert memo.get("d") == "krea2"


def test_family_memo_rejects_same_size_rewrite_with_restored_mtime(
    tmp_path, monkeypatch,
):
    from dgx_monarch.actor import model_store as ms2

    memo_path = tmp_path / "family_memo.json"
    monkeypatch.setattr(ms2, "_FAMILY_MEMO_PATH", str(memo_path))
    model = tmp_path / "model.safetensors"
    model.write_bytes(BIG)
    original = model.stat()
    ms2.memoize_family(str(model), "krea2")

    model.write_bytes(b"B" * len(BIG))
    os.utime(
        model,
        ns=(original.st_atime_ns, original.st_mtime_ns),
    )
    rewritten = model.stat()
    assert rewritten.st_size == original.st_size
    assert rewritten.st_mtime_ns == original.st_mtime_ns
    assert rewritten.st_ctime_ns != original.st_ctime_ns
    assert ms2.memoized_family(str(model)) is None


def _key(size: int, index: int = 0, ctime: int = 5) -> str:
    """A real-shaped identity: dev:ino:size:mtime_ns:ctime_ns."""
    return f"66306:{12583041 + index}:{size}:{1754300000000000000 + index}:{ctime}"


def _remember(memo_path, key, family, limit=1024, min_size=0, logger=None):
    from dgx_monarch.actor import store_family

    store_family.memoize_family(
        key, family, identity=lambda path: path, memo_path=str(memo_path),
        lock=threading.Lock(), limit=limit, min_size=min_size,
        size=lambda _path: 1 << 30,
        logger=logger or SimpleNamespace(warning=lambda *a, **k: None,
                                         info=lambda *a, **k: None))


def test_the_memo_refuses_a_file_too_small_to_be_a_checkpoint(tmp_path, monkeypatch):
    """No diffusion checkpoint is as small as 1 MiB, so a 77-byte fixture earns no row."""
    from dgx_monarch.actor import model_store as ms2

    memo_path = tmp_path / "family_memo.json"
    monkeypatch.setattr(ms2, "_FAMILY_MEMO_PATH", str(memo_path))
    fixture = tmp_path / "fixture.safetensors"
    fixture.write_bytes(b"x" * FIXTURE_BYTES)

    ms2.memoize_family(str(fixture), "krea2")

    assert not memo_path.exists() or json.loads(memo_path.read_text()) == {}
    assert ms2.memoized_family(str(fixture)) is None


def test_the_floor_is_off_by_default_and_writes_what_it_is_given(tmp_path):
    """With no floor set, a key that is not a stat identity is written as given;
    the three cap tests above depend on it."""
    memo_path = tmp_path / "family_memo.json"
    _remember(memo_path, "not-an-identity", "krea2")
    assert json.loads(memo_path.read_text()) == {"not-an-identity": "krea2"}


def test_a_stat_failure_inside_the_floor_still_only_warns(tmp_path):
    """A stat failure in the floor check only warns: a checkpoint replaced during
    a 105 s load must not throw at the end of a load that succeeded."""
    from dgx_monarch.actor import store_family

    warnings = []
    store_family.memoize_family(
        "/gone.safetensors", "krea2", identity=lambda path: path,
        memo_path=str(tmp_path / "family_memo.json"), lock=threading.Lock(),
        limit=8, min_size=1 << 20,
        size=lambda _path: (_ for _ in ()).throw(OSError("vanished")),
        logger=SimpleNamespace(warning=lambda *a, **k: warnings.append(a),
                               info=lambda *a, **k: None))
    assert warnings and "not persisted" in warnings[0][0]


def test_sub_floor_rows_are_dropped_before_the_cap_evicts(tmp_path):
    """The floor prune runs on every write, ahead of the early return and the cap.

    Below the cap the eviction loop never runs, and the common case is a repeat
    load of an already memoized checkpoint, which returns early. A prune behind
    either guard would leave the flood in place.
    """
    memo_path = tmp_path / "family_memo.json"
    seeded = {_key(FIXTURE_BYTES, i): "krea2" for i in range(20)}
    real = _key(24_480_000_000, 900)
    seeded[real] = "krea2"
    memo_path.write_text(json.dumps(seeded))

    # Far below the cap: the eviction loop cannot be what drops them.
    logged = []
    _remember(memo_path, _key(11_950_000_000, 901), "flux2", limit=1024,
              min_size=1 << 20,
              logger=SimpleNamespace(warning=lambda *a, **k: None,
                                     info=lambda *a, **k: logged.append(a)))
    memo = json.loads(memo_path.read_text())
    assert len(memo) == 2
    assert real in memo
    assert any("dropped 20 rows" in str(entry) or entry[1] == 20 for entry in logged)

    # A repeat write of a row already present still prunes and rewrites the file.
    memo_path.write_text(json.dumps({**seeded, real: "krea2"}))
    _remember(memo_path, real, "krea2", limit=1024, min_size=1 << 20)
    assert json.loads(memo_path.read_text()) == {real: "krea2"}


def test_the_prune_leaves_a_key_it_cannot_read_alone(tmp_path):
    memo_path = tmp_path / "family_memo.json"
    memo_path.write_text(json.dumps({
        "not-an-identity": "krea2",
        "1:2:three:4:5": "krea2",
        "1:2:3:4": "krea2",
        _key(FIXTURE_BYTES): "krea2",
    }))
    _remember(memo_path, _key(60_000_000_000, 7), "flux2", min_size=1 << 20)
    memo = json.loads(memo_path.read_text())
    assert set(memo) == {"not-an-identity", "1:2:three:4:5", "1:2:3:4",
                         _key(60_000_000_000, 7)}


def _remember_sized(memo_path, key, family, byte_size, limit, min_size):
    from dgx_monarch.actor import store_family

    store_family.memoize_family(
        key, family, identity=lambda path: path, memo_path=str(memo_path),
        lock=threading.Lock(), limit=limit, min_size=min_size,
        size=lambda _path: byte_size,
        logger=SimpleNamespace(warning=lambda *a, **k: None,
                               info=lambda *a, **k: None))


REAL_KREA2 = [_key(24_480_000_000, 900), _key(11_950_000_000, 901)]
OTHER_FAMILIES = [(_key(20_000_000_000 + i, 100 + i), f"fam{i % 14}")
                  for i in range(90)]
FIXTURES = [(_key(FIXTURE_BYTES, i), "krea2") for i in range(164)]


def _seed(memo_path, order):
    memo_path.write_text(json.dumps(dict(order)))
    assert len(json.loads(memo_path.read_text())) == 256


def test_the_reconstructed_flood_costs_a_real_row_at_the_cap_and_none_after(tmp_path):
    """The head's 2026-09-03 memo reordered: its two real krea2 rows oldest, then 90
    rows of other families, then the battery's 164 fixtures; 256 rows, the cap then.

    The cap charges the family holding the most rows, oldest row first. The
    fixtures are krea2 like the real rows, so one more fixture insert evicts a
    real row. Never fix this with a tier that makes vouched rows pay last: it
    would keep all 164 fixtures and charge the 90 other-family rows instead.
    """
    order = [(key, "krea2") for key in REAL_KREA2] + OTHER_FAMILIES + FIXTURES
    for min_size, survives in ((0, False), (1 << 20, True)):
        memo_path = tmp_path / f"reconstructed-{min_size}.json"
        _seed(memo_path, order)
        _remember_sized(memo_path, _key(FIXTURE_BYTES, 999), "krea2",
                        FIXTURE_BYTES, 256, min_size)
        memo = json.loads(memo_path.read_text())
        assert (REAL_KREA2[0] in memo) is survives
        if not survives:
            continue
        # The floor kept the fixture out, but the seeded flood stays until the
        # next real load prunes it in one pass.
        assert len(memo) == 256
        real = _key(60_020_000_000, 800)
        _remember_sized(memo_path, real, "flux2", 60_020_000_000, 256, min_size)
        memo = json.loads(memo_path.read_text())
        assert set(memo) == ({key for key, _ in OTHER_FAMILIES}
                             | set(REAL_KREA2) | {real})


def test_the_live_flood_is_one_battery_run_from_the_first_real_loss(tmp_path):
    """The order the head's memo was in on 2026-09-03: the two real krea2 rows newest.

    The fixtures pay first, so no real row was lost yet. One more battery run,
    165 inserts, would evict every seeded fixture and then, on its last insert,
    the oldest real krea2 row.
    """
    order = FIXTURES + OTHER_FAMILIES + [(key, "krea2") for key in REAL_KREA2]
    for min_size, survives in ((0, False), (1 << 20, True)):
        memo_path = tmp_path / f"live-{min_size}.json"
        _seed(memo_path, order)
        for index in range(165):
            _remember_sized(memo_path, _key(FIXTURE_BYTES, 1000 + index), "krea2",
                            FIXTURE_BYTES, 256, min_size)
        memo = json.loads(memo_path.read_text())
        assert (REAL_KREA2[0] in memo) is survives
        assert all(key in memo for key, _ in OTHER_FAMILIES)


def test_an_unparseable_memo_warns_before_it_starts_fresh(tmp_path):
    """A memo that does not parse loses every row, so starting fresh must warn
    and give the byte count."""
    memo_path = tmp_path / "family_memo.json"
    memo_path.write_bytes(b"{not json at all")
    warnings = []
    _remember(memo_path, _key(60_000_000_000), "flux2",
              logger=SimpleNamespace(warning=lambda *a, **k: warnings.append(a),
                                     info=lambda *a, **k: None))
    assert warnings and "does not parse" in warnings[0][0]
    assert warnings[0][2] == len(b"{not json at all")
    assert json.loads(memo_path.read_text()) == {_key(60_000_000_000): "flux2"}


def test_an_empty_or_missing_memo_is_not_a_loss_and_says_nothing(tmp_path):
    memo_path = tmp_path / "family_memo.json"
    warnings = []
    logger = SimpleNamespace(warning=lambda *a, **k: warnings.append(a),
                             info=lambda *a, **k: None)
    _remember(memo_path, _key(60_000_000_000), "flux2", logger=logger)
    memo_path.write_text("   ")
    _remember(memo_path, _key(60_000_000_000), "flux2", logger=logger)
    assert warnings == []


def test_the_shipped_limit_and_floor_are_the_sized_ones():
    from dgx_monarch.actor import model_store as ms2

    assert ms2._FAMILY_MEMO_LIMIT == 1024
    assert ms2._FAMILY_MEMO_MIN_BYTES == 1 << 20


def _safetensors(path, *, tensor_name="w", size=2 << 20):
    """A container with a real length prefix and header, padded past the floor."""
    header = json.dumps({tensor_name: {"dtype": "BF16", "shape": [1],
                                       "data_offsets": [0, 2]}}).encode()
    body = struct.pack("<Q", len(header)) + header + b"\0" * 2
    path.write_bytes(body + b"\0" * max(size - len(body), 0))
    return path


def _memo_calls(tmp_path, logger=None):
    """The two store_family entry points bound to one tmp memo and one lock."""
    from dgx_monarch.actor import store_family

    memo_path = str(tmp_path / "family_memo.json")
    lock = threading.Lock()
    log = logger or SimpleNamespace(warning=lambda *a, **k: None,
                                    info=lambda *a, **k: None)

    def write(path, family, key):
        store_family.memoize_family(
            str(path), family, identity=lambda _p: key, memo_path=memo_path,
            lock=lock, limit=8, logger=log, min_size=1 << 20)

    def read(path, key):
        return store_family.memoized_family(
            str(path), identity=lambda _p: key, memo_path=memo_path, lock=lock)

    return write, read, memo_path


def test_a_ctime_ghost_resolves_through_the_header_digest(tmp_path):
    """A metadata sweep rewrites ctime on every checkpoint and `file_identity`
    carries ctime, so every key misses at once although the bytes did not move.
    The header digest must still answer for bytes this box has loaded.
    """
    from dgx_monarch.actor import store_family

    model = _safetensors(tmp_path / "m.safetensors")
    write, read, memo_path = _memo_calls(tmp_path)
    write(model, "flux2", "dev:ino:size:mtime:ctime-before")
    assert read(model, "dev:ino:size:mtime:ctime-before") == "flux2"
    # The sweep: every field but ctime is unchanged, and the key misses.
    assert read(model, "dev:ino:size:mtime:ctime-after") == "flux2"
    index = json.loads(open(store_family._digest_index_path(memo_path)).read())
    assert index == {store_family.header_digest(str(model)): "flux2"}


def test_the_side_index_is_a_miss_path_only_and_never_the_first_answer(tmp_path,
                                                                      monkeypatch):
    """A key hit reads no file bytes; only a key miss pays for the digest reads."""
    from dgx_monarch.actor import store_family

    model = _safetensors(tmp_path / "m.safetensors")
    write, read, _ = _memo_calls(tmp_path)
    write(model, "krea2", "hit-key")
    calls = []
    real = store_family.header_digest
    monkeypatch.setattr(store_family, "header_digest",
                        lambda p: calls.append(p) or real(p))
    assert read(model, "hit-key") == "krea2"
    assert calls == []
    assert read(model, "missed-key") == "krea2"
    # Two reads of this box's own file: the lookup and its confirmation.
    assert calls == [str(model), str(model)]


def test_a_digest_hit_writes_no_memo_row(tmp_path):
    """A digest hit writes no memo row (2026-09-03): only a load writes one, and
    repairing the stat key from a hit would write a row no load earned
    (docs/DESIGN.md section 5.5).
    """
    model = _safetensors(tmp_path / "m.safetensors")
    write, read, memo_path = _memo_calls(tmp_path)
    write(model, "flux2", "old-key")
    before = open(memo_path).read()
    assert read(model, "new-key") == "flux2"
    assert open(memo_path).read() == before
    assert list(json.loads(before)) == ["old-key"]


def test_a_file_that_changes_under_the_lookup_resolves_to_nothing(tmp_path,
                                                                  monkeypatch):
    """A digest that changes on the confirming second read of this box's own
    file proves nothing about these bytes, so the lookup answers nothing."""
    from dgx_monarch.actor import store_family

    model = _safetensors(tmp_path / "m.safetensors")
    write, read, _ = _memo_calls(tmp_path)
    write(model, "flux2", "old-key")
    stable = store_family.header_digest(str(model))
    answers = iter([stable, "0" * 64])
    monkeypatch.setattr(store_family, "header_digest", lambda _p: next(answers))
    assert read(model, "new-key") is None


def test_a_container_with_no_safetensors_header_earns_no_index_row(tmp_path):
    """A .ckpt or truncated file has no header digest and gets no index row; its
    memo row is still written."""
    from dgx_monarch.actor import store_family

    other = tmp_path / "m.ckpt"
    other.write_bytes(b"PK\x03\x04" + b"\0" * (2 << 20))
    write, read, memo_path = _memo_calls(tmp_path)
    write(other, "flux2", "key")
    assert read(other, "key") == "flux2"
    assert read(other, "another-key") is None
    assert not os.path.exists(store_family._digest_index_path(memo_path))


def test_a_memo_this_call_could_not_read_is_not_a_key_miss(tmp_path, monkeypatch):
    """An unreadable memo is not a key miss, so it must not reach the index.

    Nobody knows whether the key was in a file that would not open, and an index
    that outlived a deleted memo would hand back the families the deletion
    retired.
    """
    from dgx_monarch.actor import store_family

    model = _safetensors(tmp_path / "m.safetensors")
    write, read, memo_path = _memo_calls(tmp_path)
    write(model, "flux2", "the-key")
    assert read(model, "another-key") == "flux2"        # the index does answer
    os.unlink(memo_path)
    assert os.path.exists(store_family._digest_index_path(memo_path))

    calls = []
    real = store_family.header_digest
    monkeypatch.setattr(store_family, "header_digest",
                        lambda p: calls.append(p) or real(p))
    assert read(model, "the-key") is None
    assert read(model, "another-key") is None
    assert calls == [], "an unreadable memo must not reach the side index"


def test_a_side_index_that_does_not_parse_says_so_in_its_own_words(tmp_path):
    """A memo that does not parse loses every row an operator earned; an index
    that does not parse loses nothing, because the next load of each checkpoint
    refills it. Its warning must not use the memo's sentence.
    """
    from dgx_monarch.actor import store_family

    warnings = []
    log = SimpleNamespace(warning=lambda fmt, *a: warnings.append(fmt % a),
                          info=lambda *a, **k: None,
                          debug=lambda *a, **k: None)
    model = _safetensors(tmp_path / "m.safetensors")
    write, _read, memo_path = _memo_calls(tmp_path, logger=log)
    with open(store_family._digest_index_path(memo_path), "w") as f:
        f.write("{not json at all")
    write(model, "flux2", "the-key")

    assert len(warnings) == 1
    text = warnings[0]
    assert "header-digest side index" in text
    assert "No memo row is affected" in text
    assert "every row it held is lost" not in text
    # The memo row still landed, and the index was rebuilt from this load.
    assert json.loads(open(memo_path).read()) == {"the-key": "flux2"}
    index = json.loads(open(store_family._digest_index_path(memo_path)).read())
    assert index == {store_family.header_digest(str(model)): "flux2"}


def test_the_side_index_evicts_on_the_same_rule_as_the_memo(tmp_path):
    """One busy family must not push every other family out of the index either."""
    from dgx_monarch.actor import store_family

    memo_path = str(tmp_path / "family_memo.json")
    lock = threading.Lock()
    log = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    for index in range(4):
        model = _safetensors(tmp_path / f"busy{index}.safetensors",
                             tensor_name=f"busy{index}")
        store_family.memoize_family(
            str(model), "krea2", identity=lambda p: p, memo_path=memo_path,
            lock=lock, limit=4, logger=log, min_size=1 << 20)
    rare = _safetensors(tmp_path / "rare.safetensors", tensor_name="rare")
    store_family.memoize_family(
        str(rare), "ltx", identity=lambda p: p, memo_path=memo_path,
        lock=lock, limit=4, logger=log, min_size=1 << 20)

    index = json.loads(open(store_family._digest_index_path(memo_path)).read())
    assert len(index) == 4
    assert index[store_family.header_digest(str(rare))] == "ltx"
    assert sorted(index.values()) == ["krea2", "krea2", "krea2", "ltx"]


def test_a_failed_side_index_write_does_not_cost_the_memo_its_row(tmp_path,
                                                                  monkeypatch):
    """The index is best effort; the memo row is the thing the load earned."""
    from dgx_monarch.actor import store_family

    def boom(*_a, **_k):
        raise OSError("read-only file system")

    monkeypatch.setattr(store_family, "_memoize_digest", boom)
    model = _safetensors(tmp_path / "m.safetensors")
    write, read, memo_path = _memo_calls(tmp_path)
    write(model, "flux2", "the-key")
    assert json.loads(open(memo_path).read()) == {"the-key": "flux2"}
    assert read(model, "the-key") == "flux2"


def _sweep_calls(tmp_path):
    """The two entry points bound to one real file and real-shaped stat keys.

    The ghost rule parses device, inode, size and mtime out of the key itself,
    so these tests use the key shape a worker writes, not free-form keys.
    """
    from dgx_monarch.actor import store_family

    memo_path = str(tmp_path / "family_memo.json")
    lock = threading.Lock()
    lines: list[str] = []
    log = SimpleNamespace(info=lambda fmt, *a: lines.append(fmt % a),
                          warning=lambda fmt, *a: lines.append(fmt % a),
                          debug=lambda *a, **k: None)

    def write(path, family, key):
        store_family.memoize_family(
            str(path), family, identity=lambda _p: key, memo_path=memo_path,
            lock=lock, limit=8, logger=log, min_size=1 << 20)

    def read(path, key):
        return store_family.memoized_family(
            str(path), identity=lambda _p: key, memo_path=memo_path, lock=lock)

    return write, read, memo_path, lines


def _resolves(lines):
    return [line for line in lines if "the header digest resolved it" in line]


def test_a_run_that_never_reads_the_memo_still_keeps_one_row_per_file(tmp_path):
    """A render asking for slab residency outright needs no family, so the ladder
    answers above the memo lookup and the load's write is its only memo traffic:
    that write must retire the rows a sweep ghosted. Rebuilds the 2026-09-06
    finding: four renders of one flux2 file across sweeps left four rows.
    """
    model = _safetensors(tmp_path / "flux2.safetensors")
    write, _read, memo_path, lines = _sweep_calls(tmp_path)
    size = 64_446_596_128

    write(model, "flux2", _key(size, ctime=1))
    assert _resolves(lines) == [], "a first load resolves nothing"
    for ctime in (2, 3, 4):
        write(model, "flux2", _key(size, ctime=ctime))
        assert list(json.loads(open(memo_path).read())) == [_key(size, ctime=ctime)]

    resolved = _resolves(lines)
    assert len(resolved) == 3
    assert "the identity key missed for flux2.safetensors" in resolved[0]
    assert "dropped 1 stale rows" in resolved[0]


def test_the_lookup_resolves_the_moved_key_and_the_load_retires_the_ghost(tmp_path):
    """When the ladder does read the memo, the lookup answers from the index and
    writes nothing, since only a load writes a row; the load that follows
    retires the sweep's row.
    """
    model = _safetensors(tmp_path / "flux2.safetensors")
    write, read, memo_path, lines = _sweep_calls(tmp_path)
    size = 64_446_596_128

    write(model, "flux2", _key(size, ctime=1))
    before = open(memo_path).read()
    assert read(model, _key(size, ctime=2)) == "flux2"
    assert open(memo_path).read() == before

    write(model, "flux2", _key(size, ctime=2))
    assert json.loads(open(memo_path).read()) == {_key(size, ctime=2): "flux2"}
    assert len(_resolves(lines)) == 1


def test_only_this_file_s_own_ghosts_are_retired(tmp_path):
    """A ghost matches device, inode, size and mtime and names the family this
    load detected. Another file's rows stay, and so does a same-file row naming
    another family: that is a disagreement about these bytes, not a sweep's
    leftover.
    """
    model = _safetensors(tmp_path / "flux2.safetensors")
    write, _read, memo_path, _lines = _sweep_calls(tmp_path)
    size = 64_446_596_128
    write(model, "flux2", _key(size, ctime=1))
    seeded = json.loads(open(memo_path).read())
    seeded[_key(size, ctime=2)] = "chroma"
    seeded[_key(size, index=7, ctime=1)] = "flux2"
    with open(memo_path, "w") as f:
        json.dump(seeded, f)

    write(model, "flux2", _key(size, ctime=3))

    memo = json.loads(open(memo_path).read())
    assert memo == {_key(size, ctime=2): "chroma",
                    _key(size, index=7, ctime=1): "flux2",
                    _key(size, ctime=3): "flux2"}


def test_an_index_naming_another_family_keeps_every_older_row(tmp_path):
    """An index naming another family means the header changed under a sweep, so
    this load cannot speak for the older rows: it warns and adds its own row
    beside them.
    """
    from dgx_monarch.actor import store_family

    model = _safetensors(tmp_path / "flux2.safetensors")
    write, _read, memo_path, lines = _sweep_calls(tmp_path)
    size = 64_446_596_128
    with open(memo_path, "w") as f:
        json.dump({_key(size, ctime=1): "flux2"}, f)
    with open(store_family._digest_index_path(memo_path), "w") as f:
        json.dump({store_family.header_digest(str(model)): "krea2"}, f)

    write(model, "flux2", _key(size, ctime=2))

    assert json.loads(open(memo_path).read()) == {_key(size, ctime=1): "flux2",
                                                  _key(size, ctime=2): "flux2"}
    assert _resolves(lines) == []
    assert [line for line in lines if "names krea2 where this load detected flux2"]


def test_a_memo_older_than_the_index_heals_on_the_load_after_it(tmp_path):
    """A row written before the index existed has no digest to resolve by. Its
    next load writes its own row and the index row beside it, so the load after
    the next sweep retires both earlier keys at once.
    """
    model = _safetensors(tmp_path / "flux2.safetensors")
    write, _read, memo_path, lines = _sweep_calls(tmp_path)
    size = 64_446_596_128
    with open(memo_path, "w") as f:
        json.dump({_key(size, ctime=1): "flux2"}, f)

    write(model, "flux2", _key(size, ctime=2))
    assert len(json.loads(open(memo_path).read())) == 2
    assert _resolves(lines) == []

    write(model, "flux2", _key(size, ctime=3))
    assert json.loads(open(memo_path).read()) == {_key(size, ctime=3): "flux2"}
    assert "dropped 2 stale rows" in _resolves(lines)[0]


def test_a_moved_ctime_costs_the_shipped_memo_no_row(tmp_path, monkeypatch):
    """End to end through the model_store entry points a worker calls: after a
    real ctime move with mtime held, the lookup answers and the write leaves one row.
    """
    from dgx_monarch.actor import model_store as ms2

    memo_path = tmp_path / "family_memo.json"
    monkeypatch.setattr(ms2, "_FAMILY_MEMO_PATH", str(memo_path))
    model = _safetensors(tmp_path / "flux2.safetensors")
    ms2.memoize_family(str(model), "flux2")
    before = json.loads(memo_path.read_text())

    stat = os.stat(model)
    os.utime(model, ns=(stat.st_atime_ns + 10 ** 9, stat.st_mtime_ns))
    moved = os.stat(model)
    assert moved.st_ctime_ns != stat.st_ctime_ns
    assert moved.st_mtime_ns == stat.st_mtime_ns

    assert ms2.memoized_family(str(model)) == "flux2"
    ms2.memoize_family(str(model), "flux2")

    memo = json.loads(memo_path.read_text())
    assert len(memo) == len(before) == 1
    assert list(memo) != list(before)
