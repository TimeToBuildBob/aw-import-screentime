"""Bug-hunt reproductions (each test fails on master 1297039)."""

import os
import struct
import zlib
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import aw_import_screentime.__main__ as mod

BASE_CF = 780_000_000.0  # ~2025-09


def pb(bundle, cf=None, fg=True):
    ev = mod.AppInFocusEventPb()
    ev.bundle_id = bundle
    ev.in_foreground = 1 if fg else 0
    if cf is not None:
        ev.cf_absolute_time = cf
    return ev


def write_segb2(path: Path, entries):
    """entries: list of (state, payload_bytes). Minimal SEGB v2 writer."""
    body = b""
    trailer = b""
    for state, payload in entries:
        body += struct.pack("<Ii", zlib.crc32(payload), 0) + payload
        end_offset = len(body)
        trailer += struct.pack("<2id", end_offset, state, BASE_CF)
        if len(body) % 4:
            body += b"\0" * (4 - len(body) % 4)
    header = struct.pack("<4sid16s", b"SEGB", len(entries), BASE_CF, b"\0" * 16)
    path.write_bytes(header + body + trailer)


# 1. proto3 default: a record with no cf_absolute_time reads back as 0.0, not None,
#    so the `cf_absolute_time is None` guard never fires and it is treated as
#    2001-01-01T00:00:00Z.
def test_record_without_timestamp_does_not_produce_decades_long_interval():
    events = [
        pb("com.a"),  # cf_absolute_time missing -> proto3 default 0.0
        pb("com.b", BASE_CF, True),
        pb("com.b", BASE_CF + 60, False),
    ]
    assert events[0].cf_absolute_time == 0.0  # sanity: not None
    stitched = list(mod.stitch_intervals(events, tzinfo=mod.UTC))
    longest = max(e.duration for e in stitched)
    assert longest <= timedelta(days=1), (
        f"record without a timestamp produced {stitched[0].data['app']} "
        f"from {stitched[0].timestamp.isoformat()} lasting {longest}"
    )


def test_watch_reader_drops_record_without_timestamp(monkeypatch, tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"x")
    monkeypatch.setattr(mod, "tail_device_files", lambda dev, limit: [f])
    monkeypatch.setattr(
        mod,
        "iter_app_in_focus_events",
        lambda p: iter([pb("com.a"), pb("com.b", BASE_CF, True)]),
    )
    res = mod.read_new_events_for_device("dev", mod.DeviceState())
    assert all(e.cf_absolute_time > 0 for e in res.events)


# 2. SEGB entry state is ignored: entries marked Deleted (state 3) are imported.
def test_deleted_segb_entries_are_not_imported(tmp_path):
    p = tmp_path / "stream"
    write_segb2(
        p,
        [
            (1, pb("com.kept", BASE_CF, True).SerializeToString()),
            (3, pb("com.deleted", BASE_CF + 10, True).SerializeToString()),
        ],
    )
    bundles = [e.bundle_id for e in mod.iter_app_in_focus_events(p)]
    assert "com.kept" in bundles  # sanity: writer/reader agree
    assert "com.deleted" not in bundles


# 3. `events import`/`preview` stitch records in file-mtime order and never sort
#    by event time (watch does sort). If an older file's mtime is bumped
#    (Biome rewriting/pruning it, a restore/copy), intervals are lost.
def test_import_stitches_in_event_time_order_not_mtime_order(monkeypatch, tmp_path):
    streams = tmp_path / "streams"
    d = streams / "dev"
    d.mkdir(parents=True)
    old_content = d / "100"
    new_content = d / "200"
    for p in (old_content, new_content):
        p.write_bytes(b"x")
    # older content touched later
    os.utime(new_content, (1_000_000, 1_000_000))
    os.utime(old_content, (2_000_000, 2_000_000))
    by_path = {
        old_content: [SimpleNamespace(bundle_id="com.a", cf_absolute_time=BASE_CF, in_foreground=True)],
        new_content: [
            SimpleNamespace(bundle_id="com.a", cf_absolute_time=BASE_CF + 600, in_foreground=False),
            SimpleNamespace(bundle_id="com.b", cf_absolute_time=BASE_CF + 600, in_foreground=True),
            SimpleNamespace(bundle_id="com.b", cf_absolute_time=BASE_CF + 900, in_foreground=False),
        ],
    }
    monkeypatch.setattr(mod, "STREAMS_DIR", streams)
    monkeypatch.setattr(mod, "iter_app_in_focus_events", lambda p: iter(by_path[Path(p)]))
    monkeypatch.setattr(mod, "enrich_events_with_titles", lambda *a, **k: None)

    files = mod.tail_device_files("dev", limit=5)
    events = mod.build_stitched_events_for_files(
        files, tzinfo=mod.UTC, since=None, storefronts=["us"]
    )
    apps = {e.data["app"]: e.duration.total_seconds() for e in events}
    assert apps.get("com.a") == 600.0, f"com.a's 10 minutes lost: {apps}"


# 4. Watch watermark uses `cf <= last_cf`, so a record with the same timestamp as
#    the last-read record that lands in a later write (app switch: A loses focus
#    and B gains focus at the same instant) is dropped and B's session vanishes.
def test_watch_keeps_record_with_same_timestamp_as_watermark(monkeypatch, tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"x")
    records = [SimpleNamespace(bundle_id="com.a", cf_absolute_time=BASE_CF, in_foreground=True),
               SimpleNamespace(bundle_id="com.a", cf_absolute_time=BASE_CF + 100, in_foreground=False)]
    monkeypatch.setattr(mod, "tail_device_files", lambda dev, limit: [f])
    monkeypatch.setattr(mod, "iter_app_in_focus_events", lambda p: iter(list(records)))

    state = mod.DeviceState()
    res = mod.read_new_events_for_device("dev", state)
    _, open_iv = mod.stitch_intervals_with_state(res.events, tzinfo=mod.UTC)
    state.last_file, state.last_cf, state.open_interval = res.new_last_file, res.new_last_cf, open_iv

    # Biome then appends B's gain (same instant) and later B's loss.
    records += [SimpleNamespace(bundle_id="com.b", cf_absolute_time=BASE_CF + 100, in_foreground=True),
                SimpleNamespace(bundle_id="com.b", cf_absolute_time=BASE_CF + 400, in_foreground=False)]
    res = mod.read_new_events_for_device("dev", state)
    stitched, _ = mod.stitch_intervals_with_state(
        res.events, tzinfo=mod.UTC, initial_open_interval=state.open_interval
    )
    assert [e.data["app"] for e in stitched] == ["com.b"], "com.b's 5 minutes lost"


# 5. One unreadable/partial file (e.g. a 0-byte file Biome is just creating)
#    aborts `events import` for every device instead of being skipped.
def test_import_survives_unreadable_file(monkeypatch, tmp_path):
    streams = tmp_path / "streams"
    d = streams / "dev"
    d.mkdir(parents=True)
    good = d / "100"
    write_segb2(
        good,
        [
            (1, pb("com.a", BASE_CF, True).SerializeToString()),
            (1, pb("com.a", BASE_CF + 60, False).SerializeToString()),
        ],
    )
    os.utime(good, (1_000_000, 1_000_000))
    (d / "200").write_bytes(b"")  # newest file, still being created
    monkeypatch.setattr(mod, "STREAMS_DIR", streams)
    monkeypatch.setattr(mod, "enrich_events_with_titles", lambda *a, **k: None)
    files = mod.tail_device_files("dev", limit=5)
    events = mod.build_stitched_events_for_files(
        files, tzinfo=mod.UTC, since=None, storefronts=["us"]
    )
    assert [e.data["app"] for e in events] == ["com.a"]
