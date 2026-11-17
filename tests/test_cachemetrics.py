"""Host-only regressions for the GLM published cache telemetry."""
from __future__ import annotations

import collections
import threading
from types import SimpleNamespace

import pytest

cachemetrics = pytest.importorskip("tensorfold.families.glm5_next.cuda.cachemetrics")
sessions = pytest.importorskip("tensorfold.families.glm5_next.cuda.sessions")

kvpool = pytest.importorskip("tensorfold.families.glm5_next.cuda.kvpool")


class _Seq:
    def __init__(self, *, prefill: bool, done: int, emitted: int = 0) -> None:
        self.stepper = None if prefill else object()
        self.done = done
        self.emitted = emitted


def _batch(*, pool=True):
    engine = SimpleNamespace(limit=1024, store=None)
    if not pool:
        return engine, SimpleNamespace(seqs=[_Seq(prefill=True, done=32)], queue=collections.deque(), states=[],
                                       kvp=None, n=1)
    book = kvpool.PoolBook(10, 256)
    active, idle = kvpool.SlotPages(book, 1024), kvpool.SlotPages(book, 1024)
    book.slots.extend((active, idle))
    active.reserve(4)
    active.ensure(512)
    idle.ensure(256)
    batch = SimpleNamespace(seqs=[_Seq(prefill=True, done=512), None], queue=collections.deque([object()]),
                            states=[SimpleNamespace(pages=active, pos=512), SimpleNamespace(pages=idle, pos=0)],
                            kvp=book, n=2)
    return engine, batch


def test_pool_snapshot_separates_physical_pages_reservations_and_idle_pages():
    engine, batch = _batch()
    metrics = cachemetrics.CacheMetrics(engine)
    metrics.publish_batch(batch)
    snap = metrics.snapshot()
    assert snap["pool_tokens"] == snap["pool_used_tokens"] + snap["pool_free_tokens"] == 10 * 256
    assert snap["pool_used_tokens"] == 3 * 256
    assert snap["pool_reserved_tokens"] == 2 * 256
    assert snap["pool_available_tokens"] == 5 * 256
    assert snap["pool_kept_tokens"] == 256
    metrics.publish_waiting(2)
    queued = metrics.snapshot()
    assert queued["requests_waiting"] == 2
    assert queued["pool_used_tokens"] == snap["pool_used_tokens"]
    assert snap["requests_waiting"] == 1 and snap["requests_running"] == 1
    assert snap["streams"] == {"decoding": 0, "prefilling": 1, "filling": 0, "paused": 0, "max": 2}
    assert snap["stream_kv_usage_ratio"] == 0.5


def test_scheduler_publish_cannot_overwrite_a_queue_update_with_a_stale_count():
    class Gate:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.allow = threading.Event()

        def __enter__(self):
            self.entered.set()
            assert self.allow.wait(1)
            return self

        def __exit__(self, *_):
            return False

    engine, batch = _batch(pool=False)
    batch.queue.clear()
    batch.cv = Gate()
    metrics = cachemetrics.CacheMetrics(engine)
    errors = []

    def publish() -> None:
        try:
            metrics.publish_batch(batch)
        except BaseException as exc:  # test-thread failures must not be lost
            errors.append(exc)

    producer = threading.Thread(target=publish)
    producer.start()
    assert batch.cv.entered.wait(1)
    batch.queue.append(object())
    metrics.publish_waiting(len(batch.queue))
    batch.cv.allow.set()
    producer.join()
    assert not errors
    assert metrics.snapshot()["requests_waiting"] == 1


def test_published_snapshots_are_detached_and_safe_during_republication():
    engine, batch = _batch()
    metrics = cachemetrics.CacheMetrics(engine)
    seen = []

    def publish() -> None:
        for _ in range(100):
            metrics.publish_batch(batch)
            seen.append(metrics.snapshot())

    threads = [threading.Thread(target=publish) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    seen[0]["streams"]["max"] = -1
    seen[0]["cache_restores"]["none"]["requests"] = -1
    assert metrics.snapshot()["streams"]["max"] == 2
    assert metrics.snapshot()["cache_restores"]["none"]["requests"] == 0


def test_restore_sources_and_absent_capabilities_are_honest():
    engine, batch = _batch(pool=False)
    metrics = cachemetrics.CacheMetrics(engine)
    for source, tokens in (("none", 0), ("slot", 12), ("ram", 24), ("disk", 36)):
        metrics.record_restore(source, tokens)
    metrics.publish_batch(batch)
    snap = metrics.snapshot()
    assert snap["cache_restores"] == {
        "none": {"requests": 1, "tokens": 0}, "slot": {"requests": 1, "tokens": 12},
        "ram": {"requests": 1, "tokens": 24}, "disk": {"requests": 1, "tokens": 36},
    }
    assert not {"pool_tokens", "kept_prompts", "spill"} & snap.keys()


def test_session_index_reports_budget_and_allocated_usage_without_pool_aliasing():
    index = sessions.SessionIndex(4096, (1, 1, 0, 0))
    engine = SimpleNamespace(limit=1024, store=SimpleNamespace(index=index, disk=None))
    metrics = cachemetrics.CacheMetrics(engine)
    metrics.publish_lone()
    snap = metrics.snapshot()
    assert snap["kept_prompts"] == 0
    assert snap["kept_bytes"] == snap["session_used_bytes"] == 0
    assert snap["session_budget_bytes"] == 4096


def test_lone_mode_publishes_committed_context_and_live_generation():
    engine = SimpleNamespace(limit=1000, store=None)
    metrics = cachemetrics.CacheMetrics(engine)
    metrics.set_lone("decoding", prefill_done=250, generation=7)
    snap = metrics.snapshot()
    assert snap["requests_running"] == 1
    assert snap["generation_tokens_running"] == 7
    assert snap["stream_kv_usage_ratio"] == 0.25
    metrics.set_lone(None, prefill_done=0, generation=0)
    assert metrics.snapshot()["requests_running"] == metrics.snapshot()["generation_tokens_running"] == 0


def test_snapshot_refreshes_disk_owner_view_without_republishing_cache_state():
    class Disk:
        def __init__(self) -> None:
            self.written = 0

        def telemetry_snapshot(self):
            return {"written": self.written}

    disk = Disk()
    engine = SimpleNamespace(limit=64, store=SimpleNamespace(index=None, disk=disk))
    metrics = cachemetrics.CacheMetrics(engine)
    disk.written = 3
    assert metrics.snapshot()["spill"] == {"written": 3}
