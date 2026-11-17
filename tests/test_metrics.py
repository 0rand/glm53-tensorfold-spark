"""Native CUDA request-metrics behavior."""
from __future__ import annotations

import re

import pytest

metrics_mod = pytest.importorskip("tensorfold.cuda.metrics")
health_mod = pytest.importorskip("tensorfold.cuda.health")
from tensorfold.cuda.health import Health  # noqa: E402
from tensorfold.cuda.metrics import RequestMetrics  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _histogram(text: str, name: str) -> tuple[list[int], int, float]:
    buckets = [int(value) for value in re.findall(
        rf"^tensorfold:{re.escape(name)}_bucket\{{le=\"[^\"]+\"\}} (\d+)$", text, re.M)]
    count = int(re.search(rf"^tensorfold:{re.escape(name)}_count (\d+)$", text, re.M).group(1))
    total = float(re.search(rf"^tensorfold:{re.escape(name)}_sum ([^\n]+)$", text, re.M).group(1))
    return buckets, count, total


def _render(metrics: RequestMetrics, health: Health, view=None) -> str:
    return metrics.render(served='model\\name\n"quoted"', health=health, view=view or {})


def test_terminal_accounting_is_once_and_escapes_labels():
    clock = Clock()
    health = Health(mode="basic", stall_s=0, prefill_tps=200, clock=clock)
    metrics = RequestMetrics(clock=clock)
    ticket = metrics.begin()
    clock.now = 2
    ticket.finish(500, "error")
    ticket.finish(200, "success")
    text = _render(metrics, health)
    assert 'tensorfold:requests_total{status="500",outcome="error"} 1' in text
    assert 'status="200"' not in text
    # Model labels are accepted on the legacy families without making invalid lines.
    assert 'model="model\\\\name\\n\\"quoted\\""' in health.metrics('model\\name\n"quoted"')


def test_failed_cancelled_and_zero_token_requests_omit_optional_token_phases():
    clock = Clock()
    health = Health(mode="basic", stall_s=0, prefill_tps=200, clock=clock)
    metrics = RequestMetrics(clock=clock)
    failed = metrics.begin()
    failed.engine_stats({"prefill_s": 0, "decode_s": 0})
    failed.finish(500, "error")
    cancelled = metrics.begin()
    cancelled.finish(499, "cancelled", disconnected=True)
    text = _render(metrics, health)
    assert 'tensorfold:requests_total{status="500",outcome="error"} 1' in text
    assert 'tensorfold:requests_total{status="499",outcome="cancelled"} 1' in text
    assert "tensorfold:client_disconnections_total 1" in text
    assert _histogram(text, "time_to_first_token_seconds")[1] == 0
    assert _histogram(text, "request_time_per_output_token_seconds")[1] == 0
    assert _histogram(text, "request_prefill_seconds")[1] == 0
    assert _histogram(text, "request_decode_seconds")[1] == 0


def test_finished_totals_exclude_active_engine_work_and_count_prefilled_tokens():
    clock = Clock()
    health = Health(mode="basic", stall_s=0, prefill_tps=200, clock=clock)
    request = health.begin(7)
    health.progress(request, 3)
    metrics = RequestMetrics(clock=clock)
    ticket = metrics.begin()
    ticket.finish(200, "success")
    active = _render(metrics, health)
    assert health.totals_now()["completion_tokens_total"] == 3
    assert "tensorfold:prompt_tokens_total 0" in active
    assert "tensorfold:generation_tokens_total 0" in active
    health.end(request, stats={"cached": 2, "prefill_s": 0.25}, completion_tokens=3)
    text = _render(metrics, health)
    assert "tensorfold:prompt_tokens_total 7" in text
    assert "tensorfold:generation_tokens_total 3" in text
    assert "tensorfold:prefilled_tokens_total 5" in text


def test_histograms_are_cumulative_and_include_long_duration_bucket():
    clock = Clock()
    health = Health(mode="basic", stall_s=0, prefill_tps=200, clock=clock)
    metrics = RequestMetrics(clock=clock)
    first = metrics.begin()
    clock.now = 0.1
    first.finish(200, "success")
    second = metrics.begin()
    clock.now = 901.0
    second.finish(200, "success")
    buckets, count, total = _histogram(_render(metrics, health), "request_latency_seconds")
    assert buckets == sorted(buckets)
    assert buckets[-2] == count == 2  # the long observation reaches an extended finite bucket, not +Inf
    assert buckets[-1] == count
    assert total == 901.0
