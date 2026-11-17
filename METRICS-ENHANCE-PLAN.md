# Metrics enhancement plan

Status: implemented in patch 0710 and the portable EngineMetrics consumer patch; CPU runtime verified. Production deployment and production-model GPU verification require a separately approved maintenance window.

Recorded: 2026-11-17.

## Goal and scope

Port the useful monitoring contract from the DeepSeek TensorFold-ZIG stack into this GLM TensorFold stack, using Gilesco as a secondary reference for Python exposition and request instrumentation.

The implementation must deliver:

- EngineMetrics-compatible native `tensorfold:*` metrics.
- Matching cache/session fields in `/health`.
- Reliable instrumentation for this stack's shared KV pool, RAM session store and SSD session tier.
- Preservation of the existing `tensorfold_*` operational metrics used by the watchdog and GPU watcher.

The operator reports that current cache-related metrics are nonfunctional and no SSD-cache metrics are available. Treat that as the baseline. Existing internal fields and nonzero counters are not evidence that cache monitoring works.

Port the monitoring contract, not another engine's cache implementation. Do not change production inference, memory, networking or caching settings as part of this work.

## Investigation evidence

### Current stack

The image is built from `vendor/TensorFold` plus the numbered patches, applied in order by `docker/Dockerfile`. The unpatched vendor tree is not the effective serving implementation.

- [`patches/0150-glm-mia-wins.patch`](patches/0150-glm-mia-wins.patch) implements the current `/metrics` endpoint and `Health`/`TrackedEngine` bookkeeping.
- [`patches/0600-glm-upstream-ports.patch`](patches/0600-glm-upstream-ports.patch) adds live completion-token accounting and stream information to `/health`, while deliberately leaving `/metrics` unchanged.
- [`scripts/serve.sh`](scripts/serve.sh) consumes the existing completion-token, request and decode-round counters for drafter-health checks.
- [`scripts/gpuwatch.py`](scripts/gpuwatch.py) consumes decode seconds/rounds, inflight requests and uptime.

Read-only inspection observed a running rank-0 container and HTTP 200 responses from `http://127.0.0.1:8100/metrics` and `/health`. `/metrics` exposed 13 underscore-prefixed operational families. `/health` included live completion tokens, stream counts and the context limit.

The investigation did not submit inference requests, exercise caches, restart containers or change configuration. It did not establish that cache telemetry works.

### EngineMetrics consumer

The inspected local launcher is `$HOME/vllmdash.sh`. It selects `$HOME/Development/src/EngineMetrics`, not the older checkout under `$HOME/projects/EngineMetrics`.

The active [`engines/tensorfold.py`](../../Development/src/EngineMetrics/engines/tensorfold.py) consumes both:

- Native `tensorfold:*` counters, histogram sums/counts and live gauges from `/metrics`.
- Token/timing totals, stream state, pool capacity/free space, retained-session information and the `spill` object from `/health`.

Adding Prometheus names alone would leave the cache and SSD panels incomplete.

The consumer-side fix retaining `tensorfold:kv_cache_usage_ratio`, `tensorfold:requests_running` and `tensorfold:requests_waiting` is already installed. Do not reapply it without checking the target checkout.

### Primary reference: DeepSeek TensorFold-ZIG

Local checkout: `../deepseek-v41-zig-pr65`.

Relevant sources:

- [`docs/API.md`](../deepseek-v41-zig-pr65/docs/API.md): monitoring contract and cache semantics.
- [`engine/zig/zig/src/server/metrics.zig`](../deepseek-v41-zig-pr65/engine/zig/zig/src/server/metrics.zig): native metric families, types and histogram rendering.
- [`engine/zig/zig/src/server/status_routes.zig`](../deepseek-v41-zig-pr65/engine/zig/zig/src/server/status_routes.zig): `/health` cache and spill projection.
- [`engine/zig/zig/src/core/engine_api.zig`](../deepseek-v41-zig-pr65/engine/zig/zig/src/core/engine_api.zig): typed `CacheStatus`, `SpillStatus` and CPU-only snapshot interfaces.
- [`engine/zig/zig/src/core/lane_host.zig`](../deepseek-v41-zig-pr65/engine/zig/zig/src/core/lane_host.zig): engine-thread snapshot publication.
- [`tools/enginemetrics-gauges.patch`](../deepseek-v41-zig-pr65/tools/enginemetrics-gauges.patch): portable consumer-side gauge fix.

Its strongest reusable design is an engine-thread-published CPU snapshot. HTTP readers consume the published copy rather than walking mutable cache structures or touching CUDA.

### Secondary reference: Gilesco

Local checkout: `../Gilesco-GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`.

Relevant sources:

- [`patches/0045-cuda-metrics.patch`](../Gilesco-GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/patches/0045-cuda-metrics.patch): Python exposition, TensorFold histogram conventions, health projection and broken-pipe handling.
- [`patches/0088-glm-spill-tier.patch`](../Gilesco-GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/patches/0088-glm-spill-tier.patch): historical spill telemetry contract.
- [`REBASE-NOTES.md`](../Gilesco-GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/REBASE-NOTES.md): upstream metrics and local health integration.

Use these as implementation guidance, not proof of current deployment capabilities. Prefer coherent published snapshots over Gilesco's retry-and-default reads of mutable tables. Do not transplant its different spill implementation or dynamic metric-type inference.

## Recommended viable metric set

Names below use the `tensorfold:` prefix unless explicitly identified as existing operational metrics.

| Area | Metrics or fields | Implementation decision |
|---|---|---|
| Request/token accounting | `prompt_tokens_total`, `generation_tokens_total`, `decode_rounds_total`, `mtp_drafted_total`, `mtp_accepted_total` | Connect the native contract to the current lifecycle. Distinguish finished-request totals from live tokens. |
| Scheduler activity | `requests_running`, `requests_waiting`, `generation_tokens_running`; `/health.streams` and `context_length` | Read actual admitted/queued jobs. HTTP inflight is not a substitute for scheduler running work. |
| Request latency | Histograms: `request_latency_seconds`, `time_to_first_token_seconds`, `request_prefill_seconds`, `request_decode_seconds`, `request_time_per_output_token_seconds` | Record only phases that occurred. Extend long-duration buckets beyond the reference's 300-second ceiling for long-context workloads. |
| Shared KV pool | `pool_tokens`, `pool_free_tokens`, `pool_used_tokens`, `kv_pool_usage_ratio` | New instrumentation from the actual pool allocator, not prompt counts or the context-window setting. |
| Admission reservations | Reserved-but-unmapped tokens and admission-available tokens | GLM-specific additions: physically free pages can already be promised to admitted requests. |
| Live context usage | `kv_cache_usage_ratio{pool="0"}` | Preserve the Zig meaning: maximum completed live-stream position divided by the context limit. Not shared-pool occupancy. |
| RAM sessions | `kept_prompts`, `kept_bytes`, explicit session-budget/usage fields; restored requests/tokens classified by source | Separate stored RAM snapshots, idle-slot KV pages and SSD entries. |
| SSD sessions | Capacity, entries, stored bytes, filesystem free bytes; completed reads/writes and bytes; restored tokens; restore latency; failure counters; pending writes | New lifecycle instrumentation, exposed through `/health.spill` and matching Prometheus families. |
| Operational health | Existing `tensorfold_*` request, error, fatal, stall and uptime families | Preserve their names and accounting for existing operational consumers. |

### SSD contract

Useful EngineMetrics-compatible `/health.spill` fields:

- `gib`, `entries`, `stored_bytes`, `disk_free_bytes`.
- `written`, `written_bytes`.
- `loaded`, `loaded_bytes`, `loaded_tokens`.
- Restore latency statistics, including the consumer's p50/p95 display where supported.
- Write/load/checksum failure counts.
- Pending I/O as an additional operational signal.

Use explicit metric types and units. Prometheus durations should be seconds; the existing consumer's `restore_ms_p50` and `restore_ms_p95` JSON fields are milliseconds. A bounded restore-duration histogram can provide aggregatable Prometheus latency data; document the observation window of any JSON percentile gauges.

Current internal hooks are implementation inputs, not working exported telemetry:

- `DiskTier.stats` tracks successful reads/writes, bytes and aggregate timing.
- `_Writer` owns queued work and asynchronous failures.
- `DiskTier.finish_load()` owns the local read outcome.
- `SessionStore.restore_disk()` participates in the both-rank restore verdict.

These require instrumentation and synchronization. Forwarding the last request's `stats["disk"]` is insufficient.

## Semantic constraints and exclusions

### Keep three cache quantities separate

1. Mapped pool pages: positions assigned to slot page tables.
2. Outstanding reservations: promised positions not yet mapped.
3. Live context progress: completed positions of active requests.

`PoolBook` already distinguishes physical free pages from admission-available pages. Export that distinction.

For coherent snapshots, mapped plus physically free pages must equal pool capacity. Admission-available pages must also account for outstanding reservations.

Read actual runtime capacity from the pool. Do not derive it from the context limit or hard-code the production example's capacity.

The pool's backing allocation is preallocated. Increasing page occupancy does not necessarily increase CUDA allocation by the same amount.

### Do not copy DeepSeek-specific memory assumptions

Compressed-KV, indexer, display-reservation and SWA accounting are not directly interchangeable with GLM's latent-FP8 layout.

Per-rank KV backing-byte metrics are reasonable only after deriving them from this engine's actual, deduplicated allocations. They are not required for the first viable EngineMetrics set.

Do not port macOS `process_footprint_bytes` semantics to Linux/GB10.

### SSD accounting must match this tier

The current tier shares page files between entries and writes asynchronously:

- Index admission is not a completed disk write.
- Pending entries must not appear as committed stored entries.
- Budget-accounting bytes and actual local file bytes must be distinguished.
- Logical restored tokens are not bytes read; already-present pages can avoid I/O.
- Rank-0 read success is not necessarily a usable distributed restore.
- Shared page files must be counted once.
- Missing or unsupported telemetry must not be represented as a healthy empty tier.

The current integrity checks use SHA-256. Do not label those failures as CRC failures merely to copy `crc_failed`. Expose checksum failures honestly and adjust the consumer label/mapping where necessary.

Physical SSD values in the first viable set describe rank-0-local storage, not cluster totals. Usable restored-token accounting must follow the existing both-rank verdict. Worker physical disk utilization is outside this minimum endpoint contract and should not be inferred from head-node values.

### Define timing and drafting precisely

- Start request timing before validation/tokenization/image preparation, with the exact HTTP-handler boundary documented.
- TTFT ends at the first generated token, not an initial SSE role event.
- Omit TTFT if no token was generated.
- Omit time-per-output-token observations when fewer than two tokens were generated.
- Request-prefill duration can include scheduling delays; do not label it pure GPU compute time.
- Summed per-request decode rounds are not physical batch rounds: one shared forward can advance several requests.
- Draft acceptance is accepted draft tokens divided by proposed draft tokens. Tokens per round is a different quantity.
- Count terminal HTTP status separately from engine outcome; a streaming request can fail after HTTP 200.
- Count client departures separately and finalize accounting once.

### Avoid unnecessary metric duplication

Keep the two surfaces with actual consumers:

- Native `tensorfold:*` for EngineMetrics.
- Existing `tensorfold_*` for operational tooling.

Do not add wholesale `tensorfold_health:*` mirrors, vLLM-style duplicate families or API-key labels just because references contain them. The active TensorFold adapter does not require them.

Use bounded labels. Do not label metrics with request IDs, prompts, session IDs, arbitrary exception messages or filesystem paths.

## Implementation sequence

### 1. Define the contract

Create a field-by-field implementation mapping containing:

- Metric name, type and unit.
- Request-completion versus live accounting.
- Logical engine-wide versus rank-local scope.
- Capability and absence behavior.
- Matching `/health` field consumed by EngineMetrics.
- Authoritative lifecycle owner.

Keep operational counter semantics unchanged. Use one underlying accounting implementation rather than independent counters that drift.

### 2. Publish scheduler/cache snapshots

Add a small CPU snapshot producer at the current owners:

- `batch.Batcher`: queued, admitted, prefilling and decoding jobs.
- `kvpool.PoolBook`: capacity, mapped/free pages and outstanding reservations.
- `sessions.SessionStore`: retained RAM entries, budget and allocation usage.

Publish at scheduler transitions and completed prefill/decode boundaries. Both endpoints read the same coherent snapshot. Avoid copying token lists or scanning every cached token to produce a sample.

HTTP scrapes must not launch CUDA operations, synchronize CUDA, add distributed collectives, scan directories or drain the disk writer. Refresh filesystem statistics outside HTTP scrape handling.

### 3. Instrument SSD lifecycle events

Add accounting around completed writes, reads, deletion/eviction, startup recovery and failure paths.

Distinguish:

- Committed stored entries from pending writes.
- Actual local file bytes from budget charge.
- Local read success from a usable both-rank restore.
- Checksum failure from other load failures.
- Slot, RAM-session and SSD cache reuse.

Update counters at authoritative outcomes, not merely submission. Protect asynchronous writer updates and snapshot publication without holding telemetry locks across I/O or GPU work.

### 4. Add request histograms and native exposition

Instrument the HTTP/application lifecycle around validation, generation and completion:

- Exactly one terminal accounting event.
- Streaming and nonstreaming paths.
- Validation refusals, engine failures and client cancellation.
- Optional TTFT and time-per-output-token observations only when meaningful.

Reuse the reference's bounded histogram approach. No separate Prometheus service or heavy dependency is necessary. Ship the change through the repository's numbered-patch build convention, after the existing patches, rather than modifying only an installed container.

### 5. Integrate EngineMetrics end to end

Populate expected cache/session JSON and native metrics from the same snapshots.

Consumer details:

- The current "Prefill PP (engine)" calculation divides all prompt tokens, including cached tokens, by prefill time. That is not actual uncached prefill throughput. Correct the calculation with a matching uncached-token numerator or relabel the display accurately.
- The active launcher's `--spill-cap 128` is a historical fallback. Report the actual tier budget; the adapter already prefers reported `spill.gib`.
- The live-gauge retention fix is already present.
- Unsupported fields should display as unavailable, not fabricated zeros.

The separate EngineMetrics checkout is outside this repository. Any implementation-stage changes to that checkout or its launcher require user approval under this repository's operating rules.

### 6. Verify behavior and document the contract

Extend existing behavior-oriented health, session and disk coverage rather than adding tests that merely assert metric names or source wiring.

Relevant existing coverage:

- `tests/test_health.py`.
- `tests/test_upstream_ports.py`.
- `tests/test_gpuwatch.py` and `tests/test_serve_ops.py` for existing consumers.
- `tests/test_sessdisk_index.py`.
- `tests/cuda/test_session_patches.py`.
- `tests/cuda/test_batch_sessions_patches.py`.
- `tests/cuda/test_session_disk_patches.py`.

The disk lifecycle suite already covers real tier round trips, restart recovery, corrupt/missing files and asymmetric rank failures. Attach telemetry assertions to those behaviors where appropriate.

After runtime proof, update `docs/PATCHES.md`, `docs/TRYING.md` and attribution for reused code. Remove any throwaway verification scaffolding.

## Acceptance criteria

The first deliverable must include working KV/RAM/SSD telemetry, not a request-only metrics port with cache panels deferred.

### Request lifecycle

- Successful, refused, failed and canceled requests finalize exactly once.
- Streaming and nonstreaming behavior is covered.
- TTFT excludes SSE role-only events.
- Optional timings are not fabricated as zeros.
- Native completed-request totals and live `/health` totals retain their documented meanings.
- Shared batch rounds are not misrepresented as physical forwards.

### Cache/session lifecycle

- Cold requests, slot reuse, RAM restore and SSD restore are separately observable.
- Idle retained cache can have nonzero pool occupancy while live-context ratio is zero.
- Reservations reduce admission availability without falsely increasing mapped occupancy.
- Shared pages are counted once.
- Cache reuse statistics agree with the engine's actual restored source and token counts.
- Capacity and budget fields reflect the actual instantiated configuration.

### SSD lifecycle

- Pending writes are not reported as committed storage.
- Successful writes, reads, evictions and startup recovery update occupancy consistently.
- Restores after restart are observable.
- Failed writes, checksum failures, missing/truncated files and one-rank restore failure update the correct counters.
- No successful distributed cache restore is claimed when either rank falls back to prefill.
- Physical byte counts and budget charges are not conflated.
- Absent/disabled/unsupported telemetry is distinguishable from a functional empty tier.

### Scrape and consumer safety

- Concurrent scrapes see coherent snapshots and do not trigger GPU or distributed work.
- Scraping does not wait for SSD writes to drain.
- Counter reset behavior is handled on restart.
- Existing watchdog and GPU watcher consumers continue working.
- EngineMetrics displays actual stream state, pool occupancy, RAM retention and SSD activity.

### Runtime proof

In an agreed maintenance window, exercise the actual model and EngineMetrics TUI through cold, warm, RAM-restored and SSD-restored sessions, including restart recovery. Use isolated temporary storage for destructive failure cases, not production session files.

Compare deterministic outputs with telemetry enabled and measure scrape overhead and memory use. Tests alone are not sufficient proof of the changed monitoring surface.

Implementation now includes the native/operational metric surfaces, published cache snapshots, SSD lifecycle instrumentation and a portable consumer patch. CPU runtime verification exercised real HTTP and curses interfaces, real RAM/SSD session paths and restart recovery against fresh-prefill reference output. The running production containers and external EngineMetrics checkout were not changed.

## Useful additions outside the first viable set

These are separate candidates, not prerequisites for completing the contract above:

- Per-physical-round latency distributions for slow-state detection during long replies.
- Per-drafter-arm proposal/acceptance distributions.
- Actual selected communication backend and RoCE fallback/failure telemetry.
- Precisely scoped per-rank KV backing-byte metrics.

Do not infer active RoCE from the requested environment variable, or physical batch timing from summed per-request timing.
