# 0rand fork: metrics, production tuning and benchmark notes

Branch: [`main-0rand`](https://github.com/0rand/glm53-tensorfold-spark/tree/main-0rand). Recorded: 2026-11-17.

This fork builds on Jay Leaton's GLM-5.3-Flash / TensorFold stack for two DGX Sparks. The changes here add
EngineMetrics-compatible monitoring and isolated fork image tags. The operator's local production configuration
also increases the shared KV pool, lowers two memory guards and selects maximum reasoning effort.

**1M-context operation was validated in operator tests.** The configured per-request limit remains 1,048,576 tokens
(prompt plus requested output). The saved ToolEval run scored **93/100**, with all multi-step-chain points earned.
These operator benchmark results are not an A/B measurement of the new monitoring patch.

## 1. What was implemented

### EngineMetrics and Prometheus integration

Implementation: [`patches/0710-glm-engine-metrics.patch`](patches/0710-glm-engine-metrics.patch), introduced in
[`d762617`](https://github.com/0rand/glm53-tensorfold-spark/commit/d7626170a0437e6c33fcd6ba533912f0180ce3c8).
The monitoring contract follows the DeepSeek TensorFold-ZIG stack, with Gilesco as a secondary reference; it is
implemented against this repository's patched GLM engine rather than transplanting a different cache backend.

| Area | Changes |
| --- | --- |
| Requests and generation | Native `tensorfold:*` request/token/draft accounting, actual running and waiting scheduler work, and live generation state. |
| Latency | Five histograms: request latency, time to first generated token, prefill, decode and time per output token; buckets accommodate long-context requests. |
| Shared KV pool | Instantiated capacity, mapped/free/retained tokens, reservations and admission availability. Pool occupancy is separate from live-stream context usage. |
| RAM sessions | Retained prompts/bytes and budget/usage; cache restores distinguish slot, RAM and SSD sources. |
| SSD sessions | Committed entries and physical bytes, budget charge, pending writes/bytes, local reads/writes, filesystem free bytes, restore latency and failure counters, including SHA-256 failures and distributed restore failures. |
| Scrape safety | HTTP readers consume CPU-side published snapshots instead of traversing mutable engine caches or invoking CUDA/collectives. |
| Existing operators | Existing underscore-prefixed `tensorfold_*` metrics remain available for the watchdog, canary and GPU watcher. Matching cache/SSD information is exposed through `/health`. |

The portable [`tools/enginemetrics-glm-metrics.patch`](tools/enginemetrics-glm-metrics.patch) updates the external
EngineMetrics adapter: uncached-prefill throughput, pool/live-context separation, checksum reporting and honest
unavailable displays. It is **not automatically installed into an external EngineMetrics checkout**.

Verification recorded for this implementation:

- CPU regression run: **361 passed, 40 GPU-only cases skipped**; additional pooled-scheduler regression: **2 passed**.
- Runtime smoke exercised actual HTTP and curses surfaces, real RAM/SSD storage and restart recovery with the
  deterministic CPU model harness. Cold, slot-reused, RAM-restored, SSD-restored and restarted outputs matched fresh prefill.
- Pooled-scheduler smoke observed retained pool occupancy while idle live-context usage was zero.
- These checks did not rebuild or restart production and are not production-model GPU performance/exactness proof.

The monitoring patch does not change kernels, inference settings, weights or cache file formats. See
[`METRICS-ENHANCE-PLAN.md`](METRICS-ENHANCE-PLAN.md) and [`docs/CHANGES-SUMMARY.md`](docs/CHANGES-SUMMARY.md).

### Separate fork images, shared container identities

Implemented in [`3877a93`](https://github.com/0rand/glm53-tensorfold-spark/commit/3877a93e080efff39c117c062103717b4336c3a0).

| Checkout | Effective default image |
| --- | --- |
| `main` | `glm53-tensorfold:dev` |
| `main-0rand` | `glm53-tensorfold:dev-0rand` |

[`scripts/image_name.py`](scripts/image_name.py) resolves the image after config/caller precedence. Build, worker
save/load, preparation, serving and TensorFold benchmark metadata use that same reference. Custom mutable tags
receive the suffix once; digest references remain immutable. Keep `IMAGE=glm53-tensorfold:dev` in the ignored local
config so switching back to `main` can reuse the original image without editing the config.

- [`scripts/compose.sh`](scripts/compose.sh) provides the equivalent safe image resolution for the debugging Compose stack.
- `serve.sh status` displays each container's actual image.
- Both branches deliberately use `glm53-tf-r0` and `glm53-tf-r1`: `serve.sh stop` targets those shared names,
  so it can stop the currently running version regardless of which branch created it. Stopping containers does not delete images.
- No automatic image deletion/pruning was added. Redundant images can be removed manually after settling on a version.
- Ports, volumes, prepared weights and session paths remain shared. This is image rollback, not concurrent stacks
  or a snapshot of mutable cache/session contents.

Verification: **67 tests passed**, real Compose configuration rendering selected the fork image, and an isolated
Docker/SSH smoke built/transferred/launched the fork tag before the actual `main` launcher reused the original tag
with the same config. Both original simulated node image identities were preserved. That verification did not
build, replace or delete real images. Deployment and rollback commands:
[`docs/TRYING.md: Branch images and rollback`](docs/TRYING.md#branch-images-and-rollback).

## 2. Local `prod.env` versus `prod.env.example`

The following is a sanitized comparison of the actual local `config/prod.env` with
[`config/prod.env.example`](config/prod.env.example), not a change to the template. Inline comments retaining old
values in the local file are excluded. The private config itself is not published.

| Knob | Template | Local production | Meaning |
| --- | --- | --- | --- |
| `MEM_GATE_GIB` | `108` | `0` | Disables the launcher's pre-start MemFree wait and its below-threshold preflight warning. It does not disable every engine memory check. |
| `GLM53_TF_BATCH_RESERVE_GB` | `11` | `2` | Lowers the load-time free-memory reserve used when deciding how many permanent batch slots fit. |
| `GLM53_TF_KV_POOL_TOKENS` | `1048576` | `1572864` | Enlarges the shared FP8 latent KV pool by 50%, from 4,096 to 6,144 pages of 256 tokens. |
| `GLM53_TF_DEFAULT_EFFORT` | `high` | `max` | Requests without an explicit effort use maximum reasoning effort. |
| `SERVED_NAME` | `GLM-5.3-Flash-EXL3` | `glm5.3-flash` | Changes the current API model alias, not the checkpoint. Saved benchmark reports use the earlier alias. |
| `PORT` | `8000` | `8100` | Changes the head's API port. `HOST=127.0.0.1` remains unchanged. |
| `NCCL_SOCKET_IFNAME` | `enp1s0f1np1` | `enp1s0f0np0` | Selects this pair's cabled CX7 netdev. Hardware-specific, not a general performance recommendation. |
| `NCCL_IB_HCA` | `rocep1s0f1,roceP2p1s0f1` | `rocep1s0f0,roceP2p1s0f0` | Selects the corresponding CX7 RDMA functions. Requested RoCE configuration alone does not prove the active transport. |

The local file also fills `WORKER_SSH`, `HEAD_IP`, `HEAD_HF` and `WORKER_HF`. Their addresses and home paths are
intentionally omitted here. Model/drafter snapshot references and the configured base `IMAGE` are unchanged.

### What “1.6M cache” means

The exact pool size is **1,572,864 tokens**: approximately 1.6 million decimal tokens, or 1.5 times the template's
1,048,576-token pool. It is **shared KV capacity across the batch slots**, not a 1.6M-token per-request context window
and not the RAM/SSD session-store budget. Four slots do not each receive a full pool of their own.

Unchanged settings relevant to interpreting this tuning:

| Knob | Template and local value |
| --- | --- |
| `CONTEXT` | `1048576` tokens per request, prompt plus requested output |
| `MAX_TOKENS` | `32768` |
| `GLM53_TF_BATCH` | `4` |
| `GLM53_TF_LATENT_KV` / `GLM53_TF_KV_DTYPE` | `1` / `fp8` |
| `GLM53_TF_BATCH_ADMIT_GB` | `2` |
| `GLM53_TF_SESSION_GIB` / `GLM53_TF_SESSION_RESERVE_GIB` | `2` / `6` |
| `GLM53_TF_SESSION_DISK_GIB` | `64` configured SSD-tier budget |
| `MEM_GATE_DROP_CACHES` | `1`; independent start-time cache-drop behavior is not disabled by setting the wait threshold to zero |

At the documented 7,616 bytes per token per rank, pool row storage is approximately **11.16 GiB instead of 7.44 GiB**
(an extra **3.72 GiB per rank**), excluding the null page and other overhead. This is a capacity-based estimate, not
an observed allocation from these benchmark receipts.

**Trade-off:** more KV capacity and a smaller load-time reserve leave less protection against OOM. Disabling the
launcher wait does not create free memory; admission/session guards still apply. These are operator-specific settings,
not new safe defaults for other Spark pairs. The saved runs do not establish worst-case safety for four simultaneous
long requests, nor do they isolate the effect of any one knob.

## 3. Quality and context validation

### ToolEval: 93/100

Source **Q1**, completed on 2026-11-17, tool-eval-bench `v3.0.1.dev6+g7b022ddcb`, EXL3 checkpoint, 104 scenarios,
sequential execution, seed 42, thinking enabled and a 16,384-token per-response benchmark limit.

| Metric | Recorded result |
| --- | --- |
| Final quality score | **93/100**, from **194/208 points** |
| Multi-Step Chains | **8/8**, 100% |
| Structured Output | **12/12**, 100% |
| Hard Mode | **64/70**, 91% |
| Context & State | **18/20**, 90% |
| Safety & Boundaries | **20/26**, 77% |
| Deployability | **80/100** |
| Responsiveness | **50/100**, median turn **3.0 s** |
| Reported error rate | **0.0** |

Recorded sampling settings need both fields retained: the report's base Temperature field is `0.0`, while Extra
Params contains `temperature: 0.3`, `top_p: 1` and
`chat_template_kwargs: {thinking: true, reasoning_effort: max}`. Thus the run **requested max effort and a 0.3
temperature override**; describing it simply as a greedy run would discard that override. The report is not an
independent capture of the effective wire request or server-side parameter resolution.

**Safety limitation:** Q1 flags two safety-critical failures: TC-60 (cross-turn sleeper injection) and TC-98
(multi-hop transitive prompt injection). The headline score is not a claim of safe autonomous execution of untrusted
instructions or a substitute for tool authorization and isolation.

### Comparison with Jay Leaton's published 90–91

The [upstream W21 results](docs/TOOL-CALLING.md#7-results-w21-production-b11) report 90 with thinking off and 91
with thinking **high**, at temperature 0. Low effort was recommended for agentic latency/quality and evaluated on
chains plus spark-bench agentic tasks; it was **not** the configuration behind a published full ToolEval score of 91.

Q1's headline is higher, but it is not a controlled improvement measurement: the
[upstream receipts](results/tooleval/W21-scripts/README.md) used tool-eval-bench `c7b5b95` and 69 scenarios, versus
Q1's newer 104-scenario suite, different requested sampling and max effort. Do not attribute the score difference
solely to memory tuning, reasoning effort or patch 0710.

### 1M context

The operator reports **successful 1M-context validation in tests**, with a configured limit of 1,048,576 tokens.
All saved report engine metadata also advertises that limit. The largest saved throughput cell below is a
**512,000-token prompt plus 256 output tokens**; it is separate evidence, not a full-1M throughput receipt.
No 1M TTFT/decode figure is inferred from the configured context limit or extrapolated from the shorter runs.

## 4. Performance results from `runs/`

These are selected **completed** receipts from 2026-11-17, not averages across campaigns and not a pre/post metrics
comparison. Source IDs map to the local reports in section 5. `pp`/`tg`/`d`/`c` retain the benchmark's own cell labels:
prompt tokens, generation tokens, context depth and concurrency. Rates below retain the reports' definitions.

### Depth sweep and long prompt

P1–P5 use 2,048 prompt tokens and 256 generated tokens; P6 is the separate long-prompt workload.

| Source | Benchmark cell | Reported prefill estimate (tok/s) | Reported decode (tok/s) | TTFT (s) | Total (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| P1 | `pp2048 tg256 @ d0` | 1,305 | 55.6 | 1.569 | 6.154 |
| P2 | `pp2048 tg256 @ d8192` | 1,521 | 49.3 | 6.732 | 11.909 |
| P3 | `pp2048 tg256 @ d32384` | 1,536 | 51.2 | 22.421 | 27.406 |
| P4 | `pp2048 tg256 @ d128000` | 1,499 | 48.2 | 86.785 | 92.080 |
| P5 | `pp2048 tg256 @ d256000` | 1,416 | 47.0 | 182.275 | 187.701 |
| P6 | `pp512000 tg256 @ d0` | 1,268 | 40.0 | 403.658 | 410.028 |

**Prefill caveat:** the reports derive these estimates from end-to-end TTFT. They explicitly warn that an early SSE
chunk can arrive before the first content token (llama-benchy issue #33), and that network/queue time is included.
These are not new patch-0710 engine-prefill measurements. TTFT follows the benchmark's timing definition as well.

### Concurrency sweep

Source **P7**, 2,048 prompt tokens and 128 generated tokens per cell. Values are copied as reported; no multiplication
by concurrency or reinterpretation as per-stream throughput is applied.

| Depth | Concurrency | Reported tg t/s | TTFT (s) | Total (s) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 60.3 | 1.583 | 3.689 |
| 0 | 2 | 78.5 | 3.113 | 6.150 |
| 0 | 4 | 59.3 | 5.630 | 9.790 |
| 4,096 | 1 | 52.9 | 4.186 | 6.585 |
| 4,096 | 2 | 74.3 | 9.025 | 12.172 |
| 4,096 | 4 | 35.4 | 14.404 | 17.991 |
| 8,192 | 1 | 55.0 | 6.764 | 9.074 |
| 8,192 | 2 | 72.4 | 14.876 | 18.204 |
| 8,192 | 4 | 25.0 | 22.417 | 26.040 |

The four-way cells do not show uniform scaling; deeper-context TTFT and reported decode are worse. These single
reported cells do not establish a generally optimal concurrency setting.

### Speculative decoding

Source **S1**, temperature 0, three runs per cell, depths 0 / 4,096 / 8,192, 128 output tokens. The following are the
report's per-prompt-type averages; acceptance is sourced from per-request response metrics.

| Prompt type | Average effective tok/s | Average draft-token acceptance |
| --- | ---: | ---: |
| Code | 55.0 | 69.5% |
| Filler | 49.1 | 59.1% |
| Structured | 70.1 | 80.5% |

The report labels the speculative method `unknown` and has no measured no-speculation baseline/speedup. Acceptance
is not tokens per round, and these numbers do not establish a speculative speedup or identify the active drafter.
Greedy acceptance is not a prediction of sampled-workload acceptance.

An earlier completed throughput receipt, **P0**, reported 28.4–32.4 decode tok/s. It is not folded into the later
sweeps, and no cause for that variation is established here. The initial speculative receipt ending `cbbace98` is
marked failed in the local database and is not used as the completed speculative result.

## 5. Evidence and publication boundaries

Raw reports are local, untracked operator artifacts in `runs/2026/11/`; `data/benchmarks.sqlite` supplies run status
and stored measurements. They are intentionally **not added to Git**: full traces contain local metadata and
benchmark fixture content. The summary above carries the relevant numbers without publishing the private config,
addresses, home paths, host identifiers or full tool traces. These are provenance IDs, not links to published raw reports.

| ID | Local report run ID |
| --- | --- |
| Q1 | `2026-11-17T16-38-33.858467Z_0615da2c` |
| P0 | `2026-11-17T16-26-46.768398Z_dae66973` |
| P1 | `2026-11-17T17-02-28.514074Z_99589505` |
| P2 | `2026-11-17T17-02-54.063972Z_b4f3076b` |
| P3 | `2026-11-17T17-03-51.186715Z_508a2f5b` |
| P4 | `2026-11-17T17-06-57.462017Z_df0b18b1` |
| P5 | `2026-11-17T17-13-14.883912Z_2f501f60` |
| P6 | `2026-11-17T17-28-41.356524Z_a57fa561` |
| P7 | `2026-11-17T17-59-01.218129Z_b3f98f48` |
| S1 | `2026-11-17T16-37-56.578929Z_d4e99d7a` |

The saved reports do not identify a serving-image digest or establish that patch 0710 was deployed. Current local
config values are a snapshot of the file inspected for this note, not proof of every historical run's environment.
No inference benchmark, service restart, image build, config change or host tuning was performed to write this document.

Upstream attribution and license notices remain in [`NOTICE`](NOTICE) and [`LICENSE`](LICENSE).
