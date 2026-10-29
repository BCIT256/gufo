# Ornith GPU AR checkpoint measurements

Historical P4/P4b evidence is retained below; the P4c section records the
policy choice. The M1 section records the later prefill and decode rates.
The M2 section records the same protocol on the release preset. The functional
section records the HTTP suite.

October 23, 2026; gfx1151, pinned Nix RelWithDebInfo correctness build with
assertions. These are informational measurements from eager single-token
validation, not a production benchmark or an optimization comparison.

The test times each synchronous `Step`: host launch work, the 40-layer forward,
the full vocabulary head, per-layer diagnostic downloads, and the final logits
download. It excludes upload, session creation/reset and scalar-reference work.
Prompts are teacher-forced from the pinned captures; there is no chunked
prefill or graph replay. Early development runs overlapped other validation;
only the final sequential CTest GPU-decode run is retained below.
The recorded power state at the start of that run was AC power with the
`balanced` platform profile.

Resident byte counts cover model-owned allocations (original GGUF tensor bytes
plus allocation tails) and session-owned state/scratch. The post-upload HIP
heap delta is also reported separately; runtime/library reservations and
reclaimable host file-cache pages are not model-owned allocation counts.
None of these values establishes a production context or concurrency capacity.

| History | Positions | Mean eager ms/token (traces enabled) | Session allocations, bytes |
| --- | ---: | ---: | ---: |
| reasoning_history | 87 | 81.7383 | 70284620 |
| tool_round_trip | 75 | 81.7374 | 70038860 |
| user_turn | 21 | 82.1330 | 68932940 |

Post-upload model allocations: 22,811,415,040 bytes
(21.2448 GiB). The observed HIP heap delta after upload
was 23,659,692,032 bytes (22.0348 GiB).

No pp2048/tg128 production, warm-cache, deep-context, concurrent serving or
matched speed comparison has been run for this checkpoint.

## P4b matched informational rerun

Fresh original blanket correction versus the retained one-pass policy; same
model, original teacher-forced histories, uncached CPU-reference interleave,
traces, synchronous Step scope and pinned correctness build as above. These
are a single before/after pair; they are not production or alternating-run
benchmarks, and do not establish equivalent numerical fidelity. See
[quality and experiment evidence](EXPERIMENTS.md#final-uncached-qualification).

| History | Before ms/token | After ms/token | Before / after session bytes |
| --- | ---: | ---: | --- |
| reasoning_history | 80.9294 | 49.2712 | 70284620 / 69123404 |
| tool_round_trip | 81.9418 | 49.3362 | 70038860 / 68877644 |
| user_turn | 82.1684 | 49.0838 | 68932940 / 67771724 |

AC power, balanced platform profile. Model-owned weight allocations remain
22,811,415,040 bytes. Correction scratch decreases by 1,161,216 bytes per
session; no extra persistent unpacked-weight allocation is introduced.

## P4c held-out A/B timings

Position-weighted held-out eager traced Steps: A 48.1460,
B 50.0442 ms/token over 649 positions. Kept policy: B.
Single sequential B-then-A comparison with uncached scalar-reference
interleave, not an alternating production benchmark or an equivalent-fidelity
speedup claim. See [per-prompt timings and scope](EXPERIMENTS.md#p4c-informational-timings).

## M1 prefill and greedy decode

2026-10-28, gfx1151, one session, context 4096, prefill chunks of 512.
`ornith15_bench` from the `gpu-test` preset: Nix RelWithDebInfo with
assertions. The model is loaded once. Warmup is one prefill plus 16 untimed
device-argmax steps. Each reported rate is the median of five later runs.
A run is `Reset`, `Prefill`, a host argmax of the last logits (not timed),
then 128 `Step` calls that take a device argmax and do not download logits.
Power at the start of the run: `AC0/online=1`, `BAT0/status=Charging`.

The prompt is the GGUF tokenizer's encoding of one English coding request
(merge half-open intervals, C++17, no BOS or EOS, special tokens not parsed),
repeated until it fills 2048 tokens and, separately, 261 tokens.

Lemonade numbers are the project's 2026-10-26 stock llama.cpp measurement on
this laptop and the same GGUF, autoregressive, without MTP. That run used a
2033-token prompt for the long prefill. The prompts and the builds differ.
Each column is that run's own measurement.

| | Prompt tokens | pp tok/s | tg tok/s |
| --- | ---: | ---: | ---: |
| Gufo-Ornith median | 2048 | 717.2346 | 38.4167 |
| Lemonade, 2026-10-26 | 2033 | ~995 | ~35.4 |
| Gufo-Ornith median | 261 | 544.8397 | 40.6073 |
| Lemonade, 2026-10-26 | 261 | ~626 | ~38.7 |

Five Gufo-Ornith samples, in run order:

| Prompt tokens | pp tok/s | tg tok/s |
| ---: | --- | --- |
| 2048 | 750.5527, 717.5176, 715.8051, 716.3496, 717.2346 | 38.9878, 38.4167, 38.1848, 38.7807, 38.4011 |
| 261 | 547.5602, 544.2966, 543.6358, 545.5704, 544.8397 | 42.1052, 41.1675, 40.1747, 40.6073, 40.3313 |

## M2 release preset

2026-10-28, gfx1151, same protocol as the M1 section: one session, context
4096, prefill chunks of 512, median of five after one prefill plus 16 untimed
device-argmax steps. `ornith15_bench` is the `release` preset: Nix
RelWithDebInfo, assertions off (`NDEBUG`), glibc-isolated `build/release`.
Power at the start: `AC0/online=1`, `BAT0/status=Charging`. The prompts and
the builds differ from the gpu-test run and from Lemonade. Each row is that
run's own measurement.

| | Prompt tokens | pp tok/s | tg tok/s |
| --- | ---: | ---: | ---: |
| Gufo-Ornith release median | 2048 | 740.8327 | 41.8353 |
| Gufo-Ornith gpu-test median | 2048 | 717.2346 | 38.4167 |
| Lemonade, 2026-10-26 | 2033 | ~995 | ~35.4 |
| Gufo-Ornith release median | 261 | 555.4992 | 44.3460 |
| Gufo-Ornith gpu-test median | 261 | 544.8397 | 40.6073 |
| Lemonade, 2026-10-26 | 261 | ~626 | ~38.7 |

Five release samples, in run order:

| Prompt tokens | pp tok/s | tg tok/s |
| ---: | --- | --- |
| 2048 | 766.1910, 745.4238, 740.8327, 732.2343, 730.5918 | 42.5953, 41.5434, 41.8353, 41.5521, 42.1505 |
| 261 | 559.0735, 555.4992, 555.5898, 554.0824, 554.2501 | 44.5930, 44.7125, 44.2346, 44.3460, 44.3132 |

## M2 functional suite

2026-10-28, gfx1151, `ornith15-functional-v1` through `gufo serve llm` on the
gpu-test binary. One session, context 8192, max 2048 new tokens, thinking on.
Card sampling is temperature 0.6, top_p 0.95, top_k 20, seed 20261028. Greedy
changes only the temperature to 0. All ten task runs passed. TTFT is
`usage.gufo.ttft_ms`. Decode tok/s is `completion_tokens_per_second`. A task
with several requests lists them in order. The streamed area task's client
time to the first content or reasoning delta was 179.6 ms (card) and 187.8 ms
(greedy).

| Task | Sampling | TTFT ms | Decode tok/s |
| --- | --- | --- | --- |
| write_merge | card | 314.1 | 44.9519 |
| fix_median | card | 438.4 | 43.4110 |
| tool_round_trip | card | 587.0, 725.1, 814.8 | 40.1898, 41.1109, 41.9194 |
| two_turn_clamp | card | 216.2, 406.0 | 43.2767, 45.0550 |
| rectangle_area | card | 178.6 | 43.4831 |
| write_merge | greedy | 301.7 | 44.0265 |
| fix_median | greedy | 451.9 | 43.6082 |
| tool_round_trip | greedy | 594.7, 732.8, 823.9 | 39.9973, 41.1557, 40.7600 |
| two_turn_clamp | greedy | 212.2, 484.8 | 44.2586, 43.2330 |
| rectangle_area | greedy | 187.0 | 44.7972 |
