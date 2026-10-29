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

## M3 prefix cache

2026-10-28, gfx1151, release preset `build/release/gufo`, one session, context
16384. The replay is the committed `tool-agent-loop` prompt (read/edit of
`calc0.py` and `calc1.py`, temperature 0, seed 41, thinking off, at most 192
new tokens). Cache-off is the same prompts with `cache_prompt` false. Both
files ended as `return a + b`. Tool name and arguments matched on every turn.
Turns 1 and 3 differed in assistant prose around the same call; the logit gap
is in QUALITY.md. Decode tok/s is `completion_tokens_per_second`. TTFT is
`usage.gufo.ttft_ms`.

| Turn | Prompt | Cached | Prefill | TTFT ms | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 on | 438 | 0 | 438 | 741.9 | 39.4 |
| 0 off | 438 | 0 | 438 | 757.6 | 41.8 |
| 1 on | 507 | 477 | 30 | 246.9 | 40.1 |
| 1 off | 507 | 0 | 507 | 824.1 | 39.2 |
| 2 on | 593 | 569 | 24 | 245.1 | 41.3 |
| 2 off | 593 | 0 | 593 | 1157.3 | 43.3 |
| 3 on | 648 | 618 | 30 | 253.1 | 43.6 |
| 3 off | 648 | 0 | 648 | 1205.8 | 42.2 |
| 4 on | 712 | 682 | 30 | 248.3 | 38.7 |
| 4 off | 712 | 0 | 712 | 1301.1 | 36.6 |
| 5 on | 798 | 774 | 24 | 275.8 | 40.1 |
| 5 off | 798 | 0 | 798 | 1381.4 | 40.2 |
| 6 on | 853 | 823 | 30 | 288.6 | 44.4 |
| 6 off | 853 | 0 | 853 | 1419.2 | 42.3 |

Lemonade was already running: llama-server b11580-4413a1924, same GGUF,
context 32768, `--spec-type draft-mtp` (MTP on), `--parallel 1`, no
`cache_prompt` flag. The replay omitted `cache_prompt`, and the responses
reported a prompt cache. `prompt_ms` is llama's prefill time for `prompt_n`
uncached tokens, not Gufo's TTFT. Decode tok/s is `predicted_n / predicted_ms`.

| Turn | Prompt | Cached | Uncached | prompt_ms | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 438 | 0 | 438 | 620.0 | 46.5 |
| 1 | 493 | 463 | 30 | 147.4 | 51.8 |
| 2 | 579 | 555 | 24 | 147.3 | 45.5 |
| 3 | 634 | 604 | 30 | 159.6 | 46.1 |
| 4 | 689 | 659 | 30 | 227.3 | 48.5 |
| 5 | 775 | 751 | 24 | 147.0 | 46.0 |
| 6 | 830 | 800 | 30 | 178.9 | 38.8 |

`ornith15-functional-v1` was scored again, unchanged, on this cached release
server. All ten runs passed. Greedy repeats of a card prompt are full cache
hits, so their TTFT is about 5 ms. Card TTFT is a cold prefill.

| Task | Sampling | TTFT ms | Decode tok/s | Cached |
| --- | --- | --- | ---: | ---: |
| write_merge | card | 382.4 | 43.3 | 0 |
| fix_median | card | 514.2 | 43.2 | 0 |
| tool_round_trip | card | 661.7, 257.3, 214.9 | 38.6, 41.1, 40.8 | 0, 425, 502 |
| two_turn_clamp | card | 288.4, 304.7 | 42.8, 41.9 | 0, 206 |
| rectangle_area | card | 246.2 | 43.0 | 0 |
| write_merge | greedy | 5.4 | 42.2 | 110 |
| fix_median | greedy | 5.7 | 42.3 | 214 |
| tool_round_trip | greedy | 6.5, 255.2, 214.9 | 39.7, 41.8, 39.6 | 351, 428, 505 |
| two_turn_clamp | greedy | 5.1, 303.1 | 42.1, 41.6 | 50, 188 |
| rectangle_area | greedy | 5.1 | 41.8 | 42 |

## M3 prefill tile

2026-10-29, gfx1151, release `Session::Prefill` only. The prompt is the M1
English sentence, repeated. Context 4096. `Reset` is outside the timer.
There is no decode and no host argmax, so these rates are not the M2
`ornith15_bench` medians. Baseline is one measure of tile 80. Width 48 is
the median of three measures after one warmup of each length.

| | 20 s | 20 tok/s | 261 s | 261 tok/s | 2048 s | 2048 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tile 80 | 0.1238 | 161.5 | 0.4256 | 613.3 | 2.4654 | 830.7 |
| Width 48, median | 0.1038 | 192.6 | 0.3852 | 677.6 | 2.3848 | 858.8 |

The three width-48 measures, in order: 20 tokens 196.5, 192.6, 182.8;
2048 tokens 880.4, 856.6, 858.8; 261 tokens 679.1, 677.0, 677.6.
Lemonade's ~995 tok/s at about 2048 tokens was not reached.

rocprof kernel time for those measured passes, in milliseconds. Sums add
dispatch durations, so they can exceed the GPU span when launches overlap.
The 2048 columns are the four 512-token chunks. Routed is `mul_mat_q`
with grid z 256. After the change, IQ4 gate/up at 512 tokens launch
`ntx=11`; IQ4 down stays at `ntx=86`.

| Class | 20 before | 20 after | 261 before | 261 after | 2048 before | 2048 after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Routed IQ4_XS | 29.93 | 21.93 | 110.65 | 86.35 | 560.78 | 486.12 |
| Routed Q6_K | 29.70 | 20.72 | 77.59 | 58.73 | 343.86 | 291.71 |
| Routed Q5_K | 16.35 | 12.35 | 57.14 | 45.52 | 267.51 | 240.50 |
| Dense Q6 projections | 13.63 | 13.31 | 81.47 | 77.41 | 513.49 | 497.62 |
| Shared Q8_0 | 4.07 | 4.03 | 8.07 | 7.64 | 45.42 | 43.92 |
| DeltaNet | 2.40 | 2.49 | 21.52 | 21.20 | 178.30 | 182.43 |
| Attention | 0.32 | 0.37 | 3.51 | 3.40 | 117.65 | 114.76 |
| F32 router/shexp | 2.40 | 2.37 | 17.27 | 16.77 | 130.46 | 126.77 |
| MoE epilogue/SwiGLU | 0.73 | 0.70 | 12.38 | 11.94 | 135.96 | 128.32 |
| Quantize and ids | 3.38 | 3.42 | 10.74 | 9.83 | 77.96 | 72.73 |
| Norm, residual, embed | 1.00 | 1.09 | 3.54 | 3.37 | 25.08 | 24.76 |
| Vocab GEMV | 1.92 | 1.85 | 1.88 | 1.89 | 1.89 | 1.88 |
| Runtime fill/copy | 1.54 | 1.54 | 2.71 | 2.48 | 7.57 | 7.19 |

Kernel sums: 20 tokens 107.4 to 86.2 ms, 261 tokens 408.5 to 346.5 ms,
2048 tokens 2405.9 to 2218.7 ms. The 20-token rocprof pass idled 10.4 ms
before and 11.6 ms after. Removing the per-chunk stream sync left the
unprofiled 2048-token prefill at 2.4684 s.
