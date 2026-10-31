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

## M4 staged snapshot

2026-10-29, gfx1151, release `build/release/gufo`, one session, context
16384, `--cache-disk`. Power: `AC0/online=1`, `BAT0/status=Charging`.
Lemonade's llama-server (pid 3363206, state `Sl`, port 8003, MTP) was
running on the same GGUF and was not paused. The replay is the
`tool-agent-loop` prompt (temperature 0, seed 41, thinking off, at most
192 new tokens). The live cache-on conversation was recorded, the server
restarted, then those exact message lists ran cache on, off, on, off.

`cache_snapshot_ms` is the device-stage time plus any later join of the
host export. On a partial hit, TTFT minus `prefill_ms` is smaller than
`cache_snapshot_ms`, so that join is not all on the first-token path.
`on-a-0` is the disk restore after restart. `on-b` reuses the full prompt
from RAM (`prefill_tokens` 0). A reusable pinned host bounce measured
110.4 ms on one 62-token disk hit and was not kept.

| Turn | Prompt | On cached | On TTFT | On snapshot | On restore | On prefill ms | Off TTFT | Off TTFT again | RAM TTFT | RAM restore |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 438 | 431 | 214.6 | 26.9 | 84.9 | 109.3 | 673.5 | 677.2 | 2.4 | 1.3 |
| 1 | 507 | 477 | 226.4 | 29.9 | 0 | 204.3 | 756.2 | 761.1 | 2.8 | 1.8 |
| 2 | 628 | 604 | 209.8 | 27.8 | 0 | 191.2 | 1099.4 | 1092.4 | 2.6 | 1.5 |
| 3 | 697 | 667 | 260.8 | 45.9 | 0 | 225.4 | 1129.4 | 1144.2 | 2.8 | 1.6 |
| 4 | 767 | 737 | 266.8 | 49.3 | 0 | 232.5 | 1192.6 | 1192.3 | 3.0 | 1.8 |
| 5 | 868 | 844 | 237.9 | 49.3 | 0 | 204.8 | 1320.8 | 1350.1 | 3.6 | 2.3 |
| 6 | 937 | 907 | 270.5 | 49.7 | 0 | 236.4 | 1364.3 | 1360.7 | 3.5 | 2.2 |

Suffix prefills of 24–30 tokens took 191–233 ms. Lemonade's earlier
`prompt_ms` on similar suffixes was 147–227 ms. This run does not put
those suffix TTFTs under that range. Exact RAM hits are 2.4–3.6 ms.
Disk restore of the 431-token prefix was 84.9 ms.

## M4 canonical prefill grid

2026-10-29, gfx1151, `result/bin/gufo` (nix package, 16-token grid).
One session, context 16384, `--cache-disk`. Power at the start of this
replay: `AC0/online=1`, `BAT0/status=Not charging`. Lemonade's
llama-server (pid 3363206, state `Sl`, port 8003, MTP) was running and
was not paused. Temperature 0, seed 41, thinking off, at most 192 new
tokens, tools `read` and `edit`. The live cache-on conversation was
recorded, the server restarted, then those message lists ran cache on,
off, on, off. Every turn matched on content, tool name, arguments,
finish reason and completion tokens.

A repeated request (`on-b`) re-prefills only the partial cell, 0–12
tokens here. A new turn also re-prefills tokens produced by decode since
the previous grid boundary. That tail, not the 16-token cell, is most of
`prefill_tokens`. `on-a-0` is the disk restore of the 416-token boundary.

| Turn | Prompt | Cached | Prefill tokens | TTFT ms | Snapshot ms | Restore ms | Prefill ms | Cold TTFT ms | Repeat TTFT ms | Repeat prefill |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 438 | 416 | 22 | 342.9 | 33.5 | 93.6 | 206.8 | 2599.1 | 56.1 | 6 |
| 1 | 507 | 432 | 75 | 478.4 | 3.6 | 0 | 454.1 | 2942.5 | 72.6 | 11 |
| 2 | 620 | 496 | 124 | 768.6 | 5.3 | 0 | 728.9 | 3687.3 | 81.6 | 12 |
| 3 | 689 | 608 | 81 | 595.1 | 10.8 | 0 | 523.1 | 4135.4 | 28.5 | 1 |
| 4 | 759 | 688 | 71 | 498.2 | 5.5 | 0 | 457.4 | 4519.1 | 68.2 | 7 |
| 5 | 860 | 752 | 108 | 742.0 | 5.7 | 0 | 704.0 | 5176.0 | 84.5 | 12 |
| 6 | 928 | 848 | 80 | 549.8 | 11.1 | 0 | 512.1 | 5595.2 | 2.8 | 0 |

`On` is the disk-restart pass. Cold TTFT is the first cache-off pass.
Repeat is the second cache-on pass. Decode on these turns was 35.2–39.9
tok/s. Disk restore of the 416-token prefix was 93.6 ms. RAM restore on
the repeat pass was 1.5–2.5 ms.

The same afternoon, one server at a time, the pre-grid `build/release/gufo`
and this nix package alternated on two fixed prompts (`cache_prompt`
false, 32 new tokens). The 4012-token gap is the 16-token cell launches,
not a preset difference. The same-preset comparison below replaces that
explanation.

| Prompt tokens | Order | Pre-grid pp tok/s | Grid-16 pp tok/s | Pre-grid tg tok/s | Grid-16 tg tok/s |
| ---: | --- | --- | --- | --- | --- |
| 69 | old, new, old, new | 276.5, 263.9 | 164.4, 166.0 | 48.8, 43.3 | 43.3, 41.5 |
| 4012 | old, new, old, new | 842.2, 836.1 | 173.9, 169.8 | 38.2, 37.6 | 35.6, 36.8 |

## M4 wide cells, same preset

2026-10-29T11:22:49-05:00, gfx1151. Both binaries are the release preset.
`d005bfeb` is `build/d005-src/build/release/gufo`. The candidate is
`build/release-m4/gufo` at `75e04191` (whole 16-token cells in one launch).
Order is d005, candidate, d005, candidate. Fresh process per sample,
`cache_prompt` false, temperature 0, seed 41, thinking off, context 8192,
128 new tokens. Power: `AC0/online=1`, `BAT0/status=Not charging`.
Lemonade's llama-server (pid 3363206, state `Sl`, port 8003, MTP) stayed
up. Sentence repeats land on 262, 2052 and 4012 tokens. The 69-token row
is the short tool instruction.

| Prompt tokens | d005 pp tok/s | Candidate pp tok/s | d005 tg tok/s | Candidate tg tok/s |
| ---: | --- | --- | --- | --- |
| 69 | 257.6, 261.5 | 218.9, 222.1 | 41.33, 42.50 | 42.00, 41.39 |
| 262 | 586.6, 577.3 | 531.5, 532.7 | 41.11, 41.69 | 41.23, 40.61 |
| 2052 | 851.7, 868.5 | 867.0, 855.3 | 38.61, 38.97 | 38.39, 38.95 |
| 4012 | 819.7, 812.3 | 846.5, 846.2 | 36.52, 36.38 | 36.66, 36.77 |

At 2052 and 4012 the candidate is within the run's noise of `d005bfeb`
(means +0.1% and +3.7% prefill). At 69 and 262 it is slower (means
−15% and −8.6%). Those prompts end on a short cell, so the candidate
does a second launch; the long prompts do not. Decode means stay within
1.2%. The 69-token rows generated 24 tokens; the others generated 128.

Cached agent turns on that candidate, full `read`/`edit` schemas,
temperature 0, seed 41, thinking off, at most 192 new tokens. Live
cache-on, restart, then cache on, off, on, off. Every turn matched on
content, tool name, arguments, finish reason and completion tokens.
`On` is the disk-restart pass. Cold is the first cache-off pass. Repeat
is the second cache-on pass and re-prefills only the partial cell.

| Turn | Prompt | Cached | Prefill | TTFT ms | Snapshot ms | Restore ms | Prefill ms | Cold TTFT ms | Repeat TTFT ms | Repeat prefill |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 438 | 416 | 22 | 318.4 | 24.4 | 85.7 | 194.2 | 768.2 | 55.9 | 6 |
| 1 | 503 | 432 | 71 | 272.8 | 3.6 | 0 | 254.2 | 724.9 | 66.6 | 7 |
| 2 | 589 | 496 | 93 | 312.1 | 3.8 | 0 | 296.3 | 981.4 | 88.5 | 13 |
| 3 | 644 | 576 | 68 | 368.1 | 10.4 | 0 | 308.5 | 1105.7 | 55.4 | 4 |
| 4 | 699 | 640 | 59 | 288.6 | 5.4 | 0 | 254.0 | 1063.3 | 78.5 | 11 |
| 5 | 785 | 688 | 97 | 425.1 | 11.0 | 0 | 364.3 | 1247.3 | 28.4 | 1 |
| 6 | 840 | 784 | 56 | 270.9 | 5.6 | 0 | 236.1 | 1163.9 | 68.1 | 8 |

Disk restore of the 416-token prefix was 85.7 ms. RAM restore on the
repeat pass was 1.4–2.5 ms. A new turn still re-prefills the tokens
since the previous grid boundary (55–97 here), so those TTFTs stay
above Lemonade's 147–227 ms suffix prefills.

## M4 Gufo versus llama.cpp

2026-10-29T11:33:06-05:00, gfx1151. The llama.cpp figures in `d2c585a6`
were CPU-only and are superseded by this section. Gufo is the release
preset `build/release-m4/gufo` at `75e04191`. llama.cpp is `llama-b11580`.
`LD_LIBRARY_PATH` is Lemonade's gfx1151 tree
(`therock/gfx1151-10.1.0/lib`, its `lib/llvm/lib`, and the `llama-b11580`
directory). `-ngl 99` and `-lv 4` (the default log hides the device
line). Startup log: `ROCm0 : AMD Radeon 8060S Graphics`,
`offloaded 42/42 layers to GPU`, `ROCm0 model buffer size = 21353.95 MiB`,
`ROCm_Host model buffer size = 397.85 MiB`,
`ROCm0 KV buffer size = 320.00 MiB` at context 16384. The MTP process
reported `ROCm0 model buffer size = 22211.31 MiB`. One server at a time,
fresh process per sample except the agent replay. Power: `AC0/online=1`,
`BAT0/status=Not charging`. Lemonade's llama-server (pid 3363206, state
`Sl`, port 8003, MTP) stayed up. Prompts are 25 and 204 copies of the
M1 sentence (262 and 2052 tokens). Temperature 0, seed 41, thinking off.
Each row is two cycles, Gufo then llama.cpp.

Gufo:

```
build/release-m4/gufo serve --host 127.0.0.1 --port PORT llm --model Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf --context 16384 --sessions 1 --served-model-name ornith-m4 --cache-disk artifacts/m4/m4-llama-disk --cache-disk-bytes 8589934592
```

llama.cpp autoregressive:

```
LD_LIBRARY_PATH=<therock lib>:<therock llvm lib>:<llama-b11580> llama-server -m Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf --host 127.0.0.1 --port PORT --ctx-size 16384 --jinja --parallel 1 -ngl 99 -lv 4
```

llama.cpp MTP adds `--spec-type draft-mtp`. `cache_prompt` was false for
the length and MTP tables, true for the agent replay. Length and MTP
samples generate 64 tokens. Gufo rates are `prefill_tokens / prefill_ms`
and `completion_tokens_per_second`. llama.cpp rates are
`timings.prompt_per_second` and `timings.predicted_per_second`.

| Prompt tokens | Order | Gufo pp tok/s | llama.cpp pp tok/s | Gufo tg tok/s | llama.cpp tg tok/s |
| ---: | --- | --- | --- | --- | --- |
| 262 | gufo, llama, gufo, llama | 544.9, 528.2 | 610.7, 637.3 | 43.11, 42.12 | 44.71, 45.69 |
| 2052 | gufo, llama, gufo, llama | 881.3, 858.3 | 1090.9, 1085.8 | 40.97, 40.34 | 39.48, 38.99 |

Agent replay uses one process per engine per pass, at most 8 new tokens,
and a short `read`/`edit` schema, so these prompts are 402–804 tokens
rather than the full-schema identity table above. llama.cpp columns are
`timings.prompt_ms`. Gufo's second pass reuses `--cache-disk` (turn 0
cached 384 of 402, restore 85.5 ms, TTFT 308.2 ms). llama.cpp's second
pass is a new process, so its turn 0 is cold again.

| Turn | Prompt | Gufo TTFT | llama prompt ms | Gufo TTFT again | llama prompt ms again |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 402 | 693.3 | 599.3 | 308.2 | 607.4 |
| 1 | 467 | 338.0 | 161.6 | 330.3 | 163.1 |
| 2 | 553 | 292.6 | 205.9 | 294.6 | 215.1 |
| 3 | 608 | 276.6 | 161.0 | 279.0 | 172.0 |
| 4 | 663 | 249.8 | 174.3 | 240.7 | 191.6 |
| 5 | 749 | 315.2 | 194.7 | 326.3 | 196.2 |
| 6 | 804 | 369.3 | 175.8 | 372.9 | 172.4 |

Gufo cached tokens on the first pass were 0, 400, 464, 544, 608, 656,
736. llama.cpp `cache_n` on each pass was 0, 409, 463, 560, 604, 670,
756. The second Gufo pass cached 384, 400, 464, 544, 608, 656, 736.

MTP decode is Gufo autoregressive against llama.cpp `--spec-type
draft-mtp`, 64 new tokens, cache off, same interleave. Acceptance is
`draft_n_accepted/draft_n`.

| Prompt tokens | Gufo AR tg tok/s | llama.cpp MTP tg tok/s | llama.cpp drafts accepted |
| ---: | --- | --- | --- |
| 262 | 40.33, 41.03 | 38.22, 38.03 | 28/101, 28/101 |
| 2052 | 38.61, 37.90 | 36.44, 36.21 | 29/99, 29/99 |

On these prompts llama.cpp MTP decode is slower than the same process
shape's autoregressive decode above (about 45 and 39 tok/s). This run
does not reproduce an earlier ~46 tok/s MTP figure from the agent replay.

## Gufo MTP

2026-10-29T13:33:51-05:00, gfx1151. Power: `AC0/online=1`,
`BAT0/status=Not charging`. Lemonade's llama-server (pid 3363206, state
`Sl`, port 8003, MTP) stayed up. Binary `build/release-mtp/gufo`. Fresh
process per sample, two cycles, order Gufo autoregressive, Gufo
`--speculative mtp --draft-tokens 7`, llama.cpp `--spec-type draft-mtp`.
No prefix cache. Temperature 0, seed 41, thinking off, 64 new tokens.
Prompts are the same 262- and 2052-token sentences. llama.cpp startup
logged `ROCm0 : AMD Radeon 8060S Graphics`, `offloaded 42/42 layers to
GPU`, `ROCm0 model buffer size = 22211.31 MiB`,
`ROCm_Host model buffer size = 397.85 MiB`.

```
build/release-mtp/gufo serve --host 127.0.0.1 --port PORT llm --model Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf --context 16384 --sessions 1 --served-model-name ornith-m4 --speculative mtp --draft-tokens 7
```

llama.cpp adds `--spec-type draft-mtp` to the command in the section
above. Rates are `timings.prompt_per_second` and
`timings.predicted_per_second`. Acceptance is
`draft_n_accepted/draft_n`.

| Prompt | Gufo AR pp | Gufo MTP pp | llama.cpp MTP pp | Gufo AR tg | Gufo MTP tg | llama.cpp MTP tg | Gufo accepted | llama.cpp accepted |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 262 | 512.8, 520.4 | 508.9, 511.6 | 581.3, 584.2 | 41.12, 40.72 | 35.84, 36.18 | 40.06, 38.44 | 30/63, 30/63 | 28/101, 28/101 |
| 2052 | 860.3, 856.1 | 836.9, 830.2 | 990.9, 982.9 | 38.40, 38.17 | 33.61, 33.94 | 35.80, 35.92 | 30/63, 30/63 | 29/99, 29/99 |

Gufo MTP means are 36.01 and 33.77 tok/s. llama.cpp MTP means are 39.25
and 35.86 tok/s. Gufo MTP did not beat llama.cpp on either prompt, and
this run does not reproduce ~46 tok/s. Greedy text matched Gufo
autoregressive on all four samples. A width-8 verification batch was
slower (about 14 tok/s with a second forward, about 22 tok/s after a
DeltaNet prefix rollback) and is not the server path.

The same binary's agent replay, at most 8 new tokens and cache off,
matched autoregressive output on all 7 turns. No tool call was emitted;
the budget ends first, and turn 6 stopped after one token.

## M4 remedy: one launch per call, exact boundaries

2026-10-29, gfx1151. The candidate is the nix package at `0daeb148`
(`/nix/store/41mx15vg…-gufo-0-unstable-0daeb14`). The control is the nix
package at `d005bfeb`. Both are release builds. Lemonade's llama-server
(pid 3363206, state `Sl`, port 8003, MTP) stayed up and was not paused.
Power was not logged with these runs; the machine was on its wall adapter.

The M4 grid lost short prompts to extra forward passes. A prompt that
ended on a short cell ran it as a second launch, and the server stopped
prefill at every cache boundary. Each launch costs roughly 40–65 ms even
for a few tokens, because they touch most routed expert weights. Now each
call of up to 512 tokens is one launch, a one-token launch keeps the
batched kernels, and the server grid is width 1. With a RAM cache the
boundary image is frozen inside that launch (`Session::PrefillThrough`)
and copied to the host in the background. With `--cache-disk` prefill
still stops at the boundary.

Same-preset A/B, 16:31–16:33. Order d005, candidate, three cycles. Fresh
process per cycle, one warm-up request, then the four prompts.
`cache_prompt` false, temperature 0, seed 41, thinking off, context 8192,
at most 128 new tokens. The 18-token prompt is `Reply with the word OK.`
(one new token), 69 is the short tool instruction, and 262 and 2052 are
the sentence repeats.

| Prompt tokens | d005 pp tok/s | Candidate pp tok/s | d005 TTFT ms | Candidate TTFT ms | d005 tg tok/s | Candidate tg tok/s |
| ---: | --- | --- | --- | --- | --- | --- |
| 18 | 133.5, 131.8, 123.0 | 184.4, 177.5, 167.3 | 167.9, 167.3, 174.5 | 99.5, 103.9, 110.0 | — | — |
| 69 | 275.7, 279.7, 266.7 | 349.7, 342.3, 331.8 | 282.7, 276.9, 288.3 | 198.6, 203.0, 209.8 | 40.64, 40.16, 39.91 | 40.90, 40.72, 39.76 |
| 262 | 604.9, 598.2, 585.9 | 682.4, 690.7, 657.9 | 467.4, 467.4, 478.4 | 385.5, 380.6, 399.7 | 40.15, 40.71, 39.68 | 40.58, 39.92, 39.82 |
| 2052 | 831.9, 827.5, 807.3 | 872.8, 853.9, 830.8 | 2517.3, 2523.7, 2585.0 | 2352.8, 2404.3, 2471.9 | 38.11, 38.15, 37.44 | 38.41, 38.04, 38.47 |

Median prefill gains are 34.7%, 24.2%, 14.1% and 3.2%. Decode medians
differ by +1.4%, −0.6% and +0.8%. Both columns drift down over the three cycles. On `d005bfeb`
the first token arrives 28–51 ms after `prefill_ms` ends; on the candidate
it is 1.3–2.5 ms.

Gufo against llama.cpp `llama-b11580` on the GPU, 16:33–16:35, with the
command lines and library path of the M4 llama.cpp section. Startup logged
`offloaded 42/42 layers to GPU` and `ROCm0 model buffer size = 21353.95
MiB`. Gufo used a RAM cache (no `--cache-disk`), context 16384. Fresh
process per sample, Gufo then llama.cpp, two cycles, 64 new tokens.

| Prompt tokens | Gufo pp tok/s | llama.cpp pp tok/s | Gufo tg tok/s | llama.cpp tg tok/s |
| ---: | --- | --- | --- | --- |
| 262 | 661.6, 662.7 | 591.3, 619.9 | 39.75, 40.17 | 39.82, 38.51 |
| 2052 | 863.8, 861.5 | 1044.2, 1030.4 | 38.43, 38.14 | 37.52, 38.05 |

llama.cpp's `predicted_per_second` divides by one token fewer than it
generated. Gufo's rate divides by all completion tokens. Computed the
same way as Gufo's, llama.cpp decode is 40.46, 39.12 and 38.12, 38.66
tok/s, and the means differ by under 0.4%. llama.cpp prefill means are
8.5% slower at 262 tokens and 20.2% faster at 2052.

Agent replay, same processes: one per engine per pass, the short
`read`/`edit` schema, at most 8 new tokens, `cache_prompt` true. Both
passes start cold. Gufo columns are TTFT. llama.cpp columns are
`timings.prompt_ms`.

| Turn | Prompt | Gufo cached | Gufo prefill | Gufo TTFT | llama `cache_n` | llama prompt ms | Gufo TTFT again | llama prompt ms again |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 402 | 0 | 402 | 571.2 | 0 | 628.7 | 573.6 | 633.5 |
| 1 | 467 | 402 | 65 | 194.5 | 409 | 189.4 | 188.8 | 189.4 |
| 2 | 553 | 467 | 86 | 254.6 | 463 | 244.0 | 254.9 | 236.3 |
| 3 | 608 | 553 | 55 | 206.5 | 560 | 187.1 | 208.4 | 182.7 |
| 4 | 663 | 608 | 55 | 205.5 | 604 | 200.6 | 206.6 | 202.7 |
| 5 | 749 | 663 | 86 | 246.2 | 670 | 215.6 | 254.8 | 221.8 |
| 6 | 804 | 749 | 55 | 206.7 | 756 | 183.3 | 210.1 | 186.0 |

Gufo now caches each previous prompt exactly and re-prefills the 55–86
reply and tool-result tokens. Cached turns are 188.8–254.9 ms against
llama.cpp's 182.7–244.0 ms. Gufo's cold turn 0 is 9% faster.
`cache_snapshot_ms` was 6.7–8.0 ms on cached turns.

Full-schema identity replay with the RAM cache, 16:09, release preset with
the host-only cache change before it was committed as `0daeb148`. One
process: the live cache-on conversation, then its message lists with cache
off, then cache on again. Every turn matched the live pass on content,
tool name, arguments, finish reason and completion tokens.

| Turn | Prompt | Cached | Prefill | Live TTFT ms | Cold TTFT ms | Repeat TTFT ms | Repeat restore ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 438 | 0 | 438 | 602.2 | 627.0 | 9.7 | 8.7 |
| 1 | 503 | 438 | 65 | 235.9 | 687.8 | 10.8 | 9.6 |
| 2 | 589 | 503 | 86 | 261.9 | 923.4 | 11.0 | 9.8 |
| 3 | 644 | 589 | 55 | 214.5 | 980.4 | 11.1 | 9.8 |
| 4 | 699 | 644 | 55 | 212.0 | 1006.8 | 11.3 | 10.1 |
| 5 | 785 | 699 | 86 | 267.4 | 1124.7 | 11.2 | 9.9 |
| 6 | 840 | 785 | 55 | 211.0 | 1165.6 | 11.5 | 10.2 |

Repeat restores now upload the host payload (8.7–10.2 ms). The 15:12 run
of the same replay before that change kept a device image per entry and
restored in 1.5–2.3 ms, but a long `cache-messages-loop` exhausted GPU
memory with Lemonade resident. That 15:12 run used `--cache-disk`, where
prefill stops at the boundary: cached turns were 255.0–330.2 ms, and the
disk restore of 431 tokens after restart was 98.1 ms.

## M5 prefill launch and routed tile

2026-10-31, gfx1151. Release preset. Power: `AC0/online=1`,
`BAT0/status=Not charging`. Lemonade's llama-server (pid 3363206, state
`Sl`, port 8003) stayed up and was not paused. Two kept changes, measured
separately. `3f5bce75` raises `Session::kMaxPrefillChunk` from 512 to 4096.
`f88b29e2` uses a 64-column routed MMQ tile when the prefill chunk has at
least 1024 tokens and 48 columns below that. The down projection follows
the chunk length, not its expanded expert-row count. ExplicitContract stays
on. Frozen binaries: `build/release-m5-base/gufo` at `001a6290`,
`build/release-m5-c4096/gufo` (4096, tile 48). The tile binary is
`build/release/gufo`.

rocprofv3 through `tools/prof/prof.py`, one warm `ornith15_bench` prefill
(`GUFO_ORNITH15_BENCH_SAMPLES=1`, `WARM_STEPS=0`, `DECODE=0`). Class times
are sums of dispatch durations and can exceed the GPU span. The 2048
"before" span is the sum of four 512-token chunk spans. The warm pass is
the second launch. Raw tables: `artifacts/m5/prof-classes.txt`.

| Class | 262 chunk 512 | 262 chunk 4096 | 2048 × 512 | 2048 one launch |
| --- | ---: | ---: | ---: | ---: |
| Routed IQ4_XS | 99.58 | 94.07 | 607.97 | 380.59 |
| Routed Q6_K | 71.84 | 67.00 | 394.20 | 230.07 |
| Routed Q5_K | 51.11 | 49.54 | 322.61 | 194.82 |
| Dense Q6 projections | 76.93 | 72.87 | 491.59 | 351.34 |
| Shared Q8_0 | 8.83 | 8.32 | 52.47 | 38.13 |
| DeltaNet | 18.63 | 17.42 | 169.61 | 152.81 |
| Attention | 4.74 | 4.55 | 149.17 | 137.25 |
| F32 router/shexp | 18.45 | 17.34 | 142.24 | 149.90 |
| MoE epilogue/SwiGLU | 11.89 | 11.60 | 122.04 | 140.06 |
| Quantize and ids | 10.91 | 10.51 | 81.87 | 101.09 |
| Norm, residual, embed | 3.24 | 3.04 | 25.36 | 25.44 |
| Vocab GEMV | 1.97 | 1.87 | 1.98 | 1.97 |
| Runtime fill/copy | 2.23 | 2.15 | 6.55 | 1.79 |
| GPU span | 366.4 | 365.9 | 2016.5 | 1425.1 |

The tile-64 profile uses the same classes. At 262 tokens the routed kernels
are still tile 48. At 2048 they are tile 64. Dense Q6 stays tile 80.

| Class | 262 (tile 48) | 2048 (tile 64) |
| --- | ---: | ---: |
| Routed IQ4_XS | 93.04 | 324.74 |
| Routed Q6_K | 66.66 | 193.01 |
| Routed Q5_K | 48.95 | 168.39 |
| Dense Q6 projections | 67.80 | 337.48 |
| Shared Q8_0 | 8.13 | 37.83 |
| DeltaNet | 22.63 | 177.38 |
| Attention | 4.58 | 128.21 |
| F32 router/shexp | 18.36 | 181.73 |
| MoE epilogue/SwiGLU | 11.74 | 138.91 |
| Quantize and ids | 10.81 | 100.78 |
| Norm, residual, embed | 3.64 | 27.56 |
| Vocab GEMV | 1.89 | 1.94 |
| Runtime fill/copy | 2.61 | 2.40 |
| GPU span | 374.2 | 1448.1 |

Routed IQ4_XS + Q6_K + Q5_K at 2048 tokens is 805.48 ms with tile 48 and
686.14 ms with tile 64. The GPU span did not fall with the kernel sum
(1425.1 ms to 1448.1 ms).

Server rates are interleaved, two cycles, fresh process per sample, same
prompts, temperature 0, seed 41, thinking off, `cache_prompt` false, 64 new
tokens except the 18-token prompt. Both engines use
`prefill_tokens / prefill_ms` and `predicted_n / predicted_ms`.
llama.cpp's `predicted_per_second` divides by `predicted_n - 1` and is not
this column. The 18-token Gufo completion is 1 token; llama.cpp reports
`predicted_n` 2, so that row is not a sustained decode rate. Harness:
`artifacts/m5/llama_m5.py`. Prompts: `Reply with the word OK.` (18), and
the M1 sentence repeated to 262 and 2052 tokens.

```
build/release-m5-base/gufo serve --host 127.0.0.1 --port PORT llm --model Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf --context 16384 --sessions 1 --served-model-name ornith-m5
```

The 4096 and tile binaries use the same arguments. llama.cpp:

```
LD_LIBRARY_PATH=<therock gfx1151 lib>:<therock llvm lib>:<llama-b11580> llama-server -m Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf --host 127.0.0.1 --port PORT --ctx-size 16384 --jinja --parallel 1 -ngl 99 -lv 4
```

Every llama.cpp start logged `offloaded 42/42 layers to GPU`,
`ROCm0 model buffer size = 21353.95 MiB`,
`ROCm_Host model buffer size = 397.85 MiB`,
`ROCm0 KV buffer size = 320.00 MiB`,
`ROCm0 RS buffer size = 62.81 MiB`, device AMD Radeon 8060S Graphics
(`0000:c4:00.0`).

Order base (`001a6290`), 4096 with tile 48, llama.cpp:

| Prompt | Base pp | 4096 pp | llama.cpp pp | Base tg | 4096 tg | llama.cpp tg |
| ---: | --- | --- | --- | --- | --- | --- |
| 18 | 140.7, 143.1 | 141.9, 137.3 | 120.8, 120.7 | — | — | — |
| 262 | 698.1, 669.2 | 677.5, 672.2 | 655.5, 637.5 | 40.38, 40.40 | 40.93, 40.56 | 44.41, 44.35 |
| 2052 | 904.1, 903.3 | 1053.9, 1041.8 | 1092.2, 1066.0 | 38.46, 38.42 | 39.20, 38.48 | 37.97, 38.30 |

2052 prefill means are 903.7, 1047.8 and 1079.1 tok/s. The 4096 launch is
faster than `001a6290` on both cycles and still short of llama.cpp. At 262
tokens the first cycle is 677.5 against 698.1 and the second is 672.2
against 669.2 (means 674.8 and 683.6). Decode is not slower. The 18-token
prefill stays with the baseline.

Order 4096 with tile 48, tile 64 from 1024 tokens, llama.cpp. The 18-token
path was not repeated: a chunk under 1024 tokens stays on tile 48.

| Prompt | Tile 48 pp | Tile 64 pp | llama.cpp pp | Tile 48 tg | Tile 64 tg | llama.cpp tg |
| ---: | --- | --- | --- | --- | --- | --- |
| 262 | 678.5, 688.1 | 678.3, 674.4 | 585.0, 622.4 | 41.82, 46.03 | 41.23, 40.35 | 40.22, 44.52 |
| 2052 | 1052.7, 1053.3 | 1076.8, 1079.2 | 1066.0, 1064.6 | 39.03, 38.75 | 38.98, 38.88 | 37.91, 39.59 |

2052 prefill means are 1053.0, 1078.0 and 1065.3 tok/s. Tile 64 is ahead of
llama.cpp on both cycles (1076.8 vs 1066.0, then 1079.2 vs 1064.6) and
ahead of tile 48. At 262 tokens the first cycle matches (678.3 vs 678.5)
and the second is 674.4 vs 688.1. Decode at 2052 matches. The 46.03 tok/s
tile-48 sample is one short decode interval (1390 ms); the other 262 decode
samples sit at 40–42 tok/s. Raw rows: `artifacts/m5/m5-length.json` and
`artifacts/m5/m5-tile.json`.

## M5 final prefill, same formula

2026-10-31, after `c6e64988`. The autoregressive kernels are the tile-64
launch above. `a5e5feff` and `fd08363f` do not change them; this table
was taken on `build/release/gufo` before those two rebuilds. Baseline is
`build/release-m5-base/gufo` at `001a6290`. Fresh process per sample, two
cycles, order base, final, llama.cpp. Same prompts, temperature 0, seed
41, thinking off, `cache_prompt` false, 64 new tokens except the
18-token prompt. Rates are `prefill_tokens / prefill_ms` and
`completion_tokens / decode_ms` (llama.cpp `predicted_n / predicted_ms`).
`predicted_per_second` is not this column. The 18-token Gufo completion
is 1 token. Harness: `artifacts/m5/llama_m5.py`. Rows:
`artifacts/m5/m5-final-prefill.json`.

```
build/release-m5-base/gufo serve --host 127.0.0.1 --port PORT llm --model Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf --context 16384 --sessions 1 --served-model-name ornith-m5
```

The final binary uses the same arguments. llama.cpp is the command in
the section above (`-ngl 99 -lv 4`, no `--spec-type`). Every llama.cpp
start in this table logged `offloaded 42/42 layers to GPU` and
`ROCm0 model buffer size = 21353.95 MiB`.

| Prompt | Base pp | Final pp | llama.cpp pp | Base tg | Final tg | llama.cpp tg |
| ---: | --- | --- | --- | --- | --- | --- |
| 18 | 141.8, 138.5 | 135.7, 138.4 | 121.9, 121.7 | — | — | — |
| 262 | 687.3, 677.8 | 683.8, 696.3 | 645.0, 630.4 | 40.73, 41.46 | 40.68, 46.01 | 44.59, 44.70 |
| 2052 | 914.8, 911.2 | 1094.2, 1082.6 | 1088.2, 1072.5 | 38.96, 38.80 | 38.98, 38.61 | 37.81, 38.34 |

2052 prefill means are 913.0, 1088.4 and 1080.4 tok/s. The final binary
is ahead of llama.cpp on both cycles and ahead of `001a6290`. At 262
tokens the first cycle is 683.8 against 687.3 and the second is 696.3
against 677.8. The 46.01 tok/s sample is one short decode (1391 ms); the
other final 262 decode is 40.68, next to the baseline. The 18-token
prefill is 135.7 then 138.4, against 141.8 and 138.5.

## M5 MTP decode

Same day, release `build/release/gufo` with the provisional
`kMtpRelativeCost` table, before the claim fix and the per-request
policy reset. Fresh process, two cycles, order Gufo autoregressive, Gufo
`--speculative mtp --draft-tokens 7`, llama.cpp `--spec-type draft-mtp`.
Cache off. Temperature 0, seed 41, thinking off, 64 new tokens. Same
262- and 2052-token prompts. Formula as above, every completion token.
Acceptance is `draft_tokens_accepted/draft_tokens`. Rows:
`artifacts/m5/m5-mtp-speed.json`.

```
build/release/gufo serve --host 127.0.0.1 --port PORT llm --model Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf --context 16384 --sessions 1 --served-model-name ornith-m5 --speculative mtp --draft-tokens 7
```

llama.cpp adds `--spec-type draft-mtp` and logged `offloaded 42/42 layers
to GPU`, `ROCm0 model buffer size = 22211.31 MiB`.

| Prompt | Gufo AR tg | Gufo MTP tg | llama.cpp MTP tg | Gufo accepted | llama.cpp accepted |
| ---: | --- | --- | --- | --- | --- |
| 262 | 41.01, 40.63 | 33.77, 33.53 | 38.28, 41.20 | 14/19, 14/19 | 28/101, 28/101 |
| 2052 | 39.66, 38.56 | 32.49, 32.76 | 36.64, 36.19 | 14/19, 14/19 | 29/99, 29/99 |

Gufo MTP is slower than Gufo autoregressive and slower than llama.cpp
MTP on both prompts and both cycles. It is not the default.

A fixed-width sweep on a temporary cost table (every entry 0.05, not the
shipped binary; the table was restored afterward) did not find a faster
width. Accepted draft tokens stayed near 28–30 while the draft count
grew. Rows: `artifacts/m5/m5-mtp-sweep.json`.

| Prompt | AR tg | k=1 tg | k=2 tg | k=4 tg | k=7 tg |
| ---: | --- | --- | --- | --- | --- |
| 262 | 40.34, 40.73 | 32.40, 32.21 | 30.00, 29.77 | 24.94, 24.79 | 20.59, 20.59 |
| 2052 | 39.09, 38.67 | 30.76, 30.53 | 27.71, 27.56 | 22.95, 22.74 | 18.78, 18.78 |

Acceptance, same order of cycles: 262 tokens k=1 29/34, k=2 30/66, k=4
30/130, k=7 30/221; 2052 tokens k=1 28/35, k=2 28/70, k=4 28/138, k=7
28/236. Both cycles repeated those counts.

The 7-turn agent replay, cache off, at most 128 new tokens, same
interleave (`artifacts/m5/m5-agent-speed.json`). Summed completion
tokens over decode milliseconds: cycle 0 is 37.19 autoregressive,
31.72 Gufo MTP (106/125) and 41.34 llama.cpp MTP (148/366); cycle 1 is
37.32, 31.40 (106/125) and 42.16 (148/366). Gufo MTP matched
autoregressive tool calls. llama.cpp MTP did not match the edit
arguments. Greedy prose on the speed prompts diverges; see QUALITY.md.

## M5 review

2026-10-31, gfx1151, production `nix build` packages: `0daeb14` (M4
remedy, the same kernels as `001a6290`) and `72e51cb` (M5 final).
Power: `AC0/online=1`, `BAT0/status=Not charging`. Lemonade's
llama-server (pid 3363206, port 8003) stayed `Sl`. One process per
engine per cycle, one 8-token warm-up request, then the prompts in
order; engine order rotates each cycle. Cache off, temperature 0, seed
41, thinking off, 64 new tokens. Rates use every completion token on
both engines. llama.cpp `b11580` ran `-ngl 99 -lv 4` with Lemonade's
ROCm libraries and logged `offloaded 42/42 layers to GPU` at every
start. Harness: `artifacts/m5-review/review_ab.py` (reuses
`artifacts/m5/m5_mtp_speed.py`). Rows: `artifacts/m5-review/review-ar.json`
and `review-mtp.json`.

Autoregressive, four cycles, median (per-cycle values in the rows):

| Prompt | `0daeb14` pp | `72e51cb` pp | llama.cpp pp | `0daeb14` tg | `72e51cb` tg | llama.cpp tg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 183.8 | 183.6 | 171.1 | — | — | — |
| 262 | 706.2 | 699.3 | 697.2 | 39.91 | 39.55 | 36.72 |
| 2052 | 859.0 | 1035.2 | 1046.9 | 37.76 | 38.04 | 36.97 |

2,052-token prefill rose 20.5% over the M4 remedy. llama.cpp was ahead
on all four cycles (1077.0 vs 1075.5, 1054.8 vs 1051.6, 1039.0 vs
1011.2, 1038.0 vs 1018.9), so the M5 section's "ahead of llama.cpp" is
not reproduced: the two are level within about 1%. 18- and 262-token
prefill and decode are unchanged against `0daeb14`.

MTP (`--speculative mtp --draft-tokens 7`) against autoregressive on the
same `72e51cb` binary and llama.cpp `--spec-type draft-mtp`, three
cycles, median tg (accepted/drafted were identical every cycle):

| Prompt | Gufo AR | Gufo MTP | llama.cpp MTP | Gufo accepted | llama.cpp accepted |
| ---: | ---: | ---: | ---: | --- | --- |
| 262 | 39.16 | 34.85 | 35.57 | 14/19 | 28/101 |
| 2052 | 37.55 | 33.84 | 35.91 | 14/19 | 29/99 |

Gufo MTP is 11% and 10% slower than Gufo autoregressive, which is faster
than llama.cpp MTP on both prompts.

MTP with `--cache-disk`, 7-turn agent replay (`m5_agent.py identity`):
full-prompt RAM repeats reached the first token in 9.5–10.6 ms (restore
8.3–9.4 ms); turns that prefill a new suffix took 282–340 ms, against
189–255 ms for autoregressive cached turns in the M4 remedy review. The
disk hit after restart restored 431 tokens in 116.7 ms (first token
252 ms).

## M6 review

2026-10-31, gfx1151, production `nix build` packages: `72e51cb` (M5 final)
and `4406552` (M6). Same method as the M5 review: AC power, one process per
engine per cycle, one warm-up, rotated order, cache off, temperature 0,
every completion token in the rate. llama.cpp `b11580` logged
`offloaded 42/42 layers to GPU` at each of its nine starts. Harness and rows:
`artifacts/m6-review/`.

Autoregressive, four cycles, median:

| Prompt | `72e51cb` pp | `4406552` pp | llama.cpp pp | `72e51cb` tg | `4406552` tg | llama.cpp tg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 189.4 | 193.5 | 192.5 | — | — | — |
| 262 | 718.4 | 757.5 | 738.3 | 43.74 | 43.37 | 40.94 |
| 2052 | 1087.4 | 1148.8 | 1071.6 | 40.91 | 40.57 | 40.14 |

`Project` now launches an F32 matrix as one grouped grid instead of one launch
per eight tokens. `4406552` led llama.cpp at 2,052 tokens on all four cycles
(1205.9 vs 1112.7, 1182.8 vs 1068.5, 1114.7 vs 1074.6, 1112.3 vs 1063.8).
18-token prefill is level with llama.cpp.

MTP (`--speculative mtp --draft-tokens 7`) against autoregressive on
`4406552` and llama.cpp `--spec-type draft-mtp`, three cycles, median tg:

| Prompt | Gufo AR | Gufo MTP | llama.cpp MTP | Gufo accepted | llama.cpp accepted |
| ---: | ---: | ---: | ---: | --- | --- |
| 262 | 41.12 | 48.20 | 37.21 | 28/41 | 28/101 |
| 2052 | 40.01 | 45.88 | 37.62 | 28/40 | 29/99 |

7-turn agent replay, cache off, at most 128 new tokens, summed completion
tokens over decode milliseconds, two cycles: autoregressive 37.38 / 36.56,
Gufo MTP 45.87 / 45.43 (117/142 accepted), llama.cpp MTP 42.50 / 42.18
(148/366 accepted, 277 tokens rather than 235 because its output differs).

One verification launch with the decode dots, host wall time on a working-tree
release build at 261 / 2,052 prompt tokens (`GUFO_ORNITH15_BENCH_MTP=1`):
decode step 25.2 / 26.8 ms, draft 2.8 / 2.9 ms, width 2 29.6 / 31.1 ms, width 4
42.0 / 45.1 ms, width 8 72.8 / 75.5 ms. `kMtpRelativeCost` is the mean of those
rounds over one step.

MTP with `--cache-disk`, agent replay: cold, RAM repeat and disk after restart
gave identical output on all 7 turns. RAM repeats reached the first token in
9.7–14.6 ms; turns that prefill a new suffix took 289–364 ms, almost all suffix
prefill (260–330 ms). No paired autoregressive run was taken on these turns.
