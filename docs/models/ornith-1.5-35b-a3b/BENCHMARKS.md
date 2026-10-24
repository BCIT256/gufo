# Ornith GPU AR checkpoint measurements

Original P4 evidence is retained below; the P4b follow-up at the end records
the current one-pass policy.

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
