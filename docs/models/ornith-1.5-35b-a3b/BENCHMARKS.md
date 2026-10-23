# Ornith GPU AR checkpoint measurements

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
