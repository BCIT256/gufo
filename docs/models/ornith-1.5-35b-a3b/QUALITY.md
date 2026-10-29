# Ornith GPU AR checkpoint quality

Historical P4/P4b evidence is retained below. The P4c section records the
policy choice. The M1 section replaces the GPU acceptance gate.

October 23, 2026; synthetic operators and teacher-forced full-vocabulary
comparisons at 183 positions in three fixed histories. This is a numerical
checkpoint, not broad model capability, tool-use or long-context qualification.

The GPU session and independent scalar `ReferenceModel` consume exactly the
token IDs in the llama.cpp captures. The scalar reference's semantic graph
was separately validated against llama.cpp; it shares no GPU operators.
KL is computed in F64 over the entire vocabulary, in nats. The acceptance
rules remain fixed:

- No GPU top-1 flip where the scalar reference's top-two margin exceeds 1.0.
- KL(reference || GPU) at most 0.1 when reference entropy is below 0.5.
- Per-prompt mean KL(llama batch || GPU) at most twice llama.cpp's own mean
  KL(batch || sequential) on the same tokens.

The maximum absolute logit error and top-1 agreement are reported at every
position. Both implementations optionally expose the residual after each
layer. On numerical failure the test reports the first layer with residual
relative RMS above 0.1. This is a diagnostic threshold, not an additional or
replacement acceptance bound. Context exhaustion and invalid calls are
checked. Reset, fresh-session state and traced/untraced GPU replay are checked
at the first token of each history.

The model SHA256 is recorded in [the checkpoint guide](README.md). The fork
base is `53b0f5b728e0ba2225c02c26e8c3f29b640297af`; llama.cpp is
`68d9053afd4f4d0752ced6187585f862355a40be` (CPU reference build).
The pinned GPU toolchain uses nixpkgs
`2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`, ROCm 7.2.3, HIP clang 22.0.0,
GCC 15.3.0, and CMake 4.3.4; RelWithDebInfo with assertions enabled.
`flake.lock` SHA256 is
`faabad820ec09caab7351bf48910cec895a38cf1766e098e3065aea86f405ed6`.

Capture identities (ORNL v1; each batch/sequential pair has identical tokens):

| History | Batch SHA256 | Sequential SHA256 |
| --- | --- | --- |
| reasoning_history | `2f3efea0ffa746fce2543b9dff10aaeb382585c757b3230bc81378885d712552` | `f6c4122c5e7ec7c2ea01ceb2d489f4719976579af3c3b0cbba2e13c10b649738` |
| tool_round_trip | `4414159be4397c0bd2fadd4da9c89655148225f3a2cffa01484ebd28a42e2b75` | `6a7b9a57713dfa813e1594e32de10618aa04d3649ca0c6720e7e64008fd42bff` |
| user_turn | `0bfcc756a3ecd9c750cd117913dea2d137a74688fce7164feac16aafdf601f41` | `5c595f639f7a704dddb06547b6590361da012a0b23ad8b0b8a604d66234a81cd` |

A temporary sigmoid-for-SiLU GPU gate mutation failed at reasoning-history
position 0: KL(reference || GPU) 12.66098026, KL(llama || GPU) 12.66150897,
maximum logit error 24.45965654, and both top-1 comparisons disagreed. The trace
reported layer 0 first (relative RMS 1.022276042). CTest exited 8. SiLU was
restored before the final full-suite run; the mutation was not committed.

The default Flash-Next GDN path's 384 instrumented output/state/history/rollback
download hashes match its pre-change baseline exactly. Its existing tests are
unchanged. This is bounded operator evidence, not a new Flash-Next model sweep.
Raw captures remain in the preparation checkout's ignored
`artifacts/reference/oracle/`. Traces, per-position rows, build/test logs and
identity manifests remain outside the fork in ignored `artifacts/p4/`.

Full GPU lockstep results (all fixed acceptance rules passed):

| History | Positions | Top-1 vs reference / llama | KL vs reference mean / max | KL vs llama mean / max | Confident max KL vs reference / llama | Max absolute logit error |
| --- | ---: | --- | --- | --- | --- | ---: |
| reasoning_history | 87 | 87/87 / 77/87 | 0.0009218999007 / 0.03443386001 | 0.06561303982 / 1.788048549 | 0.0002805819905 / 0.02214198211 | 2.565742761 |
| tool_round_trip | 75 | 74/75 / 71/75 | 0.004077427532 / 0.0953593984 | 0.05441090237 / 0.5700931291 | 0.002146017432 / 0.03571511562 | 2.462987423 |
| user_turn | 21 | 21/21 / 19/21 | 6.722697007e-08 / 2.888812344e-07 | 0.02487865833 / 0.1960749618 | 7.269799356e-08 / 0.01230913494 | 0.2022695541 |

No clear-margin reference top-1 flips occurred. The sole reference top-1
disagreement was below the fixed 1.0-logit margin. Each prompt's llama
batch/sequential mean KL was 0.05104379883, 0.08534505311, 0.0324381, respectively.

Required configure/build succeeded. CTest: 29 passed, zero failed/skipped
(883.33 seconds), including all 21 unchanged Flash-Next tests. Formatting:
650 C++ files passed. Hosted PR suite: 66 passed, zero failed/skipped
(53.83 seconds). The model was rehashed and matched its pinned SHA256.

## P4b follow-up: one weight pass per projection

The tables above retain the original P4 checkpoint. P4b removes every
activation-residual pass and keeps F32 activation inputs only at full
attention output projections. All 183 original positions pass the unchanged
criteria, with no clear-margin flips. The independent CPU and GPU oracle
test sources are byte-identical to `49113bb2`; final qualification is uncached.

| History | Top-1 reference / llama | Mean KL reference / llama | Confident maximum KL vs reference |
| --- | --- | --- | ---: |
| reasoning_history | 84/87 / 78/87 | 0.02666456063 / 0.04671976705 | 0.02819847754 |
| tool_round_trip | 70/75 / 67/75 | 0.04959738944 / 0.07962550927 | 0.08232510347 |
| user_turn | 19/21 / 19/21 | 0.009889620099 / 0.02028499337 | 0.001946743154 |

KL against the scalar reference is higher than with blanket correction;
this is qualification within the fixed envelope, not numerical equivalence.
See [the full before/after record](EXPERIMENTS.md#final-uncached-qualification)
for maxima, rejected controls, remaining uncertainties and test evidence.

## P4c held-out policy decision

Current policy: **B** under the rule committed before measurement.
The current clear-flip criterion requires a reference margin above 1.0
logit and agreement of both llama.cpp captures with reference top-1.
Reference entropy < 0.5, confident KL <= 0.1, and per-prompt mean
KL(llama || GPU) <= 2x KL(llama batch || sequential) remain unchanged.
The test completes all histories and returns failure if any criterion fails.

Held-out failures: A 6, B 3.
This is a bounded policy decision, not a general capability, long-context
or exact-logit-equivalence claim. Full per-prompt counts, confidence maxima,
KL directions and original-prompt before/after results are in
[EXPERIMENTS.md](EXPERIMENTS.md#p4c-held-out-results).

## P4d: scalar-oracle failures and control limits

October 24, 2026 diagnostic record. Production policy remains B. No threshold
changed. The details, identities and control limits are in
[EXPERIMENTS.md](EXPERIMENTS.md#p4d-prior-evidence-recorded-october-24-2026-not-a-new-measurement).

The scalar reference fails its existing native-oracle contract on held-out
`code_diff`, `plain_prose` and `tool_two_parameters`. `kept_reasoning`,
`long_reasoning` and `spanish` pass. Native batch and sequential captures
themselves reach confident KL 5.0404 on code and 1.4487 on the tool history.
That spread does not waive B's three GPU failures and does not relax the
GPU gates.

F32 K/V storage alone did not change llama.cpp captures, because Flash
Attention casts K/V back to F16. Disabling Flash Attention as well, and
evaluating Q8_0, Q5_K, Q6_K and IQ4_XS dots in F64 on dequantized blocks,
brings sequential prose and tool top-1 to full agreement with the scalar
reference and leaves code at 108/110 with mean KL about 0.00364. F32-weight
projections, including routing, were not part of that change. An unmodified
quantized rebuild reproduced the three original sequential captures
byte-for-byte. These controls are not replacement goldens.

A uniform second activation-residual pass on the remaining Q8_1 projections
removed the tool flips and created confident KL 5.280833756 at code position
54. It was rejected. B's binary was restored to
`a5358cb088996ee0b69dec483edcedd21789c062535a8bb76e9cb2c5fb7d332d`.
The next measurement protocol is registered in
[EXPERIMENTS.md](EXPERIMENTS.md#p4e-protocol-registered-october-24-2026-before-new-scoring)
and is not itself a result.

## M1 acceptance reset

User-approved replacement of the GPU gate. The historical per-position rules
above stay in the record. They are no longer acceptance criteria for
`ornith15.gpu_decode`.

The gate is distribution-level, on whichever capture directory
`GUFO_ORNITH15_LLAMA_LOGITS` points at:

- Per prompt, mean KL(llama batch || GPU) <= 2.0 times mean KL(llama batch ||
  llama sequential).
- Every position's KL(llama batch || GPU) <= 3.0 times the largest
  per-position KL(llama batch || llama sequential) in that capture set.
  A zero self-spread rejects any positive GPU KL.

Clear flips and confident-position KL are diagnostics. They are computed only
when `GUFO_ORNITH15_SCALAR_REFERENCE=1`. The scalar reference and
`ornith15.reference_oracle` are unchanged. The GPU test leaves the reference
off unless that variable is set. Thresholds are not tuned from results.
Chunked prefill is scored at 16, 48, 64 and 512 tokens.

## Session snapshot

`ornith15.gpu_snapshot` uses the same M1 ratios, unchanged. A continuation
after `SaveSnapshot` must match the continuation after `RestoreSnapshot`
bit for bit: into a fresh session, into a session that has already prefilled
another prompt, and back into the source session. The checked frontiers are
P=1, P=7, P=17 inside a 16-token chunk, P=48 with chunk 64 when the history
is longer than 50, and P=17 immediately after a decode `Step`.

Oracle (3 histories, set max self-KL 1.039941167) and held-out (6 histories,
set max self-KL 5.040428549) both passed. Every compared continuation logit
was bit-identical (max abs error 0). Two independent reaches of the same
frontier produced equal payloads. Zeroing layer 0's recurrent state changed
the next token's logits on every history; a truncated payload was rejected
and left the position unchanged; layer 3, a full-attention layer, has no
recurrent region. Restored full sequences passed M1 on every frontier.

Payload bytes depend only on position: 66877504 at P=1, 67000384 at P=7,
67205184 at P=17, and 67840064 at P=48. Each additional token adds 20480
bytes, ten F16 K/V layers.

The same gate covers a staged device image. At each frontier the test
copies with `SaveSnapshot`, then `StageSnapshot`, prefills past that
frontier, and `ExportStagedSnapshot`. The exported bytes must equal the
stop-at-boundary payload. `RestoreStagedSnapshot` into that same session
must leave KV in place, and into a fresh session must copy KV. Both
continuations must match the source continuation bit for bit. Oracle and
held-out both passed that case: exported bytes matched, same-session
restore skipped KV, foreign restore copied KV, and every continuation
logit was bit-identical (max abs error 0). Payload sizes were unchanged.

The server reuses that payload only at a canonical 16-token prefill grid.
`Session::SetCanonicalGrid(16)` makes every cell start at an absolute
multiple of 16. The direct session API leaves the grid off, so
`ornith15.gpu_decode` still scores chunks of 16, 48, 64 and 512.
Checkpoints are stored only at a grid boundary that prefill produced.
Decode steps and a short final cell are not reused; the next turn
re-prefills them. A cached suffix therefore runs the same cells as a cold
prefill of the same prompt. Whole cells inside one call are a single
launch. `ornith15.gpu_chunk` checks that launch against 16-token cells
and against grid-aligned splits (32+rest, 400+112, 16+middle+tail) on
both capture sets and one agent prefix. Logits and boundary snapshots
matched. A test-only column-dependent Q6_K/Q5_K product made the wide
call diverge. Oracle `^(ornith15|qwen38_flash_next)\.` on that tree was
32/32, including `gpu_chunk`. Held-out `gpu_decode` and `gpu_snapshot`
passed. Thresholds were not changed. Flash-Next leaves the explicit
product off, and its tests in that oracle run passed.

`ornith15.gpu_grid` checks that schedule on each capture, repeated to 128
tokens, and on a 128-token prefix of a rendered tool-agent chat. A cold
prefill is compared with a RAM rewind after four teacher-forced steps, a
restore into a session that already prefilled other tokens, and a host
payload round trip. Oracle (4 histories) and held-out (7 histories) were
bit-identical at every compared logit (max abs error 0). The distribution
gate is unchanged: oracle `^(ornith15|qwen38_flash_next)\.` was 31/31,
including `gpu_decode`, `gpu_snapshot` and `gpu_grid`. Held-out
`gpu_decode` and `gpu_snapshot` passed. Thresholds were not changed.

The same prompt served greedy (temperature 0, seed 41) from RAM, from disk
after a restart, and with `cache_prompt` false produced the same content,
tool name, arguments, finish reason and completion-token count on all 7
turns of the tool-agent replay. The cache suite's `legacy-tool-cancel`,
`legacy-tool-cancel-disk`, `thinking-tool-cancel`,
`thinking-tool-cancel-disk`, `disk-spacing` and `disk-spacing-disk` passed.
`text-cancel` failed: after a discarded assistant the next prompt refilled
26 tokens (cached 160 of 186). That 160-token hit is the stable message
prefix, already a multiple of 16. The unstable region is the thinking-off
generation prompt plus the new user turn, which is longer than the suite's
16-token budget. The older chunk-split gap is recorded here so the failed
case stays visible. On a 507-token turn the two leaders were `I` (id 40)
and `<tool_call>` (id 248058). One 512-token prefill scored them 20.8312
and 20.4856 (gap 0.3456). The same prefill split at token 500 scored
20.8926 and 20.7317 (gap 0.1609) and still chose `I`. Splitting at the
cached position 477 chose `<tool_call>` by 0.2029. Prefill, 39
teacher-forced steps, then the suffix chose `<tool_call>` by 1.2751.
Chunks of 64 stayed with the single 512-token prefill.

The wide-cell release binary was scored again with `--suite cache`.
`legacy-tool-cancel`, `legacy-tool-cancel-disk`, `thinking-tool-cancel`,
`thinking-tool-cancel-disk`, `disk-spacing` and `disk-spacing-disk`
passed their own reports. `text-cancel` failed the same way (cached 160,
prefill 26). The runner's shared-boundary check still fails with
`stored_tokens` [2432, 6272], the same summary as the 16-cell run.

The width-48 routed prefill tile was scored with those same ratios.
Oracle `ctest` of `^(ornith15|qwen38_flash_next)\.` was 30/30, including
`ornith15.gpu_decode`, `ornith15.gpu_ops` and `ornith15.gpu_snapshot`.
Held-out `ornith15.gpu_decode` passed. Chunk sizes stayed 16, 48, 64 and
512. Thresholds were not changed.

## MTP

`--speculative mtp` drafts with the bundled NextN block and commits each
kept token with one decode-dot step, the same graph as autoregressive
`Step`. The draft runs on its own stream and scratch. A rejected draft
stops the chain. Other speculative backends are rejected. A different
`--mtp-model` is rejected. `--cache-disk` and a RAM cache capacity are
rejected with `Ornith MTP does not support the prefix cache`. The load
line is `event=ornith_mtp draft_tokens=N cache=off`.

Sampled acceptance is the Flash-Next rule. `ornith15.gpu_mtp` passed on
this tree: trunk logits with the NextN consume stayed at max abs 0
against a session that did not draft (threshold 1e-4). A 4-token
`VerifyForward` differed from four single steps by max abs 0.85989475.
Greedy argmax still agreed 12/12 on that synthetic chain and 24/24 on an
81-token sentence (9 of 41 drafts accepted). Synthetic token ids accepted
0 of 30. The server does not sample the batch row, so that gap is not the
continuation it emits.

`ornith15-functional-v1` was scored unchanged against this server, card
and greedy. Both reports were 5/5. Tool calls parsed. All 8 greedy
answers matched an autoregressive server: content, reasoning, tool name
and arguments, finish reason, and completion-token count. The full
`^(ornith15|qwen38_flash_next)\.` oracle was not re-run after the
decode-step path. `RunChunk` with prefix rollback off is unchanged.
Thresholds were not changed.
