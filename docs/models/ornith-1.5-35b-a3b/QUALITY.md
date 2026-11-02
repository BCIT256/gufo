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

This record is the earlier server, which committed each kept token with
one decode-dot step. M5 verifies the chain in one launch and keeps the
prefix cache; see M5 MTP below. On the earlier server `--cache-disk` and
a RAM cache were rejected with `Ornith MTP does not support the prefix
cache`. The loader still prints `event=ornith_mtp draft_tokens=N
cache=off`; that string is not a rejection. Other speculative backends
are rejected. A different `--mtp-model` is rejected.

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

## M4 remedy: exact boundaries

The explicit Q6_K/Q5_K product from `75e04191` stays; Flash-Next still
leaves it off. On a grid session a one-token launch now keeps the batched
kernels instead of the decode dots, so a prefill row has the same bits
wherever the prompt is split. The server grid is width 1. Each call of up
to 512 tokens is one launch. The server reuses a cached prompt at its
exact boundary and stores only frontiers that prefill produced. Decode
uses other kernels, so the next turn re-prefills the generated reply. (Superseded:
see [Reply reuse from decode state](#reply-reuse-from-decode-state).)
With a RAM cache, `Session::PrefillThrough` freezes the boundary inside the
launch. DeltaNet conv and recurrent state are checkpointed at the boundary
token in the same kernel. Gated-attention K/V below the boundary is copied
after the launch. The boundary row's logits come from that row's hidden
state through the output head. With `--cache-disk`, prefill stops at the
boundary as before.

`ornith15.gpu_chunk` covers both capture sets and the agent prefix,
against 16-token reference cells. 207 schedules passed, none failed, and
the logits and boundary snapshots were bit-identical. New schedules split
off any grid: 37+rest, 5+11+rest, rest+3, rest+1, and whole cells with a
lone final token. On the agent prompt, final cells are 1, 2, 3, 8, 9 and
15 tokens. Five in-pass checks on a 448-token prompt (start 0 at
boundaries 1, 37, 441 and 447; start 32 at boundary 443) required three
things: the image payload byte-identical to the stop-at-boundary
snapshot, equal last logits, and an equal continuation from the image.
All passed. A column-dependent product diverged at token 0. Decode dots
for a lone token diverged at token 96.

On that tree (later committed as `8508275c` and `7d8a38b3`), the oracle
`^(ornith15|qwen38_flash_next)\.` passed 34/34. Held-out `gpu_decode`,
`gpu_snapshot`, `gpu_grid`, `gpu_chunk` and `gpu_mtp` passed 5/5. After
the host-only cache change (`0daeb148`), `gpu_snapshot`, `gpu_grid` and
`gpu_chunk` passed again. The full oracle was not re-run on that tree.
Thresholds were not changed.

Greedy (temperature 0, seed 41) tool-agent replay, 7 turns, full schemas,
compared on content, tool name, arguments, finish reason and completion
tokens. With `--cache-disk` at 15:12, before the host-only change (live,
restart, then on, off, on, off), and with the RAM cache at 16:09 (live,
off, on), every turn matched the live pass. Turn 0
after the restart restored 431 tokens from disk.

On `0daeb14`, 12 of the 13 upstream suites passed: `cache-edits`,
`tool-agent`, `tool-agent-loop`, `cache-messages-loop`, `text-cancel`,
`thinking-tool-cancel`, `legacy-tool-cancel`, `disk-spacing` and their
`-disk` variants. `text-cancel` failed on the 16-token grid and now passes.
`cache-messages-loop` had exhausted GPU memory before the host-only change.
`tools` failed in `auto-tools`. Asked to omit the optional parameter, the
model sent `{"value":"alpha","optional":1}`, as it did on the M3 release
build.

`ornith15-functional-v1` was scored unchanged on `0daeb14`. Greedy was
5/5. Card was 4/5. Card `write_merge` reasoned for 2048 completion tokens,
the suite's limit (`finish_reason` `length`). Its content was an opening
python fence and `def`, so the answer was scored as a failure. The cut-off
reasoning was coherent and already held a passing draft. It had 118
non-empty lines, 82 of them distinct; the repeats are code drafts. At the
frozen seed, earlier builds produced 380 (M2), 922 (M3), and 1367 and 1904
(M4 MTP) completion tokens.

A diagnostic on 2026-10-30, outside the scored suite, sampled the same
prompt at card settings over the frozen seed and seeds 1000–1031. The
builds were `d005bfeb` and `0daeb14`, and a private llama.cpp `b11580` on
the GPU, which was sent `min_p` 0 as Gufo uses. Gufo builds alternated in
blocks of five seeds.

| Build | Seeds | Pass | `length` | Failed tests |
| --- | ---: | ---: | ---: | ---: |
| `d005bfeb` | 33 | 32 | 0 | 1 |
| `0daeb14` | 33 | 22 | 5 | 6 |
| llama.cpp | 13 | 9 | 3 | 1 |

The failed tests were mostly `unsorted overlap`: the answer seeded the
merge with the unsorted first pair. `release-m4` (`75e04191`) produced
byte-identical content and reasoning to `0daeb14` on the five seeds
compared, so this prompt's state predates the remedy. llama.cpp's sampler
differs from Gufo's, so its seeds do not pair with Gufo's.

Every seed of one build shares the same 110-token prefill state. Four
rewordings of the same task, scored with the same tests, gave each build
four more states, five seeds each (2000–2004).

| Rewording | `d005bfeb` pass | `0daeb14` pass |
| --- | ---: | ---: |
| named | 5/5 | 5/5 |
| context | 4/5 | 3/5 |
| stdlib | 5/5 | 5/5 |
| reordered | 5/5 | 5/5 |

The rewordings do not separate the builds. On the frozen prompt,
`0daeb14` matches llama.cpp's rate, and `d005bfeb` is above both. The card
`write_merge` failure stays recorded as a failure. These samples do not
show a systematic loss from the M4 prefill product. The suite, its seed
and its 2048-token limit are unchanged.

## M5 prefill

ExplicitContract stays on. A 2048-token prefill is one launch, and a routed
chunk of at least 1024 tokens uses a 64-column tile. Shorter chunks stay at
48, including every 16-token reference cell. A 400+112+rest split therefore
mixes tile 48 and tile 64 against a one-shot tile 64. The bits still match.

On `3f5bce75` (4096-token launch, tile 48), `ornith15.gpu_chunk` passed in
381.54 s. The log reports `chunk identity bit-identical`, the 2048-token
one-shot and both in-pass captures, and the two mutations diverging
(`artifacts/m5/gpu-chunk-4096-detail.log`). The oracle and held-out
distribution filters were not re-run on that commit alone.

On `f88b29e2` (tile 64 from 1024 tokens), the same test passed again inside
both filters below. The oracle run's `gpu_chunk` log (314.90 s) reports
`agent/chunk2048 one PASS`, `in_pass start=0 boundary=1000 end=2048 PASS`,
`in_pass start=400 boundary=1500 end=2048 PASS`, `mutation DIVERGED
first_bad_token=0`, `mutation_lone_token DIVERGED first_bad_token=96`, and
`chunk identity bit-identical`.

Held-out `GUFO_ORNITH15_LLAMA_LOGITS` pointed at `artifacts/reference/heldout`.
`ctest --preset gpu-full -R '^ornith15\.(gpu_decode|gpu_snapshot|gpu_grid|gpu_chunk|gpu_mtp)$'`
passed 5/5 in 604.35 s. Oracle logits, `ctest --preset gpu-full -R
'^(ornith15|qwen38_flash_next)\.'`, passed 34/34 in 782.44 s, including
every `qwen38_flash_next` test. Thresholds were not changed. The shared MMQ
default was not edited. Log: `artifacts/m5/gates-tile64.log`.

## M5 MTP

One round drafts serially from the NextN block, at most 7 tokens
(`kMaxMtpDraftTokens`). `VerifyForward` then scores the anchor plus those
drafts in one trunk launch, using the prefill kernels (`RunChunk` and
`ProjectAllLogits`), so the rows are not decode-step logits. A partial
accept calls `RollbackMtpBatch(keep)` and does not run the trunk again.
A full accept drafts the last token while the chain is still hot, then
`RewindMtp`. The version-2 snapshot stores NextN K/V and the trunk carry
(`h_carry`). The chain is cold at a prefill frontier. The draft-length
controller is not in the snapshot. It resets in `Invalidate` and at the
start of `PreviewFirstToken`, which is the request boundary when a cache
hit skips `Invalidate`.

`ornith15.gpu_mtp` on this path
(`artifacts/m5/gpu-mtp-batched-oracle.log`): a 4-token `VerifyForward`
differed from four `Step`s by max abs 0.85989475. Greedy argmax agreed
12/12 on that chain (0 of 30 synthetic drafts accepted) and 24/24 on an
81-token sentence (9 of 41 drafts accepted). Verify-row distribution
against the captures passed on `reasoning_history`, `tool_round_trip`
and `user_turn` (mean KL ratios 1.154, 1.105 and 0.719; outlier
failures 0). Thresholds were not changed.

On `c6e64988`, before the logit-batch claim fix, held-out
`gpu_decode`, `gpu_snapshot`, `gpu_grid`, `gpu_chunk` and `gpu_mtp`
passed 5/5 (`artifacts/m5/gates-mtp-heldout.log`). The oracle filter
`^(ornith15|qwen38_flash_next)\.` passed 34/34
(`artifacts/m5/gates-mtp-oracle.log`), including `gpu_chunk`
bit-identical. Those two GPU filters were not re-run after `a5e5feff`
or `fd08363f`. Neither commit changes kernel math: the first allocates
the verify logit batch inside `EnableMtp`, before the resource claim;
the second resets the draft-length controller per request. The host
`pr` preset, outside `nix develop`, passed 67/67 after `fd08363f`.

`ornith15-functional-v1` on the MTP server (`--speculative mtp
--draft-tokens 7`, context 8192) passed card and greedy, 5/5 each.
An autoregressive greedy run also passed 5/5. Tool calls still parse.
Greedy message text is not all identical to autoregressive text:
`rectangle_area` matches; `tool_round_trip` turns 0 and 1 match on
content, tool name and arguments; turn 2 differs at the first comma
versus a newline. `fix_median` differs at `median_ms()` versus
`median_ms`. `two_turn_clamp` uses `max(lo, min(value, hi))` on the
autoregressive server and `min(max(value, lo), hi)` with MTP. `write_merge`
unpacks the interval on the autoregressive server and indexes
`interval[0]` and `interval[1]` with MTP. Both forms passed the suite.
Answers: `artifacts/m5/functional-v1-mtp/` and `functional-v1-ar/`.

Temperature 0 on the 7-turn agent replay, cache off, matched
autoregressive content, tool name, arguments and finish on all 7 turns
in both cycles. The prose speed prompts do not: the 262-token completion
first differs at character 73, the 2052-token completion at character
217, the same place in both cycles.

Cached-versus-cold identity at temperature 0 on that agent replay held
on the claim-fix binary, before the per-request policy reset. RAM hits
restored the full prompt (`prefill_ms` 0, cached tokens 438, 503, 589,
644, 699, 785, 840) and the completion matched. After restart the first
turn was a disk hit: cached 431 of 438, 7 tokens prefilled,
`cache_restore_ms` 110.6, completion matched. Later turns in that
process were RAM prefix hits (`min_step` skips further disk
checkpoints). The disk payload of the 431-token checkpoint was
76574784 bytes. This identity was not re-measured after `fd08363f`.

Upstream cache suites, tier `pr`, `--record-baseline`, one session,
context 32768, `--speculative mtp --draft-tokens 7`, on the post-reset
release binary (`artifacts/m5/cache-mtp-32k/report.json`). Passed:
`cache-edits`, `cache-depth`, `cache-rotation`, `text-cancel`,
`thinking-tool-cancel`, `legacy-tool-cancel`, `disk-spacing` and the
four `-disk` variants. Failed: `cache-growth` (cold greedy, empty text,
`finish` `length`, 128 completion tokens, never `BETA`);
`cache-shared-prefix` (reply `ALPHA.` where the suite requires the bare
code; miss `prefix_changed`, common prefix 4); `cache-compaction`
(shared probe text `GAMMA` with cached tokens 0, miss `prefix_changed`,
common prefix 2263 of checkpoint 2277); `cache-bridge` (unqualified:
64 fills published 7374985728 of 10224906240 budget bytes).
`cache-concurrency` was not run. Thresholds and the suites were not
edited. A context-8192 run of the same suites is an earlier setup miss
and is not this result.

## M5 review

On `72e51cbd` (after `a5e5feff` and `fd08363f`), rebuilt `gpu-test`:
held-out `gpu_decode`, `gpu_snapshot`, `gpu_grid`, `gpu_chunk`,
`gpu_mtp` 5/5; oracle `^(ornith15|qwen38_flash_next)\.` 34/34,
including `gpu_chunk` bit-identical. Thresholds unchanged. Logs:
`artifacts/m5-review/gates-*.log`.

MTP cache identity on the production `72e51cb` binary, temperature 0,
7-turn agent replay with `--cache-disk`: a cache-on pass, a full repeat
in the same process (every turn a full-prompt RAM hit) and a pass after
restart (turn 0 a disk hit of 431 tokens) gave the same content, tool
name, arguments and finish on all 7 turns. The repeat pass is where the
pre-`a5e5feff` server returned HTTP 500 (`text runner measured state
exceeds its resource claim`); it completed.

The four upstream suites that failed with MTP, re-run on `72e51cb` with
and without MTP (tier `pr`, `--record-baseline`, context 32768, one
session, inside `nix develop`; `artifacts/m5-review/suites-{ar,mtp}/`):

| Suite | Autoregressive | MTP |
| --- | --- | --- |
| `cache-compaction` | passed | passed |
| `cache-shared-prefix` | failed: reply `ALPHA.`, not the bare code | failed, same reply |
| `cache-bridge` | unqualified: 9,030,868,992 of 15,459,983,360 budget bytes | unqualified: 9,079,234,560 of 10,099,662,848 |
| `cache-growth` | failed: `keep_thinking` turns 1–3, cache did not cover the replayed reply | failed earlier: a thinking turn used all 128 tokens without answering `BETA` |

`cache-shared-prefix` and `cache-bridge` fail the same way with and
without MTP. The autoregressive `cache-growth` failure is the design
recorded under the M4 remedy: only prefill frontiers are stored, so the
previous reply (63, 7, ... tokens) is re-prefilled; the suite requires
the cache to cover it. With MTP, greedy reasoning differs from
autoregressive (see M5 MTP), and on that case it ran past the suite's
128-token limit before the coverage checks. `cache-compaction` failed
in Grok's MTP run and passed in both of these. None of these suites ran
in M4.

## M6 review

`VerifyForward` runs up to 8 tokens on the decode dots. `ornith15.gpu_mtp` now
requires max abs 0 between each verified row and `Step` on the same token at
widths 1, 2, 3, 4, 7 and 8, over the eager body, graph capture and replay, and
between MTP-session and plain `Step` (was `< 1e-4`). This supersedes the
verify-versus-decode gap and greedy differences in M5 MTP. On `4406552`, MTP
greedy text equals autoregressive on the 262- and 2,052-token prompts in every
cycle and on all 7 agent-replay turns.

Gates at `44065526`: oracle 34/34 including `gpu_chunk` bit-identical and verify
distribution PASS; held-out 5/5; host PR suite 67/67. `ornith15-functional-v1`
unchanged on `4406552`: autoregressive greedy 5/5, card 4/5 (`write_merge`
reasoned to the 2,048-token limit, the same failure as the M4 remedy, kept as a
failure); MTP greedy 5/5, card 5/5. The MTP card pass is one seed and is not
evidence that MTP fixes `write_merge`.

Upstream suites, with and without MTP: `cache-compaction` passes.
`cache-shared-prefix` fails on `ALPHA.`; llama.cpp `b11580` gave the same answer
on that request. `cache-bridge` is unqualified: 64 side snapshots publish 9.03 GB
against the 1.5x fill required of a 15.46 GB budget. `cache-growth` fails with
`cached=2907, previous_prompt=2907, previous_completion=63, prefilled=262`, now
the same way with MTP: the reply is re-prefilled next turn.

## Idle reply refill

Removed in `a1aa52ba`; see [Reply reuse from decode state](#reply-reuse-from-decode-state).
The results below are the state at `6116b0d7`.

The refill prefills the reply with the prefill kernels, so a later turn repeats
a cold prefill. Decode state is never published. In the resend test, 8 requests
on one conversation matched a cache-off run in text, tool name and arguments on
every pass, at gaps of 0, 50, 300 and 1,000 ms, with and without MTP. Gates at
`6116b0d7`: oracle 34/34, held-out 5/5, host PR suite 67/67 with two new
scheduler tests. On production `6116b0d`, MTP cache identity holds for RAM repeat
and disk after restart. `ornith15-functional-v1` unchanged: autoregressive greedy
5/5, card 4/5 (`write_merge`, kept as a failure); MTP greedy 5/5, card 5/5.

`cache-growth` still fails, unchanged: `cached=2907, previous_prompt=2907,
previous_completion=63, prefilled=262`. The suite sends the next turn at once,
inside the 100 ms delay. A build with no delay covered the reply at a 0 ms gap in
the resend test, but cost about 100 ms to first token there, and it was not run
through this suite. The suite is not changed.

## MTP by default and prompt lookup

Rule fixed before scoring: the five `ornith15-functional-v1` tasks at the frozen
seed plus seeds 1000-1009 (55 runs per mode), and `write_merge` through seed 1031
(33 runs). MTP holds if its total is within 3 of autoregressive, no task is worse
by more than 2 of 11, and `write_merge` is within 4. Suite prompts and tests are
unchanged; card sampling.

| Build | 55 runs | `write_merge`, 33 seeds |
| --- | ---: | ---: |
| Autoregressive, `7e2268e2` | 51 | 22 |
| MTP, `7e2268e2` | 51 | 27 |
| MTP + lookup, production `2f4e1d0` | 51 | 27 |

The other four tasks passed 11/11 in every row. All 55-run failures are
`write_merge`, recorded as failures. Sampled output differs by seed between
modes, so these are rates. The lookup row ran in a later session, against the
earlier autoregressive row.

Greedy and seeded (temperature 0.6) output on production, 5 turns with tool
calls and whole-file replies: gap 0, gap 1,000 ms, disk and disk after restart
all matched cache off. Lookup counters were not recorded on these turns.
`ornith15.gpu_mtp` checks that a draft after `CatchUpMtp` is bit-identical to one
built token by token, with all copies kept, a rejected copy, and a neural draft
followed by copies, and that the trunk is untouched.

A server with no `--speculative` drafts when the GGUF bundles NextN. A copy of
the APEX file with its metadata set to no NextN block loaded autoregressively, and
`--speculative mtp` on that copy failed at load.

Gates at `2f4e1d05`: oracle 34/34, held-out 5/5, host PR suite 67/67 with one new
scheduler test. `ornith15-functional-v1` on production with no flag: greedy 5/5,
card 5/5 on the frozen seed.

Upstream suites, same list and order on `6116b0d` and `2f4e1d0` with `--speculative
mtp`: `cache-edits`, `tool-agent`, `tool-agent-loop`, `cache-messages-loop` and
`long-context` pass on both. `tools` (`auto-tools`), `cache-growth`,
`cache-shared-prefix`, `cache-compaction` and `cache-bridge` fail on both the same
way. `cache-compaction` passes alone on both. After the other suites on one server
its retry resumes at 27,525 of 27,542 tokens and fails its unchanged-retry check.
The cache is full (9.94 of 10.1 GB retained), so the retry copy is refused and the
retry restores the stable boundary before the final user turn, as
[KV-CACHE.md](../../KV-CACHE.md) specifies. That turn plus its assistant opening
counts 17 tokens by `/messages/count_tokens`; this suite allows a fixed 16, where
`tool_agent.py` and `tool_choice.py` measure the bound. `2f4e1d0` alone with
`--cache-ram-bytes 1500000000` reproduces the failure with `--speculative mtp`.
A local copy using the measured bound passes every check, cold controls included,
with `--speculative mtp` and `off`. This is a suite bound, not lost state, and a
retry costs one 17-token prefill (220 ms). `run.py` reads a missing `--speculative`
as autoregressive and fails on the drafts it then sees, so MTP runs must pass the
flag. The suites are not changed.

`auto-tools` fails on `auto_chat_thinking_high`: told to omit `optional`, the model
reasons that it will, then sends `{"value":"alpha","optional":1}`. The suite requires
exactly `{"value":"alpha"}`. Greedy gives this with and without drafting. Sampled at
temperature 0.7, top_p 0.85, top_k 20, 150 seeds, same scorer, with the suite's tool
definition / llama.cpp's rendering of it (no `strict`, empty `description`), Gufo
omits it 27 / 4 times autoregressive and 33 / 1 by default; llama.cpp `b11580` on
ROCm omits it 81 of 150 on both, since it renders them identically. Raw `/completion`
on one prompt, no tools or grammar, min_p 0: 5 of 150 against 57 of 150. Forced to
the decision point: 0 of 100 against 33 of 100.

The difference is one near-tie, between `<` (`<parameter=optional>`) and `</`
(`</function>`). logit(`<`) minus logit(`</`) at that row: llama.cpp ROCm 0.40 (its
`n_probs`), scalar reference about 0.9 (solved from its KL against the batch capture,
two tokens only), Gufo 2.1 ± 0.7 (temperature-2 samples, 300), llama.cpp CPU sequential
2.48, CPU batch 4.41. Every one prefers `<`, and llama.cpp's own builds span more than
Gufo's distance from them. `ornith15.gpu_decode` passes on this capture (mean ratio
1.01, largest position ratio 1.89), so this is not a defect under the acceptance gate.
The scalar reference fails its stricter check here on this row (KL 0.29). No engine
change. It is one prompt: whether Gufo adds optional arguments more often in real
agent work is what `scripts/workflow_eval.py` reports, not this row.

`tests/functional/pi_agent.py` pins Pi 0.87.0. A local copy with the pin moved to
the installed Pi 1.3.0 passed all 8 cases, one pass, on the production default. Its
prompts are French: it covers the tool loop, not the English product workflow.

## English coding evaluation

`ornith-workflow-v1`, rules frozen before scoring: ten tasks driven by Pi 1.3.0,
five seeds, Gufo `2f4e1d0` against llama.cpp b11580 ROCm, each task scored by
real tests or an exact check. Gufo 50/50, llama.cpp 48/50; both failures are
`write_tests`, where its tests missed the planted fault. Tool calls 299 and 313,
tool errors 35 and 36. Optional `read` arguments 4 of 92 and 5 of 104, so the
`auto-tools` near-tie does not appear as different behavior in this work.
Capability is accepted. Five seeds cannot separate 5/5 from 3/5 on one task.

## Reply reuse from decode state

Rules fixed on 2026-11-01 before any result, for publishing a finished reply's
decode frontier. Cached turns then follow decode rows, so they stop being
bit-identical to a cold prompt. Baselines are `2f4e1d0`. The change stays only
if all of these hold; otherwise it is reverted and the result recorded.

1. `ornith15.gpu_decode` passes on the oracle and held-out captures at unchanged
   thresholds, with a new agent-loop row: prefill resumes on decode state at
   turns of 64+16, 16+4 and 1+1 tokens. Other gates and the host PR suite pass.
2. `ornith15-functional-v1` on one seed 10/10. The sampled rule at no less than
   its baseline minus 2: at least 49 of 55 runs, and `write_merge` at least 25
   of 33 seeds, both with MTP and lookup.
3. `continuation` passes. `cache-growth` is reported.
4. `ornith-workflow-v1` with its harness unchanged: capability and latency
   accepted against llama.cpp, and Gufo's first-token sum at most 170 s (205.1 s
   before, 135.2 s on llama.cpp, so at least half the gap).

Result: kept, at `88334a5c`. Every rule held.

1. `ornith15` gates 13/13 on the oracle. Held-out `gpu_decode` passes with 18
   new resume rows. Host PR suite 67/67.
2. `ornith15-functional-v1` 10/10. Sampled 51 of 55; `write_merge` 27 of 33.
3. `long-context` passes (no suite is named `continuation`; the first run named
   it and ran nothing, so this one came after the evaluation). `cache-growth`
   now passes. `tools` still fails `auto-tools` on the near-tie optional argument,
   as it did before this change. `run.py` also marks these runs failed because
   `disk-spacing` expects no drafts while the server now drafts by default.
4. `ornith-workflow-v1`: capability accepted, 48 against 50; latency accepted,
   247 against 257 s; first-token sum 169.8 s against the 170 s limit, a margin
   of 0.2 s.

Both Gufo failures are `resume_followup`, where the model wrote
`self.assertRaises(ValueError, median, [])`. That test is valid, and the frozen
scorer rejects it because it looks for the text `median([])`. The harness is
unchanged and the failures stand in the score; correcting this needs a new suite
id.

Cached turns are no longer bit-identical to cold prompts. `a1aa52ba` then removed
the unused idle refill: host PR 67/67, and on that binary `long-context` and the
cancellation and disk suites pass.

## Split decode attention

Decode dots split each row's key tiles over eight blocks and merge the partial
rows, so a decode row's rounding differs from the unsplit kernel at the same
precision. Heads per block do not change bits at a given split count, and prefill
stays unsplit and bit-identical. Rules fixed before scoring: `ornith-workflow-v1`
capability and latency accepted, sampled functional-v1 at least 49 of 55,
`write_merge` at least 25 of 33. Results: accepted at 50/50 and 50/50, 55/55, 29/33.
Gates 35/35, held-out `gpu_decode` passes: worst mean ratio 1.39 (1.40 before, limit
2), worst position KL 5.6 (5.58 before, limit 15.1). Host PR 101/101, functional-v1
10/10. MTP rows stay bit-identical to a decode step (`ornith15.gpu_mtp`).
