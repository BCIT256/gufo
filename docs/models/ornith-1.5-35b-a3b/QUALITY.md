# Ornith GPU AR checkpoint quality

Original P4 evidence is retained below; the P4b follow-up at the end records
the current one-pass policy.

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
