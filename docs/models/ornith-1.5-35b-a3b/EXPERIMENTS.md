# Ornith GPU checkpoint experiments

October 23, 2026. Correctness checkpoint; no optimization or speedup claim.
The original P4 decisions below precede the P4b follow-up, which supersedes
the blanket residual correction.

| Decision | Observation |
| --- | --- |
| Reject one Q8_1 activation pass | Reasoning-history position 15 failed the fixed per-position contract: KL(reference\|\|GPU) 0.9641056443; residual relative RMS first exceeded 0.1 at layer 15. |
| Retain activation residual correction | Project the original activation with MMVQ, reconstruct its actual Q8_1 values (including the rounded F16 scales), and project the remaining F32 residual through the same packed weights. Add the two outputs. This raises activation fidelity and doubles weight reads; it changes neither experts nor stored weight precision. |
| Retain explicit DeltaNet semantics | SiLU output gating, `x / max(norm(x), eps)` Q/K normalization, and value head `h` reading key head `h % k_heads`, following the validated scalar reference. |
| Reject sigmoid output gate for Ornith (temporary fault) | Failed at position 0, KL(reference\|\|GPU) 12.66098026; first diverging layer 0. Restored SiLU before final qualification. |

The independent synthetic check covers Q6_K, Q5_K, IQ4_XS and Q8_0 with shared
and distinct routed experts, normal and small activations (seed 12345, three
experts, top-2, three rows, M=16, K=512). Corrected maximum
F64 errors are smaller than the single-pass errors and below the unchanged
0.01 times maximum absolute reference bound. Q6_K embedding gathers match the
host dequantizer byte for byte.

Flash-Next's default GDN path retains its arithmetic. Temporary instrumentation
hashed every download in its existing GDN test before and after the shared
change: all 384 output/state/history/rollback downloads matched bit for bit.
Instrumentation is not part of the committed tests. Raw logs are kept outside
the fork.

## P4b: isolate activation quantization (October 23, 2026)

Follow-up to `49113bb2`; same APEX MTP Quality trunk, oracle captures,
`flake.lock`, assertions and acceptance constants as the checkpoint above.
The temporary projection selector, dumps, reference cache and experimental
Q8_K route are diagnostic tools only. Raw logs and diagnostic source snapshots
are retained outside the fork under the preparation repository's ignored
`artifacts/p4b/`. No raw activations, machine-private paths or model weights
belong in this repository.

The first isolation window uses the first 16 **unchanged** reasoning-history
tokens and logits from the pinned captures. The executable's acceptance rules
are unchanged, including the mean-KL rule. A mean-KL result on this shorter
window is **not** a full-history qualification: its llama batch/sequential
floor is 0.02818325388 rather than the full history's 0.05104379883.
Every unlisted projection retains the committed two-pass correction.

| Single-pass classes | Position 15 KL(reference \|\| GPU) | Top-1 matches reference / llama | 16-position diagnostic |
| --- | ---: | --- | --- |
| All | 0.9641056443 | no / no | Per-position failure at 15 |
| Attention q/k/v/o | 0.4954318485 | no / no | Per-position failure at 15 |
| q | 0.01378199628 | yes / yes | Pass |
| k | 0.02723799885 | yes / yes | Pass |
| v | 0.1060572278 | yes / yes | Mean-KL failure only |
| o | 0.009362684219 | yes / yes | Pass |
| q + v | 0.01246906592 | yes / yes | Pass |
| v + o | 0.2401483912 | no / no | Per-position failure at 15 |
| DeltaNet qkv/gate/alpha/beta/out | 0.2131580882 | yes / yes | Mean-KL failure only |
| Shared gate/up/down | 0.1971773472 | yes / yes | Mean-KL failure only |
| Routed gate/up/down | 0.2443209522 | yes / yes | Pass |
| Output head | 0.00003940683907 | yes / yes | Pass |
| qkv | 0.4324013842 | yes / yes | Mean-KL failure only |
| gate | 0.2575534124 | yes / yes | Mean-KL failure only |
| alpha | 0.5389866999 | no / no | Per-position failure at 15 |
| beta | 7.969291384e-05 | yes / yes | Pass |
| ssm_out | 0.04543987902 | yes / yes | Pass |
| shared_gate | 0.008057482967 | yes / yes | Pass |
| shared_up | 0.01263526538 | yes / yes | Pass |
| shared_down | 0.03438065422 | yes / yes | Pass |
| routed_gate | 0.03840979196 | yes / yes | Pass |
| routed_up | 0.2123057898 | yes / yes | Mean-KL failure only |
| routed_down | 0.07254470458 | yes / yes | Pass |
| All except o | 0.04695266223 | yes / yes | Pass |

The smallest isolated class that reproduces the original per-position
failure is **ssm_alpha alone**: KL 0.5389866999, top-1 disagreeing with both
references. The v/o pair is a second independent reproducer while either
class alone retains top-1 at that position. These projections all store Q6_K
weights. Neither reproducer implies an incorrect packed-weight decoder.

### Arithmetic audit and measured input diagnostics

At the pinned llama.cpp revision `68d9053afd4f4d0752ced6187585f862355a40be`,
`ggml/src/ggml-cpu/ggml-cpu.c` maps Q5_K, Q6_K and IQ4_XS weights to Q8_K
activation dots. On x86, `arch/x86/quants.c::quantize_row_q8_K` calls
`ggml-quants.c::quantize_row_q8_K_ref`: 256 values, a signed absmax, nearest
integer rounding, an FP32 scale and 16-element integer sums. Q8_0 weights
instead dot against Q8_0 activations. The F32 router and shared expert's scalar
gate keep F32 inputs in both engines; Gufo never applied Q8_1 to those tensors.
Gufo's MMVQ route uses 32-element Q8_1 blocks with FP16 `d` and `s`.

Measured at reasoning-history position 15, layers 14–16 in the original
single-pass run: no nonfinite input, nonfinite scale, overflow or nonzero
activation block rounded to zero `d`. The Q6_K MMVQ decoder consumes `d` and
integer codes, not `s`. Layer 14/16 DeltaNet output has 4/128 and 15/128
subnormal `d` values, respectively; the largest relative `d` rounding error
there is 0.000719. Attention layer 15 has no subnormal `d` or `s`. Its q/k/v
input peaks at 22.1031 with RMS 0.847901 (peak/RMS 26.0681), and its o input
peaks at 0.602800 with RMS 0.0263594 (22.8685). Ordinary quantization error
from these outliers dominates the much smaller scale-rounding contribution.

An identical-input replay on the q-only-single-pass run's activations unpacks the
original Q6_K weights with the pinned CPU decoder and accumulates in F64.
Across 128 evenly spaced output rows per projection, the GPU's first pass
matches the exact reconstructed-Q8_1 dot to about `1e-7` relative RMS. For
layer 14/16 alpha, all 32 output rows have Q8_1 relative RMS errors
0.008907/0.007978 versus CPU Q8_K 0.026486/0.014855. For
layer 15 v/o, relative RMS output error against the unquantized-input dot is
0.01588/0.01903 with Q8_1, versus 0.04097/0.03879 with the actual pinned
llama.cpp Q8_K CPU dot. Thus this case does **not** establish that Q8_1 has
larger local error or an FP16 range defect. The different quantization grids
produce different error directions, which the nonlinear/recurrent model can
amplify differently. That last mechanism is architectural inference; the
operator errors and full-model outcomes are measured observations.

As a control, one CPU-compatible Q8_K pass for Q6_K projections (all residual
passes disabled) retains top-1 at position 15, KL(reference || GPU)
0.2496339382 and KL(llama || GPU) 0.1500483978. It passes this diagnostic window;
it is not a full-history qualification or a retained production route.

The isolated-alpha-only capture independently confirms this for the smallest
reproducer. At layers 14/16 its input peak/RMS ratios are 21.3961/20.3257;
`d` ranges are [0.00830078, 0.162109] / [0.00717926, 0.153687], with no
zero, subnormal or nonfinite scales. Q8_1 input relative RMS errors are
0.009899/0.008735; scale rounding alone contributes 0.000167/0.000177.
Its 32 output rows have Q8_1 relative RMS errors 0.008018/0.009067 versus
CPU Q8_K 0.036462/0.017240; GPU-versus-exact-Q8_1 errors stay below `1e-7`
relative RMS. Alpha controls the exponential recurrent decay, so perturbing
it is a plausible amplifier; the experiment proves sensitivity, not a unique
bad channel or an arithmetic error in that decay operator.

### Retained policy and rejected alternatives

Retain F32 activations only for full attention's output projection (`o`). A
model-owned Q6_K/Q8_0 kernel unpacks each original weight in registers and
accumulates against the F32 input in one pass. All other quantized projections
use one existing Q8_1 MMVQ pass. Delete the blanket correction, its extra
quantization launch, Q8 reconstruction kernel and 1,161,216 session scratch
bytes. No expert, weight precision, token history or reasoning length changes.
There is no new runtime selector and no unpacked persistent weight copy.

This is a precision choice at a measured sensitive boundary, not a repair of
an incorrect Q6_K integer dot. Single-class isolation identifies alpha as a
smallest reproducer; preserving the attention output input instead is enough
to satisfy the complete fixed contract with all other classes single-pass.
It does **not** establish that alpha quantization is harmless in other policies.
The model's response to mixed-precision changes is not monotonic.

Rejected full-history controls (same acceptance rules):

| Policy | Outcome |
| --- | --- |
| F32 Q6_K dense inputs except vocabulary head; all others single-pass | Reasoning passes; tool position 59 fails |
| F32 v+o; all others single-pass | Reasoning position 77 fails |
| F32 q/k/v/o; all others single-pass | Reasoning position 67 fails |
| F32 alpha+o; all others single-pass | Reasoning position 67 fails |
| F32 alpha+o plus routed-up correction | Reasoning position 15 fails |
| F32 Q6_K dense inputs except head, plus correction for all routed projections | Reasoning position 77 fails |

The prototype F32 selector supported Q6_K only: Q8_0 shared projections in
these controls still used their selected MMVQ policy. Some early o-only runs
stopped on an incomplete diagnostic reference cache; those were not numerical
failures or passes. The completed o-only diagnostic subsequently passed all
183 positions. Reference-cache files for tool positions 58/59 were identical
byte for byte across two independently computed reloads. Both oracle tests
are restored byte for byte before final qualification; no cache, shortened
capture or diagnostic selector remains in production or committed tests.

The retained F32 operator has independent unpacked-weight/F64 coverage for
Q6_K and Q8_0, K=256/512/4096 and K=32/96/512/4096 respectively, M=1/3/17,
zero/tiny/ordinary/large activations, and a 100x outlier. Outputs must be finite
and maximum error at most `1e-5` times the maximum absolute F64 output.
Existing routed-format checks retain their original `0.01` bound against F64;
they now check the retained single pass rather than a removed correction.
The four full-decode acceptance constants and all failure comparisons are
unchanged; the entire `gpu_decode_test.cpp` is byte-identical to the base.

What remains unexplained: a channel-by-channel causal account of why the
Q8_K error direction is less damaging at the original failing position, or
which recurrent/routing decisions amplify each perturbation. The evidence
rules out a Q6_K dot bug and FP16 scale-range failure for the inspected inputs;
it does not prove universal superiority of either activation format. The
qualification below is bounded to the 183 pinned teacher-forced positions,
not a general capability, long-context or production benchmark claim.

### Final uncached qualification

All 183 positions pass the original full-decode contract with no clear-margin
flips. Before is the freshly rerun `49113bb2` blanket-correction executable;
after is the retained one-pass policy. Both recompute the independent CPU
reference, use the original complete captures and include per-layer traces.

| History | Policy | Top-1 reference / llama | KL reference mean / max | KL llama mean / max | Confident max KL reference / llama | Max absolute logit error |
| --- | --- | --- | --- | --- | --- | ---: |
| reasoning_history | Before | 87/87 / 77/87 | 0.0009218999007 / 0.03443386001 | 0.06561303982 / 1.788048549 | 0.0002805819905 / 0.02214198211 | 2.565742761 |
| reasoning_history | After | 84/87 / 78/87 | 0.02666456063 / 0.640025068 | 0.04671976705 / 0.9642286617 | 0.02819847754 / 0.009705740038 | 4.763941765 |
| tool_round_trip | Before | 74/75 / 71/75 | 0.004077427532 / 0.0953593984 | 0.05441090237 / 0.5700931291 | 0.002146017432 / 0.03571511562 | 2.462987423 |
| tool_round_trip | After | 70/75 / 67/75 | 0.04959738944 / 0.8508555326 | 0.07962550927 / 1.531945698 | 0.08232510347 / 0.08362369473 | 6.589832067 |
| user_turn | Before | 21/21 / 19/21 | 6.722697007e-08 / 2.888812344e-07 | 0.02487865833 / 0.1960749618 | 7.269799356e-08 / 0.01230913494 | 0.2022695541 |
| user_turn | After | 19/21 / 19/21 | 0.009889620099 / 0.0972345182 | 0.02028499337 / 0.1682137146 | 0.001946743154 / 0.006774671916 | 2.163137913 |

At the original failing reasoning-history position 15, the rejected all-Q8_1
single pass had KL(reference || GPU) 0.9641056443 and KL(llama || GPU)
0.3805854377, disagreeing with both top-1 tokens. The retained one-pass policy
has 0.06924758689 / 0.1345883603 and agrees with both top-1 tokens. This
comparison uses the rejected single-pass run; the before rows above instead
use the committed two-pass baseline.

The retained policy has greater KL against the scalar reference than the
blanket correction. Passing the unchanged envelope is not logit equivalence
or a claim of identical numerical fidelity. Mean KL against llama improves
on some histories and worsens on others, as the table shows.

| History | Before ms/token | After ms/token | Before / after session bytes |
| --- | ---: | ---: | --- |
| reasoning_history | 80.9294 | 49.2712 | 70284620 / 69123404 |
| tool_round_trip | 81.9418 | 49.3362 | 70038860 / 68877644 |
| user_turn | 82.1684 | 49.0838 | 68932940 / 67771724 |

Informational single before/after pair, not an alternating production
benchmark. Eager synchronous Step includes traces and the full-logit download,
excludes CPU-reference work/loading/reset, and is interleaved with uncached
CPU-reference steps in both runs. Cached diagnostic timings are excluded.
AC power and balanced platform profile; pinned toolchain and model unchanged.
No pp2048/tg128, serving or long-context speed claim.

Required configure/build succeeded. Full selected CTest suite: **29 passed,
zero failed/skipped**, 644.07 seconds, including all 21 unchanged Flash-Next
tests and both unchanged independent oracle tests. Format: 650 C++ files
passed. PR suite: 66 passed, zero failed/skipped. Model SHA256 was reverified
as `c83373e4c502c6d4929339406ffbb1f442598b153a3ade02b628752e00a282fd`.
