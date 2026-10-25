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

## P4c pre-registration: held-out activation policy (October 23, 2026)

Registered before any P4c logit capture, inference or policy measurement.
The six fixed ChatML histories in `tools/ornith15/prompts/heldout/` cover
a code edit with a diff, a long single-turn reasoning question, a two-parameter
tool call and response, kept multi-turn reasoning, plain prose and Spanish.
Each ends with the template's open assistant thinking prefix. These prompts
were not used to choose the P4b policy. Token counts include all framing and
are constrained to 60–150 using the pinned model tokenizer before capture.

Compare only A (all quantized projections use one Q8_1 activation pass) and B
(current F32 activations for full-attention o_proj only). F32 stored weights
retain their existing F32 path in both. No experts, reasoning, stored weight
precision, token histories or other arithmetic change. Use the same model,
llama.cpp build/dumper, pinned toolchain, uncached scalar reference and traces.
Run both policies on the original three histories as controls and all six
held-out histories; only held-out results determine the policy.

**Decision rule:** keep B only if it has fewer criterion failures than A on
the held-out set **OR** lowers total mean KL(llama batch || GPU) by **more than
20%**; otherwise remove B and keep the simpler, faster A. Total mean is the
position-weighted mean across all held-out histories (sum of KL / total
positions). Criterion failures count each clear-flip position, each position
exceeding the confident-reference KL bound, and each prompt exceeding the
mean-KL bound separately. Finish every prompt even after a criterion failure
to avoid censoring either policy; the test must still return failure.

Before comparison, align clear flips with the independent reference oracle's
self-consistency principle: GPU top-1 differs from scalar-reference top-1,
reference top-two margin > 1.0 logit, and both llama.cpp batch and sequential
captures agree with that reference top-1. Keep confident entropy 0.5 nats,
confident KL 0.1 nats and per-prompt mean-KL ratio 2.0 unchanged. Record original
prompt outcomes before and after alignment. No threshold adjustment, extra
policy search, prompt replacement or result-driven exclusion is allowed.
Timings from synchronous traced Steps are informational only. Raw captures,
logs, traces and private identities stay in ignored artifacts; check in bounded
result summaries and remove the temporary A code toggle.

### P4c held-out results

Pre-registration: `577030c4`; criterion alignment: `34ad50f1`; starting
policy: `b5968851`. Six histories, 649 held-out positions. All
prompt bytes match the pinned independent Jinja template, and all paired
captures contain identical token IDs. The existing CPU llama.cpp dumper/build
was reused at `68d9053afd4f4d0752ced6187585f862355a40be`, with 28 threads,
no BOS and special-token parsing. No prompt or threshold changed after
pre-registration. Every comparison recomputed the scalar reference uncached.

A = all quantized projections use single-pass Q8_1; B = F32 input only for
full-attention o_proj. F32 stored weights retain their normal F32 path.
KL directions below are reference || GPU, llama batch || GPU, and llama
batch || sequential. Confident maxima use each named teacher's entropy.

| Set / history | Policy | Positions | Top-1 reference / llama | Clear flips | Confident failures | Mean-KL failures | KL reference mean / max | KL llama mean / max | Confident max KL reference / llama | Llama batch/sequential mean |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| heldout / code_diff | A | 110 | 94/110 / 90/110 | 0 | 0 | 0 | 0.070468512 / 1.0313216 | 0.094288373 / 1.1443894 | 0.0056009285 / 0.46160684 | 0.17893345 |
| heldout / code_diff | B | 110 | 99/110 / 91/110 | 0 | 1 | 0 | 0.073933167 / 2.8894865 | 0.088656158 / 1.1275335 | 0.12203016 / 0.081545139 | 0.17893345 |
| heldout / kept_reasoning | A | 105 | 92/105 / 81/105 | 2 | 0 | 0 | 0.13640859 / 3.6799383 | 0.14094481 / 3.0633738 | 0.015614306 / 0.034344576 | 0.14903544 |
| heldout / kept_reasoning | B | 105 | 88/105 / 81/105 | 0 | 0 | 0 | 0.049041231 / 0.70830451 | 0.1023016 / 2.2514763 | 0.022650992 / 0.016602408 | 0.14903544 |
| heldout / long_reasoning | A | 113 | 94/113 / 88/113 | 0 | 1 | 0 | 0.048527514 / 1.2393714 | 0.055289752 / 0.56602558 | 0.12334056 / 0.00019811087 | 0.055639588 |
| heldout / long_reasoning | B | 113 | 99/113 / 84/113 | 0 | 0 | 0 | 0.04206219 / 1.3063301 | 0.052702931 / 0.62567909 | 0.04689614 / 9.29108e-05 | 0.055639588 |
| heldout / plain_prose | A | 100 | 87/100 / 76/100 | 0 | 0 | 0 | 0.025913674 / 0.31505611 | 0.12046446 / 3.4974406 | 0.033517009 / 1.99137 | 0.074674158 |
| heldout / plain_prose | B | 100 | 89/100 / 79/100 | 0 | 0 | 0 | 0.063396081 / 3.1470488 | 0.094277856 / 2.9361005 | 0.027702202 / 0.035650701 | 0.074674158 |
| heldout / spanish | A | 100 | 81/100 / 79/100 | 0 | 0 | 0 | 0.037649034 / 0.33639265 | 0.063203406 / 1.0661992 | 0.022700281 / 0.012956418 | 0.056702974 |
| heldout / spanish | B | 100 | 78/100 / 77/100 | 0 | 0 | 0 | 0.036191724 / 0.4110262 | 0.052564251 / 0.59908337 | 0.022550202 / 0.020454059 | 0.056702974 |
| heldout / tool_two_parameters | A | 121 | 108/121 / 101/121 | 3 | 0 | 0 | 0.073414922 / 1.3188008 | 0.093053286 / 1.5166993 | 0.048204906 / 1.0630969 | 0.13134111 |
| heldout / tool_two_parameters | B | 121 | 110/121 / 106/121 | 2 | 0 | 0 | 0.071854248 / 0.72470664 | 0.098024796 / 1.0103834 | 0.055388525 / 0.21469149 | 0.13134111 |
| original / reasoning_history | A | 87 | 76/87 / 80/87 | 1 | 1 | 0 | 0.082250022 / 1.9611435 | 0.087975424 / 2.1981747 | 0.54480881 / 0.58774378 | 0.051043799 |
| original / reasoning_history | B | 87 | 84/87 / 78/87 | 0 | 0 | 0 | 0.026664561 / 0.64002507 | 0.046719767 / 0.96422866 | 0.028198478 / 0.00970574 | 0.051043799 |
| original / tool_round_trip | A | 75 | 62/75 / 65/75 | 0 | 1 | 0 | 0.069785238 / 1.3760381 | 0.089850426 / 2.1064567 | 0.12531203 / 0.075479761 | 0.085345053 |
| original / tool_round_trip | B | 75 | 70/75 / 67/75 | 0 | 0 | 0 | 0.049597389 / 0.85085553 | 0.079625509 / 1.5319457 | 0.082325103 / 0.083623695 | 0.085345053 |
| original / user_turn | A | 21 | 18/21 / 16/21 | 0 | 0 | 0 | 0.025165854 / 0.3006784 | 0.029226662 / 0.19225252 | 0.0031156069 / 0.010902197 | 0.0324381 |
| original / user_turn | B | 21 | 19/21 / 19/21 | 0 | 0 | 0 | 0.0098896201 / 0.097234518 | 0.020284993 / 0.16821371 | 0.0019467432 / 0.0067746719 | 0.0324381 |

Original-prompt before/after alignment, current B policy: all three pass
under both versions, with identical top-1 counts and every KL statistic above.
The old-criterion binary was retained before rebuilding the aligned test.

| Original history | B old / aligned clear flips | A old-margin / aligned clear flips |
| --- | ---: | ---: |
| reasoning_history | 0 / 0 | 2 / 1 |
| tool_round_trip | 0 / 0 | 1 / 0 |
| user_turn | 0 / 0 | 0 / 0 |

A's old-margin counts are computed alongside the aligned run; they are
not a second A run under the old test's early-exit behavior.

Held-out criterion failures: A **6**, B **3**.
Position-weighted total mean KL(llama || GPU): A **0.094059943**,
B **0.081655605**; B's relative reduction is
**13.1877%**. The pre-registered rule
therefore keeps **B**. Original prompts do not contribute to this choice.

### P4c informational timings

Same synchronous traced Step scope as P4b: includes GPU launches, full logits
and layer downloads; excludes model upload, scalar-reference work and reset.
Measurements are interleaved with uncached CPU-reference steps. All inference
and capture jobs ran sequentially, on AC power with the balanced platform
profile. These are single runs in B-then-A order, not an alternating production
benchmark or an equivalent-fidelity speedup claim.

| Set / history | A ms/token | B ms/token |
| --- | ---: | ---: |
| heldout / code_diff | 48.3580 | 50.0372 |
| heldout / kept_reasoning | 48.1195 | 50.1268 |
| heldout / long_reasoning | 47.9679 | 50.0629 |
| heldout / plain_prose | 47.8842 | 49.9818 |
| heldout / spanish | 48.1723 | 50.0421 |
| heldout / tool_two_parameters | 48.3374 | 50.0149 |
| original / reasoning_history | 48.3775 | 50.2090 |
| original / tool_round_trip | 47.9242 | 49.9192 |
| original / user_turn | 48.2836 | 50.0510 |

Held-out position-weighted timing: A 48.1460, B 50.0442 ms/token.

Identity hashes and all numeric rows are in [p4c-results.json](p4c-results.json).
Raw rows, build/test logs, traces and private identities remain in ignored
`artifacts/p4c/`; held-out ORNL captures remain in ignored
`artifacts/reference/heldout/`. Toolchain and model are unchanged from P4b;
the full model digest was reverified before comparison.

### P4c limits and unexplained behavior

B is retained because it has fewer held-out criterion failures (3 versus 6),
not because it clears the 20% KL-reduction arm (13.1877%). Neither policy
passes the complete held-out numerical contract. B fails confident-reference
KL at code_diff position 75 (0.1220301624) and has corroborated clear flips
at tool_two_parameters positions 96 and 104. A instead fails kept_reasoning
with two clear flips, long_reasoning with one confident-KL violation, and
the tool history with three criterion failures. All six held-out prompt
mean-KL criteria pass for both policies.

The direction and location of the failures depend on the activation policy.
These measured observations do not establish a monotonic benefit from more
F32 arithmetic, an operator bug, or universal generalization. A causal account
of the recurrent/routing decisions amplifying each perturbation remains
unexplained. No further policies were tried and no thresholds were changed.
The o-F32 kernel and its independent F64 tests remain; the temporary A source
toggle is removed.

CTest repeats verbose output for a failed test. The initial orchestration
counted those repeated rows twice and stopped after B completed; its
bookkeeping was corrected to count unique history names, then A was run.
All completed B measurements were retained without a rerun or exclusion.

Reproduction: use the held-out batch/sequential capture loop in
[tools/ornith15/README.md](../../../tools/ornith15/README.md#held-out-activation-policy-captures),
then the documented GPU configure/build and decode test with each capture
directory. To reproduce A, temporarily replace only the full-attention
`Session::Attention` call to `DenseMatVecF32` with
`Dense(l.attn_out, ctx_, block_out_)`, rebuild, and restore the source after
measurement. B is the committed default; no runtime policy selector exists.

### P4c final validation

GPU configure and both target builds passed. Retained B's rebuilt executable
has exactly the pre-comparison B SHA256; production source is unchanged from
`b5968851`. The final combined Ornith and Flash-Next suite passed **29/29**,
zero failures/skips, **571.07 seconds**, using the original oracle directory.
Its B results reproduce every original-prompt top-1/KL/failure statistic above.
This original-set suite pass does not erase the held-out failures.
Formatting passed **650 C++ files**. Plain host `cmake --preset cpu-test` and
`cmake --build --preset pr -j 28` passed **66/66**, zero failures/skips,
**36.01 seconds**. AC power and balanced profile matched at start and finish.

The first CPU attempt incorrectly wrapped the existing host-configured
CPU build in `nix develop`: 56/66 tests failed at startup with glibc 2.43
requirements versus Nix's 2.42 runtime, including loader errors and aborts.
The plain host commands above resolve that environment mismatch without
source changes. Both attempts remain in the raw logs (`pr-final.log` and
`pr-host.log`); this did not affect the GPU-policy comparison or its pinned
Nix toolchain. CPU host tool versions are recorded in `p4c-results.json`.

## P4d prior evidence (recorded October 24, 2026; not a new measurement)

P4d ran after `a97a8495` and did not change production sources. On October 24
the following identities were rehashed and matched the P4c pins, so the P4d
artifacts below are adopted as prior evidence rather than rerun:

| Identity | SHA256 |
| --- | --- |
| Engine HEAD | `a97a84950d11191ab1dfa9fe0081ef0e011d0009` |
| APEX MTP Quality GGUF | `c83373e4c502c6d4929339406ffbb1f442598b153a3ade02b628752e00a282fd` |
| Restored B `ornith15_gpu_decode_test` | `a5358cb088996ee0b69dec483edcedd21789c062535a8bb76e9cb2c5fb7d332d` |
| Native dumper | `f5bd6218664e66bc837e81f4727f79536a70cb5850d7b4034d9dd0edccb1137b` |
| llama.cpp archive `68d9053a` | `7ec6ef2e02c29a15879cd4c997b55976f1f0bbce4a042de15b71b376df52dc40` |

`reference.cpp`, `executor.cpp`, `gpu_decode_test.cpp` and
`reference_oracle_test.cpp` match `artifacts/p4d/identities.json`. All six
original captures, all twelve held-out captures, the five reference libraries,
the six held-out prompt bytes and the three frozen confirmation prompts match
the recorded digests. Confirmation prompts were hashed only; they were not
scored. Git status at this record is `ornith-apex` with those production
sources clean. No push was done.

The scalar reference's unchanged native-oracle contract fails 3 of 6 held-out
histories. Original-set passage did not qualify these histories.

| History | Positions | Outcome | Clear-flip positions | Confident-KL positions | Confident max KL | Mean KL | Native self-spread confident max |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| code_diff | 110 | fail | 62 | 48 | 0.3352508399 | 0.1136313756 | 5.040428549 |
| kept_reasoning | 105 | pass | none | none | 0.01184742515 | 0.09683127252 | 0.008572688449 |
| long_reasoning | 113 | pass | none | none | 0.001439062827 | 0.05190873378 | 0.001561245914 |
| plain_prose | 100 | fail | 47 | 87 | 1.676610564 | 0.1138889323 | 0.04933347484 |
| spanish | 100 | pass | none | none | 0.003007087172 | 0.06015610973 | 0.03758152676 |
| tool_two_parameters | 121 | fail | 86 | 44, 58, 82 | 0.8428643482 | 0.09802616422 | 1.448712299 |

Native batch-versus-sequential confident KL reaches 5.0404 on code and 1.4487
on the tool history. That is reference numerical spread. It does not erase a
GPU criterion and it is not a reason to change a threshold.

Precision controls kept packed weights, token IDs and the pinned llama.cpp
revision. Modified Q8_0, Q5_K, Q6_K and IQ4_XS dots used original F32
activations and F64 accumulation over dequantized blocks. F32-weight
projections, including the router, kept native accumulation. F32 K/V alone
left captures byte-identical because Flash Attention casts K/V back to F16.
With F32 K/V and Flash Attention disabled, sequential control versus scalar:

| History | Top-1 | Mean KL(scalar \|\| control) |
| --- | ---: | ---: |
| code_diff | 108/110 | 0.003641090934 |
| plain_prose | 100/100 | 3.512098409e-10 |
| tool_two_parameters | 121/121 | 2.212404796e-10 |

A matched unmodified quantized build with the same optional CPU backends
disabled reproduced all three original native sequential captures exactly.
These controls are not goldens and do not make every operation equivalent to
the scalar reference. The code-history remainder is unexplained.

At B's three failing positions the first differing expert sets are at layers
0, 6 and 13. Independent F64 routing on the GPU inputs supported selected-set
membership only. It did not establish ordered IDs, normalized weights or the
final expert reduction. The first residual relative RMS above the diagnostic
marker 0.1 was after MoE at layers 29, 23 and 17. That marker is not an
operator tolerance and not a causal attribution.

Uniform control C added a second activation-residual pass on every remaining
quantized projection while keeping B's o-F32 path. Tool flips disappeared and
code position 54 reached confident KL 5.280833756. C failed the zero-violation
guard, was rejected, and B was restored. That run used cached scalar rows. It
is not qualification and its timings are not comparable to P4c. Frozen
confirmation histories were not captured or scored.

P4d also contains failed startup and build attempts. Those are provenance,
not numerical results. Raw logs stay in ignored `artifacts/p4d/`. CTest
repeats failed output; histories are counted once.

## P4e protocol (registered October 24, 2026, before new scoring)

This section is the measurement protocol for the next diagnosis. It defines
references, the token target, boundaries, the identical-state matrix, the
arithmetic distinctions, operator bounds and the unchanged gates. No P4e
candidate is scored above this heading. Operator bounds are derived from the
declared representation, the reduction length and the rounding operations.
They are not fitted to observed GPU error. The 0.1 residual-RMS marker is
not reused as a tolerance.

### Three references

| Reference | Definition | What a disagreement means | Limit |
| --- | --- | --- | --- |
| N, native llama.cpp | Existing batch and sequential ORNL captures. Batch is the compatibility direction in the mean-KL gate. | A GPU criterion uses N as written. Native batch/sequential spread is reported beside it. | N is not a mathematical oracle. Spread does not waive a GPU failure. |
| S, scalar graph, with P4d controls as context | `ReferenceModel` at this revision: F32 activations, F64 dots where that file accumulates in F64, libm `exp`/`log1p`. P4d's four-format F64 llama control is supporting context for the quantized dots only. | S versus GPU mixes representation cost, reduction order and any implementation defect. | Router and other F32-weight dots were not changed in the llama control. S already fails three held-out native contracts. S is not replaced by a control. |
| B-eval, independent evaluator of policy B | Host code that does not call the GPU quantizer or the GPU dot. M evaluates packed activation fields and the consumed packed-weight formula. F evaluates original F32 activations times independently dequantized weights, summed in F64, then rounded to binary32. G is the production GPU output on the same input. | G−M is implementation arithmetic. M−F is representation cost. | B-eval cannot replace N, S or the acceptance gates. A smaller whole-model KL does not attribute an operator. |

Policy B, unchanged: one Q8_1 activation pass for quantized projections,
except full-attention `o_proj`, which keeps F32 activations and reads the
original packed weights once. Stored F32 weights stay on the F32 path.
Experts, packed weight formats, token histories and reasoning length stay
as they are. AR skips `blk.40`.

### Graph fact used by the layer-0 replay

In both `ReferenceModel::Step` and `rocm::Session::Step`, a token's layer-0
input is the RMSNorm of that token's embedding. Layer-0 MoE and every later
layer run after linear attention returns. They do not write the layer-0
convolution history or the layer-0 recurrent state. The next token's layer-0
call reads only its own embedding and those two histories. Replaying tokens
0 through 75 of layer 0 from the embeddings therefore reconstructs the
production layer-0 state. It does not reconstruct the other 39 layers.
State orientation in both sources is `[value head][value row][key column]`,
with value head `h` reading key head `h mod n_k_heads`.

### Target and fixed gates

Diagnostic replay uses token IDs from native `code_diff.ologits`, positions
0 through 75 inclusive. The same IDs are in the sequential capture. Captures
are not regenerated. Tool positions 96 and 104 are in scope only after the
code prefix has located a boundary, and only with the same method.

GPU acceptance gates stay exactly:

- Clear flip: GPU top-1 differs from the scalar reference, the reference
  top-two margin is greater than 1.0, and both native captures agree with
  the reference top-1 at that position.
- Confident reference: entropy below 0.5 nats, and KL(reference \|\| GPU)
  at most 0.1 there.
- Per prompt, mean KL(llama batch \|\| GPU) at most 2.0 times mean
  KL(llama batch \|\| sequential).

Every position is reported, including positions after a failure. Histories,
prompts, denominators and weights are not changed after results are seen.
Final qualification recomputes S uncached. Cached scalar rows are allowed
only in a run labelled diagnostic. Missing model, device or captures are
skips, never passes. Confirmation prompts stay untouched until a candidate
has zero registered violations on all original and all exposed held-out
histories. Their hashes are
`schedule_confirmation.txt` `c0e732b0…81803`,
`french_confirmation.txt` `46b54896…108b3`,
`json_confirmation.txt` `5cf20820…c359f`.

### Q8_1 encoding contract

Production decode quantizes with `quantize_q8_1` into `block_q8_1`, not the
MMQ tile layout. Block length B = 32. The allocated row length is
`ne0 = ceil(K / 512) * 512`. Elements at indices `>= K` are exact zeros and
participate in the block maximum and the auxiliary sum. The independent
encoder uses that same tail. Tests keep the tail at this allocation boundary.

For finite `x` in a block, let `a = max |x_i|`. Absolute value and maximum
of finite binary32 values are exact. If `a = 0`, every code is 0 and both
stored halves are `+0`. Otherwise:

- `d = RN32(a / 127)`, IEEE-754 binary32 division.
- `q_i = round_half_away_from_zero(RN32(x_i / d))`, the C `roundf` rule.
- Because `|d - a/127| <= 2^{-24} |a/127|` for a normal result,
  `|a/d - 127| < 2e-5`, so the maximum-magnitude code is exactly `±127` and
  every code lies in `[-127, 127]`. An implementation that emits `-128` or
  that saturates by truncation is outside the contract.
- `sumq = Σ q_i` is an exact integer. `|sumq| <= 4064`, so it is an exact
  binary32 integer.
- The stored scale is `RN16(d)`. The stored auxiliary is
  `RN16(RN32(d * sumq))`. The scale that created the codes is `d`, not the
  stored half. The auxiliary is not `RN16(RN16(d) * sumq)`.

Binary16 rounding is ties-to-even. GPU-emitted `qs`, scale and auxiliary
must be bit-identical to this encoder. There is no ulp allowance on the
encoding. Nonfinite inputs are rejected by the evaluator. A GPU NaN is not
a pass.

Which field a consumer reads is part of the formula:

- Q8_0 and Q6_K MMVQ read the stored scale only, promoted exactly from
  binary16 to binary32. They do not read the auxiliary sum.
- Q4_1, Q5_1, Q4_K and Q5_K dots read the auxiliary sum as its own rounded
  value. A generic dot of reconstructed activations `scale * q` is not the
  consumed formula for those types.
- IQ4_XS is specified from its MMVQ consumer when a routed projection of
  that type is evaluated. It is not assumed to match Q6_K.

Layer-0 linear projections are the first tensors inventoried. The o-F32
kernel is in scope when a shared-kernel regression needs it. The whole
model is not expanded into a second persistent F32 copy.

### M, F, D and the dot bounds

Let `u = 2^{-24}` and `γ_k = k u / (1 - k u)` for `k u < 1`. Let `P` be the
sum of absolute exact per-element products under the consumed formula.
`M_real` is that sum in real arithmetic. `M = RN32(M_real)`.

`F_real` is the F64 dot of the original finite F32 activations with
independently dequantized weights. Dequantization uses the stored integer
codes and the stored scales as exact dyadics: for Q6_K,
`d_half * int8_scale * (q6 - 32)`; for Q8_0, `d_half * q8`. `F = RN32(F_real)`.
Padding zeros are included. `M − F` is reported as representation cost.
It is not an implementation failure.

`G` is the production GPU result. `G − M` is an implementation discrepancy
only when `G` matches none of the declared binary32 sequences below. Those
sequences are the source expression with and without contraction of a
multiply into a following add. They are not a tolerance fitted to `G`.

Q6_K MMVQ, `VDR = 1`, `qi = 32`, one warp, superblock 256. Lane `L` owns
`iqs = L`. Each superblock partial is two exact 4-wide integer dots
(`dp4a`), each multiplied by its int8 scale (the integer product is an
exact binary32 value), then by the Q8_1 scale. The two terms sum from zero,
then multiply by the Q6_K superblock scale. Lanes walk superblocks in
increasing order and add into a zero accumulator. The warp then butterfly-
adds with xor offsets 16, 8, 4, 2, 1. Declared sequences:

- `D_mul`: every multiply and add rounded separately.
- `D_fma_inner`: the `sumf += d8 * (dot * scale)` step is one `fma`.
- `D_fma_outer`: that inner `fma`, and `tmp += d6 * sumf` is one `fma`.

Q8_0 MMVQ uses `qi = 8`, `VDR = 2`, so a lane's block stride is 8. The same
three sequences apply to `d_w * d_a * integer_dot` and to the running sum.
Q8_0 does not read the auxiliary sum.

The o-F32 kernel is a different formula. Each lane sequentially sums
reconstructed weights times the original F32 activation. Q6_K
reconstruction is `(RN32(d_half * scale)) * code`, then times `x`. The
warp reduction is a shuffle-down tree of offsets 16, 8, 4, 2, 1. `D_mul`
and the contracted `sum += weight * x` are the two declared sequences.
This path does not replace nonfinite outputs with zero. The Q8_1 MMVQ
writer does replace a nonfinite row sum with zero; a finite fixture must
not rely on that replacement.

`G` satisfies the implementation contract for a projection when it is
bit-identical to one declared `D`. Otherwise the difference is unexplained.
Separately, before any GPU result is interpreted, each `D` must satisfy

`|D - M_real| <= (γ_n + (1 + γ_n) ((1 + u)^3 - 1)) P`

where `n` is the number of binary32 additions on one element's path:
inner group adds, plus `n_partials - 1` sequential partial adds, plus 5
butterfly adds. For a layer-0 Q6_K projection with input length 2048,
`n_partials = 8` and `n = 1 + 7 + 5 = 13`. For `ssm_out`, input length
4096, `n_partials = 16` and `n = 21`. If a `D` exceeds this bound, that
diagnostic is wrong and is not used to excuse `G`. The bound is not a
GPU pass/fail substitute for bit-identity with `D`.

Relative RMS uses the reference Euclidean norm. If both vectors are zero,
relative RMS is 0. If the reference norm is 0 and the other vector is not,
relative RMS is reported as infinite and the absolute error is kept.
Nonfinite values fail the comparison.

### Layer-0 boundaries

For every token `t` in `0..75`, and for every evaluator that applies,
record the first nonzero difference and, separately, the first difference
that is outside its declared bound or outside its declared sequence.
Drift does not have to grow. Do not binary-search a threshold.

| Boundary | Data | Declared distinction |
| --- | --- | --- |
| Embedding | Q6_K or Q8_0 row for the token id. No dot reduction. | `M` is the binary32 reconstruction `(d * scale) * code`. `F` is the F64 dyadic rounded to binary32. |
| Attention RMSNorm | Exact input and output bytes, weight, `rms_eps`. | Scalar sum of squares is F64. GPU norm is its own kernel. Compare bytes; do not treat a later 0.1 residual marker as this bound. |
| QKV, z, alpha, beta | Projection output and the Q8_1 bytes actually stored. | `G−M` and `M−F` as above. Alpha and beta have 32 outputs and K = 2048. |
| Convolution | History in, current projection, taps, pre-SiLU value, post-SiLU value, history out. | Four products. Scalar association is `((w3 v + w0 h0) + w1 h1) + w2 h2`. GPU `SsmConv4Kernel` association is `((w0 h0 + w1 h1) + w2 h2) + w3 v`. Both then apply SiLU. Association difference of the pre-SiLU value is at most `γ_3` times the sum of absolute products. SiLU itself is libm on the scalar side and `__expf` on the device. |
| Q/K preparation | Head index, pre-norm and post-norm values, scale. | Both use clamp-epsilon L2: `1 / max(sqrt(sum of squares), eps)`, not `rsqrt(sum + eps)`. Scalar sum of squares is F64. GPU sum is a binary32 wave reduction. The clamp form is the selected semantics, not a discrepancy by itself. |
| Decay and beta | Alpha, beta, `a`, `dt`, decay, sigmoid. | Scalar: `exp(softplus(alpha + dt) * a)` and `1/(1+exp(-beta))` with libm, softplus threshold 20. GPU: `__expf` and `log1pf(__expf)`. No host guess at an `__expf` ulp budget is added. An expression-tree mismatch is a defect. A libm-versus-device numeric gap is reported as math-library drift and is causal only if an intervention shows it. |
| Recurrent update | Incoming state, decayed state, `u`, error or delta, outgoing state. | Scalar dots are F64 then rounded, and the update is `(v - u) * beta` added as `delta * key`. GPU non-tail lanes use `fma(beta, RN(error * key), decayed)`. Each lane's last key, indices 31, 63, 95 and 127, uses `fma(pre_decay, decay, RN(RN(error * key) * beta))`. Lane width is 32, four lanes, xor offsets 2 then 1. These sequences are predeclared. A nonzero scalar-versus-GPU gap is not by itself a defect. |
| Recurrent output | Raw output and the query scale. | Scalar scale is `1/sqrt(float(128))`. GPU scale is `rsqrtf(128)`. Output dots differ in the same way as `u`. |
| Gated norm | RMSNorm output, SiLU of z, product. | Same norm and SiLU distinctions as above. |
| ssm_out | Q8_1 bytes if quantized, projection, residual add. | Same `G−M` / `M−F` contract. K = 4096 for this projection. |

Raw arrays stay in ignored `artifacts/p4e/`. Each array's manifest records
dtype, shape, layout, little-endian, token, layer, boundary, source hashes
and the incoming-state hash. The recurrent state is 32×128×128 binary32,
2 MiB per boundary. Identical incoming and outgoing states may be stored
once. Production replay uses the production launcher and the real weights.
A lookalike kernel is not `G`. Instrumentation is test-only. Instrumented
and uninstrumented outputs must match where both are collected; a mismatch
means the instrumented run is not production evidence.

### Identical-input, identical-state matrix

At the earliest unexplained update, freeze one operand bundle and one
incoming state from each source. If the operation includes convolution, the
incoming convolution history is part of the bundle. Reset state before
every trial. Hashes of the fixtures are recorded.

| Operands | Incoming state | Question |
| --- | --- | --- |
| Reference | Reference | Reference trajectory control |
| GPU | GPU | GPU trajectory control |
| Reference | GPU | State contribution with reference operands |
| GPU | Reference | Operand contribution with reference state |

These crosses are counterfactuals. They are not claims about the real
trajectory. Outgoing state and output are both compared. Inherited history
is reported separately from the local update. The local recurrence
comparison uses the scalar formula and the GPU non-tail and tail formulas
above on identical operands. It does not rewrite the GPU sequence into the
scalar algebra.

### Routing audit, only at implicated boundaries

On identical logits the audit reports full-softmax probabilities, ordered
top-8 IDs, the 8th-versus-9th margin, renormalized weights, the clamp
`max(sum, 6.103515625e-5)`, expert-to-slot mapping, the weighted reduction
and the shared-expert sigmoid gate. Selected-set equality is not enough.

The GPU router keeps the higher probability and breaks remaining ties by
the lower expert index (`other > best || (other == best && oi < index)`).
The scalar `partial_sort` comparator is `probs[a] > probs[b]` with no tie
key. Exact ties and probabilities that match only after rounding are
tested on synthetic logits. The scalar comparator is not silently changed.
Nothing in P4d shows that a tie caused the three registered GPU failures.

Expert reduction is checked with ordered expert outputs and weights held
fixed, so selection is not confused with reduction order. The epilogue
adds `weight * down` from a zero accumulator and then `sigmoid(gate) * shared`.

### Causal intervention rule

Before changing any output, write the prediction: which local quantity
becomes explained, which later route, weight or state changes, and which
of the three known logit violations moves. State whether the substitute is
`M` (matched representation) or `F` (higher-precision mathematical output).
Those answer different questions.

The harness is run first with the original output and state. That control
must reproduce the uninstrumented baseline. Only the suspect operation,
and its outgoing state when it is stateful, is replaced. The suffix tokens,
packed weights and unrelated state stay as they were. Continuing the full
model requires every layer's state through the prefix. Layer-0 state alone
cannot initialize layers 1–39. A stateful intervention continues from the
intervened state.

A correction is causal only when the local unexplained error disappears
and the predicted downstream route, state or logit response occurs. A
lower final KL by itself is compensation. One operation may explain one
failure and not the other two. An unsuccessful intervention is kept in
the record.

### What this protocol does not authorize

No expert-count change, no shorter history, no truncated reasoning, no
artifact substitution, no runtime precision selector, and no per-layer
precision sweep. No CLI, server, MTP, vision, driver or Q6_K comparison
work. If every checked B operation matches its declared sequence and the
contract still fails, B is reported insufficient on the checked scope and
a mechanism-based next policy is proposed without running it. If the first
unexplained boundary is not closed, the result is inconclusive and names
that boundary. B remains the production policy until a replacement has
zero registered violations on the original set, the exposed held-out set
and the untouched confirmation set.

### P4e operator comparison (synthetic fixtures, October 24, 2026)

This is not a model-logit score and not a qualification. The host
evaluator is `tools/ornith15/representation.py`. The dump binary
`ornith15_representation_gpu_test` calls `qfn_mmq_quantize_q8_1`,
`qfn_mmq_moe_vec` and `DenseMatVecF32` only. Production decode was not
rebuilt: its SHA256 remains `a5358cb0…7d332d`. Raw dumps are in ignored
`artifacts/p4e/wp1-representation/`.

Layer 0 of the selected GGUF, from the header only: token embedding
Q6_K; linear projections `attn_qkv`, `attn_gate`, `ssm_alpha`,
`ssm_beta` and `ssm_out` are Q6_K; routed experts are Q6_K; shared
experts are Q8_0; norms, convolution, `ssm_a`, `ssm_dt` and both router
inputs are F32. There is no layer-0 `o_proj`. The first full-attention
output, `blk.3.attn_output.weight`, is Q6_K and is the o-F32 path.
Q5_K routed experts begin at `blk.5`. IQ4_XS is present and is not a
layer-0 tensor. `blk.40.attn_output` is Q8_0; decode skips that block.

On the synthetic dump, 55 scored rows:

- Q8_1 bytes for K = 32, 512 and 544, including the zero row, the
  half-away row and the K = 544 tail, match the independent encoder.
  No ulp slack was allowed.
- Every Q6_K MMVQ row (K = 256 and K = 2048, duplicate expert slots
  included) matches `D_fma_inner` or `D_fma_outer`, and some also match
  `D_mul`. Every predeclared D is inside the registered gamma bound.
  Rounded G−M is 0 or about 2^-13 to 2^-12. Rounded M−F on the same
  rows is orders of magnitude larger, up to about 2^4 on a K = 2048
  row. That gap is representation cost.
- Every o-F32 row matches `D_fma`, and some also match `D_mul`. On
  these fixtures M−F is 0: o-F32 has no Q8_1 substitution, and the F64
  sum rounded to binary32 matched the real sum. G−M is 0 or about
  2^-18 to 2^-13.
- Q8_0 MMVQ at K = 32 matches every left-associated sequence. At
  K = 512 and K = 2048, 10 of 16 rows match only `D_compiled`:
  `acc = fma(RN(d_w * dot), d_a, acc)`, which is the association
  written beside `mul_mat_vec_q8_ahead`. Those rows are one ulp or less
  from M (about 2^-18 to 2^-15, or zero) and do not match the
  left-associated `(d_w * d_a) * dot` sequences. Their D values still
  satisfy the gamma bound. M−F remains the larger term.

No row is unexplained once `D_compiled` is reported separately. The
left-associated Q8_0 sequences are not bit-identical to G on those 10
rows. Layer-0 Q6_K projections, which are the linear-attention dots,
match a predeclared sequence on every synthetic row tried here. This
does not locate the three held-out logit failures.

### P4e layer-0 replay (held-out code_diff tokens 0–75)

`ornith15_layer0_replay_test` replays layer 0 only, from a zero
convolution history and a zero recurrent state, using `EmbedQ6`,
`RmsNormRows`, `qfn_mmq_quantize_q8_1`, `qfn_mmq_moe_vec` and
`GatedDeltaNet`. It does not run MoE or layers 1–39. Token IDs are the
first 76 entries of held-out `code_diff.ologits`. The host Q6 sequences
were checked against the synthetic GPU dump before this replay: 8 of 8
rows at K = 256 and 8 of 8 at K = 2048 matched a declared sequence.
The log is ignored `artifacts/p4e/wp2-layer0-replay.log`.

On all 76 tokens, embedding bytes match the host Q6 reconstruction, and
every row of `ssm_qkv`, `ssm_gate`, `ssm_alpha`, `ssm_beta` and
`ssm_out` matches `D_mul`, `D_fma_inner` or `D_fma_outer`. There is no
first unexplained Q6 dot.

The independent encoder matches the GPU Q8_1 codes and stored scale on
every block of all 76 norm rows (K = 2048, 64 blocks each). Two blocks,
token 63 block 16 and token 66 block 47, differ only in the auxiliary
half, by one binary16 ulp. Nearby binary32 scales that keep the same
codes and the same stored scale reproduce that auxiliary, so the device
`amax / 127` division is not always the correctly rounded quotient.
Q6_K MMVQ does not read the auxiliary. The layer-0 dots are unaffected.

Host SiLU of the GPU convolution association stays within 1.9e-6
absolute of the GPU post-SiLU value. The pairwise gap between the two
f32 add associations exceeds the published one-summation `γ_3` bound on
some channels (40 tokens, first at token 4; that log is retained). A
host-only search also finds inputs whose two orders exceed `γ_3 P`, up
to about 1.23 times that allowance. That does not justify a fitted
constant. Each order is within `γ_3` of the exact sum of the same four
rounded products, so the triangle inequality gives a pairwise allowance
of `2 γ_3 Σ|p_i|`. The corrected replay has zero channels above that
pairwise allowance. Prep and one-step recurrence gaps stay small. They
are unallocated numerical differences, not a math-library attribution
and not a cleared layer-0 path.

### Review corrections (same prefix, exit status enforced)

The replay now returns non-zero for mandatory mismatches. An injected
embed, Q6, Q8, nonfinite, GPU-rerun and pairwise failure each produce a
non-zero disposition; the old one-summation convolution count does not.
Missing model or captures return 77. Self-check is a separate mode.
The rerun log is `artifacts/p4e/wp2-layer0-replay-r2.log`. It exits 5.

Q8 codes and stored scales still match on all 4,864 blocks. The two
auxiliary-only mismatches are in the table (`q8_aux_bad` 1 at tokens 63
and 66) and fail the process. Q6 rows still match a registered sequence
on every token, and a second GPU launch from the saved inputs and state
is byte-identical (`gg_bad` 0).

An independent scalar trajectory, using the reference matvec, SiLU, L2
and recurrence, runs beside the GPU trajectory from the same token IDs
and a zero initial state. The largest absolute gaps are about 3.8e-6 in
the attention norm, 0.104 in `ssm_qkv`, 0.108 in recurrent state and
0.00240 in `ssm_out`. At token 75 the state gap is about 0.0576.
Crossing scalar projections with GPU state, and GPU projections with
scalar state, moves the outgoing state by up to about 0.091. These are
trajectory differences, not a located operator defect. The saved
boundaries, both state trajectories and fixture hashes are under
`artifacts/p4e/wp2-layer0/`.

The synthetic Q8_0 score is split the same way. Ten rows match only
`D_compiled`. `original_protocol_pass` is false and the scorer exits 1.
`supplemental_explained` is 10. The earlier aggregate that treated those
rows as a pass is kept as `score-compiled-aggregate.json`.

### Crossed evaluators and one-operand swaps (same prefix)

`ornith15.layer0_replay` now gets `GUFO_ORNITH15_SOURCE_DIR` from CMake
and CTest runs it with the engine source tree as its working directory.
The encoder script resolves from that root. A missing script still leaves
the Q8 check unmeasured. The measured rerun is
`artifacts/p4e/wp2-layer0-replay-swap.log`. It exits 5. The original
one-summation log `artifacts/p4e/wp2-layer0-replay.log` stays in place:
40 tokens, first at token 4. This run counts the same 40 tokens, first
at token 4, and zero channels above the pairwise `2 γ_3 Σ|p_i|` allowance.

Each of the four input/state combinations ran on the production GPU
launcher and on the scalar evaluator. Incoming input means the qkv, gate,
alpha, beta and convolution history. State means the recurrent matrix.
Outgoing state, gated output and `ssm_out` were compared and saved for
all eight cells. GPU/GPU is byte-identical on those three tensors
(`gg_bad` 0). Scalar/scalar matches the independent scalar trajectory
(`scalar_rr_bad` 0). With the GPU bundle and the GPU incoming state, the
scalar state stays within 2.861e-6 (token 20) and the scalar gated output
stays within 7.153e-7 (token 24). In that same cell `ssm_out` still
differs by up to 0.002423 (token 17), beside Q6 rows that match a
registered sequence.

The reference-input/GPU-state cell moves state by up to 0.09116 (token
6). The GPU-input/reference-state cell moves state by up to 0.08638
(token 7) and is the larger cell on 70 of the 76 tokens. At token 0 the
history and the recurrent state are zero, so the trajectory gap of
0.004060 is entirely the reference-input cell. At token 75 the trajectory
state gap is 0.05761: 0.05595 of that step follows the incoming reference
state, and 0.01595 follows the current reference bundle.

A further production launch replaced one scalar operand at a time and
kept every other projection, the history and the incoming state on the
GPU trajectory. Replacing the gate changes the state by 0 on all 76
tokens and changes `ssm_out` by up to 0.001378 (token 59). At token 0,
alpha and history replacements also leave the state unchanged, while qkv
and beta move it by 0.002572 and 0.002429. The token-1 history gap,
0.017666, equals the token-0 qkv gap. Over the prefix the largest
one-step state movement is the convolution history on 64 tokens, beta on
7 and the current qkv on 5. Those peaks are 0.09007 (history, token 6),
0.04801 (beta, token 59), 0.02629 (current qkv, token 4) and 0.02094
(alpha, token 55).

The one-step state movements are not additive contributions. They
support a carried test of qkv/history propagation. They do not establish
that a continued substitution shrinks the state gap, and they do not
establish that the `ssm_out` gap keeps the matched-operand magnitude of
about 0.0024. That magnitude forecast is withdrawn.

The frozen substitute is an F64 matvec of `ssm_qkv` on the exact GPU
RMSNorm output of that token. Gate, alpha, beta, `GatedDeltaNet` and
`ssm_out` stay on the production launch. The convolution history and the
recurrent state carried forward are the outgoing values of this chain,
starting from zero. The qkv taken from the scalar trajectory uses the
scalar RMSNorm output as well as the F64 matvec. The one-step
`swap_scalar_qkv` column is that broader substitution.

The frozen prediction has no magnitude. It says that the harness, which
reinserts the production qkv and carries its own history and state, is
byte-identical to the production state, gated output, `ssm_out` and
outgoing history at all 76 positions. The substituted trajectory is
compared at all 76 positions and is not assigned a forecast. A local response that justifies
a later routing and logit test is a matching harness together with a
carried state or `ssm_out` that differs from production. Production stays
policy B. The auxiliary-field failure still fails the process. The three
registered logit failures stay unresolved until that later test is run.
Q8 codes and scales still match; tokens 63 and 66 still differ by one
auxiliary ulp. The table and fixtures are under `artifacts/p4e/wp2-layer0/`.

### Carried GPU-norm qkv substitution (same prefix)

The definition and prediction in the previous section were committed
before this run. The log is
`artifacts/p4e/wp2-layer0-replay-qkv-carry.log`. The process exits 5.
Q8 codes and scales match. Tokens 63 and 66 still fail the auxiliary
half. The one-summation count is again 40, first at token 4, and the
pairwise count is 0. The harness mismatches on 0 of 76 tokens.
Reinserting the production qkv and carrying that chain's history and
state reproduces the production state, gated output, `ssm_out` and
outgoing history.

The F64 `ssm_qkv` on the GPU RMSNorm output differs from the production
Q6 row at all 76 positions. Its maximum absolute difference is 0.103668
at token 21. The broader scalar-norm column has the same maximum. The
two printed per-token maxima differ by at most 3.8e-6, and they are
equal on 55 tokens. At token 0 both are 0.017666.

The carried chain differs from production at every position. The state
gap is 0.002572 at token 0, reaches 0.108024 at token 6, and is 0.062038
at token 75. The outgoing history gap reaches 0.103668. The gated-output
gap reaches 0.001230 at token 0. The `ssm_out` gap is the production Q6
kernel applied to the substituted gated output. It reaches 0.000828 at
token 6 and is 0.000130 at token 75. These magnitudes are the
measurement. The one-step table was not used to assign them.

The frozen decision rule is met. The harness matches, and the carried
state and `ssm_out` differ from production, so a later routing and logit
test on an unchanged suffix is justified. That test has not been run.
Production remains policy B. The code_diff/75 KL failure and the two
tool flips remain unresolved.
