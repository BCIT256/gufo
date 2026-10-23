# Ornith GPU checkpoint experiments

October 23, 2026. Correctness checkpoint; no optimization or speedup claim.

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
