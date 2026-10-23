# Quantized HIP kernels

Adapted from `ggml-org/llama.cpp` at
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT; see `LICENSE`).
The upstream source directory is `ggml/src/ggml-cuda`; Gufo builds these
sources with HIP for AMD Strix Halo (`gfx1151`).

`qfn_mmq.h` is the public adapter. It supplies Q8_0 dense projections and
Q4_K, Q5_K, Q6_K, IQ4_XS, Q5_1 and Q8_0 routed expert projections, including
the paired Q4_K gate/up path. `qfn_ggml_stubs.*` supplies the minimal context and scratch
pool without the ggml graph runtime. Gufo owns scheduling and HIP graphs.

The kernels use native HIP runtime, hipBLAS and HIP numeric types. The
upstream runtime aliases, NVIDIA capability dispatch and PTX branches have
been removed; retained matrix fragments use AMD WMMA/MFMA. Headers use the
`.hpp` suffix. Gufo owns scheduling and HIP graphs.

When importing an upstream fix, preserve the Gufo namespace, adapter and
HIP-only dispatch, update the pin, and run the Flash-Next operator tests
under `tests/models/qwen38_flash_next/` plus the AR/MTP session replay.
The routed GEMM test compares supported formats against a floating-point
reference and checks the paired adapter against two separate projections.
Build and validate through Nix as described in `AGENTS.md`.

The Ornith format addition enables the pinned upstream `vec_dot_q6_K_q8_1`
and `vec_dot_iq4_xs_q8_1` in MMVQ (separate, paired and grouped gate/up),
and the existing Q6_K/IQ4_XS tiles in MMQ/MUL_MAT_ID. No upstream pin or
dot-product arithmetic changed. Both formats require whole 256-value weight
blocks; Ornith's K=2048 gate/up and K=512 down projections meet this contract.
Grouped gate/up accepts at most eight tokens and 32 selected experts (Ornith
uses eight); the wider Q4_K route remains format-specific. The K=640
expert-ordered down specialization remains restricted to Q5_1/Q8_0.

`routed_kquant_ops_test.cpp` forces both MMVQ and MMQ for these two formats
with 256 experts, top-8 routing, both Ornith projections and 1/2/3/8/64 tokens.
Down inputs are independent slot rows (eight per token). Its independent
host unpacking and F64 dots use the existing routed test's activation
quantization bound: maximum error below 0.01 times maximum absolute reference
output. Output guards, paired projections and scalar/grouped gate identity
are also checked. These are synthetic operator checks, not full-model
Ornith AR/MTP qualification.

The [operator result summary](../../../../../../docs/models/qwen3.8-flash-next/artifacts/ornith-routed-kquant.json)
records the toolchain, fixture identities and per-shape errors. All 21
Flash-Next tests (the existing 20 plus this check), formatting and the 66-test
PR suite passed on October 23, 2026. Raw logs remain outside the repository.
