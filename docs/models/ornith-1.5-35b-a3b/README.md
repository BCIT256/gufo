# Ornith 1.5 35B-A3B: GPU AR checkpoint

Experimental `qwen35moe` library support for single-token autoregressive
decode on gfx1151. This checkpoint has no CLI/server integration, chunked
prefill, graphs, batching, vision, YaRN or MTP execution. Feed a prompt by
repeated `Session::Step` calls. Each successful call returns the full vocabulary
and advances one position; invalid calls leave state untouched. A device
execution failure requires `Reset` before reuse.

Bind with `ModelWeights::Bind(reader, BindOptions{})`, upload with
`rocm::DeviceModel::Upload`, then create `rocm::Session` with an explicit context
capacity. The device model must outlive the session. `blk.40` stays unbound.
The tested artifact is APEX MTP Quality, SHA256
`c83373e4c502c6d4929339406ffbb1f442598b153a3ade02b628752e00a282fd`.
Other artifacts accepted by the binder are not runtime-qualified here.

Composition follows Flash-Next at `53b0f5b7`, with a plain residual stream and
none of its hyper-connections, indexer, PLE or sidecar machinery. Flash-Next's
kernel and MMQ libraries are linked directly. DeltaNet selects SiLU output
gating and clamped L2 normalization explicitly; Flash-Next's defaults retain
sigmoid gating and additive epsilon. Attention uses the generic dense path
with F16 KV. Conv/recurrent states remain F32. All capacities come from Config.

Each quantized projection dispatches its own GGUF format. Most dense matrices
use the one-expert view of MMVQ; routed gate/up/down retain all eight selected
experts and the per-layer Q6_K/Q5_K/IQ4_XS formats. Shared experts stay Q8_0;
router and shared-gate weights stay F32. Attention output projections retain
F32 activations and multiply their original packed Q6_K/Q8_0 weights directly;
all remaining quantized projections use one Q8_1 MMVQ pass. No projection uses
a second residual pass. See [experiments](EXPERIMENTS.md) for the isolation and
quality evidence, including the pre-registered held-out comparison. Weights
are never narrowed or replaced. The Ornith device
helpers are the Q6_K embedding gather and the F32-input dense projection.

`ornith15.gpu_ops` checks embedding bytes, single-pass routed projections and
the F32-input dense projection against a host-unpacked F64 oracle.
`ornith15.gpu_decode` compares the GPU and scalar reference on the captured histories when both `GUFO_ORNITH15_APEX_MTP_MODEL`
and `GUFO_ORNITH15_LLAMA_LOGITS` are set. Missing inputs yield skip code 77.
An optional `LayerTrace` records the residual after each layer on either
implementation without changing their arithmetic.

See [quality](QUALITY.md) and [measurement scope](BENCHMARKS.md).
