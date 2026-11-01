# Ornith 1.5 35B-A3B: GPU AR checkpoint

`qwen35moe` autoregressive decode and chunked prefill on gfx1151.
`gufo serve llm --model <gguf>` and `gufo chat --model <gguf>` load this
architecture on ROCm. The server keeps an exact snapshot of each session
frontier and reuses it when a later prompt has the same prefix. A miss
re-prefills. Capture queues a private device image at the boundary and copies
it to the host after that boundary, so a cached turn does not wait for the
host copy before its first token. A later restore of that image skips KV
rows that are already the live prefix; a disk payload uploads every region.
On a grid session (`Session::SetCanonicalGrid`; the server uses width 1)
prefill gives the same bits for a token wherever the prompt is split: batched
Q6_K and Q5_K products use a fixed order, and a one-token launch keeps the
batched kernels instead of the decode dots. Each call of up to 4096 tokens is one
launch. Routed prefill uses a 64-column MMQ tile when that chunk has at least
1024 tokens and a 48-column tile below that; dense projections keep the
default tile. The server reuses a cached prompt at its exact boundary and also
publishes a finished reply's decode frontier, so the next turn resumes through
the reply. Decode steps use other kernels, so such a turn is held to the
distribution gate, not bit identity with a cold prompt. A cancelled request still
returns to a prefill frontier. The direct session API leaves the grid off, so an
explicit chunk size is unchanged. With a RAM cache, `Session::PrefillThrough`
freezes the cache boundary into a staged image inside the same launch and the
host copy runs in the background, so the boundary costs no second forward
pass. The host payload is the disk cache when `--cache-disk` is set; then
prefill stops at snapshot boundaries, as upstream requires for disk.
The server drafts with the bundled NextN block when the file has one, unless
`--speculative off` is given; `--speculative mtp` requires it. It runs at most 7
drafts and verifies that chain in one trunk launch on the decode dots, so
greedy output equals autoregressive and each verified row is bit-identical
to a decode step. It is faster than autoregressive decode on the measured
prompts (see BENCHMARKS.md). The prefix cache accepts it.
Prompt lookup adds proposals copied from the request's own prompt and output
(`speculative::PromptLookup`, the Flash-Next rules), and the same verification
checks them. `Session::CatchUpMtp` then writes NextN K/V for the copied tokens, bit-identical to
drafting them one at a time. The index is rebuilt from each request's prompt, so
it is not in the snapshot.
An MTP snapshot is version
2 and stores NextN K/V plus the trunk hidden carry. The draft chain is
cold at a prefill frontier; it is not rebuilt from a side buffer. The
draft-length controller resets at each request and is not in the
snapshot. Vision and YaRN are absent. Other `--speculative` backends,
`--mmproj` and `--enable-yarn` are rejected. `Session::Prefill` runs a prompt
in chunks of at most 4096 tokens. `Session::Step` advances one token. A plain decode step
may replay a HIP graph. Pass a vocabulary-sized buffer to receive logits, or
an empty buffer plus a sampled id to take a device argmax without downloading
logits. Invalid calls leave state untouched. A device execution failure
requires `Reset` before reuse. `SaveSnapshot` copies the position, the
gated-attention KV below it, every DeltaNet recurrent and short-conv state,
and the logits into a version-1 host payload. With MTP enabled the
same call writes version 2, which adds NextN K/V and the trunk carry.
`StageSnapshot` queues that
copy into a private device image and returns without a host sync.
`ExportStagedSnapshot` writes that payload from the image after the
session has moved on. Version 1 is the autoregressive image; version 2
is the MTP image. `RestoreSnapshot` loads a host payload into a fresh or
already used session. `RestoreStagedSnapshot` loads the device image and
skips KV when the destination still holds that prefix. `SnapshotPayloadBytes`
reports the size. A prefill that returns every row also stores its last row in the
decode logit buffer, so the payload matches the published frontier.

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
all remaining quantized projections use one Q8_1 MMVQ pass on decode. No
projection uses a second residual pass. See [experiments](EXPERIMENTS.md) for the isolation and
quality evidence, including the pre-registered held-out comparison. Weights
are never narrowed or replaced. The Ornith device
helpers are the Q6_K embedding gather, the alpha/beta pack, and the F32-input
dense projection. Prefill uses the linked MMQ batched projections, including
Q6_K and IQ4_XS. Full-attention output projections stay on the F32-input dot
for decode; prefill uses the matching MMQ path.

`ornith15.gpu_ops` checks embedding bytes, single-pass routed projections and
the F32-input dense projection against a host-unpacked F64 oracle.
`ornith15.gpu_decode` scores the GPU against the llama.cpp batch and
sequential captures when `GUFO_ORNITH15_APEX_MTP_MODEL` and
`GUFO_ORNITH15_LLAMA_LOGITS` are set. `ornith15.gpu_snapshot` requires a
restored continuation to match the source continuation bit for bit.
Missing inputs yield skip code 77.
The gate is distribution-level; see [quality](QUALITY.md). Set
`GUFO_ORNITH15_SCALAR_REFERENCE=1` to also run the scalar reference. That
comparison is a diagnostic. An optional `LayerTrace` records the residual
after each layer when the caller supplies one, without changing the arithmetic.

See [quality](QUALITY.md) and [measurement scope](BENCHMARKS.md).
