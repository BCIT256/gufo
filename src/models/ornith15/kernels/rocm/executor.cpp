#include "src/models/ornith15/kernels/rocm/executor.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/core/hip/managed_alloc.hpp"
#include "src/models/ornith15/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::ornith15::rocm {
namespace fn = qwen38_flash_next::rocm;
using core::GgmlType;
namespace {

void Check(hipError_t rc, const char* what) {
  if (rc != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(rc));
}

// Packed host image of one session frontier. Fields are native-endian and
// the struct is 64 bytes so a payload has no header padding.
struct SnapshotHeader {
  char magic[4];
  std::uint32_t version;
  std::uint32_t position;
  std::uint32_t max_context;
  std::uint32_t num_layers;
  std::uint32_t full_attention_interval;
  std::uint32_t conv_floats;
  std::uint32_t recurrent_floats;
  std::uint32_t kv_dim;
  std::uint32_t vocab;
  std::uint32_t hidden_size;
  std::uint32_t ssm_conv_kernel;
  std::uint32_t ssm_num_v_heads;
  std::uint32_t ssm_head_dim;
  std::uint32_t num_kv_heads;
  std::uint32_t head_dim;
};
static_assert(sizeof(SnapshotHeader) == 64, "snapshot header is packed");

constexpr char kSnapshotMagic[4] = {'O', 'R', 'N', 'S'};

std::uint32_t ConvFloats(const Config& c) {
  return (c.ssm_conv_kernel - 1) * c.SsmConvChannels();
}

std::uint32_t RecurrentFloats(const Config& c) {
  return c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim;
}

bool HeaderMatches(const SnapshotHeader& header, const Config& c,
                   std::uint32_t max_context) {
  return std::memcmp(header.magic, kSnapshotMagic, 4) == 0 &&
         (header.version == Session::kSnapshotPayloadVersion ||
          header.version == Session::kMtpSnapshotPayloadVersion) &&
         header.max_context == max_context &&
         header.num_layers == c.num_layers &&
         header.full_attention_interval == c.full_attention_interval &&
         c.full_attention_interval != 0 &&
         header.conv_floats == ConvFloats(c) &&
         header.recurrent_floats == RecurrentFloats(c) &&
         header.kv_dim == c.AttentionKvDim() && header.vocab == c.vocab_size &&
         header.hidden_size == c.hidden_size &&
         header.ssm_conv_kernel == c.ssm_conv_kernel &&
         header.ssm_num_v_heads == c.ssm_num_v_heads &&
         header.ssm_head_dim == c.ssm_head_dim &&
         header.num_kv_heads == c.num_kv_heads &&
         header.head_dim == c.head_dim && header.position <= max_context;
}

std::size_t PayloadBytes(const SnapshotHeader& header) {
  std::size_t bytes = sizeof(SnapshotHeader);
  const std::size_t conv = std::size_t(header.conv_floats) * sizeof(float);
  const std::size_t recurrent =
      std::size_t(header.recurrent_floats) * sizeof(float);
  const std::size_t kv_row = std::size_t(header.kv_dim) * sizeof(__half);
  for (std::uint32_t layer = 0; layer < header.num_layers; ++layer) {
    const bool linear = ((layer + 1) % header.full_attention_interval) != 0;
    if (linear)
      bytes += conv + recurrent;
    else
      bytes += 2 * std::size_t(header.position) * kv_row;
  }
  if (header.version == Session::kMtpSnapshotPayloadVersion) {
    bytes += 2 * std::size_t(header.position) * kv_row;
    bytes += std::size_t(header.hidden_size) * sizeof(float);
  }
  bytes += std::size_t(header.vocab) * sizeof(float);
  return bytes;
}

SnapshotHeader MakeHeader(const Config& c, std::uint32_t position,
                          std::uint32_t max_context, bool mtp) {
  SnapshotHeader header{};
  std::memcpy(header.magic, kSnapshotMagic, 4);
  header.version = mtp ? Session::kMtpSnapshotPayloadVersion
                       : Session::kSnapshotPayloadVersion;
  header.position = position;
  header.max_context = max_context;
  header.num_layers = c.num_layers;
  header.full_attention_interval = c.full_attention_interval;
  header.conv_floats = ConvFloats(c);
  header.recurrent_floats = RecurrentFloats(c);
  header.kv_dim = c.AttentionKvDim();
  header.vocab = c.vocab_size;
  header.hidden_size = c.hidden_size;
  header.ssm_conv_kernel = c.ssm_conv_kernel;
  header.ssm_num_v_heads = c.ssm_num_v_heads;
  header.ssm_head_dim = c.ssm_head_dim;
  header.num_kv_heads = c.num_kv_heads;
  header.head_dim = c.head_dim;
  return header;
}

std::uint64_t NextLineage() {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

StagedSnapshot::~StagedSnapshot() {
  if (ready != nullptr) {
    (void)hipEventSynchronize(ready);
    (void)hipEventDestroy(ready);
    ready = nullptr;
  }
  if (device != nullptr) {
    (void)hipFree(device);
    device = nullptr;
  }
}

Session::~Session() {
  if (mtp_stream_ != nullptr) {
    (void)hipStreamSynchronize(mtp_stream_);
    qfn_mmq_release_stream(mtp_stream_);
  }
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
    qfn_mmq_release_stream(stream_);
  }
  if (copy_stream_ != nullptr)
    (void)hipStreamSynchronize(copy_stream_);
  if (graph_exec_ != nullptr)
    (void)hipGraphExecDestroy(graph_exec_);
  for (hipGraphExec_t exec : verify_graph_) {
    if (exec != nullptr)
      (void)hipGraphExecDestroy(exec);
  }
  for (void* p : allocations_)
    (void)hipFree(p);
  if (mtp_stream_ != nullptr)
    (void)hipStreamDestroy(mtp_stream_);
  if (copy_stream_ != nullptr)
    (void)hipStreamDestroy(copy_stream_);
  if (stream_ != nullptr)
    (void)hipStreamDestroy(stream_);
}

void* Session::Allocate(std::size_t bytes) {
  void* p = nullptr;
  Check(hip::AllocateDevice(&p, bytes), "session allocation");
  allocations_.push_back(p);
  bytes_ += bytes;
  return p;
}

float* Session::Floats(std::size_t count) {
  return static_cast<float*>(Allocate(count * sizeof(float)));
}

void Session::EnsureLogitsBatch() {
  if (logits_batch_ != nullptr)
    return;
  logits_batch_ =
      Floats(std::size_t(kMaxPrefillChunk) * model_->config().vocab_size);
}

void Session::NoteGraphFailure() {
  graph_failed_ = true;
  if (graph_noted_)
    return;
  graph_noted_ = true;
  std::fprintf(stderr,
               "ornith15: decode graph capture failed; staying eager\n");
}

std::unique_ptr<Session> Session::Create(const DeviceModel& model,
                                         std::uint32_t max_context,
                                         std::string* error) {
  auto s = std::unique_ptr<Session>(new Session);
  try {
    const auto& c = model.config();
    if (max_context == 0 || max_context > c.context_length)
      throw std::runtime_error("invalid AR context capacity");
    s->model_ = &model;
    s->max_context_ = max_context;
    s->lineage_ = NextLineage();
    Check(hipStreamCreateWithFlags(&s->stream_, hipStreamNonBlocking),
          "stream");
    Check(hipStreamCreateWithFlags(&s->copy_stream_, hipStreamNonBlocking),
          "snapshot stream");
    const std::uint32_t n = kMaxPrefillChunk;
    const std::uint32_t used = c.num_experts_used;
    s->zero_ids_ = static_cast<std::int32_t*>(
        s->Allocate(std::size_t(n) * sizeof(std::int32_t)));
    s->zero_id_ = s->zero_ids_;
    s->token_ids_ = static_cast<std::int32_t*>(
        s->Allocate(std::size_t(n) * sizeof(std::int32_t)));
    s->device_pos_ =
        static_cast<std::uint32_t*>(s->Allocate(sizeof(std::uint32_t)));
    Check(hipMemset(s->zero_ids_, 0, std::size_t(n) * sizeof(std::int32_t)),
          "zero expert ids");
    s->res_ = s->Floats(std::size_t(n) * c.hidden_size);
    s->xn_ = s->Floats(std::size_t(n) * c.hidden_size);
    s->block_out_ = s->Floats(std::size_t(n) * c.hidden_size);
    s->logits_ = s->Floats(c.vocab_size);
    s->qkv_ = s->Floats(std::size_t(n) * c.SsmConvChannels());
    s->z_ = s->Floats(std::size_t(n) * c.SsmValueDim());
    s->ab_ = s->Floats(std::size_t(n) * 2 * c.ssm_num_v_heads);
    s->alpha_ = s->Floats(std::size_t(n) * c.ssm_num_v_heads);
    s->beta_ = s->Floats(std::size_t(n) * c.ssm_num_v_heads);
    s->conv_ =
        s->Floats(std::size_t(n + c.ssm_conv_kernel) * c.SsmConvChannels());
    s->qn_ = s->Floats(std::size_t(n) * c.SsmKeyDim());
    s->kn_ = s->Floats(std::size_t(n) * c.SsmKeyDim());
    s->raw_ = s->Floats(std::size_t(n) * c.SsmValueDim());
    s->gdn_out_ = s->Floats(std::size_t(n) * c.SsmValueDim());
    s->qg_ = s->Floats(std::size_t(n) * 2 * c.AttentionQDim());
    s->q_ = s->Floats(std::size_t(n) * c.AttentionQDim());
    s->gate_ = s->Floats(std::size_t(n) * c.AttentionQDim());
    s->k_ = s->Floats(std::size_t(n) * c.AttentionKvDim());
    s->v_ = s->Floats(std::size_t(n) * c.AttentionKvDim());
    s->ctx_ = s->Floats(std::size_t(n) * c.AttentionQDim());
    s->router_ = s->Floats(std::size_t(n) * c.num_experts);
    s->route_weights_ = s->Floats(std::size_t(n) * used);
    s->ids_ = static_cast<std::int32_t*>(
        s->Allocate(std::size_t(n) * used * sizeof(std::int32_t)));
    s->expert_gate_ = s->Floats(std::size_t(n) * used * c.expert_ff);
    s->expert_up_ = s->Floats(std::size_t(n) * used * c.expert_ff);
    s->expert_down_ = s->Floats(std::size_t(n) * used * c.hidden_size);
    s->shared_gate_ = s->Floats(std::size_t(n) * c.shared_expert_ff);
    s->shared_up_ = s->Floats(std::size_t(n) * c.shared_expert_ff);
    s->shared_out_ = s->Floats(std::size_t(n) * c.hidden_size);
    s->shared_inp_ = s->Floats(n);
    s->argmax_scratch_ =
        s->Allocate(fn::kArgmaxParts * sizeof(fn::ArgmaxCandidate));
    s->sampled_dev_ =
        static_cast<std::int32_t*>(s->Allocate(sizeof(std::int32_t)));
    s->state_.resize(c.num_layers);
    for (std::size_t i = 0; i < c.num_layers; ++i) {
      auto& state = s->state_[i];
      if (c.IsLinearLayer(i)) {
        state.conv = s->Floats((c.ssm_conv_kernel - 1) * c.SsmConvChannels());
        state.recurrent = s->Floats(std::size_t(c.ssm_num_v_heads) *
                                    c.ssm_head_dim * c.ssm_head_dim);
      } else {
        const auto bytes =
            std::size_t(max_context) * c.AttentionKvDim() * sizeof(__half);
        state.k = static_cast<__half*>(s->Allocate(bytes));
        state.v = static_cast<__half*>(s->Allocate(bytes));
      }
    }
    if (!s->Reset(error))
      return nullptr;
    return s;
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = e.what();
    return nullptr;
  }
}

bool Session::Reset(std::string* error) {
  try {
    if (mtp_stream_ != nullptr)
      Check(hipStreamSynchronize(mtp_stream_), "reset mtp drain");
    mtp_draft_inflight_ = false;
    Check(hipStreamSynchronize(stream_), "reset drain");
    const auto& c = model_->config();
    for (auto& s : state_) {
      if (s.conv != nullptr) {
        Check(hipMemsetAsync(
                  s.conv, 0,
                  (c.ssm_conv_kernel - 1) * c.SsmConvChannels() * sizeof(float),
                  stream_),
              "conv reset");
        Check(hipMemsetAsync(s.recurrent, 0,
                             std::size_t(c.ssm_num_v_heads) * c.ssm_head_dim *
                                 c.ssm_head_dim * sizeof(float),
                             stream_),
              "recurrent reset");
      }
      // KV beyond Position() is never read; it is overwritten before use.
    }
    if (mtp_enabled_) {
      Check(
          hipMemsetAsync(h_carry_, 0,
                         model_->config().hidden_size * sizeof(float), stream_),
          "mtp carry reset");
    }
    Check(hipStreamSynchronize(stream_), "reset");
    position_ = 0;
    mtp_position_ = 0;
    mtp_chain_ = false;
    lineage_ = NextLineage();
    kv_high_water_ = 0;
    failed_ = false;
    return true;
  } catch (const std::exception& e) {
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

void Session::Dense(const DeviceTensor& w, const float* x, float* out) {
  if (w.type == GgmlType::kF32) {
    fn::SmallGemm(w.data, fn::WeightType::kF32, x, out, 1, w.rows, w.cols,
                  stream_);
  } else {
    // A dense quantized matrix is one expert with ID zero. This uses the
    // existing Q6_K/Q8_0 dot products without converting the weights.
    Quantized(w, x, zero_id_, out, 1, 1);
  }
}

void Session::Project(const DeviceTensor& w, const float* x, float* out,
                      std::uint32_t n) {
  if (DecodeDots(n)) {
    if (n == 1) {
      Dense(w, x, out);
      return;
    }
    // Verification rows: each row takes the dot Dense gives one token.
    // F32 below already keeps each token's FMA order at any width.
    if (w.type != GgmlType::kF32) {
      Quantized(w, x, zero_ids_, out, n, 1);
      return;
    }
  }
  if (w.type == GgmlType::kF32) {
    // Every full group of eight tokens is one grid; each token keeps the
    // FMA order a lone token takes.
    fn::SmallGemm(w.data, fn::WeightType::kF32, x, out, n, w.rows, w.cols,
                  stream_);
    return;
  }
  if (w.type == GgmlType::kQ8_0 && w.experts == 1) {
    if (qfn_mmq_q8_0_dense(w.data, x, out, static_cast<int>(w.rows),
                           static_cast<int>(n), static_cast<int>(w.cols),
                           stream_) != 0)
      throw std::runtime_error("dense Q8_0 projection failed");
    return;
  }
  MoeRaw(w, x, zero_ids_, out, n, 1);
}

void Session::Quantized(const DeviceTensor& w, const float* x,
                        const std::int32_t* ids, float* out, std::uint32_t rows,
                        std::uint32_t used) {
  if (qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, x, ids, out, w.rows,
                      w.cols, rows, w.experts, used, stream_) != 0)
    throw std::runtime_error("quantized projection failed");
}

// Q6_K and Q5_K otherwise pair a four-factor product differently on
// adjacent tile columns, so a token's result depended on how many rows
// the call had placed ahead of it. The explicit order does not. Cleared
// on return. Decode stays on the vector dots and does not set this.
struct ExplicitContract {
  bool armed = false;
  explicit ExplicitContract(bool enable) : armed(enable) {
    if (armed)
      qfn_mmq_set_explicit_contract(1);
  }
  ~ExplicitContract() {
    if (armed)
      qfn_mmq_set_explicit_contract(0);
  }
  ExplicitContract(const ExplicitContract&) = delete;
  ExplicitContract& operator=(const ExplicitContract&) = delete;
};

// Routed prefill only. Below 1,024 tokens the tile is 48 columns; a longer
// chunk uses 64. The column grid is capped at the token count passed in:
// gate and up pass the chunk, so tiles past that are not launched; down
// already passes one row per selected slot, and the cap does not shrink it.
// Down's tile follows the prefill chunk, not its expanded row count.
// Cleared on return. Dense projections and decode keep the default selector.
struct RoutedTileHint {
  bool active = false;
  RoutedTileHint(int tokens, int experts, int used, int chunk_tokens) {
    if (experts <= 1 || tokens <= 0 || used <= 0)
      return;
    qfn_mmq_set_routed_max_expert_rows(tokens);
    const int rows = chunk_tokens > 0 ? chunk_tokens : tokens;
    qfn_mmq_set_routed_tile_cols(rows >= 1024 ? 64 : 48);
    active = true;
  }
  ~RoutedTileHint() {
    if (!active)
      return;
    qfn_mmq_set_routed_max_expert_rows(0);
    qfn_mmq_set_routed_tile_cols(0);
  }
  RoutedTileHint(const RoutedTileHint&) = delete;
  RoutedTileHint& operator=(const RoutedTileHint&) = delete;
};

void Session::MoeRaw(const DeviceTensor& w, const float* x,
                     const std::int32_t* ids, float* out,
                     std::uint32_t n_tokens, std::uint32_t used,
                     std::uint32_t chunk_tokens) {
  const int m = static_cast<int>(w.rows);
  const int k = static_cast<int>(w.cols);
  const int n = static_cast<int>(n_tokens);
  const int experts = static_cast<int>(w.experts);
  const int slots = static_cast<int>(used);
  const ExplicitContract contract(!chunk_dependent_reduction_);
  const RoutedTileHint tile(n, experts, slots, static_cast<int>(chunk_tokens));
  int rc = -1;
  switch (w.type) {
    case GgmlType::kQ8_0:
      rc = qfn_mmq_q8_0_moe_raw(w.data, x, ids, out, m, k, n, experts, slots,
                                stream_);
      break;
    case GgmlType::kQ4_K:
      rc = qfn_mmq_q4_K_moe_raw(w.data, x, ids, out, m, k, n, experts, slots,
                                stream_);
      break;
    case GgmlType::kQ5_1:
      rc = qfn_mmq_q5_1_moe_raw(w.data, x, ids, out, m, k, n, experts, slots,
                                stream_);
      break;
    case GgmlType::kQ5_K:
      rc = qfn_mmq_q5_K_moe_raw(w.data, x, ids, out, m, k, n, experts, slots,
                                stream_);
      break;
    case GgmlType::kQ6_K:
      rc = qfn_mmq_q6_K_moe_raw(w.data, x, ids, out, m, k, n, experts, slots,
                                stream_);
      break;
    case GgmlType::kIQ4_XS:
      rc = qfn_mmq_iq4_xs_moe_raw(w.data, x, ids, out, m, k, n, experts, slots,
                                  stream_);
      break;
    default:
      throw std::runtime_error("unsupported batched weight format");
  }
  if (rc != 0)
    throw std::runtime_error("batched projection failed");
}

void Session::LinearAttention(const DeviceLayer& l, std::size_t layer,
                              std::uint32_t n) {
  const auto& c = model_->config();
  Project(l.ssm_qkv, xn_, qkv_, n);
  Project(l.ssm_gate, xn_, z_, n);
  if (n == 1 && DecodeDots(n)) {
    Dense(l.ssm_alpha, xn_, ab_);
    Dense(l.ssm_beta, xn_, ab_ + c.ssm_num_v_heads);
  } else {
    Project(l.ssm_alpha, xn_, alpha_, n);
    Project(l.ssm_beta, xn_, beta_, n);
    PackAlphaBeta(alpha_, beta_, ab_, n, c.ssm_num_v_heads, stream_);
  }
  // PrefillThrough: freeze this layer's conv history and recurrent state at
  // the boundary token inside the same launch.
  fn::GdnCheckpoint checkpoint{};
  if (in_pass_.base != nullptr && layer < in_pass_.conv.size()) {
    checkpoint.state =
        reinterpret_cast<float*>(in_pass_.base + in_pass_.recurrent[layer]);
    checkpoint.history =
        reinterpret_cast<float*>(in_pass_.base + in_pass_.conv[layer]);
    checkpoint.tokens = in_pass_.tokens;
  }
  auto& state = state_[layer];
  fn::RollbackRows state_snaps{};
  fn::RollbackRows conv_snaps{};
  if (gdn_rollback_ && n > 1 && n <= 8 && layer < gdn_saved_.size()) {
    const auto& saved = gdn_saved_[layer];
    for (int row = 0; row < 7; ++row) {
      state_snaps.rows[row] = saved.state[row];
      conv_snaps.rows[row] = saved.conv[row];
    }
  }
  fn::GatedDeltaNet(
      qkv_, c.SsmConvChannels(), z_, c.SsmValueDim(), ab_, l.ssm_conv1d.f32(),
      l.ssm_a.f32(), l.ssm_dt.f32(), l.ssm_norm.f32(), state.conv, conv_, qn_,
      kn_, raw_, state.recurrent, gdn_out_, nullptr, state_snaps, conv_snaps, n,
      c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim, c.ssm_conv_kernel,
      false, false, c.rms_eps, stream_, nullptr, checkpoint, 0, nullptr,
      fn::GdnGate::kSilu, fn::GdnL2Norm::kClampEps);
  Project(l.ssm_out, gdn_out_, block_out_, n);
}

void Session::Attention(const DeviceLayer& l, State& state, std::uint32_t n) {
  const auto& c = model_->config();
  Project(l.attn_q, xn_, qg_, n);
  Project(l.attn_k, xn_, k_, n);
  Project(l.attn_v, xn_, v_, n);
  fn::UnpackQGate(qg_, 2 * c.AttentionQDim(), q_, gate_, nullptr, nullptr, n,
                  c.num_heads, c.head_dim, 0, stream_);
  fn::RmsNormRows(q_, l.attn_q_norm.f32(), q_, n * c.num_heads, c.head_dim, 1,
                  c.rms_eps, stream_);
  fn::RmsNormRows(k_, l.attn_k_norm.f32(), k_, n * c.num_kv_heads, c.head_dim,
                  1, c.rms_eps, stream_);
  fn::Rope(q_, n, c.num_heads, c.head_dim, c.rotary_dim, device_pos_,
           c.rope_theta, stream_);
  fn::Rope(k_, n, c.num_kv_heads, c.head_dim, c.rotary_dim, device_pos_,
           c.rope_theta, stream_);
  fn::StoreKv(k_, state.k, n, c.AttentionKvDim(), device_pos_, stream_);
  fn::StoreKv(v_, state.v, n, c.AttentionKvDim(), device_pos_, stream_);
  fn::Attention(q_, state.k, state.v, nullptr, 0, ctx_, nullptr, 1, n,
                device_pos_, c.num_heads, c.num_kv_heads, c.head_dim, 1,
                stream_);
  fn::SigmoidMul(ctx_, gate_, std::size_t(n) * c.AttentionQDim(), stream_);
  if (DecodeDots(n)) {
    // Keep this input in F32: quantizing both V and the gated context can
    // amplify activation rounding through the attention value path. This dot
    // reads the original packed weights once, without a residual pass.
    for (std::uint32_t row = 0; row < n; ++row)
      DenseMatVecF32(l.attn_out.type, l.attn_out.data,
                     ctx_ + std::size_t(row) * l.attn_out.cols,
                     block_out_ + std::size_t(row) * l.attn_out.rows,
                     l.attn_out.rows, l.attn_out.cols, stream_);
  } else {
    Project(l.attn_out, ctx_, block_out_, n);
  }
}

void Session::Moe(const DeviceLayer& l, std::uint32_t n) {
  const auto& c = model_->config();
  Project(l.router, xn_, router_, n);
  fn::RouterTopK(router_, c.num_experts, ids_, route_weights_, n, c.num_experts,
                 c.num_experts_used, stream_);
  if (DecodeDots(n)) {
    Quantized(l.ffn_gate_exps, xn_, ids_, expert_gate_, n, c.num_experts_used);
    Quantized(l.ffn_up_exps, xn_, ids_, expert_up_, n, c.num_experts_used);
  } else {
    MoeRaw(l.ffn_gate_exps, xn_, ids_, expert_gate_, n, c.num_experts_used);
    MoeRaw(l.ffn_up_exps, xn_, ids_, expert_up_, n, c.num_experts_used);
  }
  fn::Swiglu(expert_gate_, expert_up_,
             std::size_t(n) * c.num_experts_used * c.expert_ff, stream_);
  if (DecodeDots(n)) {
    Quantized(l.ffn_down_exps, expert_gate_, ids_, expert_down_,
              n * c.num_experts_used, 1);
  } else {
    MoeRaw(l.ffn_down_exps, expert_gate_, ids_, expert_down_,
           n * c.num_experts_used, 1, n);
  }
  Project(l.shexp_gate, xn_, shared_gate_, n);
  Project(l.shexp_up, xn_, shared_up_, n);
  fn::Swiglu(shared_gate_, shared_up_, std::size_t(n) * c.shared_expert_ff,
             stream_);
  Project(l.shexp_down, shared_gate_, shared_out_, n);
  Project(l.shexp_gate_inp, xn_, shared_inp_, n);
  fn::MoeEpilogue(expert_down_, route_weights_, shared_out_, shared_inp_, 1,
                  block_out_, n, c.num_experts_used, c.hidden_size, stream_);
}

void Session::RunChunk(std::uint32_t n, LayerTrace* trace) {
  const auto& c = model_->config();
  if (n == 0 || n > kMaxPrefillChunk)
    throw std::runtime_error("invalid chunk width");
  if (model_->token_embd().type == GgmlType::kQ6_K) {
    EmbedQ6Rows(model_->token_embd().data, token_ids_, res_, n, c.hidden_size,
                stream_);
  } else {
    fn::EmbedTokens(model_->token_embd().data, fn::WeightType::kQ8_0,
                    token_ids_, res_, n, c.hidden_size, 1, stream_);
  }
  if (trace != nullptr)
    trace->assign(c.num_layers, std::vector<float>(c.hidden_size));
  for (std::size_t i = 0; i < c.num_layers; ++i) {
    const auto& l = model_->layers()[i];
    fn::RmsNormRows(res_, l.attn_norm.f32(), xn_, n, c.hidden_size, 1,
                    c.rms_eps, stream_);
    if (l.linear)
      LinearAttention(l, i, n);
    else
      Attention(l, state_[i], n);
    fn::MtpAddEmbedding(block_out_, res_, n, c.hidden_size, 1, stream_);
    fn::RmsNormRows(res_, l.post_attention_norm.f32(), xn_, n, c.hidden_size, 1,
                    c.rms_eps, stream_);
    Moe(l, n);
    fn::MtpAddEmbedding(block_out_, res_, n, c.hidden_size, 1, stream_);
    if (trace != nullptr && n == 1)
      Check(hipMemcpyAsync((*trace)[i].data(), res_,
                           c.hidden_size * sizeof(float), hipMemcpyDeviceToHost,
                           stream_),
            "layer trace");
  }
}

void Session::ProjectLastLogits(std::uint32_t n) {
  const auto& c = model_->config();
  fn::RmsNormRows(res_ + std::size_t(n - 1) * c.hidden_size,
                  model_->output_norm().f32(), xn_, 1, c.hidden_size, 1,
                  c.rms_eps, stream_);
  Dense(model_->output(), xn_, logits_);
}

void Session::ProjectAllLogits(std::uint32_t n) {
  const auto& c = model_->config();
  EnsureLogitsBatch();
  fn::RmsNormRows(res_, model_->output_norm().f32(), xn_, n, c.hidden_size, 1,
                  c.rms_eps, stream_);
  if (DecodeDots(n) && model_->output().type != GgmlType::kF32) {
    // Each verification row takes the decode head's dot.
    Quantized(model_->output(), xn_, zero_ids_, logits_batch_, n, 1);
    return;
  }
  // One path for every position. The last-logit head stays on the decode dot.
  MoeRaw(model_->output(), xn_, zero_ids_, logits_batch_, n, 1);
}

void Session::DecodeBody(LayerTrace* trace) {
  RunChunk(1, trace);
  ProjectLastLogits(1);
}

void Session::TryCapture() {
  hipGraph_t captured = nullptr;
  auto begin = hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
  if (begin != hipSuccess) {
    (void)hipGetLastError();
    if (hipStreamSynchronize(stream_) != hipSuccess) {
      NoteGraphFailure();
      return;
    }
    begin = hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
    if (begin != hipSuccess) {
      (void)hipGetLastError();
      NoteGraphFailure();
      return;
    }
  }
  try {
    DecodeBody(nullptr);
    Check(hipGetLastError(), "graph record");
    const hipError_t end = hipStreamEndCapture(stream_, &captured);
    if (end != hipSuccess || captured == nullptr)
      throw std::runtime_error("graph capture end");
    hipGraphExec_t exec = nullptr;
    const hipError_t inst =
        hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0);
    (void)hipGraphDestroy(captured);
    captured = nullptr;
    if (inst != hipSuccess || exec == nullptr)
      throw std::runtime_error("graph instantiate");
    graph_exec_ = exec;
  } catch (...) {
    if (captured != nullptr) {
      (void)hipGraphDestroy(captured);
    } else {
      hipGraph_t junk = nullptr;
      (void)hipStreamEndCapture(stream_, &junk);
      if (junk != nullptr)
        (void)hipGraphDestroy(junk);
    }
    (void)hipGetLastError();
    NoteGraphFailure();
  }
}

void Session::TryCaptureVerify(std::uint32_t n) {
  if (n >= 9 || verify_graph_[n] != nullptr || verify_graph_failed_[n])
    return;
  hipGraph_t captured = nullptr;
  auto begin = hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
  if (begin != hipSuccess) {
    (void)hipGetLastError();
    verify_graph_failed_[n] = true;
    return;
  }
  try {
    RunChunk(n, nullptr);
    ProjectAllLogits(n);
    Check(hipGetLastError(), "verify graph record");
    const hipError_t end = hipStreamEndCapture(stream_, &captured);
    if (end != hipSuccess || captured == nullptr)
      throw std::runtime_error("verify graph capture end");
    hipGraphExec_t exec = nullptr;
    const hipError_t inst =
        hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0);
    (void)hipGraphDestroy(captured);
    captured = nullptr;
    if (inst != hipSuccess || exec == nullptr)
      throw std::runtime_error("verify graph instantiate");
    verify_graph_[n] = exec;
  } catch (...) {
    if (captured != nullptr) {
      (void)hipGraphDestroy(captured);
    } else {
      hipGraph_t junk = nullptr;
      (void)hipStreamEndCapture(stream_, &junk);
      if (junk != nullptr)
        (void)hipGraphDestroy(junk);
    }
    (void)hipGetLastError();
    verify_graph_failed_[n] = true;
  }
}

bool Session::Step(std::int32_t token, std::span<float> logits,
                   std::string* error, LayerTrace* trace,
                   std::int32_t* sampled) {
  const auto& c = model_->config();
  const bool want_logits = logits.size() == c.vocab_size;
  const bool want_sample = sampled != nullptr;
  if (failed_ || token < 0 ||
      static_cast<std::uint32_t>(token) >= c.vocab_size ||
      position_ >= max_context_ ||
      !(want_logits || (want_sample && logits.empty()))) {
    if (error != nullptr)
      *error =
          "invalid AR step (failed session, token, context or logits size)";
    return false;
  }
  const bool plain = trace == nullptr;
  try {
    Check(hipMemcpyAsync(device_pos_, &position_, sizeof(position_),
                         hipMemcpyHostToDevice, stream_),
          "position upload");
    Check(hipMemcpyAsync(token_ids_, &token, sizeof(token),
                         hipMemcpyHostToDevice, stream_),
          "token upload");
    // The graph is the trunk only; NextN consumes the token after it.
    if (plain && graph_exec_ != nullptr) {
      Check(hipGraphLaunch(graph_exec_, stream_), "graph launch");
    } else {
      DecodeBody(trace);
      Check(hipGetLastError(), "AR launch");
      if (plain) {
        const bool capture =
            warmed_ && graph_exec_ == nullptr && !graph_failed_;
        warmed_ = true;
        if (capture)
          TryCapture();
      }
    }
    if (mtp_enabled_ && !mtp_suspend_ && mtp_position_ == position_)
      MtpConsume(1);
    if (want_sample) {
      fn::Argmax(logits_, static_cast<fn::ArgmaxCandidate*>(argmax_scratch_),
                 sampled_dev_, 1, c.vocab_size, stream_);
      Check(hipMemcpyAsync(sampled, sampled_dev_, sizeof(std::int32_t),
                           hipMemcpyDeviceToHost, stream_),
            "sample download");
    }
    if (want_logits)
      Check(hipMemcpyAsync(logits.data(), logits_, logits.size_bytes(),
                           hipMemcpyDeviceToHost, stream_),
            "logits download");
    Check(hipStreamSynchronize(stream_), "AR step");
    ++position_;
    kv_high_water_ = std::max(kv_high_water_, position_);
    return true;
  } catch (const std::exception& e) {
    (void)hipStreamSynchronize(stream_);
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

bool Session::SetCanonicalGrid(std::uint32_t grid) {
  if (position_ != 0 || grid > kMaxPrefillChunk)
    return false;
  canonical_grid_ = grid;
  return true;
}

bool Session::Prefill(std::span<const std::int32_t> tokens,
                      std::span<float> last_logits, std::string* error,
                      std::uint32_t chunk_tokens, std::span<float> all_logits) {
  const auto& c = model_->config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  const bool want_all = !all_logits.empty();
  bool tokens_ok =
      !tokens.empty() && chunk_tokens > 0 && chunk_tokens <= kMaxPrefillChunk &&
      last_logits.size() == c.vocab_size &&
      (!want_all || all_logits.size() == std::size_t(n) * c.vocab_size) &&
      static_cast<std::uint64_t>(position_) + n <= max_context_ &&
      (canonical_grid_ == 0 || position_ % canonical_grid_ == 0);
  if (tokens_ok) {
    for (std::int32_t token : tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
        tokens_ok = false;
        break;
      }
    }
  }
  if (failed_ || !tokens_ok) {
    if (error != nullptr)
      *error =
          "invalid prefill (failed session, token, context, chunk, grid or "
          "logits size)";
    return false;
  }
  // On the grid, one-token launches take the batched kernels rather than the
  // decode dots, so a lone final token matches its row in a wider launch.
  const bool saved_grid_rows = grid_rows_;
  grid_rows_ = canonical_grid_ != 0 && !lone_token_decode_dots_;
  struct RestoreGridRows {
    bool& flag;
    bool value;
    ~RestoreGridRows() { flag = value; }
  } restore_grid_rows{grid_rows_, saved_grid_rows};
  try {
    std::uint32_t done = 0;
    while (done < n) {
      std::uint32_t count = std::min(chunk_tokens, n - done);
      if (canonical_grid_ != 0 && count > canonical_grid_ && done + count < n) {
        // A chunk that does not finish the call ends on a cell boundary. The
        // final chunk keeps its short cell: every launch, a lone 1–15 token
        // cell included, runs the batched kernels, so that cell matches the
        // same tokens at the end of a wider launch without a second pass.
        count -= count % canonical_grid_;
      }
      const bool last = done + count == n;
      Check(hipMemcpyAsync(device_pos_, &position_, sizeof(position_),
                           hipMemcpyHostToDevice, stream_),
            "position upload");
      Check(hipMemcpyAsync(token_ids_, tokens.data() + done,
                           std::size_t(count) * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream_),
            "token upload");
      RunChunk(count, nullptr);
      if (want_all) {
        ProjectAllLogits(count);
        // The snapshot and a later Step read logits_, which the batched head
        // does not write. Keep that one row equal to the published last row.
        if (last) {
          Check(hipMemcpyAsync(
                    logits_,
                    logits_batch_ + std::size_t(count - 1) * c.vocab_size,
                    std::size_t(c.vocab_size) * sizeof(float),
                    hipMemcpyDeviceToDevice, stream_),
                "prefill frontier logits");
        }
        Check(
            hipMemcpyAsync(all_logits.data() + std::size_t(done) * c.vocab_size,
                           logits_batch_,
                           std::size_t(count) * c.vocab_size * sizeof(float),
                           hipMemcpyDeviceToHost, stream_),
            "prefill logits");
      } else if (last) {
        ProjectLastLogits(count);
        Check(hipMemcpyAsync(last_logits.data(), logits_,
                             last_logits.size_bytes(), hipMemcpyDeviceToHost,
                             stream_),
              "prefill last logits");
      }
      if (mtp_enabled_ && !mtp_suspend_ && mtp_position_ == position_)
        MtpConsume(count);
      // Grid cells of one call stay ordered on the compute stream. Syncing
      // each cell would add a host round trip per cell on a long prefill.
      if (canonical_grid_ == 0 || last) {
        Check(hipGetLastError(), "prefill launch");
        Check(hipStreamSynchronize(stream_), "prefill");
        if (want_all && last) {
          const float* row =
              all_logits.data() + std::size_t(n - 1) * c.vocab_size;
          std::copy(row, row + c.vocab_size, last_logits.begin());
        }
      }
      position_ += count;
      done += count;
      kv_high_water_ = std::max(kv_high_water_, position_);
    }
    return true;
  } catch (const std::exception& e) {
    (void)hipStreamSynchronize(stream_);
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

std::size_t Session::SnapshotPayloadBytes() const noexcept {
  return SnapshotPayloadBytes(position_);
}

std::size_t Session::SnapshotPayloadBytes(
    std::uint32_t position) const noexcept {
  if (model_ == nullptr)
    return 0;
  return PayloadBytes(
      MakeHeader(model_->config(), position, max_context_, mtp_enabled_));
}

bool Session::PrefillThrough(std::span<const std::int32_t> tokens,
                             std::span<float> last_logits,
                             std::uint32_t boundary,
                             std::shared_ptr<StagedSnapshot>* staged,
                             std::string* error) {
  if (model_ == nullptr || staged == nullptr || failed_ ||
      canonical_grid_ == 0 || boundary <= position_ ||
      tokens.size() > kMaxPrefillChunk ||
      std::uint64_t{boundary} >= std::uint64_t{position_} + tokens.size() ||
      (mtp_enabled_ && (mtp_chain_ || mtp_position_ != position_))) {
    if (error != nullptr)
      *error =
          "invalid in-pass checkpoint (grid, MTP chain, boundary or width)";
    return false;
  }
  const auto& c = model_->config();
  const SnapshotHeader header =
      MakeHeader(c, boundary, max_context_, mtp_enabled_);
  if (!HeaderMatches(header, c, max_context_)) {
    if (error != nullptr)
      *error = "session configuration cannot be snapshotted";
    return false;
  }
  const std::uint32_t start = position_;
  auto image = std::make_shared<StagedSnapshot>();
  image->position = boundary;
  image->lineage = lineage_;
  image->bytes = PayloadBytes(header);
  std::memcpy(image->header_bytes, &header, sizeof(header));
  // Offsets follow TransferPayload at `boundary`.
  const std::size_t conv_bytes = std::size_t(ConvFloats(c)) * sizeof(float);
  const std::size_t recurrent_bytes =
      std::size_t(RecurrentFloats(c)) * sizeof(float);
  const std::size_t kv_bytes =
      std::size_t(boundary) * c.AttentionKvDim() * sizeof(__half);
  InPassImage layout;
  layout.conv.assign(c.num_layers, 0);
  layout.recurrent.assign(c.num_layers, 0);
  layout.boundary = boundary;
  layout.mtp = mtp_enabled_;
  std::vector<std::pair<std::size_t, std::uint32_t>> kv;  // offset, layer
  std::size_t offset = sizeof(SnapshotHeader);
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    if (c.IsLinearLayer(layer)) {
      layout.conv[layer] = offset;
      offset += conv_bytes;
      layout.recurrent[layer] = offset;
      offset += recurrent_bytes;
    } else {
      kv.emplace_back(offset, layer);
      offset += 2 * kv_bytes;
    }
  }
  if (mtp_enabled_) {
    layout.mtp_k = offset;
    offset += 2 * kv_bytes;
    layout.mtp_carry = offset;
    offset += std::size_t(c.hidden_size) * sizeof(float);
  }
  const std::size_t logit_offset = offset;
  if (logit_offset + std::size_t(c.vocab_size) * sizeof(float) !=
      image->bytes) {
    if (error != nullptr)
      *error = "in-pass snapshot layout diverged";
    return false;
  }
  // The image is optional: without device memory for it, prefill the same
  // tokens and report no checkpoint.
  if (hipMalloc(&image->device, image->bytes) != hipSuccess) {
    (void)hipGetLastError();
    image->device = nullptr;
    staged->reset();
    return Prefill(tokens, last_logits, error);
  }
  try {
    Check(hipEventCreateWithFlags(&image->ready, hipEventDisableTiming),
          "in-pass event");
    Check(hipMemcpyAsync(image->device, image->header_bytes, sizeof(header),
                         hipMemcpyHostToDevice, stream_),
          "in-pass header");
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = e.what();
    return false;
  }
  layout.base = static_cast<std::uint8_t*>(image->device);
  layout.tokens = boundary - start;
  in_pass_ = std::move(layout);
  struct ClearInPass {
    InPassImage& image;
    ~ClearInPass() { image = {}; }
  } clear_in_pass{in_pass_};
  if (!Prefill(tokens, last_logits, error))
    return false;
  try {
    // K/V below the boundary are final once written; later rows do not
    // touch them. The boundary logits use the same one-row head a prefill
    // ending there runs, from that row's final residual.
    auto* base = static_cast<std::uint8_t*>(image->device);
    for (const auto& [at, layer] : kv) {
      Check(hipMemcpyAsync(base + at, state_[layer].k, kv_bytes,
                           hipMemcpyDeviceToDevice, stream_),
            "in-pass K");
      Check(hipMemcpyAsync(base + at + kv_bytes, state_[layer].v, kv_bytes,
                           hipMemcpyDeviceToDevice, stream_),
            "in-pass V");
    }
    const std::uint32_t row = boundary - start - 1;
    if (in_pass_.mtp) {
      // MtpConsume has overwritten res_ with the NextN residual. h_rows_
      // still holds this chunk's trunk output-norm, and NextN K/V through
      // the live end includes the boundary prefix.
      const float* carry = h_rows_ + std::size_t(row) * c.hidden_size;
      Check(hipMemcpyAsync(base + in_pass_.mtp_k, mtp_state_.k, kv_bytes,
                           hipMemcpyDeviceToDevice, stream_),
            "in-pass mtp K");
      Check(hipMemcpyAsync(base + in_pass_.mtp_k + kv_bytes, mtp_state_.v,
                           kv_bytes, hipMemcpyDeviceToDevice, stream_),
            "in-pass mtp V");
      Check(hipMemcpyAsync(base + in_pass_.mtp_carry, carry,
                           std::size_t(c.hidden_size) * sizeof(float),
                           hipMemcpyDeviceToDevice, stream_),
            "in-pass mtp carry");
      Dense(model_->output(), carry,
            reinterpret_cast<float*>(base + logit_offset));
    } else {
      fn::RmsNormRows(res_ + std::size_t(row) * c.hidden_size,
                      model_->output_norm().f32(), xn_, 1, c.hidden_size, 1,
                      c.rms_eps, stream_);
      Dense(model_->output(), xn_,
            reinterpret_cast<float*>(base + logit_offset));
    }
    Check(hipEventRecord(image->ready, stream_), "in-pass event record");
    Check(hipGetLastError(), "in-pass launch");
  } catch (const std::exception& e) {
    (void)hipStreamSynchronize(stream_);
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
  *staged = std::move(image);
  return true;
}

void Session::TransferPayload(std::uint8_t* base, bool into_base,
                              hipMemcpyKind kind, hipStream_t stream,
                              bool skip_kv, std::uint32_t position,
                              std::uint32_t version) const {
  // Callers have already checked that the header matches this session, so
  // the live config describes `base`. Version 2 inserts NextN K/V and the
  // carried trunk hidden before the logits.
  const auto& c = model_->config();
  std::size_t offset = sizeof(SnapshotHeader);
  const std::size_t conv_bytes = std::size_t(ConvFloats(c)) * sizeof(float);
  const std::size_t recurrent_bytes =
      std::size_t(RecurrentFloats(c)) * sizeof(float);
  const std::size_t kv_bytes =
      std::size_t(position) * c.AttentionKvDim() * sizeof(__half);
  const auto copy = [&](void* state, std::size_t count, const char* what) {
    if (count == 0)
      return;
    void* dst = into_base ? static_cast<void*>(base + offset) : state;
    const void* src =
        into_base ? state : static_cast<const void*>(base + offset);
    Check(hipMemcpyAsync(dst, src, count, kind, stream), what);
  };
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    const auto& state = state_[layer];
    if (c.IsLinearLayer(layer)) {
      copy(state.conv, conv_bytes,
           into_base ? "snapshot conv" : "restore conv");
      offset += conv_bytes;
      copy(state.recurrent, recurrent_bytes,
           into_base ? "snapshot recurrent" : "restore recurrent");
      offset += recurrent_bytes;
    } else if (!skip_kv) {
      copy(state.k, kv_bytes, into_base ? "snapshot K" : "restore K");
      offset += kv_bytes;
      copy(state.v, kv_bytes, into_base ? "snapshot V" : "restore V");
      offset += kv_bytes;
    } else {
      offset += 2 * kv_bytes;
    }
  }
  if (version == kMtpSnapshotPayloadVersion) {
    if (!skip_kv) {
      copy(mtp_state_.k, kv_bytes,
           into_base ? "snapshot mtp K" : "restore mtp K");
      offset += kv_bytes;
      copy(mtp_state_.v, kv_bytes,
           into_base ? "snapshot mtp V" : "restore mtp V");
      offset += kv_bytes;
    } else {
      offset += 2 * kv_bytes;
    }
    copy(h_carry_, std::size_t(c.hidden_size) * sizeof(float),
         into_base ? "snapshot mtp carry" : "restore mtp carry");
    offset += std::size_t(c.hidden_size) * sizeof(float);
  }
  const std::size_t logit_bytes = std::size_t(c.vocab_size) * sizeof(float);
  copy(logits_, logit_bytes, into_base ? "snapshot logits" : "restore logits");
  offset += logit_bytes;
  if (offset != PayloadBytes(MakeHeader(c, position, max_context_,
                                        version == kMtpSnapshotPayloadVersion)))
    throw std::runtime_error("snapshot size diverged");
}

bool Session::SaveSnapshot(std::vector<std::uint8_t>* payload,
                           std::string* error) const {
  if (payload == nullptr || model_ == nullptr || failed_ ||
      stream_ == nullptr) {
    if (error != nullptr)
      *error = "invalid snapshot (no destination or failed session)";
    return false;
  }
  const auto& c = model_->config();
  const SnapshotHeader header =
      MakeHeader(c, position_, max_context_, mtp_enabled_);
  if (!HeaderMatches(header, c, max_context_)) {
    if (error != nullptr)
      *error = "session configuration cannot be snapshotted";
    return false;
  }
  std::vector<std::uint8_t> bytes(PayloadBytes(header));
  std::memcpy(bytes.data(), &header, sizeof(header));
  try {
    if (mtp_stream_ != nullptr)
      Check(hipStreamSynchronize(mtp_stream_), "snapshot mtp drain");
    Check(hipStreamSynchronize(stream_), "snapshot drain");
    TransferPayload(bytes.data(), true, hipMemcpyDeviceToHost, stream_, false,
                    position_, header.version);
    Check(hipStreamSynchronize(stream_), "snapshot");
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = e.what();
    return false;
  }
  *payload = std::move(bytes);
  return true;
}

std::shared_ptr<StagedSnapshot> Session::StageSnapshot(std::string* error) {
  if (model_ == nullptr || failed_ || stream_ == nullptr) {
    if (error != nullptr)
      *error = "invalid snapshot stage (failed session)";
    return nullptr;
  }
  const auto& c = model_->config();
  const SnapshotHeader header =
      MakeHeader(c, position_, max_context_, mtp_enabled_);
  if (!HeaderMatches(header, c, max_context_)) {
    if (error != nullptr)
      *error = "session configuration cannot be snapshotted";
    return nullptr;
  }
  auto staged = std::make_shared<StagedSnapshot>();
  staged->position = position_;
  staged->lineage = lineage_;
  staged->bytes = PayloadBytes(header);
  std::memcpy(staged->header_bytes, &header, sizeof(header));
  try {
    Check(hipMalloc(&staged->device, staged->bytes), "snapshot image");
    Check(hipEventCreateWithFlags(&staged->ready, hipEventDisableTiming),
          "snapshot event");
    Check(hipMemcpyAsync(staged->device, staged->header_bytes, sizeof(header),
                         hipMemcpyHostToDevice, stream_),
          "snapshot header");
    if (mtp_stream_ != nullptr)
      Check(hipStreamSynchronize(mtp_stream_), "snapshot mtp drain");
    TransferPayload(static_cast<std::uint8_t*>(staged->device), true,
                    hipMemcpyDeviceToDevice, stream_, false, position_,
                    header.version);
    Check(hipEventRecord(staged->ready, stream_), "snapshot event record");
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = e.what();
    return nullptr;
  }
  return staged;
}

bool ExportStagedImage(const StagedSnapshot& staged,
                       std::vector<std::uint8_t>* payload, std::string* error) {
  if (staged.device == nullptr || staged.ready == nullptr ||
      payload == nullptr || staged.bytes < 64) {
    if (error != nullptr)
      *error = "invalid staged image export";
    return false;
  }
  hipStream_t stream = nullptr;
  try {
    Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
          "image export stream");
    Check(hipEventSynchronize(staged.ready), "image stage");
    std::vector<std::uint8_t> bytes(staged.bytes);
    Check(hipMemcpyAsync(bytes.data(), staged.device, staged.bytes,
                         hipMemcpyDeviceToHost, stream),
          "image export");
    Check(hipStreamSynchronize(stream), "image export");
    if (std::memcmp(bytes.data(), staged.header_bytes, 64) != 0)
      throw std::runtime_error("staged image header mismatch");
    *payload = std::move(bytes);
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = e.what();
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
    return false;
  }
  (void)hipStreamDestroy(stream);
  return true;
}

bool Session::ExportStagedSnapshot(
    const std::shared_ptr<StagedSnapshot>& staged,
    std::vector<std::uint8_t>* payload, std::string* error) const {
  if (staged == nullptr || staged->device == nullptr ||
      staged->ready == nullptr || payload == nullptr ||
      copy_stream_ == nullptr || staged->bytes < 64) {
    if (error != nullptr)
      *error = "invalid staged snapshot export";
    return false;
  }
  try {
    Check(hipEventSynchronize(staged->ready), "snapshot stage");
    std::vector<std::uint8_t> bytes(staged->bytes);
    Check(hipMemcpyAsync(bytes.data(), staged->device, staged->bytes,
                         hipMemcpyDeviceToHost, copy_stream_),
          "snapshot export");
    Check(hipStreamSynchronize(copy_stream_), "snapshot export");
    if (std::memcmp(bytes.data(), staged->header_bytes, 64) != 0)
      throw std::runtime_error("staged snapshot header mismatch");
    *payload = std::move(bytes);
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = e.what();
    return false;
  }
  return true;
}

bool Session::RestoreStagedSnapshot(const StagedSnapshot& staged,
                                    std::span<float> logits,
                                    std::string* error) {
  if (model_ == nullptr || stream_ == nullptr || staged.device == nullptr ||
      staged.ready == nullptr || staged.bytes < sizeof(SnapshotHeader)) {
    if (error != nullptr)
      *error = "invalid staged snapshot restore";
    return false;
  }
  const auto& c = model_->config();
  SnapshotHeader header{};
  std::memcpy(&header, staged.header_bytes, sizeof(header));
  if (!HeaderMatches(header, c, max_context_) ||
      header.position != staged.position ||
      staged.bytes != PayloadBytes(header) ||
      (header.version == kMtpSnapshotPayloadVersion) != mtp_enabled_ ||
      (logits.size() != c.vocab_size && !logits.empty())) {
    if (error != nullptr)
      *error = "staged snapshot does not match this session";
    return false;
  }
  const bool skip_kv = staged.lineage != 0 && staged.lineage == lineage_ &&
                       kv_high_water_ >= staged.position;
  try {
    Check(hipEventSynchronize(staged.ready), "staged restore wait");
    Check(hipStreamSynchronize(stream_), "staged restore drain");
    TransferPayload(static_cast<std::uint8_t*>(staged.device), false,
                    hipMemcpyDeviceToDevice, stream_, skip_kv, staged.position,
                    header.version);
    const std::size_t logit_bytes = std::size_t(c.vocab_size) * sizeof(float);
    const std::size_t logit_offset = staged.bytes - logit_bytes;
    if (!logits.empty()) {
      Check(hipMemcpyAsync(
                logits.data(),
                static_cast<std::uint8_t*>(staged.device) + logit_offset,
                logit_bytes, hipMemcpyDeviceToHost, stream_),
            "staged restore logits");
    }
    Check(hipStreamSynchronize(stream_), "staged restore");
  } catch (const std::exception& e) {
    failed_ = true;
    last_restore_skipped_kv_ = false;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
  position_ = staged.position;
  if (header.version == kMtpSnapshotPayloadVersion) {
    mtp_position_ = staged.position;
    mtp_chain_ = false;
    mtp_draft_inflight_ = false;
  }
  if (!skip_kv) {
    lineage_ = staged.lineage == 0 ? NextLineage() : staged.lineage;
    kv_high_water_ = staged.position;
  }
  last_restore_skipped_kv_ = skip_kv;
  failed_ = false;
  return true;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::span<float> logits, std::string* error) {
  if (model_ == nullptr || stream_ == nullptr) {
    if (error != nullptr)
      *error = "invalid snapshot restore";
    return false;
  }
  const auto& c = model_->config();
  SnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    if (error != nullptr)
      *error = "snapshot payload is truncated";
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (!HeaderMatches(header, c, max_context_) ||
      payload.size() != PayloadBytes(header) ||
      (header.version == kMtpSnapshotPayloadVersion) != mtp_enabled_ ||
      (logits.size() != c.vocab_size && !logits.empty())) {
    if (error != nullptr)
      *error = "snapshot does not match this session";
    return false;
  }
  // Pinning the caller buffer makes the upload one DMA rather than a
  // bounce through unpinned pages. Failure leaves the ordinary path.
  // Restores of one payload share the pin; two threads must not register
  // or unregister the same pages together.
  static std::mutex pin_mu;
  std::lock_guard<std::mutex> pin_lock(pin_mu);
  void* registered = const_cast<std::uint8_t*>(payload.data());
  const bool pinned = hipHostRegister(registered, payload.size(),
                                      hipHostRegisterReadOnly) == hipSuccess;
  if (!pinned)
    (void)hipGetLastError();
  try {
    Check(hipStreamSynchronize(stream_), "restore drain");
    TransferPayload(const_cast<std::uint8_t*>(payload.data()), false,
                    hipMemcpyHostToDevice, stream_, false, header.position,
                    header.version);
    Check(hipStreamSynchronize(stream_), "restore");
    if (!logits.empty()) {
      const std::size_t logit_bytes = std::size_t(c.vocab_size) * sizeof(float);
      std::memcpy(logits.data(), payload.data() + payload.size() - logit_bytes,
                  logit_bytes);
    }
  } catch (const std::exception& e) {
    if (pinned)
      (void)hipHostUnregister(registered);
    failed_ = true;
    last_restore_skipped_kv_ = false;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
  if (pinned)
    (void)hipHostUnregister(registered);
  position_ = header.position;
  if (header.version == kMtpSnapshotPayloadVersion) {
    mtp_position_ = header.position;
    mtp_chain_ = false;
    mtp_draft_inflight_ = false;
  }
  lineage_ = NextLineage();
  kv_high_water_ = header.position;
  last_restore_skipped_kv_ = false;
  failed_ = false;
  return true;
}

bool Session::SnapshotRecurrentRegion(std::span<const std::uint8_t> payload,
                                      std::uint32_t layer, std::size_t* offset,
                                      std::size_t* bytes) {
  if (offset == nullptr || bytes == nullptr ||
      payload.size() < sizeof(SnapshotHeader))
    return false;
  SnapshotHeader header{};
  std::memcpy(&header, payload.data(), sizeof(header));
  if (std::memcmp(header.magic, kSnapshotMagic, 4) != 0 ||
      (header.version != kSnapshotPayloadVersion &&
       header.version != kMtpSnapshotPayloadVersion) ||
      header.full_attention_interval == 0 || layer >= header.num_layers ||
      payload.size() != PayloadBytes(header))
    return false;
  if (((layer + 1) % header.full_attention_interval) == 0)
    return false;
  std::size_t cursor = sizeof(header);
  const std::size_t conv_bytes =
      std::size_t(header.conv_floats) * sizeof(float);
  const std::size_t recurrent_bytes =
      std::size_t(header.recurrent_floats) * sizeof(float);
  const std::size_t kv_bytes =
      std::size_t(header.position) * header.kv_dim * sizeof(__half);
  for (std::uint32_t index = 0; index < layer; ++index) {
    const bool linear = ((index + 1) % header.full_attention_interval) != 0;
    cursor += linear ? conv_bytes + recurrent_bytes : 2 * kv_bytes;
  }
  *offset = cursor + conv_bytes;
  *bytes = recurrent_bytes;
  return cursor + conv_bytes + recurrent_bytes <= payload.size();
}

bool Session::EnableMtp(std::string* error) {
  if (failed_ || model_ == nullptr || position_ != 0 || mtp_enabled_ ||
      !model_->has_mtp()) {
    if (error != nullptr)
      *error = "MTP requires a fresh session and the bundled NextN block";
    return false;
  }
  const auto& block = model_->mtp();
  if (block.nextn_eh_proj.empty() || block.nextn_enorm.empty() ||
      block.nextn_hnorm.empty() || block.nextn_shared_head_norm.empty() ||
      block.attn_q.empty()) {
    if (error != nullptr)
      *error = "NextN block is missing a projection";
    return false;
  }
  try {
    const auto& c = model_->config();
    const std::uint32_t n = kMaxPrefillChunk;
    const std::uint32_t hidden = c.hidden_size;
    h_rows_ = Floats(std::size_t(n) * hidden);
    h_in_ = Floats(std::size_t(n) * hidden);
    h_carry_ = Floats(hidden);
    h_carry_saved_ = Floats(hidden);
    chain_h_ = Floats(hidden);
    e_buf_ = Floats(std::size_t(n) * hidden);
    e_norm_ = Floats(std::size_t(n) * hidden);
    h_norm_ = Floats(std::size_t(n) * hidden);
    concat_ = Floats(std::size_t(n) * 2 * hidden);
    // The drafter shares no buffer with the captured decode graph. Its
    // quantized dots take a pool keyed by this stream, not the trunk's.
    Check(hipStreamCreateWithFlags(&mtp_stream_, hipStreamNonBlocking),
          "mtp stream");
    const auto used = c.num_experts_used;
    mtp_token_ = static_cast<std::int32_t*>(Allocate(sizeof(std::int32_t)));
    mtp_pos_ = static_cast<std::uint32_t*>(Allocate(sizeof(std::uint32_t)));
    mtp_res_ = Floats(hidden);
    mtp_xn_ = Floats(hidden);
    mtp_block_ = Floats(hidden);
    mtp_logits_ = Floats(c.vocab_size);
    mtp_hidden_ = Floats(hidden);
    mtp_qg_ = Floats(2 * c.AttentionQDim());
    mtp_q_ = Floats(c.AttentionQDim());
    mtp_gate_ = Floats(c.AttentionQDim());
    mtp_k_ = Floats(c.AttentionKvDim());
    mtp_v_ = Floats(c.AttentionKvDim());
    mtp_ctx_ = Floats(c.AttentionQDim());
    mtp_router_ = Floats(c.num_experts);
    mtp_route_w_ = Floats(used);
    mtp_ids_ = static_cast<std::int32_t*>(
        Allocate(std::size_t(used) * sizeof(std::int32_t)));
    mtp_expert_gate_ = Floats(std::size_t(used) * c.expert_ff);
    mtp_expert_up_ = Floats(std::size_t(used) * c.expert_ff);
    mtp_expert_down_ = Floats(std::size_t(used) * hidden);
    mtp_shared_gate_ = Floats(c.shared_expert_ff);
    mtp_shared_up_ = Floats(c.shared_expert_ff);
    mtp_shared_out_ = Floats(hidden);
    mtp_shared_inp_ = Floats(1);
    const auto bytes =
        std::size_t(max_context_) * c.AttentionKvDim() * sizeof(__half);
    mtp_state_.k = static_cast<__half*>(Allocate(bytes));
    mtp_state_.v = static_cast<__half*>(Allocate(bytes));
    Check(hipMemset(h_carry_, 0, hidden * sizeof(float)), "mtp carry");
    // Prefix images exist only for the 128-wide, 4-tap DeltaNet this model
    // uses. A wider verification batch keeps the full-trunk snapshot path.
    if (c.ssm_head_dim == 128 && c.ssm_conv_kernel == 4) {
      gdn_saved_.assign(c.num_layers, {});
      const auto channels = c.SsmConvChannels();
      const auto hist = std::size_t(c.ssm_conv_kernel - 1) * channels;
      for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
        if (!c.IsLinearLayer(layer))
          continue;
        for (std::uint32_t row = 0; row < 7; ++row) {
          const auto floats = fn::GdnRollbackRowFloats(
              row, c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim);
          gdn_saved_[layer].state[row] = Floats(floats);
          gdn_saved_[layer].conv[row] = Floats(hist);
        }
      }
      gdn_rollback_ready_ = true;
    }
    // The runner measures this session once, at the end of EnableMtp, and
    // refuses a later restore whose resident size is larger. VerifyForward
    // projects every row, so the logit batch has to exist before that
    // measurement rather than on first use.
    EnsureLogitsBatch();
    mtp_enabled_ = true;
    mtp_position_ = 0;
    mtp_chain_ = false;
    return true;
  } catch (const std::exception& e) {
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

void Session::CaptureTrunkHidden(std::uint32_t n) {
  const auto& c = model_->config();
  fn::RmsNormRows(res_, model_->output_norm().f32(), h_rows_, n, c.hidden_size,
                  1, c.rms_eps, stream_);
  Check(hipMemcpyAsync(h_carry_, h_rows_ + std::size_t(n - 1) * c.hidden_size,
                       c.hidden_size * sizeof(float), hipMemcpyDeviceToDevice,
                       stream_),
        "mtp carry");
}

void Session::MtpRun(std::uint32_t n, std::uint32_t start_pos,
                     const float* hidden, bool write_head) {
  const auto& c = model_->config();
  const auto& l = model_->mtp();
  Check(hipMemcpyAsync(device_pos_, &start_pos, sizeof(start_pos),
                       hipMemcpyHostToDevice, stream_),
        "mtp position");
  if (model_->token_embd().type == GgmlType::kQ6_K) {
    EmbedQ6Rows(model_->token_embd().data, token_ids_, e_buf_, n, c.hidden_size,
                stream_);
  } else {
    fn::EmbedTokens(model_->token_embd().data, fn::WeightType::kQ8_0,
                    token_ids_, e_buf_, n, c.hidden_size, 1, stream_);
  }
  fn::RmsNormRows(e_buf_, l.nextn_enorm.f32(), e_norm_, n, c.hidden_size, 1,
                  c.rms_eps, stream_);
  fn::RmsNormRows(hidden, l.nextn_hnorm.f32(), h_norm_, n, c.hidden_size, 1,
                  c.rms_eps, stream_);
  ConcatRows(e_norm_, h_norm_, concat_, n, c.hidden_size, stream_);
  Project(l.nextn_eh_proj, concat_, res_, n);
  fn::RmsNormRows(res_, l.attn_norm.f32(), xn_, n, c.hidden_size, 1, c.rms_eps,
                  stream_);
  Attention(l, mtp_state_, n);
  fn::MtpAddEmbedding(block_out_, res_, n, c.hidden_size, 1, stream_);
  fn::RmsNormRows(res_, l.post_attention_norm.f32(), xn_, n, c.hidden_size, 1,
                  c.rms_eps, stream_);
  Moe(l, n);
  fn::MtpAddEmbedding(block_out_, res_, n, c.hidden_size, 1, stream_);
  if (!write_head)
    return;
  fn::RmsNormRows(res_ + std::size_t(n - 1) * c.hidden_size,
                  l.nextn_shared_head_norm.f32(), chain_h_, 1, c.hidden_size, 1,
                  c.rms_eps, stream_);
  Dense(model_->output(), chain_h_, logits_);
}

void Session::MtpConsume(std::uint32_t n) {
  const auto& c = model_->config();
  fn::RmsNormRows(res_, model_->output_norm().f32(), h_rows_, n, c.hidden_size,
                  1, c.rms_eps, stream_);
  ShiftHidden(h_rows_, h_carry_, h_in_, n, c.hidden_size, stream_);
  MtpRun(n, mtp_position_, h_in_, false);
  Check(hipMemcpyAsync(h_carry_, h_rows_ + std::size_t(n - 1) * c.hidden_size,
                       c.hidden_size * sizeof(float), hipMemcpyDeviceToDevice,
                       stream_),
        "mtp carry");
  mtp_position_ += n;
  mtp_chain_ = false;
}

void Session::SwapDraftScratch() noexcept {
  std::swap(stream_, mtp_stream_);
  std::swap(token_ids_, mtp_token_);
  std::swap(device_pos_, mtp_pos_);
  std::swap(res_, mtp_res_);
  std::swap(xn_, mtp_xn_);
  std::swap(block_out_, mtp_block_);
  std::swap(logits_, mtp_logits_);
  std::swap(qg_, mtp_qg_);
  std::swap(q_, mtp_q_);
  std::swap(gate_, mtp_gate_);
  std::swap(k_, mtp_k_);
  std::swap(v_, mtp_v_);
  std::swap(ctx_, mtp_ctx_);
  std::swap(router_, mtp_router_);
  std::swap(route_weights_, mtp_route_w_);
  std::swap(ids_, mtp_ids_);
  std::swap(expert_gate_, mtp_expert_gate_);
  std::swap(expert_up_, mtp_expert_up_);
  std::swap(expert_down_, mtp_expert_down_);
  std::swap(shared_gate_, mtp_shared_gate_);
  std::swap(shared_up_, mtp_shared_up_);
  std::swap(shared_out_, mtp_shared_out_);
  std::swap(shared_inp_, mtp_shared_inp_);
}

bool Session::BeginDraft(std::int32_t token, std::string* error) {
  const auto& c = model_->config();
  if (!mtp_enabled_ || failed_ || mtp_draft_inflight_ || token < 0 ||
      static_cast<std::uint32_t>(token) >= c.vocab_size ||
      mtp_position_ >= max_context_ || mtp_stream_ == nullptr) {
    if (error != nullptr)
      *error = "invalid MTP draft";
    return false;
  }
  try {
    const float* hidden = mtp_chain_ ? chain_h_ : h_carry_;
    // Copy before the trunk's CaptureTrunkHidden overwrites h_carry_.
    Check(hipMemcpyAsync(mtp_hidden_, hidden, c.hidden_size * sizeof(float),
                         hipMemcpyDeviceToDevice, mtp_stream_),
          "mtp hidden");
    Check(hipStreamSynchronize(mtp_stream_), "mtp hidden");
    SwapDraftScratch();
    try {
      Check(hipMemcpyAsync(token_ids_, &token, sizeof(token),
                           hipMemcpyHostToDevice, stream_),
            "mtp token");
      MtpRun(1, mtp_position_, mtp_hidden_, true);
      Check(hipGetLastError(), "mtp draft launch");
    } catch (...) {
      SwapDraftScratch();
      throw;
    }
    SwapDraftScratch();
    ++mtp_position_;
    mtp_chain_ = true;
    mtp_draft_inflight_ = true;
    return true;
  } catch (const std::exception& e) {
    mtp_draft_inflight_ = false;
    (void)hipStreamSynchronize(mtp_stream_);
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

bool Session::FinishDraft(std::span<float> logits, std::string* error) {
  if (!mtp_enabled_ || !mtp_draft_inflight_ || mtp_stream_ == nullptr) {
    if (error != nullptr)
      *error = "invalid MTP draft";
    return false;
  }
  const auto& c = model_->config();
  try {
    if (logits.size() != c.vocab_size)
      throw std::runtime_error("invalid MTP draft");
    Check(hipMemcpyAsync(logits.data(), mtp_logits_, logits.size_bytes(),
                         hipMemcpyDeviceToHost, mtp_stream_),
          "mtp logits");
    Check(hipStreamSynchronize(mtp_stream_), "mtp draft");
    mtp_draft_inflight_ = false;
    if (failed_)
      throw std::runtime_error("MTP draft after a failed session");
    return true;
  } catch (const std::exception& e) {
    mtp_draft_inflight_ = false;
    (void)hipStreamSynchronize(mtp_stream_);
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

bool Session::DraftToken(std::int32_t token, std::span<float> logits,
                         std::string* error) {
  const auto& c = model_->config();
  if (logits.size() != c.vocab_size) {
    if (error != nullptr)
      *error = "invalid MTP draft";
    return false;
  }
  if (!BeginDraft(token, error))
    return false;
  return FinishDraft(logits, error);
}

bool Session::VerifyForward(std::span<const std::int32_t> tokens,
                            std::span<float> all_logits, std::string* error) {
  const auto& c = model_->config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  bool tokens_ok = mtp_enabled_ && !failed_ && n > 0 && n <= kMaxPrefillChunk &&
                   all_logits.size() == std::size_t(n) * c.vocab_size &&
                   static_cast<std::uint64_t>(position_) + n <= max_context_;
  if (tokens_ok) {
    for (std::int32_t token : tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
        tokens_ok = false;
        break;
      }
    }
  }
  if (!tokens_ok) {
    if (error != nullptr)
      *error = "invalid MTP verification batch";
    return false;
  }
  mtp_suspend_ = true;
  mtp_batch_tokens_ = 0;
  const bool snap = gdn_rollback_ready_ && n > 1 && n <= 8;
  gdn_rollback_ = snap;
  vector_rows_ = n <= kMaxVectorRows;
  try {
    Check(hipMemcpyAsync(device_pos_, &position_, sizeof(position_),
                         hipMemcpyHostToDevice, stream_),
          "verify position");
    Check(hipMemcpyAsync(token_ids_, tokens.data(),
                         std::size_t(n) * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream_),
          "verify tokens");
    if (snap && verify_graph_[n] != nullptr) {
      Check(hipGraphLaunch(verify_graph_[n], stream_), "verify graph");
    } else {
      RunChunk(n, nullptr);
      ProjectAllLogits(n);
      if (snap && verify_warmed_[n] && !verify_graph_failed_[n])
        TryCaptureVerify(n);
      if (snap)
        verify_warmed_[n] = true;
    }
    gdn_rollback_ = false;
    vector_rows_ = false;
    Check(hipMemcpyAsync(logits_,
                         logits_batch_ + std::size_t(n - 1) * c.vocab_size,
                         std::size_t(c.vocab_size) * sizeof(float),
                         hipMemcpyDeviceToDevice, stream_),
          "verify frontier");
    Check(
        hipMemcpyAsync(all_logits.data(), logits_batch_,
                       all_logits.size_bytes(), hipMemcpyDeviceToHost, stream_),
        "verify logits");
    CaptureTrunkHidden(n);
    Check(hipGetLastError(), "verify launch");
    Check(hipStreamSynchronize(stream_), "verify");
    position_ += n;
    kv_high_water_ = std::max(kv_high_water_, position_);
    mtp_batch_tokens_ = snap ? n : 0;
    mtp_suspend_ = false;
    gdn_rollback_ = false;
    return true;
  } catch (const std::exception& e) {
    mtp_suspend_ = false;
    gdn_rollback_ = false;
    vector_rows_ = false;
    (void)hipStreamSynchronize(stream_);
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

bool Session::VerifyDecode(std::int32_t token, std::span<float> logits,
                           std::string* error) {
  const auto& c = model_->config();
  if (!mtp_enabled_ || failed_ || token < 0 ||
      static_cast<std::uint32_t>(token) >= c.vocab_size ||
      position_ >= max_context_ || logits.size() != c.vocab_size) {
    if (error != nullptr)
      *error = "invalid MTP verify step";
    return false;
  }
  mtp_suspend_ = true;
  try {
    Check(hipMemcpyAsync(device_pos_, &position_, sizeof(position_),
                         hipMemcpyHostToDevice, stream_),
          "verify position");
    Check(hipMemcpyAsync(token_ids_, &token, sizeof(token),
                         hipMemcpyHostToDevice, stream_),
          "verify token");
    if (graph_exec_ != nullptr) {
      Check(hipGraphLaunch(graph_exec_, stream_), "verify decode graph");
    } else {
      DecodeBody(nullptr);
      Check(hipGetLastError(), "verify decode");
      const bool capture = warmed_ && !graph_failed_;
      warmed_ = true;
      if (capture)
        TryCapture();
    }
    CaptureTrunkHidden(1);
    Check(hipMemcpyAsync(logits.data(), logits_, logits.size_bytes(),
                         hipMemcpyDeviceToHost, stream_),
          "verify decode logits");
    Check(hipStreamSynchronize(stream_), "verify decode");
    ++position_;
    kv_high_water_ = std::max(kv_high_water_, position_);
    mtp_suspend_ = false;
    return true;
  } catch (const std::exception& e) {
    mtp_suspend_ = false;
    (void)hipStreamSynchronize(stream_);
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

bool Session::RollbackMtpBatch(std::uint32_t keep, std::string* error) {
  if (!mtp_enabled_ || failed_ || mtp_batch_tokens_ < 2 || keep == 0 ||
      keep > mtp_batch_tokens_ || !gdn_rollback_ready_) {
    if (error != nullptr)
      *error = "invalid MTP prefix rollback";
    return false;
  }
  if (keep == mtp_batch_tokens_) {
    mtp_batch_tokens_ = 0;
    return true;
  }
  const auto& c = model_->config();
  const auto conv_bytes =
      std::size_t(c.ssm_conv_kernel - 1) * c.SsmConvChannels() * sizeof(float);
  const auto slot = keep - 1;
  try {
    for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
      if (!c.IsLinearLayer(layer))
        continue;
      const auto& saved = gdn_saved_[layer];
      fn::RollbackRows state_snaps{};
      for (int row = 0; row < 7; ++row)
        state_snaps.rows[row] = saved.state[row];
      fn::RestoreGdnState(state_[layer].recurrent, state_snaps, keep,
                          c.ssm_num_k_heads, c.ssm_num_v_heads, stream_);
      Check(hipMemcpyAsync(state_[layer].conv, saved.conv[slot], conv_bytes,
                           hipMemcpyDeviceToDevice, stream_),
            "mtp conv rollback");
    }
    Check(hipMemcpyAsync(h_carry_, h_rows_ + std::size_t(slot) * c.hidden_size,
                         c.hidden_size * sizeof(float), hipMemcpyDeviceToDevice,
                         stream_),
          "mtp hidden rollback");
    position_ = position_ - mtp_batch_tokens_ + keep;
    kv_high_water_ = position_;
    mtp_batch_tokens_ = 0;
    return true;
  } catch (const std::exception& e) {
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

void Session::RewindMtp(std::uint32_t position) noexcept {
  mtp_position_ = position;
  mtp_chain_ = false;
}

bool Session::CatchUpMtp(std::span<const std::int32_t> tokens,
                         std::uint32_t first_row, std::string* error) {
  const auto& c = model_->config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (!mtp_enabled_ || failed_ || n == 0 || first_row + n > kMaxVectorRows ||
      mtp_position_ + n > max_context_) {
    if (error != nullptr)
      *error = "invalid MTP catch-up";
    return false;
  }
  vector_rows_ = true;
  try {
    Check(hipMemcpyAsync(token_ids_, tokens.data(),
                         std::size_t(n) * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream_),
          "mtp catch-up tokens");
    ShiftHidden(h_rows_, h_carry_saved_, h_in_, first_row + n, c.hidden_size,
                stream_);
    MtpRun(n, mtp_position_, h_in_ + std::size_t(first_row) * c.hidden_size,
           false);
    vector_rows_ = false;
    Check(hipGetLastError(), "mtp catch-up launch");
    Check(hipStreamSynchronize(stream_), "mtp catch-up");
    mtp_position_ += n;
    mtp_chain_ = false;
    return true;
  } catch (const std::exception& e) {
    vector_rows_ = false;
    (void)hipStreamSynchronize(stream_);
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

bool Session::StoreFrontierLogits(std::span<const float> logits,
                                  std::string* error) {
  if (model_ == nullptr || failed_ || stream_ == nullptr ||
      logits.size() != model_->config().vocab_size) {
    if (error != nullptr)
      *error = "invalid frontier logits";
    return false;
  }
  try {
    Check(hipMemcpyAsync(logits_, logits.data(), logits.size_bytes(),
                         hipMemcpyHostToDevice, stream_),
          "frontier logits");
    Check(hipStreamSynchronize(stream_), "frontier logits");
    return true;
  } catch (const std::exception& e) {
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
}

void Session::SaveMtpCarry() {
  if (!mtp_enabled_)
    return;
  const auto bytes = model_->config().hidden_size * sizeof(float);
  Check(hipMemcpyAsync(h_carry_saved_, h_carry_, bytes, hipMemcpyDeviceToDevice,
                       stream_),
        "save mtp carry");
}

void Session::RestoreMtpCarry() {
  if (!mtp_enabled_)
    return;
  const auto bytes = model_->config().hidden_size * sizeof(float);
  Check(hipMemcpyAsync(h_carry_, h_carry_saved_, bytes, hipMemcpyDeviceToDevice,
                       stream_),
        "restore mtp carry");
}

}  // namespace gufo::models::ornith15::rocm
