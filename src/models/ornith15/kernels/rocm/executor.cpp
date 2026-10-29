#include "src/models/ornith15/kernels/rocm/executor.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

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
         header.version == Session::kSnapshotPayloadVersion &&
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
  bytes += std::size_t(header.vocab) * sizeof(float);
  return bytes;
}

SnapshotHeader MakeHeader(const Config& c, std::uint32_t position,
                          std::uint32_t max_context) {
  SnapshotHeader header{};
  std::memcpy(header.magic, kSnapshotMagic, 4);
  header.version = Session::kSnapshotPayloadVersion;
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
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
    qfn_mmq_release_stream(stream_);
  }
  if (copy_stream_ != nullptr)
    (void)hipStreamSynchronize(copy_stream_);
  if (graph_exec_ != nullptr)
    (void)hipGraphExecDestroy(graph_exec_);
  for (void* p : allocations_)
    (void)hipFree(p);
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
    Check(hipStreamSynchronize(stream_), "reset");
    position_ = 0;
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
  if (n == 1) {
    Dense(w, x, out);
    return;
  }
  if (w.type == GgmlType::kF32) {
    // Groups of eight stay on the width that keeps each token's FMA order.
    std::uint32_t done = 0;
    while (done < n) {
      const std::uint32_t rows = std::min<std::uint32_t>(8, n - done);
      fn::SmallGemm(
          w.data, fn::WeightType::kF32, x + std::size_t(done) * w.cols,
          out + std::size_t(done) * w.rows, rows, w.rows, w.cols, stream_);
      done += rows;
    }
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

// Routed prefill only. Width 48 beat the default 80-column tile at 20,
// 261 and 2048 tokens on gfx1151. The column grid is capped at the
// token count passed in: gate and up pass the chunk, so tiles past that
// are not launched; down already passes one row per selected slot, and
// the cap does not shrink it. Cleared on return. Dense projections and
// decode keep the default selector.
struct RoutedTileHint {
  bool active = false;
  RoutedTileHint(int tokens, int experts, int used) {
    if (experts <= 1 || tokens <= 0 || used <= 0)
      return;
    qfn_mmq_set_routed_max_expert_rows(tokens);
    qfn_mmq_set_routed_tile_cols(48);
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
                     std::uint32_t n_tokens, std::uint32_t used) {
  const int m = static_cast<int>(w.rows);
  const int k = static_cast<int>(w.cols);
  const int n = static_cast<int>(n_tokens);
  const int experts = static_cast<int>(w.experts);
  const int slots = static_cast<int>(used);
  const RoutedTileHint tile(n, experts, slots);
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
  if (n == 1) {
    Dense(l.ssm_alpha, xn_, ab_);
    Dense(l.ssm_beta, xn_, ab_ + c.ssm_num_v_heads);
  } else {
    Project(l.ssm_alpha, xn_, alpha_, n);
    Project(l.ssm_beta, xn_, beta_, n);
    PackAlphaBeta(alpha_, beta_, ab_, n, c.ssm_num_v_heads, stream_);
  }
  (void)layer;
  auto& state = state_[layer];
  fn::GatedDeltaNet(
      qkv_, c.SsmConvChannels(), z_, c.SsmValueDim(), ab_, l.ssm_conv1d.f32(),
      l.ssm_a.f32(), l.ssm_dt.f32(), l.ssm_norm.f32(), state.conv, conv_, qn_,
      kn_, raw_, state.recurrent, gdn_out_, nullptr, {}, {}, n,
      c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim, c.ssm_conv_kernel,
      false, false, c.rms_eps, stream_, nullptr, {}, 0, nullptr,
      fn::GdnGate::kSilu, fn::GdnL2Norm::kClampEps);
  Project(l.ssm_out, gdn_out_, block_out_, n);
}

void Session::Attention(const DeviceLayer& l, std::size_t layer,
                        std::uint32_t n) {
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
  auto& state = state_[layer];
  fn::StoreKv(k_, state.k, n, c.AttentionKvDim(), device_pos_, stream_);
  fn::StoreKv(v_, state.v, n, c.AttentionKvDim(), device_pos_, stream_);
  fn::Attention(q_, state.k, state.v, nullptr, 0, ctx_, nullptr, 1, n,
                device_pos_, c.num_heads, c.num_kv_heads, c.head_dim, 1,
                stream_);
  fn::SigmoidMul(ctx_, gate_, std::size_t(n) * c.AttentionQDim(), stream_);
  if (n == 1) {
    // Keep this input in F32: quantizing both V and the gated context can
    // amplify activation rounding through the attention value path. This dot
    // reads the original packed weights once, without a residual pass.
    DenseMatVecF32(l.attn_out.type, l.attn_out.data, ctx_, block_out_,
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
  if (n == 1) {
    Quantized(l.ffn_gate_exps, xn_, ids_, expert_gate_, 1, c.num_experts_used);
    Quantized(l.ffn_up_exps, xn_, ids_, expert_up_, 1, c.num_experts_used);
  } else {
    MoeRaw(l.ffn_gate_exps, xn_, ids_, expert_gate_, n, c.num_experts_used);
    MoeRaw(l.ffn_up_exps, xn_, ids_, expert_up_, n, c.num_experts_used);
  }
  fn::Swiglu(expert_gate_, expert_up_,
             std::size_t(n) * c.num_experts_used * c.expert_ff, stream_);
  if (n == 1) {
    Quantized(l.ffn_down_exps, expert_gate_, ids_, expert_down_,
              c.num_experts_used, 1);
  } else {
    MoeRaw(l.ffn_down_exps, expert_gate_, ids_, expert_down_,
           n * c.num_experts_used, 1);
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
      Attention(l, i, n);
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
  try {
    std::uint32_t done = 0;
    while (done < n) {
      std::uint32_t count = std::min(chunk_tokens, n - done);
      if (canonical_grid_ != 0) {
        const std::uint32_t room =
            canonical_grid_ - (position_ % canonical_grid_);
        count = std::min(count, room);
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
  if (model_ == nullptr)
    return 0;
  return PayloadBytes(MakeHeader(model_->config(), position_, max_context_));
}

void Session::TransferPayload(std::uint8_t* base, bool into_base,
                              hipMemcpyKind kind, hipStream_t stream,
                              bool skip_kv, std::uint32_t position) const {
  // Layout is the version-1 payload. Callers have already checked that the
  // header matches this session, so the live config describes `base`.
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
  const std::size_t logit_bytes = std::size_t(c.vocab_size) * sizeof(float);
  copy(logits_, logit_bytes, into_base ? "snapshot logits" : "restore logits");
  offset += logit_bytes;
  if (offset != PayloadBytes(MakeHeader(c, position, max_context_)))
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
  const SnapshotHeader header = MakeHeader(c, position_, max_context_);
  if (!HeaderMatches(header, c, max_context_)) {
    if (error != nullptr)
      *error = "session configuration cannot be snapshotted";
    return false;
  }
  std::vector<std::uint8_t> bytes(PayloadBytes(header));
  std::memcpy(bytes.data(), &header, sizeof(header));
  try {
    Check(hipStreamSynchronize(stream_), "snapshot drain");
    TransferPayload(bytes.data(), true, hipMemcpyDeviceToHost, stream_, false,
                    position_);
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
  const SnapshotHeader header = MakeHeader(c, position_, max_context_);
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
    TransferPayload(static_cast<std::uint8_t*>(staged->device), true,
                    hipMemcpyDeviceToDevice, stream_, false, position_);
    Check(hipEventRecord(staged->ready, stream_), "snapshot event record");
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = e.what();
    return nullptr;
  }
  return staged;
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
                    hipMemcpyDeviceToDevice, stream_, skip_kv, staged.position);
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
                    hipMemcpyHostToDevice, stream_, false, header.position);
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
      header.version != kSnapshotPayloadVersion ||
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

}  // namespace gufo::models::ornith15::rocm
