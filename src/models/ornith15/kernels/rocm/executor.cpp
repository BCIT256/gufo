#include "src/models/ornith15/kernels/rocm/executor.hpp"

#include <algorithm>
#include <cstdio>
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

}  // namespace

Session::~Session() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
    qfn_mmq_release_stream(stream_);
  }
  if (graph_exec_ != nullptr)
    (void)hipGraphExecDestroy(graph_exec_);
  for (void* p : allocations_)
    (void)hipFree(p);
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
    Check(hipStreamCreateWithFlags(&s->stream_, hipStreamNonBlocking),
          "stream");
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

void Session::MoeRaw(const DeviceTensor& w, const float* x,
                     const std::int32_t* ids, float* out,
                     std::uint32_t n_tokens, std::uint32_t used) {
  const int m = static_cast<int>(w.rows);
  const int k = static_cast<int>(w.cols);
  const int n = static_cast<int>(n_tokens);
  const int experts = static_cast<int>(w.experts);
  const int slots = static_cast<int>(used);
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
    return true;
  } catch (const std::exception& e) {
    (void)hipStreamSynchronize(stream_);
    failed_ = true;
    if (error != nullptr)
      *error = e.what();
    return false;
  }
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
      static_cast<std::uint64_t>(position_) + n <= max_context_;
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
          "invalid prefill (failed session, token, context, chunk or logits "
          "size)";
    return false;
  }
  try {
    std::uint32_t done = 0;
    while (done < n) {
      const std::uint32_t count = std::min(chunk_tokens, n - done);
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
      Check(hipGetLastError(), "prefill launch");
      Check(hipStreamSynchronize(stream_), "prefill");
      if (want_all && last) {
        const float* row =
            all_logits.data() + std::size_t(n - 1) * c.vocab_size;
        std::copy(row, row + c.vocab_size, last_logits.begin());
      }
      position_ += count;
      done += count;
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

}  // namespace gufo::models::ornith15::rocm
