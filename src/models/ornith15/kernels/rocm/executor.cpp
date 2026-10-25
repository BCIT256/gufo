#include "src/models/ornith15/kernels/rocm/executor.hpp"

#include <stdexcept>

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
  if (stream_) {
    (void)hipStreamSynchronize(stream_);
    qfn_mmq_release_stream(stream_);
  }
  for (void* p : allocations_)
    (void)hipFree(p);
  if (stream_)
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
    s->zero_id_ = static_cast<std::int32_t*>(s->Allocate(sizeof(std::int32_t)));
    s->device_pos_ =
        static_cast<std::uint32_t*>(s->Allocate(sizeof(std::uint32_t)));
    Check(hipMemset(s->zero_id_, 0, sizeof(std::int32_t)), "zero expert ID");
    s->res_ = s->Floats(c.hidden_size);
    s->xn_ = s->Floats(c.hidden_size);
    s->block_out_ = s->Floats(c.hidden_size);
    s->logits_ = s->Floats(c.vocab_size);
    s->qkv_ = s->Floats(c.SsmConvChannels());
    s->z_ = s->Floats(c.SsmValueDim());
    s->ab_ = s->Floats(2 * c.ssm_num_v_heads);
    s->conv_ = s->Floats((1 + c.ssm_conv_kernel) * c.SsmConvChannels());
    s->qn_ = s->Floats(c.SsmKeyDim());
    s->kn_ = s->Floats(c.SsmKeyDim());
    s->raw_ = s->Floats(c.SsmValueDim());
    s->gdn_out_ = s->Floats(c.SsmValueDim());
    s->qg_ = s->Floats(2 * c.AttentionQDim());
    s->q_ = s->Floats(c.AttentionQDim());
    s->gate_ = s->Floats(c.AttentionQDim());
    s->k_ = s->Floats(c.AttentionKvDim());
    s->v_ = s->Floats(c.AttentionKvDim());
    s->ctx_ = s->Floats(c.AttentionQDim());
    s->router_ = s->Floats(c.num_experts);
    s->route_weights_ = s->Floats(c.num_experts_used);
    s->ids_ = static_cast<std::int32_t*>(
        s->Allocate(c.num_experts_used * sizeof(std::int32_t)));
    s->expert_gate_ = s->Floats(c.num_experts_used * c.expert_ff);
    s->expert_up_ = s->Floats(c.num_experts_used * c.expert_ff);
    s->expert_down_ = s->Floats(c.num_experts_used * c.hidden_size);
    s->shared_gate_ = s->Floats(c.shared_expert_ff);
    s->shared_up_ = s->Floats(c.shared_expert_ff);
    s->shared_out_ = s->Floats(c.hidden_size);
    s->shared_inp_ = s->Floats(1);
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
    if (error)
      *error = e.what();
    return nullptr;
  }
}

bool Session::Reset(std::string* error) {
  try {
    Check(hipStreamSynchronize(stream_), "reset drain");
    const auto& c = model_->config();
    for (auto& s : state_) {
      if (s.conv) {
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
    if (error)
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

void Session::Quantized(const DeviceTensor& w, const float* x,
                        const std::int32_t* ids, float* out, std::uint32_t rows,
                        std::uint32_t used) {
  if (qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, x, ids, out, w.rows,
                      w.cols, rows, w.experts, used, stream_))
    throw std::runtime_error("quantized projection failed");
}

void Session::LinearAttention(const DeviceLayer& l, std::size_t layer,
                               Layer0Probe* probe) {
  const auto& c = model_->config();
  Dense(l.ssm_qkv, xn_, qkv_);
  Dense(l.ssm_gate, xn_, z_);
  Dense(l.ssm_alpha, xn_, ab_);
  Dense(l.ssm_beta, xn_, ab_ + c.ssm_num_v_heads);
  const bool layer0 = probe && layer == 0;
  if (layer0 && (probe->norm_out || probe->qkv_before || probe->qkv_out ||
                 probe->edit_qkv)) {
    if (probe->norm_out)
      Check(hipMemcpyAsync(probe->norm_out, xn_, c.hidden_size * sizeof(float),
                           hipMemcpyDeviceToHost, stream_),
            "layer0 norm");
    if (probe->qkv_before)
      Check(hipMemcpyAsync(probe->qkv_before, qkv_,
                           c.SsmConvChannels() * sizeof(float),
                           hipMemcpyDeviceToHost, stream_),
            "layer0 qkv before");
    if (probe->edit_qkv)
      probe->edit_qkv(probe->edit_ctx, qkv_, xn_, c.SsmConvChannels(),
                      c.hidden_size, stream_);
    if (probe->qkv_out)
      Check(hipMemcpyAsync(probe->qkv_out, qkv_,
                           c.SsmConvChannels() * sizeof(float),
                           hipMemcpyDeviceToHost, stream_),
            "layer0 qkv out");
  }
  auto& state = state_[layer];
  fn::GatedDeltaNet(
      qkv_, c.SsmConvChannels(), z_, c.SsmValueDim(), ab_, l.ssm_conv1d.f32(),
      l.ssm_a.f32(), l.ssm_dt.f32(), l.ssm_norm.f32(), state.conv, conv_, qn_,
      kn_, raw_, state.recurrent, gdn_out_, nullptr, {}, {}, 1,
      c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim, c.ssm_conv_kernel,
      false, false, c.rms_eps, stream_, nullptr, {}, 0, nullptr,
      fn::GdnGate::kSilu, fn::GdnL2Norm::kClampEps);
  if (layer0 && probe->conv_out)
    Check(hipMemcpyAsync(probe->conv_out, state.conv,
                         (c.ssm_conv_kernel - 1) * c.SsmConvChannels() *
                             sizeof(float),
                         hipMemcpyDeviceToHost, stream_),
          "layer0 conv");
  if (layer0 && probe->state_out)
    Check(hipMemcpyAsync(probe->state_out, state.recurrent,
                         std::size_t(c.ssm_num_v_heads) * c.ssm_head_dim *
                             c.ssm_head_dim * sizeof(float),
                         hipMemcpyDeviceToHost, stream_),
          "layer0 state");
  Dense(l.ssm_out, gdn_out_, block_out_);
}

void Session::Attention(const DeviceLayer& l, std::size_t layer) {
  const auto& c = model_->config();
  Dense(l.attn_q, xn_, qg_);
  Dense(l.attn_k, xn_, k_);
  Dense(l.attn_v, xn_, v_);
  fn::UnpackQGate(qg_, 2 * c.AttentionQDim(), q_, gate_, nullptr, nullptr, 1,
                  c.num_heads, c.head_dim, 0, stream_);
  fn::RmsNormRows(q_, l.attn_q_norm.f32(), q_, c.num_heads, c.head_dim, 1,
                  c.rms_eps, stream_);
  fn::RmsNormRows(k_, l.attn_k_norm.f32(), k_, c.num_kv_heads, c.head_dim, 1,
                  c.rms_eps, stream_);
  fn::Rope(q_, 1, c.num_heads, c.head_dim, c.rotary_dim, device_pos_,
           c.rope_theta, stream_);
  fn::Rope(k_, 1, c.num_kv_heads, c.head_dim, c.rotary_dim, device_pos_,
           c.rope_theta, stream_);
  auto& state = state_[layer];
  fn::StoreKv(k_, state.k, 1, c.AttentionKvDim(), device_pos_, stream_);
  fn::StoreKv(v_, state.v, 1, c.AttentionKvDim(), device_pos_, stream_);
  fn::Attention(q_, state.k, state.v, nullptr, 0, ctx_, nullptr, 1, 1,
                device_pos_, c.num_heads, c.num_kv_heads, c.head_dim, 1,
                stream_);
  fn::SigmoidMul(ctx_, gate_, c.AttentionQDim(), stream_);
  // Keep this input in F32: quantizing both V and the gated context can
  // amplify activation rounding through the attention value path. This dot
  // reads the original packed weights once, without a residual pass.
  DenseMatVecF32(l.attn_out.type, l.attn_out.data, ctx_, block_out_,
                 l.attn_out.rows, l.attn_out.cols, stream_);
}

void Session::Moe(const DeviceLayer& l, std::size_t layer, Layer0Probe* probe) {
  const auto& c = model_->config();
  Dense(l.router, xn_, router_);
  fn::RouterTopK(router_, c.num_experts, ids_, route_weights_, 1, c.num_experts,
                 c.num_experts_used, stream_);
  if (probe && probe->route_ids && probe->route_weights) {
    const auto k = c.num_experts_used;
    Check(hipMemcpyAsync(probe->route_ids + layer * k, ids_,
                         k * sizeof(std::int32_t), hipMemcpyDeviceToHost,
                         stream_),
          "route ids");
    Check(hipMemcpyAsync(probe->route_weights + layer * k, route_weights_,
                         k * sizeof(float), hipMemcpyDeviceToHost, stream_),
          "route weights");
  }
  // Dispatch each tensor's format independently; no model-wide expert type.
  Quantized(l.ffn_gate_exps, xn_, ids_, expert_gate_, 1, c.num_experts_used);
  Quantized(l.ffn_up_exps, xn_, ids_, expert_up_, 1, c.num_experts_used);
  fn::Swiglu(expert_gate_, expert_up_, c.num_experts_used * c.expert_ff,
             stream_);
  Quantized(l.ffn_down_exps, expert_gate_, ids_, expert_down_,
            c.num_experts_used, 1);
  Dense(l.shexp_gate, xn_, shared_gate_);
  Dense(l.shexp_up, xn_, shared_up_);
  fn::Swiglu(shared_gate_, shared_up_, c.shared_expert_ff, stream_);
  Dense(l.shexp_down, shared_gate_, shared_out_);
  Dense(l.shexp_gate_inp, xn_, shared_inp_);
  if (probe && probe->shared_gate)
    Check(hipMemcpyAsync(probe->shared_gate + layer, shared_inp_, sizeof(float),
                         hipMemcpyDeviceToHost, stream_),
          "shared gate");
  fn::MoeEpilogue(expert_down_, route_weights_, shared_out_, shared_inp_, 1,
                  block_out_, 1, c.num_experts_used, c.hidden_size, stream_);
}

bool Session::Step(std::int32_t token, std::span<float> logits,
                   std::string* error, LayerTrace* trace, Layer0Probe* probe) {
  const auto& c = model_->config();
  // Reject invalid calls before any session state or caller output changes.
  if (failed_ || token < 0 ||
      static_cast<std::uint32_t>(token) >= c.vocab_size ||
      position_ >= max_context_ || logits.size() != c.vocab_size) {
    if (error)
      *error =
          "invalid AR step (failed session, token, context or logits size)";
    return false;
  }
  try {
    Check(hipMemcpyAsync(device_pos_, &position_, sizeof(position_),
                         hipMemcpyHostToDevice, stream_),
          "position upload");
    if (model_->token_embd().type == GgmlType::kQ6_K) {
      EmbedQ6(model_->token_embd().data, token, res_, c.hidden_size, stream_);
    } else {
      Check(hipMemcpyAsync(zero_id_, &token, sizeof(token),
                           hipMemcpyHostToDevice, stream_),
            "token upload");
      fn::EmbedTokens(model_->token_embd().data, fn::WeightType::kQ8_0,
                      zero_id_, res_, 1, c.hidden_size, 1, stream_);
      Check(hipMemsetAsync(zero_id_, 0, sizeof(token), stream_),
            "restore zero expert ID");
    }
    if (trace)
      trace->assign(c.num_layers, std::vector<float>(c.hidden_size));
    for (std::size_t i = 0; i < c.num_layers; ++i) {
      const auto& l = model_->layers()[i];
      fn::RmsNormRows(res_, l.attn_norm.f32(), xn_, 1, c.hidden_size, 1,
                      c.rms_eps, stream_);
      if (l.linear)
        LinearAttention(l, i, probe);
      else
        Attention(l, i);
      fn::MtpAddEmbedding(block_out_, res_, 1, c.hidden_size, 1, stream_);
      fn::RmsNormRows(res_, l.post_attention_norm.f32(), xn_, 1, c.hidden_size,
                      1, c.rms_eps, stream_);
      Moe(l, i, probe);
      fn::MtpAddEmbedding(block_out_, res_, 1, c.hidden_size, 1, stream_);
      if (trace)
        Check(hipMemcpyAsync((*trace)[i].data(), res_,
                             c.hidden_size * sizeof(float),
                             hipMemcpyDeviceToHost, stream_),
              "layer trace");
    }
    fn::RmsNormRows(res_, model_->output_norm().f32(), xn_, 1, c.hidden_size, 1,
                    c.rms_eps, stream_);
    Dense(model_->output(), xn_, logits_);
    Check(hipGetLastError(), "AR launch");
    Check(hipMemcpyAsync(logits.data(), logits_, logits.size_bytes(),
                         hipMemcpyDeviceToHost, stream_),
          "logits download");
    Check(hipStreamSynchronize(stream_), "AR step");
    ++position_;
    return true;
  } catch (const std::exception& e) {
    (void)hipStreamSynchronize(stream_);
    failed_ = true;
    if (error)
      *error = e.what();
    return false;
  }
}

}  // namespace gufo::models::ornith15::rocm
