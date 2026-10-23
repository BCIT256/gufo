#include "src/models/ornith15/reference.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "src/models/ornith15/cpu_ops.hpp"

namespace gufo::models::ornith15 {

using cpu::Sigmoid;
using cpu::Silu;

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               std::uint32_t max_context)
    : w_(weights), c_(weights.config), max_context_(max_context) {
  linear_.resize(c_.num_layers);
  attention_.resize(c_.num_layers);
  Reset();
}

void ReferenceModel::Reset() {
  position_ = 0;
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (c_.IsLinearLayer(il)) {
      auto& s = linear_[il];
      s.conv.assign(static_cast<std::size_t>(c_.ssm_conv_kernel - 1) *
                        c_.SsmConvChannels(),
                    0.0F);
      s.state.assign(static_cast<std::size_t>(c_.ssm_num_v_heads) *
                         c_.ssm_head_dim * c_.ssm_head_dim,
                     0.0F);
    } else {
      auto& s = attention_[il];
      s.k.clear();
      s.v.clear();
    }
  }
}

void ReferenceModel::LinearAttention(const LayerWeights& l, LinearState& s,
                                     std::span<const float> x,
                                     std::span<float> out) {
  const std::uint32_t channels = c_.SsmConvChannels();
  const std::uint32_t key_dim = c_.SsmKeyDim();
  const std::uint32_t value_dim = c_.SsmValueDim();
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;

  std::vector<float> qkv(channels);
  std::vector<float> z(value_dim);
  std::vector<float> alpha(c_.ssm_num_v_heads);
  std::vector<float> beta(c_.ssm_num_v_heads);
  cpu::MatVec(l.ssm_qkv, 0, x, qkv);
  cpu::MatVec(l.ssm_gate, 0, x, z);
  cpu::MatVec(l.ssm_alpha, 0, x, alpha);
  cpu::MatVec(l.ssm_beta, 0, x, beta);

  // Causal depthwise conv over the last `kern` projections, then SiLU.
  const auto* conv_w = static_cast<const float*>(l.ssm_conv1d.data);
  std::vector<float> conv(channels);
  for (std::uint32_t ch = 0; ch < channels; ++ch) {
    float acc =
        conv_w[static_cast<std::size_t>(ch) * kern + kern - 1] * qkv[ch];
    for (std::uint32_t k = 0; k + 1 < kern; ++k) {
      acc += conv_w[static_cast<std::size_t>(ch) * kern + k] *
             s.conv[static_cast<std::size_t>(k) * channels + ch];
    }
    conv[ch] = Silu(acc);
  }
  if (kern > 1) {
    std::copy(s.conv.begin() + channels, s.conv.end(), s.conv.begin());
    std::copy(qkv.begin(), qkv.end(), s.conv.end() - channels);
  }

  float* q = conv.data();
  float* k = conv.data() + key_dim;
  float* v = conv.data() + 2 * key_dim;
  for (std::uint32_t h = 0; h < c_.ssm_num_k_heads; ++h) {
    cpu::L2Norm(std::span<float>(q + static_cast<std::size_t>(h) * d, d),
                c_.rms_eps);
    cpu::L2Norm(std::span<float>(k + static_cast<std::size_t>(h) * d, d),
                c_.rms_eps);
  }

  const auto* a = static_cast<const float*>(l.ssm_a.data);
  const auto* dt = static_cast<const float*>(l.ssm_dt.data);
  const auto* norm_w = static_cast<const float*>(l.ssm_norm.data);
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(d));
  std::vector<float> attn(value_dim);
  std::vector<float> u(d);
  for (std::uint32_t h = 0; h < c_.ssm_num_v_heads; ++h) {
    // Value head h reads key head h % n_k_heads, the tiled layout the GGUF
    // converter permuted the value-side tensors into.
    const float* qh = q + static_cast<std::size_t>(h % c_.ssm_num_k_heads) * d;
    const float* kh = k + static_cast<std::size_t>(h % c_.ssm_num_k_heads) * d;
    const float* vh = v + static_cast<std::size_t>(h) * d;
    float* S = s.state.data() + static_cast<std::size_t>(h) * d * d;
    // ssm_a holds -exp(A_log), so the decay is below one.
    const float decay = std::exp(cpu::Softplus(alpha[h] + dt[h]) * a[h]);
    const float b = Sigmoid(beta[h]);
    // S[j][i]: j over value dim, i over key dim.
    for (std::uint32_t j = 0; j < d; ++j) {
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] *= decay;
        acc += static_cast<double>(row[i]) * kh[i];
      }
      u[j] = static_cast<float>(acc);
    }
    float* o = attn.data() + static_cast<std::size_t>(h) * d;
    for (std::uint32_t j = 0; j < d; ++j) {
      const float delta = (vh[j] - u[j]) * b;
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] += delta * kh[i];
        acc += static_cast<double>(row[i]) * qh[i];
      }
      o[j] = static_cast<float>(acc) * q_scale;
    }
    // Per-head RMSNorm, then the SiLU output gate of Qwen3.5.
    cpu::RmsNorm(std::span<float>(o, d), norm_w, c_.rms_eps);
    for (std::uint32_t j = 0; j < d; ++j) {
      o[j] *= Silu(z[static_cast<std::size_t>(h) * d + j]);
    }
  }
  cpu::MatVec(l.ssm_out, 0, attn, out);
}

void ReferenceModel::Attention(const LayerWeights& l, AttentionState& s,
                               std::span<const float> x, std::uint32_t pos,
                               std::span<float> out) {
  const std::uint32_t hd = c_.head_dim;
  const std::uint32_t nh = c_.num_heads;
  const std::uint32_t nkv = c_.num_kv_heads;
  const std::uint32_t group = nh / nkv;

  std::vector<float> qg(2 * c_.AttentionQDim());
  std::vector<float> k(c_.AttentionKvDim());
  std::vector<float> v(c_.AttentionKvDim());
  cpu::MatVec(l.attn_q, 0, x, qg);
  cpu::MatVec(l.attn_k, 0, x, k);
  cpu::MatVec(l.attn_v, 0, x, v);

  // wq interleaves [q | gate] per head.
  std::vector<float> q(c_.AttentionQDim());
  std::vector<float> gate(c_.AttentionQDim());
  for (std::uint32_t h = 0; h < nh; ++h) {
    std::copy_n(qg.begin() + static_cast<std::size_t>(h) * 2 * hd, hd,
                q.begin() + static_cast<std::size_t>(h) * hd);
    std::copy_n(qg.begin() + static_cast<std::size_t>(h) * 2 * hd + hd, hd,
                gate.begin() + static_cast<std::size_t>(h) * hd);
    cpu::RmsNorm(
        std::span<float>(q.data() + static_cast<std::size_t>(h) * hd, hd),
        static_cast<const float*>(l.attn_q_norm.data), c_.rms_eps);
  }
  for (std::uint32_t h = 0; h < nkv; ++h) {
    cpu::RmsNorm(
        std::span<float>(k.data() + static_cast<std::size_t>(h) * hd, hd),
        static_cast<const float*>(l.attn_k_norm.data), c_.rms_eps);
  }
  cpu::Rope(q.data(), nh, hd, c_.rotary_dim, pos, c_.rope_theta);
  cpu::Rope(k.data(), nkv, hd, c_.rotary_dim, pos, c_.rope_theta);
  s.k.insert(s.k.end(), k.begin(), k.end());
  s.v.insert(s.v.end(), v.begin(), v.end());

  // Dense causal attention over every cached position.
  const std::uint32_t n_kv = pos + 1;
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  std::vector<float> ctx(c_.AttentionQDim(), 0.0F);
  std::vector<float> scores(n_kv);
  for (std::uint32_t h = 0; h < nh; ++h) {
    const std::uint32_t kvh = h / group;
    const float* qh = q.data() + static_cast<std::size_t>(h) * hd;
    float max_score = -INFINITY;
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const float* kj =
          s.k.data() + (static_cast<std::size_t>(j) * nkv + kvh) * hd;
      double dot = 0.0;
      for (std::uint32_t i = 0; i < hd; ++i) {
        dot += static_cast<double>(qh[i]) * kj[i];
      }
      scores[j] = static_cast<float>(dot) * scale;
      max_score = std::max(max_score, scores[j]);
    }
    double denom = 0.0;
    for (float& sc : scores) {
      sc = std::exp(sc - max_score);
      denom += sc;
    }
    float* oh = ctx.data() + static_cast<std::size_t>(h) * hd;
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const float p = static_cast<float>(scores[j] / denom);
      const float* vj =
          s.v.data() + (static_cast<std::size_t>(j) * nkv + kvh) * hd;
      for (std::uint32_t i = 0; i < hd; ++i) {
        oh[i] += p * vj[i];
      }
    }
    for (std::uint32_t i = 0; i < hd; ++i) {
      oh[i] *= Sigmoid(gate[static_cast<std::size_t>(h) * hd + i]);
    }
  }
  cpu::MatVec(l.attn_out, 0, ctx, out);
}

void ReferenceModel::Moe(const LayerWeights& l, std::span<const float> x,
                         std::span<float> out) {
  std::vector<float> probs(c_.num_experts);
  cpu::MatVec(l.router, 0, x, probs);
  // Softmax over every expert, top-k of the probabilities, renormalized.
  const float max_logit = *std::max_element(probs.begin(), probs.end());
  double denom = 0.0;
  for (float& v : probs) {
    v = std::exp(v - max_logit);
    denom += v;
  }
  for (float& v : probs) {
    v = static_cast<float>(v / denom);
  }
  std::vector<std::uint32_t> order(c_.num_experts);
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(
      order.begin(), order.begin() + c_.num_experts_used, order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return probs[a] > probs[b]; });
  float sum = 0.0F;
  for (std::uint32_t i = 0; i < c_.num_experts_used; ++i) {
    sum += probs[order[i]];
  }
  sum = std::max(sum, 6.103515625e-5F);

  std::fill(out.begin(), out.end(), 0.0F);
  std::vector<float> gate(c_.expert_ff);
  std::vector<float> up(c_.expert_ff);
  std::vector<float> down(c_.hidden_size);
  for (std::uint32_t i = 0; i < c_.num_experts_used; ++i) {
    const std::uint32_t e = order[i];
    const float weight = probs[e] / sum;
    cpu::MatVec(l.ffn_gate_exps, e, x, gate);
    cpu::MatVec(l.ffn_up_exps, e, x, up);
    for (std::uint32_t j = 0; j < c_.expert_ff; ++j) {
      gate[j] = Silu(gate[j]) * up[j];
    }
    cpu::MatVec(l.ffn_down_exps, e, gate, down);
    for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
      out[j] += weight * down[j];
    }
  }

  // Shared expert with its own scalar sigmoid gate.
  std::vector<float> sgate(c_.shared_expert_ff);
  std::vector<float> sup(c_.shared_expert_ff);
  cpu::MatVec(l.shexp_gate, 0, x, sgate);
  cpu::MatVec(l.shexp_up, 0, x, sup);
  for (std::uint32_t j = 0; j < c_.shared_expert_ff; ++j) {
    sgate[j] = Silu(sgate[j]) * sup[j];
  }
  cpu::MatVec(l.shexp_down, 0, sgate, down);
  float shared_gate = 0.0F;
  cpu::MatVec(l.shexp_gate_inp, 0, x, std::span<float>(&shared_gate, 1));
  const float sg = Sigmoid(shared_gate);
  for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
    out[j] += sg * down[j];
  }
}

bool ReferenceModel::Step(std::int32_t token, std::span<float> logits,
                          std::string* error_msg) {
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "reference context is full";
    }
    return false;
  }
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token out of range";
    }
    return false;
  }
  if (!logits.empty() && logits.size() != c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "logits span must hold the vocabulary";
    }
    return false;
  }
  const std::uint32_t hidden = c_.hidden_size;

  std::vector<float> res(hidden);
  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     res.data());

  std::vector<float> normed(hidden);
  std::vector<float> block_out(hidden);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const LayerWeights& l = w_.layers[il];
    normed = res;
    cpu::RmsNorm(normed, static_cast<const float*>(l.attn_norm.data),
                 c_.rms_eps);
    if (l.linear) {
      LinearAttention(l, linear_[il], normed, block_out);
    } else {
      Attention(l, attention_[il], normed, position_, block_out);
    }
    for (std::uint32_t i = 0; i < hidden; ++i) {
      res[i] += block_out[i];
    }

    normed = res;
    cpu::RmsNorm(normed, static_cast<const float*>(l.post_attention_norm.data),
                 c_.rms_eps);
    Moe(l, normed, block_out);
    for (std::uint32_t i = 0; i < hidden; ++i) {
      res[i] += block_out[i];
    }
  }
  if (!logits.empty()) {
    cpu::RmsNorm(res, static_cast<const float*>(w_.output_norm.data),
                 c_.rms_eps);
    cpu::MatVec(w_.output, 0, res, logits);
  }
  ++position_;
  return true;
}

}  // namespace gufo::models::ornith15
