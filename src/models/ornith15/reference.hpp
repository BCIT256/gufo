#ifndef GUFO_MODELS_ORNITH15_REFERENCE_HPP_
#define GUFO_MODELS_ORNITH15_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/ornith15/trace.hpp"
#include "src/models/ornith15/weights.hpp"

namespace gufo::models::ornith15 {

/// Single-token, float32, scalar reference of the Ornith (qwen35moe) text
/// graph. It pins the semantics every execution route must reproduce:
/// pre-norm residual blocks, Gated DeltaNet with SiLU-gated output norm, gated
/// GQA, and softmax top-k MoE with the sigmoid-gated shared expert. Weights
/// are decoded on the fly; activations stay float32. Speed is irrelevant.
class ReferenceModel {
public:
  ReferenceModel(const ModelWeights& weights, std::uint32_t max_context);

  /// Runs one token at the next position; `logits` (vocab floats) receives
  /// the output distribution when non-empty.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::string* error_msg = nullptr,
                          LayerTrace* trace = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  struct LinearState {
    std::vector<float> conv;   ///< [kernel-1][channels], oldest first
    std::vector<float> state;  ///< [v_heads][head_dim(v)][head_dim(k)]
  };
  struct AttentionState {
    std::vector<float> k;  ///< [pos][kv_heads][head_dim], rotated
    std::vector<float> v;  ///< [pos][kv_heads][head_dim]
  };

  void LinearAttention(const LayerWeights& l, LinearState& s,
                       std::span<const float> x, std::span<float> out);
  void Attention(const LayerWeights& l, AttentionState& s,
                 std::span<const float> x, std::uint32_t pos,
                 std::span<float> out);
  void Moe(const LayerWeights& l, std::span<const float> x,
           std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
};

}  // namespace gufo::models::ornith15

#endif  // GUFO_MODELS_ORNITH15_REFERENCE_HPP_
