#ifndef GUFO_MODELS_ORNITH15_CPU_OPS_HPP_
#define GUFO_MODELS_ORNITH15_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/ornith15/weights.hpp"

/// Scalar CPU operators for the reference model: one row at a time, every
/// format decoded on the fly. Correctness oracle only; nothing here is tuned.
namespace gufo::models::ornith15::cpu {

/// Decodes one row of `t` (expert `e`) into `out` (t.cols floats).
void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out);

/// y[r] = sum_c W[e][r][c] * x[c] for every row of the matrix, accumulated
/// in double precision.
void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y);

/// RMSNorm over `x` in place with gamma `w`.
void RmsNorm(std::span<float> x, const float* w, float eps);
/// x / max(||x||, eps), ggml's l2_norm used for DeltaNet queries and keys.
void L2Norm(std::span<float> x, float eps);

[[nodiscard]] float Sigmoid(float x) noexcept;
[[nodiscard]] float Silu(float x) noexcept;
[[nodiscard]] float Softplus(float x) noexcept;

/// NEOX-style partial rotary embedding on the first `rotary_dim` elements of
/// each `head_dim` head: pair (i, i + rotary_dim/2) rotates by pos * theta_i.
/// Text-only IMRoPE gives every section the same position, so it is exactly
/// this rotation.
void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta);

}  // namespace gufo::models::ornith15::cpu

#endif  // GUFO_MODELS_ORNITH15_CPU_OPS_HPP_
