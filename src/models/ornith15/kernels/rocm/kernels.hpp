#ifndef GUFO_MODELS_ORNITH15_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_ORNITH15_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>

namespace gufo::models::ornith15::rocm {

/// Gather a single Q6_K embedding row.
void EmbedQ6(const void* table, std::uint32_t token, float* out,
             std::uint32_t hidden, hipStream_t stream);

/// Gather `n` Q6_K embedding rows. `tokens` is device memory.
void EmbedQ6Rows(const void* table, const std::int32_t* tokens, float* out,
                 std::uint32_t n, std::uint32_t hidden, hipStream_t stream);

/// Pack per-token alpha and beta rows into [t][alpha | beta].
void PackAlphaBeta(const float* alpha, const float* beta, float* out,
                   std::uint32_t n, std::uint32_t v_heads, hipStream_t stream);

/// Dense Q6_K/Q8_0 dot retaining F32 activations. One weight pass, no
/// activation quantization or persistent unpacked-weight copy. Only the
/// numerically sensitive attention output projection uses this route.
void DenseMatVecF32(core::GgmlType type, const void* weights, const float* x,
                    float* out, std::uint32_t rows, std::uint32_t cols,
                    hipStream_t stream);

/// `dst[0] = carry`, `dst[t] = rows[t - 1]` for `t > 0`. Rows are `width`
/// floats. Used to pair each MTP token with the previous trunk hidden.
void ShiftHidden(const float* rows, const float* carry, float* dst,
                 std::uint32_t n, std::uint32_t width, hipStream_t stream);

/// `out[t] = concat(a[t], b[t])` along the hidden axis.
void ConcatRows(const float* a, const float* b, float* out, std::uint32_t n,
                std::uint32_t width, hipStream_t stream);

}  // namespace gufo::models::ornith15::rocm

#endif
