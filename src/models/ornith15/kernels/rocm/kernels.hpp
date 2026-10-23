#ifndef GUFO_MODELS_ORNITH15_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_ORNITH15_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::ornith15::rocm {

/// Single Q6_K embedding row; the remaining operators come from Flash-Next.
void EmbedQ6(const void* table, std::uint32_t token, float* out,
             std::uint32_t hidden, hipStream_t stream);

/// Residual of the original F32 activations after MMQ's actual Q8_1
/// reconstruction (including its rounded F16 scale). Used by a second MMVQ
/// pass to retain more activation precision without changing weight formats.
void Q8Residual(const float* x, const void* q8, float* residual,
                std::uint32_t rows, std::uint32_t cols,
                std::uint32_t padded_cols, hipStream_t stream);

}  // namespace gufo::models::ornith15::rocm

#endif
