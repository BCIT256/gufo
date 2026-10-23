#include "src/models/ornith15/kernels/rocm/kernels.hpp"

#include <hip/hip_fp16.h>

namespace gufo::models::ornith15::rocm {
namespace {

struct Q8Block {
  __half2 ds;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8Block) == 36);

__global__ void Q8ResidualKernel(const float* x, const Q8Block* q8,
                                 float* residual, std::uint32_t count,
                                 std::uint32_t cols,
                                 std::uint32_t padded_cols) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const auto j = (i / cols) * padded_cols + i % cols;
    const auto& b = q8[j / 32];
    residual[i] = x[i] - __half2float(__low2half(b.ds)) *
                             static_cast<float>(b.qs[j % 32]);
  }
}

__global__ void EmbedQ6Kernel(const unsigned char* table, std::uint32_t token,
                              float* out, std::uint32_t hidden) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden)
    return;
  const auto* p = table + (std::size_t(token) * hidden / 256 + i / 256) * 210;
  const auto index = i % 256;
  const auto half = index / 128, segment = index % 128 / 32, lane = index % 32;
  const auto low = p[half * 64 + segment % 2 * 32 + lane];
  const int code = (((low >> (segment / 2 * 4)) & 15) |
                    (((p[128 + half * 32 + lane] >> (2 * segment)) & 3) << 4)) -
                   32;
  const auto scale = reinterpret_cast<const std::int8_t*>(p + 192)[index / 16];
  const float d = __half2float(*reinterpret_cast<const __half*>(p + 208));
  out[i] = (d * static_cast<float>(scale)) * static_cast<float>(code);
}

}  // namespace

void EmbedQ6(const void* table, std::uint32_t token, float* out,
             std::uint32_t hidden, hipStream_t stream) {
  EmbedQ6Kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
      static_cast<const unsigned char*>(table), token, out, hidden);
}

void Q8Residual(const float* x, const void* q8, float* residual,
                std::uint32_t rows, std::uint32_t cols,
                std::uint32_t padded_cols, hipStream_t stream) {
  const auto count = rows * cols;
  Q8ResidualKernel<<<(count + 255) / 256, 256, 0, stream>>>(
      x, static_cast<const Q8Block*>(q8), residual, count, cols, padded_cols);
}

}  // namespace gufo::models::ornith15::rocm
