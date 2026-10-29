#include "src/models/ornith15/kernels/rocm/kernels.hpp"

#include <hip/hip_fp16.h>

#include <stdexcept>

namespace gufo::models::ornith15::rocm {
namespace {

// Each wave reduces one output row. Weights stay in their GGUF encoding;
// only the activation quantization boundary is removed.
template<core::GgmlType type>
__global__ void DenseF32Kernel(const unsigned char* weights, const float* x,
                               float* out, std::uint32_t rows,
                               std::uint32_t cols) {
  constexpr bool q6 = type == core::GgmlType::kQ6_K;
  constexpr int block = q6 ? 256 : 32;
  constexpr int bytes = q6 ? 210 : 34;
  const auto lane = threadIdx.x % 32;
  const auto row = blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows)
    return;
  const auto* w = weights + std::size_t(row) * (cols / block) * bytes;
  float sum = 0;
  for (std::uint32_t b = 0; b < cols / block; ++b) {
    const auto* p = w + b * bytes;
    const float d =
        __half2float(*reinterpret_cast<const __half*>(p + (q6 ? 208 : 0)));
    for (std::uint32_t j = lane; j < block; j += 32) {
      float weight;
      if constexpr (q6) {
        const auto half = j / 128, segment = j % 128 / 32;
        const auto low = p[half * 64 + segment % 2 * 32 + lane];
        const int code =
            (((low >> (segment / 2 * 4)) & 15) |
             (((p[128 + half * 32 + lane] >> (2 * segment)) & 3) << 4)) -
            32;
        const auto scale =
            reinterpret_cast<const std::int8_t*>(p + 192)[j / 16];
        weight = (d * static_cast<float>(scale)) * static_cast<float>(code);
      } else {
        weight = d * static_cast<float>(
                         reinterpret_cast<const std::int8_t*>(p + 2)[j]);
      }
      sum += weight * x[b * block + j];
    }
  }
  for (int delta = 16; delta; delta /= 2)
    sum += __shfl_down(sum, delta, 32);
  if (lane == 0)
    out[row] = sum;
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

__global__ void EmbedQ6RowsKernel(const unsigned char* table,
                                  const std::int32_t* tokens, float* out,
                                  std::uint32_t n, std::uint32_t hidden) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = std::size_t(n) * hidden;
  if (i >= total)
    return;
  const std::uint32_t t = i / hidden;
  const std::uint32_t elem = i - t * hidden;
  const auto token = static_cast<std::uint32_t>(tokens[t]);
  const auto* p =
      table + (std::size_t(token) * hidden / 256 + elem / 256) * 210;
  const auto index = elem % 256;
  const auto half = index / 128, segment = index % 128 / 32, lane = index % 32;
  const auto low = p[half * 64 + segment % 2 * 32 + lane];
  const int code = (((low >> (segment / 2 * 4)) & 15) |
                    (((p[128 + half * 32 + lane] >> (2 * segment)) & 3) << 4)) -
                   32;
  const auto scale = reinterpret_cast<const std::int8_t*>(p + 192)[index / 16];
  const float d = __half2float(*reinterpret_cast<const __half*>(p + 208));
  out[i] = (d * static_cast<float>(scale)) * static_cast<float>(code);
}

__global__ void PackAlphaBetaKernel(const float* alpha, const float* beta,
                                    float* out, std::uint32_t n,
                                    std::uint32_t v) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * v)
    return;
  const std::uint32_t t = i / v;
  const std::uint32_t h = i - t * v;
  out[t * (2 * v) + h] = alpha[i];
  out[t * (2 * v) + v + h] = beta[i];
}

}  // namespace

void EmbedQ6(const void* table, std::uint32_t token, float* out,
             std::uint32_t hidden, hipStream_t stream) {
  EmbedQ6Kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
      static_cast<const unsigned char*>(table), token, out, hidden);
}

void EmbedQ6Rows(const void* table, const std::int32_t* tokens, float* out,
                 std::uint32_t n, std::uint32_t hidden, hipStream_t stream) {
  const auto count = static_cast<std::uint32_t>(std::size_t(n) * hidden);
  if (count == 0)
    return;
  EmbedQ6RowsKernel<<<(count + 255) / 256, 256, 0, stream>>>(
      static_cast<const unsigned char*>(table), tokens, out, n, hidden);
}

void PackAlphaBeta(const float* alpha, const float* beta, float* out,
                   std::uint32_t n, std::uint32_t v_heads, hipStream_t stream) {
  const auto count = n * v_heads;
  if (count == 0)
    return;
  PackAlphaBetaKernel<<<(count + 255) / 256, 256, 0, stream>>>(alpha, beta, out,
                                                               n, v_heads);
}

void DenseMatVecF32(core::GgmlType type, const void* weights, const float* x,
                    float* out, std::uint32_t rows, std::uint32_t cols,
                    hipStream_t stream) {
  const auto* w = static_cast<const unsigned char*>(weights);
  switch (type) {
    case core::GgmlType::kQ6_K:
      DenseF32Kernel<core::GgmlType::kQ6_K>
          <<<(rows + 3) / 4, 128, 0, stream>>>(w, x, out, rows, cols);
      break;
    case core::GgmlType::kQ8_0:
      DenseF32Kernel<core::GgmlType::kQ8_0>
          <<<(rows + 3) / 4, 128, 0, stream>>>(w, x, out, rows, cols);
      break;
    default:
      throw std::invalid_argument("unsupported F32-input dense weight format");
  }
}

}  // namespace gufo::models::ornith15::rocm
