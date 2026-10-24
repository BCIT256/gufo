#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/models/ornith15/cpu_ops.hpp"
#include "src/models/ornith15/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace orn = gufo::models::ornith15;
using gufo::core::GgmlType;
namespace {

void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("GPU operator check failed");
}

template<typename T>
struct Buffer {
  T* data{nullptr};
  explicit Buffer(std::size_t n) {
    Require(hipMalloc(&data, n * sizeof(T)) == hipSuccess);
  }
  explicit Buffer(const std::vector<T>& x) : Buffer(x.size()) {
    Require(hipMemcpy(data, x.data(), x.size() * sizeof(T),
                      hipMemcpyHostToDevice) == hipSuccess);
  }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  std::vector<T> Download(std::size_t n) {
    std::vector<T> x(n);
    Require(hipMemcpy(x.data(), data, n * sizeof(T), hipMemcpyDeviceToHost) ==
            hipSuccess);
    return x;
  }
};

void Run(GgmlType type) {
  constexpr int k = 512, m = 16, experts = 3, rows = 3, used = 2;
  const int block = type == GgmlType::kQ8_0 ? 32 : 256;
  const int bytes = type == GgmlType::kQ6_K     ? 210
                    : type == GgmlType::kQ5_K   ? 176
                    : type == GgmlType::kIQ4_XS ? 136
                                                : 34;
  std::vector<std::uint8_t> packed(std::size_t(experts) * m * k / block *
                                   bytes);
  std::uint32_t seed = 12345;
  const auto next = [&]() {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
  };
  for (auto& b : packed)
    b = static_cast<std::uint8_t>(next());
  for (std::size_t b = 0; b < packed.size(); b += bytes) {
    const __half d = __float2half(0.0003F);
    std::memcpy(packed.data() + b + (type == GgmlType::kQ6_K ? 208 : 0), &d, 2);
    if (type == GgmlType::kQ5_K)
      std::memcpy(packed.data() + b + 2, &d, 2);
  }
  const orn::TensorRef w{packed.data(), type, k, m, experts, 0, 0, "synthetic"};
  std::vector<float> x(rows * k);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = (static_cast<int>(next() % 65536) - 32768) / 32768.0F *
           (i / k == 1 ? 0.007F : 2.0F);
  const std::vector<std::int32_t> ids{0, 2, 1, 2, 0, 1};
  const Buffer<std::uint8_t> dw(packed);
  const Buffer<float> dx(x);
  const Buffer<std::int32_t> di(ids);
  Buffer<float> out(rows * used * m);
  Require(qfn_mmq_moe_vec(static_cast<int>(type), dw.data, dx.data, di.data,
                          out.data, m, k, rows, experts, used, nullptr) == 0);
  const auto single = out.Download(rows * used * m);
  double scale = 0, one_error = 0;
  std::vector<float> unpacked(k);
  for (int r = 0; r < rows; ++r)
    for (int s = 0; s < used; ++s)
      for (int j = 0; j < m; ++j) {
        orn::cpu::DequantizeRow(w, ids[r * used + s], j, unpacked.data());
        double expected = 0;
        for (int i = 0; i < k; ++i)
          expected += static_cast<double>(unpacked[i]) * x[r * k + i];
        const auto index = (r * used + s) * m + j;
        Require(std::isfinite(single[index]));
        scale = std::max(scale, std::abs(expected));
        one_error = std::max(one_error, std::abs(expected - single[index]));
      }
  std::cout << "type=" << static_cast<int>(type)
            << " single_error=" << one_error << " tolerance=" << 0.01 * scale
            << '\n';
  Require(one_error < 0.01 * scale);
  if (type == GgmlType::kQ6_K) {
    Buffer<float> embedding(k);
    for (const auto token : {0U, 7U, 15U}) {
      orn::rocm::EmbedQ6(dw.data, token, embedding.data, k, nullptr);
      orn::cpu::DequantizeRow(w, 0, token, unpacked.data());
      Require(embedding.Download(k) == unpacked);
    }
  }
}

// Independent unpacked-weight/F64 check, including scales that Q8_1 cannot
// represent and an outlier among otherwise unsaturated channels.
void RunF32() {
  for (const auto type : {GgmlType::kQ6_K, GgmlType::kQ8_0}) {
    const int block = type == GgmlType::kQ6_K ? 256 : 32;
    const int bytes = type == GgmlType::kQ6_K ? 210 : 34;
    const std::vector<int> sizes = type == GgmlType::kQ6_K
                                       ? std::vector<int>{256, 512, 4096}
                                       : std::vector<int>{32, 96, 512, 4096};
    for (int k : sizes) {
      for (int m : {1, 3, 17}) {
        std::uint32_t seed = 98765;
        const auto next = [&]() {
          seed ^= seed << 13;
          seed ^= seed >> 17;
          seed ^= seed << 5;
          return seed;
        };
        std::vector<std::uint8_t> packed(std::size_t(m) * (k / block) * bytes);
        for (auto& b : packed)
          b = static_cast<std::uint8_t>(next());
        for (std::size_t b = 0; b < packed.size(); b += bytes) {
          const __half d = __float2half(0.0003F);
          std::memcpy(packed.data() + b + (type == GgmlType::kQ6_K ? 208 : 0),
                      &d, 2);
        }
        const orn::TensorRef w{packed.data(),
                               type,
                               static_cast<std::uint64_t>(k),
                               static_cast<std::uint64_t>(m),
                               1,
                               0,
                               0,
                               "synthetic"};
        const Buffer<std::uint8_t> dw(packed);
        for (float magnitude : {0.0F, 1e-7F, 0.03F, 20.0F, 1e6F}) {
          std::vector<float> x(k), row(k);
          for (auto& v : x)
            v = (static_cast<int>(next() % 65536) - 32768) / 32768.0F *
                magnitude;
          x[k / 2] *= 100;
          const Buffer<float> dx(x);
          Buffer<float> out(m);
          orn::rocm::DenseMatVecF32(type, dw.data, dx.data, out.data, m, k,
                                    nullptr);
          const auto actual = out.Download(m);
          double worst = 0, scale = 0;
          for (int r = 0; r < m; ++r) {
            orn::cpu::DequantizeRow(w, 0, r, row.data());
            double expected = 0;
            for (int i = 0; i < k; ++i)
              expected += static_cast<double>(row[i]) * x[i];
            Require(std::isfinite(actual[r]));
            worst = std::max(worst, std::abs(expected - actual[r]));
            scale = std::max(scale, std::abs(expected));
          }
          Require(worst <= 1e-5 * scale);
          std::cout << "dense_f32 type=" << static_cast<int>(type) << " k=" << k
                    << " m=" << m << " magnitude=" << magnitude
                    << " max_error=" << worst << " bound=" << 1e-5 * scale
                    << '\n';
        }
      }
    }
  }
}

}  // namespace

int main() {
  try {
    Require(qfn_mmq_init(0) == 0);
    for (auto type :
         {GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kIQ4_XS, GgmlType::kQ8_0})
      Run(type);
    RunF32();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
