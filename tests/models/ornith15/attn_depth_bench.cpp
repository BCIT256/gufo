// Times Ornith's dense attention for one token pass: ten layers, each with its
// own K/V cache, so the working set is the model's and not the MALL's. Tries
// each heads-per-block and split count at the given rows and depths and prints
// an FNV-1a hash of the outputs, which must match across unsplit groupings.
//
// ornith15_attn_depth_bench [reps] [depths...]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kLayers = 10;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

std::vector<float> Gaussianish(std::size_t count, std::uint32_t seed) {
  std::vector<float> values(count);
  std::uint32_t state = seed | 1u;
  for (auto& value : values) {
    float sum = -2.0f;
    for (int i = 0; i < 4; ++i)
      sum += static_cast<float>(Next(&state) & 0xFFFF) / 65536.0f;
    value = sum;
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 9;
  std::vector<std::uint32_t> depths;
  for (int i = 2; i < argc; ++i)
    depths.push_back(static_cast<std::uint32_t>(std::atoi(argv[i])));
  if (depths.empty())
    depths = {2048, 8192, 16384, 30000};
  const std::uint32_t capacity =
      *std::max_element(depths.begin(), depths.end());
  const std::size_t kv_count = std::size_t(capacity + 2064) * kKvHeads * kDim;

  std::vector<__half*> k_caches;
  std::vector<__half*> v_caches;
  std::vector<float*> queries;
  constexpr std::uint32_t kMaxRows = 2048;
  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    for (std::uint32_t kind = 0; kind < 2; ++kind) {
      const auto values = Gaussianish(kv_count, layer * 2 + kind + 7);
      std::vector<__half> halves(kv_count);
      for (std::size_t i = 0; i < kv_count; ++i)
        halves[i] = __float2half(values[i]);
      (kind == 0 ? k_caches : v_caches).push_back(Upload(halves));
    }
    queries.push_back(Upload(
        Gaussianish(std::size_t(kMaxRows) * kHeads * kDim, layer + 1000)));
  }
  float* out =
      Upload(std::vector<float>(std::size_t(kMaxRows) * kHeads * kDim));
  float* partials =
      Upload(std::vector<float>(std::size_t(8) * kHeads * 32 * (kDim + 2)));
  std::uint32_t position = 0;
  std::uint32_t* device_pos = Upload(std::vector<std::uint32_t>{position});
  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "hipStreamCreate");

  struct Variant {
    std::uint32_t group;
    std::uint32_t splits;
  };
  const std::vector<Variant> narrow = {{2, 1},  {2, 2},  {2, 4}, {4, 2},
                                       {4, 4},  {4, 8},  {8, 4}, {8, 8},
                                       {8, 16}, {8, 32}, {4, 16}};
  const std::vector<Variant> wide = {{2, 1}, {4, 1}, {8, 1}};
  std::printf("depth rows group splits median_ms min_ms hash\n");
  for (const std::uint32_t depth : depths) {
    position = depth;
    CheckHip(hipMemcpy(device_pos, &position, sizeof(position),
                       hipMemcpyHostToDevice),
             "position");
    for (const std::uint32_t rows : {1u, 2u, 8u, 64u, 512u, 2048u}) {
      for (const Variant& variant : rows <= 8 ? narrow : wide) {
        std::vector<double> times;
        std::uint64_t hash = 14695981039346656037ULL;
        for (int rep = 0; rep < reps; ++rep) {
          const auto start = std::chrono::steady_clock::now();
          for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            q::Attention(
                queries[layer], k_caches[layer], v_caches[layer], nullptr, 0,
                out, variant.splits > 1 ? partials : nullptr, variant.splits,
                rows, device_pos, kHeads, kKvHeads, kDim, 1, stream, nullptr,
                nullptr, 0, 0, variant.group);
          }
          CheckHip(hipStreamSynchronize(stream), "attention");
          times.push_back(std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count());
        }
        std::vector<std::uint8_t> bytes(std::size_t(rows) * kHeads * kDim *
                                        sizeof(float));
        CheckHip(
            hipMemcpy(bytes.data(), out, bytes.size(), hipMemcpyDeviceToHost),
            "hash");
        for (const std::uint8_t byte : bytes) {
          hash ^= byte;
          hash *= 1099511628211ULL;
        }
        std::sort(times.begin(), times.end());
        std::printf("%5u %4u %5u %6u %9.2f %6.2f %016llx\n", depth, rows,
                    variant.group, variant.splits, times[times.size() / 2],
                    times.front(), static_cast<unsigned long long>(hash));
        std::fflush(stdout);
      }
    }
  }
  CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
  return 0;
}
