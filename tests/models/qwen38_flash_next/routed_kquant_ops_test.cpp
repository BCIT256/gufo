#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

using gufo::core::GgmlType;
namespace {

constexpr int kExperts = 256;
constexpr int kUsed = 8;
constexpr std::size_t kGuard = 16;
constexpr float kPoison = -1234567.0F;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

// Exact allocation sizes expose ragged input overreads. All buffers outlive
// the queued work; downloads synchronize before destruction.
template<typename T>
struct Device {
  T* data = nullptr;
  explicit Device(const std::vector<T>& host) {
    CheckHip(hipMalloc(&data, host.size() * sizeof(T)), "allocation");
    CheckHip(hipMemcpy(data, host.data(), host.size() * sizeof(T),
                       hipMemcpyHostToDevice),
             "upload");
  }
  ~Device() { (void)hipFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  std::vector<T> Download(std::size_t count) const {
    std::vector<T> host(count);
    CheckHip(
        hipMemcpy(host.data(), data, count * sizeof(T), hipMemcpyDeviceToHost),
        "download");
    return host;
  }
};

struct Weights {
  std::vector<std::uint8_t> packed;
  std::vector<float> values;
};

// Independently unpack the GGUF layouts on the host. Do not call the device
// tile loaders, dot helpers or the production CPU dequantizer for the oracle.
Weights MakeWeights(GgmlType type, int rows, int cols, std::uint32_t seed) {
  const bool q6 = type == GgmlType::kQ6_K;
  const std::size_t bytes = q6 ? 210 : 136;
  const std::size_t blocks = std::size_t(kExperts) * rows * cols / 256;
  Weights w{std::vector<std::uint8_t>(blocks * bytes),
            std::vector<float>(blocks * 256)};
  constexpr int levels[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                              1,    13,   25,  38,  53,  69,  89,  113};
  for (std::size_t b = 0; b < blocks; ++b) {
    auto* p = w.packed.data() + b * bytes;
    for (std::size_t i = 0; i < bytes; ++i)
      p[i] = static_cast<std::uint8_t>(Next(seed));
    const __half d = __float2half(0.0001F * (1 + Next(seed) % 5));
    std::memcpy(p + (q6 ? 208 : 0), &d, 2);
    const float df = __half2float(d);
    std::uint16_t high = 0;
    if (!q6)
      std::memcpy(&high, p + 2, 2);
    for (int i = 0; i < 256; ++i) {
      int code, scale;
      if (q6) {
        const int half = i / 128, segment = i % 128 / 32, lane = i % 32;
        const int low = p[half * 64 + segment % 2 * 32 + lane];
        code = ((low >> (segment / 2 * 4)) & 15) |
               (((p[128 + half * 32 + lane] >> (2 * segment)) & 3) << 4);
        code -= 32;
        scale = static_cast<std::int8_t>(p[192 + i / 16]);
      } else {
        const int group = i / 32, lane = i % 32;
        const int packed = p[8 + group * 16 + lane % 16];
        code = levels[(packed >> (lane / 16 * 4)) & 15];
        scale = ((p[4 + group / 2] >> (4 * (group % 2))) & 15) |
                (((high >> (2 * group)) & 3) << 4);
        scale -= 32;
      }
      w.values[b * 256 + i] = df * static_cast<float>(scale) * code;
    }
  }
  return w;
}

std::vector<float> Output(const Device<float>& device, std::size_t count) {
  const auto host = device.Download(count + 2 * kGuard);
  for (std::size_t i = 0; i < host.size(); ++i) {
    if (i < kGuard || i >= kGuard + count) {
      if (host[i] != kPoison)
        throw std::runtime_error("routed projection overwrote a guard");
    } else if (!std::isfinite(host[i])) {
      throw std::runtime_error("nonfinite routed projection");
    }
  }
  return {host.begin() + kGuard, host.end() - kGuard};
}

void Run(GgmlType type, bool down) {
  const char* name = type == GgmlType::kQ6_K ? "Q6_K" : "IQ4_XS";
  const int rows = down ? 2048 : 512, cols = down ? 512 : 2048;
  const auto weights = MakeWeights(type, rows, cols, 17);
  const auto paired = down ? Weights{} : MakeWeights(type, rows, cols, 79);
  const Device<std::uint8_t> dw(weights.packed);
  // Use independent gate/up weights to detect accidental projection reuse.
  const Device<std::uint8_t> db(down ? weights.packed : paired.packed);
  const auto mmq =
      type == GgmlType::kQ6_K ? qfn_mmq_q6_K_moe_raw : qfn_mmq_iq4_xs_moe_raw;
  for (const int tokens : {1, 2, 3, 8, 64}) {
    const int inputs = down ? tokens * kUsed : tokens;
    const int used = down ? 1 : kUsed;
    const std::size_t count = std::size_t(tokens) * kUsed * rows;
    std::uint32_t seed = 37;
    std::vector<float> x(std::size_t(inputs) * cols);
    for (std::size_t i = 0; i < x.size(); ++i) {
      // Unsaturated rows alongside ordinary activations, as in the existing
      // routed vector grouping test.
      const float scale = (i / cols) % 2 ? 1.0F / 128.0F : 2.0F;
      x[i] = scale * (static_cast<int>(Next(seed) & 0xFFFFU) - 32768) / 32768;
    }
    std::vector<std::int32_t> ids(tokens * kUsed);
    for (int t = 0; t < tokens; ++t) {
      // Four shared experts (including both boundaries) and four dispersed
      // experts. Each token selects eight distinct experts out of 256.
      for (int j = 0; j < kUsed; ++j)
        ids[t * kUsed + j] = j < 4 ? j * 85 : (t * 4 + j - 4) % 252 + 1;
      for (int j = 4; j < kUsed; ++j) {
        int& e = ids[t * kUsed + j];
        if (e >= 85)
          ++e;
        if (e >= 170)
          ++e;
      }
    }
    const Device<float> dx(x);
    const Device<std::int32_t> di(ids);
    const std::vector<float> dirty(count + 2 * kGuard, kPoison);
    const Device<float> dv(dirty), dm(dirty), dup(dirty), dg(dirty);
    if (qfn_mmq_moe_vec(static_cast<int>(type), dw.data, dx.data, di.data,
                        dv.data + kGuard, rows, cols, inputs, kExperts, used,
                        nullptr, down ? nullptr : db.data,
                        down ? nullptr : dup.data + kGuard) ||
        mmq(dw.data, dx.data, di.data, dm.data + kGuard, rows, cols, inputs,
            kExperts, used, nullptr))
      throw std::runtime_error("routed K-quant launch failed");
    if (tokens == 1) {
      if (mmq(dw.data, dx.data, di.data, dg.data + kGuard, rows, 640, inputs,
              kExperts, used, nullptr) != -1 ||
          qfn_mmq_moe_vec(static_cast<int>(type), dw.data, dx.data, di.data,
                          dg.data + kGuard, rows, 640, inputs, kExperts, used,
                          nullptr) != -1 ||
          qfn_mmq_moe_gated_vec(static_cast<int>(type), dw.data, db.data,
                                dx.data, di.data, dg.data + kGuard, rows, cols,
                                1, kExperts, 33, nullptr) != -1 ||
          qfn_mmq_moe_gated_vec(static_cast<int>(type), dw.data, db.data,
                                dx.data, di.data, dg.data + kGuard, rows, cols,
                                9, kExperts, kUsed, nullptr) != -1)
        throw std::runtime_error("K-quant adapter accepted invalid geometry");
      if (dg.Download(dirty.size()) != dirty)
        throw std::runtime_error("rejected projection modified output");
    }
    const auto vec = Output(dv, count), mat = Output(dm, count);
    double scale = 0.0, error_vec = 0.0, error_mat = 0.0;
    for (std::size_t s = 0; s < ids.size(); ++s) {
      for (int row = 0; row < rows; ++row) {
        double ref = 0.0;
        const auto* w =
            weights.values.data() + (std::size_t(ids[s]) * rows + row) * cols;
        const auto* input = x.data() + s / used * cols;
        for (int k = 0; k < cols; ++k)
          ref += static_cast<double>(w[k]) * input[k];
        scale = std::max(scale, std::abs(ref));
        error_vec = std::max(error_vec, std::abs(ref - vec[s * rows + row]));
        error_mat = std::max(error_mat, std::abs(ref - mat[s * rows + row]));
      }
    }
    // Same activation-quantization bound as routed_wmma_ops_test::Ok:
    // max |MMQ - F64| < 1e-2 * max |F64|. MMVQ consumes the same Q8_1
    // activation precision, so apply that identical bound to both routes.
    const double tolerance = 1e-2 * scale;
    for (const bool vector : {true, false}) {
      const double error = vector ? error_vec : error_mat;
      std::cout << std::setprecision(10) << name << ' '
                << (vector ? "MMVQ" : "MMQ") << " tokens=" << tokens
                << " E=256 top=8 M=" << rows << " K=" << cols
                << " max_error=" << error << " tolerance=" << tolerance
                << " scale=" << scale << std::endl;
      if (!(error < tolerance))
        throw std::runtime_error("routed projection exceeds F64 tolerance");
    }
    if (!down) {
      const auto up = Output(dup, count);
      const Device<float> separate(dirty);
      if (qfn_mmq_moe_vec(static_cast<int>(type), db.data, dx.data, di.data,
                          separate.data + kGuard, rows, cols, inputs, kExperts,
                          used, nullptr))
        throw std::runtime_error("separate up projection failed");
      if (up != Output(separate, count))
        throw std::runtime_error("paired up differs from separate projection");
      if (qfn_mmq_moe_vec(static_cast<int>(type), dw.data, dx.data, di.data,
                          separate.data + kGuard, rows, cols, inputs, kExperts,
                          used, nullptr))
        throw std::runtime_error("separate gate projection failed");
      if (vec != Output(separate, count))
        throw std::runtime_error(
            "paired gate differs from separate projection");
      // Gated MMVQ retains its established 1..8-token contract. Wider
      // ungated decode and MMQ batches above are still exercised at 64.
      if (tokens <= 8) {
        if (qfn_mmq_moe_gated_vec(static_cast<int>(type), dw.data, db.data,
                                  dx.data, di.data, dg.data + kGuard, rows,
                                  cols, tokens, kExperts, kUsed, nullptr))
          throw std::runtime_error("gated projection failed");
        const auto gated = Output(dg, count);
        double gated_scale = 0.0, gated_error = 0.0;
        for (std::size_t s = 0; s < ids.size(); ++s) {
          for (int row = 0; row < rows; ++row) {
            const auto offset = (std::size_t(ids[s]) * rows + row) * cols;
            const auto* gate_weights = weights.values.data() + offset;
            const auto* up_weights = paired.values.data() + offset;
            const auto* input = x.data() + s / kUsed * cols;
            double ref_gate = 0.0, ref_up = 0.0;
            for (int k = 0; k < cols; ++k) {
              ref_gate += static_cast<double>(gate_weights[k]) * input[k];
              ref_up += static_cast<double>(up_weights[k]) * input[k];
            }
            const double expected =
                (ref_gate / (1.0 + std::exp(-ref_gate))) * ref_up;
            gated_scale = std::max(gated_scale, std::abs(expected));
            gated_error = std::max(gated_error,
                                   std::abs(expected - gated[s * rows + row]));
          }
        }
        const double gated_tolerance = 1e-2 * gated_scale;
        std::cout << std::setprecision(10) << name
                  << " gated MMVQ tokens=" << tokens
                  << " E=256 top=8 M=" << rows << " K=" << cols
                  << " max_error=" << gated_error
                  << " tolerance=" << gated_tolerance
                  << " scale=" << gated_scale << std::endl;
        if (!(gated_error < gated_tolerance))
          throw std::runtime_error("gated projection exceeds F64 tolerance");
        // Use the existing grouping test's bit-identity contract against
        // separate scalar device gates (same exponential implementation).
        for (int t = 0; t < tokens; ++t) {
          if (qfn_mmq_moe_gated_vec(static_cast<int>(type), dw.data, db.data,
                                    dx.data + t * cols, di.data + t * kUsed,
                                    separate.data + kGuard + t * kUsed * rows,
                                    rows, cols, 1, kExperts, kUsed, nullptr))
            throw std::runtime_error("scalar gated projection failed");
        }
        if (gated != Output(separate, count))
          throw std::runtime_error("grouped gate differs from scalar gate");
      }
    }
  }
}

}  // namespace

int main() {
  try {
    if (qfn_mmq_init(0))
      throw std::runtime_error("HIP initialization failed");
    for (const auto type : {GgmlType::kQ6_K, GgmlType::kIQ4_XS})
      for (const bool down : {false, true})
        Run(type, down);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
