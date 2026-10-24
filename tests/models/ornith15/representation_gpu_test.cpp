// Production-launcher dump for the policy-B representation evaluator.
// This binary does not decide whether G matches a declared sequence. The
// host scorer in tools/ornith15/representation.py does that. A missing
// device is a skip.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/ornith15/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace orn = gufo::models::ornith15;
using gufo::core::GgmlType;

namespace {

void Require(bool ok, const char* what) {
  if (!ok)
    throw std::runtime_error(what);
}

template<typename T>
struct Buffer {
  T* data{nullptr};
  explicit Buffer(std::size_t n) {
    Require(
        hipMalloc(&data, std::max<std::size_t>(n, 1) * sizeof(T)) == hipSuccess,
        "hipMalloc");
  }
  explicit Buffer(const std::vector<T>& host) : Buffer(host.size()) {
    if (!host.empty())
      Require(hipMemcpy(data, host.data(), host.size() * sizeof(T),
                        hipMemcpyHostToDevice) == hipSuccess,
              "upload");
  }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  std::vector<T> Download(std::size_t n) const {
    std::vector<T> host(n);
    Require(hipDeviceSynchronize() == hipSuccess, "synchronize");
    if (n)
      Require(hipMemcpy(host.data(), data, n * sizeof(T),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "download");
    return host;
  }
};

std::uint32_t Next(std::uint32_t& seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}

void WriteBytes(const std::string& path, const void* data, std::size_t n) {
  std::ofstream out(path, std::ios::binary);
  Require(out.good(), "open dump");
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
  Require(out.good(), "write dump");
}

std::uint16_t HalfBits(float value) {
  const __half half = __float2half(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

void StampScales(std::vector<std::uint8_t>& packed, int block_bytes,
                 int d_offset, float scale) {
  const std::uint16_t bits = HalfBits(scale);
  for (std::size_t b = 0;
       b + static_cast<std::size_t>(block_bytes) <= packed.size();
       b += block_bytes)
    std::memcpy(packed.data() + b + d_offset, &bits, sizeof(bits));
}

struct Dump {
  std::string dir;
  std::string manifest = "{\n  \"cases\": [\n";
  bool first = true;

  explicit Dump(std::string directory) : dir(std::move(directory)) {}

  void Add(const std::string& body) {
    if (!first)
      manifest += ",\n";
    first = false;
    manifest += body;
  }

  void Finish() {
    manifest += "\n  ]\n}\n";
    WriteBytes(dir + "/manifest.json", manifest.data(), manifest.size());
  }
};

std::string Quote(const std::string& name) {
  return "\"" + name + "\"";
}

void DumpQuantize(Dump& dump, const std::string& name, int k, int rows,
                  const std::vector<float>& values) {
  Require(static_cast<int>(values.size()) == rows * k, "quantize shape");
  const Buffer<float> x(values);
  const std::size_t bytes = qfn_mmq_q8_1_bytes(rows, k);
  Buffer<std::uint8_t> q8(bytes);
  Require(qfn_mmq_quantize_q8_1(x.data, q8.data, rows, k, nullptr) == 0,
          "quantize");
  const auto host = q8.Download(bytes);
  WriteBytes(dump.dir + "/" + name + "_x.f32", values.data(),
             values.size() * sizeof(float));
  WriteBytes(dump.dir + "/" + name + ".q8", host.data(), host.size());
  dump.Add("    {\"name\": " + Quote(name) + ", \"kind\": \"q8_1\", \"k\": " +
           std::to_string(k) + ", \"rows\": " + std::to_string(rows) +
           ", \"x\": " + Quote(name + "_x.f32") +
           ", \"q8\": " + Quote(name + ".q8") + "}");
}

void DumpMmvq(Dump& dump, const std::string& name, GgmlType type, int k,
              int rows, int experts, int tokens, int used,
              const std::vector<float>& values,
              const std::vector<std::int32_t>& ids,
              const std::vector<std::uint8_t>& weight) {
  const Buffer<float> x(values);
  const Buffer<std::int32_t> id_buf(ids);
  const Buffer<std::uint8_t> w(weight);
  Buffer<float> y(static_cast<std::size_t>(tokens) * used * rows);
  Require(qfn_mmq_moe_vec(static_cast<int>(type), w.data, x.data, id_buf.data,
                          y.data, rows, k, tokens, experts, used, nullptr) == 0,
          "moe_vec");
  const auto host = y.Download(static_cast<std::size_t>(tokens) * used * rows);
  const char* format = type == GgmlType::kQ6_K ? "q6_k" : "q8_0";
  WriteBytes(dump.dir + "/" + name + "_x.f32", values.data(),
             values.size() * sizeof(float));
  WriteBytes(dump.dir + "/" + name + "_w.bin", weight.data(), weight.size());
  WriteBytes(dump.dir + "/" + name + "_y.f32", host.data(),
             host.size() * sizeof(float));
  std::string id_json;
  for (std::size_t i = 0; i < ids.size(); ++i)
    id_json += (i ? ", " : "") + std::to_string(ids[i]);
  dump.Add("    {\"name\": " + Quote(name) +
           ", \"kind\": \"mmvq\", \"format\": " + Quote(format) + ", \"k\": " +
           std::to_string(k) + ", \"rows\": " + std::to_string(rows) +
           ", \"experts\": " + std::to_string(experts) + ", \"tokens\": " +
           std::to_string(tokens) + ", \"used\": " + std::to_string(used) +
           ", \"ids\": [" + id_json + "], \"x\": " + Quote(name + "_x.f32") +
           ", \"weights\": " + Quote(name + "_w.bin") +
           ", \"y\": " + Quote(name + "_y.f32") + "}");
}

void DumpOf32(Dump& dump, const std::string& name, GgmlType type, int k,
              int rows, const std::vector<float>& values,
              const std::vector<std::uint8_t>& weight) {
  const Buffer<float> x(values);
  const Buffer<std::uint8_t> w(weight);
  Buffer<float> y(rows);
  orn::rocm::DenseMatVecF32(type, w.data, x.data, y.data, rows, k, nullptr);
  const auto host = y.Download(rows);
  const char* format = type == GgmlType::kQ6_K ? "q6_k" : "q8_0";
  WriteBytes(dump.dir + "/" + name + "_x.f32", values.data(),
             values.size() * sizeof(float));
  WriteBytes(dump.dir + "/" + name + "_w.bin", weight.data(), weight.size());
  WriteBytes(dump.dir + "/" + name + "_y.f32", host.data(),
             host.size() * sizeof(float));
  dump.Add("    {\"name\": " + Quote(name) +
           ", \"kind\": \"of32\", \"format\": " + Quote(format) + ", \"k\": " +
           std::to_string(k) + ", \"rows\": " + std::to_string(rows) +
           ", \"x\": " + Quote(name + "_x.f32") + ", \"weights\": " +
           Quote(name + "_w.bin") + ", \"y\": " + Quote(name + "_y.f32") + "}");
}

std::vector<float> Activations(int n, int salt) {
  std::vector<float> values(n);
  for (int i = 0; i < n; ++i)
    values[i] = static_cast<float>(((i * 17 + salt * 13) % 201) - 100) / 25.0F;
  return values;
}

std::vector<std::uint8_t> Weights(GgmlType type, int k, int rows, int experts,
                                  std::uint32_t seed, float scale) {
  const int block = type == GgmlType::kQ6_K ? 256 : 32;
  const int bytes = type == GgmlType::kQ6_K ? 210 : 34;
  std::vector<std::uint8_t> packed(static_cast<std::size_t>(experts) * rows *
                                   (k / block) * bytes);
  for (auto& byte : packed)
    byte = static_cast<std::uint8_t>(Next(seed));
  StampScales(packed, bytes, type == GgmlType::kQ6_K ? 208 : 0, scale);
  return packed;
}

}  // namespace

int main(int argc, char** argv) {
  const char* env = std::getenv("GUFO_ORNITH15_REPRESENTATION_DUMP");
  const char* dir = argc > 1 ? argv[1] : env;
  if (!dir || !dir[0]) {
    std::cout << "SKIP: pass a dump directory\n";
    return 77;
  }
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
    std::cout << "SKIP: no HIP device\n";
    return 77;
  }
  try {
    Require(qfn_mmq_init(0) == 0, "qfn_mmq_init");
    Dump dump(dir);

    std::vector<float> crafted(32, 0.0F);
    crafted[0] = 127.0F;
    crafted[1] = 0.5F;
    crafted[2] = -0.5F;
    crafted[3] = 1.5F;
    crafted[4] = -1.5F;
    std::vector<float> q8_rows = crafted;
    auto quiet = Activations(32, 2);
    auto mixed = Activations(32, 3);
    mixed[0] = 4.0F;
    mixed[1] = -4.0F;
    q8_rows.insert(q8_rows.end(), 32, 0.0F);
    q8_rows.insert(q8_rows.end(), quiet.begin(), quiet.end());
    q8_rows.insert(q8_rows.end(), mixed.begin(), mixed.end());
    DumpQuantize(dump, "q8_k32", 32, 4, q8_rows);
    DumpQuantize(dump, "q8_k512", 512, 2, [&] {
      auto a = Activations(512, 4);
      auto b = Activations(512, 5);
      a.insert(a.end(), b.begin(), b.end());
      return a;
    }());
    std::vector<float> tail(544, 0.0F);
    tail[543] = 1.0F;
    for (int i = 0; i < 40; ++i)
      tail[i] = Activations(40, 6)[i];
    DumpQuantize(dump, "q8_k544", 544, 1, tail);

    const std::vector<std::int32_t> ids{0, 0, 1, 0};
    for (const int k : {256, 2048}) {
      const int rows = 2;
      auto x = Activations(k, 7);
      auto y = Activations(k, 8);
      x.insert(x.end(), y.begin(), y.end());
      DumpMmvq(dump, "q6_k" + std::to_string(k), GgmlType::kQ6_K, k, rows, 2, 2,
               2, x, ids,
               Weights(GgmlType::kQ6_K, k, rows, 2, 1000 + k, 0.02F));
    }
    for (const int k : {32, 512, 2048}) {
      const int rows = 2;
      auto x = Activations(k, 9);
      auto y = Activations(k, 10);
      x.insert(x.end(), y.begin(), y.end());
      DumpMmvq(dump, "q8_0_k" + std::to_string(k), GgmlType::kQ8_0, k, rows, 2,
               2, 2, x, ids,
               Weights(GgmlType::kQ8_0, k, rows, 2, 2000 + k, 0.05F));
    }
    for (const int k : {256, 2048}) {
      DumpOf32(dump, "of32_q6_k" + std::to_string(k), GgmlType::kQ6_K, k, 3,
               Activations(k, 11),
               Weights(GgmlType::kQ6_K, k, 3, 1, 3000 + k, 0.015625F));
    }
    for (const int k : {32, 512}) {
      DumpOf32(dump, "of32_q8_k" + std::to_string(k), GgmlType::kQ8_0, k, 3,
               Activations(k, 12),
               Weights(GgmlType::kQ8_0, k, 3, 1, 4000 + k, 0.125F));
    }
    dump.Finish();
    std::cout << "representation dump written; numerics are not judged here\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
