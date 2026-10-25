// Layer-6 expert rows for tool_two_parameters position 96, on the frozen
// GPU post-attention norm and the GPU's ordered expert ids. Production B
// is not modified. The scalar trajectory's expert set is not substituted.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/ornith15/cpu_ops.hpp"
#include "src/models/ornith15/kernels/rocm/device_model.hpp"
#include "src/models/ornith15/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace orn = gufo::models::ornith15;
namespace fn = gufo::models::qwen38_flash_next::rocm;
namespace fs = std::filesystem;
using gufo::core::GgmlType;
using gufo::quant::Fp16ToFloat;
using gufo::quant::block_q5_K;

#pragma pack(push, 1)
struct Q8Block {
  std::uint16_t d;
  std::uint16_t s;
  std::int8_t qs[32];
};
#pragma pack(pop)
static_assert(sizeof(Q8Block) == 36, "q8_1 block");

namespace {

constexpr int kIds[8] = {202, 120, 108, 37, 165, 33, 106, 218};
constexpr int kUsed = 8;
constexpr int kHidden = 2048;
constexpr int kExpert = 512;
constexpr int kExperts = 256;

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

const char* TypeName(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
      return "F32";
    case GgmlType::kQ8_0:
      return "Q8_0";
    case GgmlType::kQ5_K:
      return "Q5_K";
    case GgmlType::kQ6_K:
      return "Q6_K";
    case GgmlType::kQ4_K:
      return "Q4_K";
    case GgmlType::kIQ4_XS:
      return "IQ4_XS";
    default:
      return "other";
  }
}

std::uint32_t LoadU32(const std::uint8_t* p) {
  std::uint32_t v = 0;
  std::memcpy(&v, p, 4);
  return v;
}

int Dp4a(std::uint32_t a, std::uint32_t b, int c) {
  int sum = c;
  for (int i = 0; i < 4; ++i) {
    const auto xa = static_cast<std::int8_t>((a >> (8 * i)) & 0xff);
    const auto xb = static_cast<std::int8_t>((b >> (8 * i)) & 0xff);
    sum += static_cast<int>(xa) * static_cast<int>(xb);
  }
  return sum;
}

float LowHalf(const Q8Block& block) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &block, 2);
  return Fp16ToFloat(bits);
}

// One Q5_K MMVQ partial. d8 is the stored Q8 scale, not the auxiliary half.
// The code sum in the min term is recomputed from the codes.
float Q5Partial(const block_q5_K& weight, const Q8Block* row, int kbx,
                int iqs, bool contract) {
  const int bq8_offset = 2 * ((iqs / 2) / 4);
  const auto vl0 = static_cast<std::int32_t>(
      LoadU32(weight.qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4)));
  const auto vl1 = static_cast<std::int32_t>(
      LoadU32(weight.qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4) + 16));
  const auto vh0 = static_cast<std::int32_t>(
                       LoadU32(weight.qh + 4 * ((iqs / 2) % 4))) >>
                   bq8_offset;
  const auto vh1 = static_cast<std::int32_t>(
                       LoadU32(weight.qh + 4 * ((iqs / 2) % 4) + 16)) >>
                   bq8_offset;
  std::uint16_t aux[2];
  const auto* scales = reinterpret_cast<const std::uint16_t*>(weight.scales);
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = static_cast<std::uint16_t>(((scales[j + 2] >> 0) & 0x0f0f) |
                                       ((scales[j - 2] & 0xc0c0) >> 2));
    aux[1] = static_cast<std::uint16_t>(((scales[j + 2] >> 4) & 0x0f0f) |
                                       ((scales[j] & 0xc0c0) >> 2));
  }
  const auto* sc = reinterpret_cast<const std::uint8_t*>(aux);
  const auto* m = sc + 2;
  float sum_d = 0;
  float sum_m = 0;
  for (int i = 0; i < 2; ++i) {
    const std::uint32_t v0 =
        ((static_cast<std::uint32_t>(vl0) >> (4 * i)) & 0x0F0F0F0Fu) |
        ((static_cast<std::uint32_t>(vh0 >> i) << 4) & 0x10101010u);
    const std::uint32_t v1 =
        ((static_cast<std::uint32_t>(vl1) >> (4 * i)) & 0x0F0F0F0Fu) |
        ((static_cast<std::uint32_t>(vh1 >> i) << 4) & 0x10101010u);
    const Q8Block& q8 = row[static_cast<std::size_t>(kbx) * 8 + bq8_offset + i];
    const int lane = (iqs / 2) % 4;
    std::uint32_t u0 = 0, u1 = 0;
    std::memcpy(&u0, q8.qs + 4 * lane, 4);
    std::memcpy(&u1, q8.qs + 16 + 4 * lane, 4);
    const int dot = Dp4a(v0, u0, Dp4a(v1, u1, 0));
    const int code_sum = Dp4a(0x01010101u, u0, Dp4a(0x01010101u, u1, 0));
    const float d8 = LowHalf(q8);
    // sc and the code sum are integers in the device expression, so the
    // product is exact before it meets the Q8 scale.
    const float dot_sc = static_cast<float>(dot * static_cast<int>(sc[i]));
    const float sum_mterm =
        static_cast<float>(code_sum * static_cast<int>(m[i]));
    if (contract) {
      sum_d = std::fmaf(d8, dot_sc, sum_d);
      sum_m = std::fmaf(d8, sum_mterm, sum_m);
    } else {
      sum_d += d8 * dot_sc;
      sum_m += d8 * sum_mterm;
    }
  }
  const float d = Fp16ToFloat(weight.d);
  const float dmin = Fp16ToFloat(weight.dmin);
  const float scale_term = d * sum_d;
  const float min_term = dmin * sum_m;
  if (contract)
    return std::fmaf(d, sum_d, -min_term);
  return scale_term - min_term;
}

float Q5Row(const block_q5_K* blocks, int nblocks, const Q8Block* row,
            bool contract) {
  float lane[32] = {};
  for (int tid = 0; tid < 32; ++tid) {
    float acc = 0;
    for (int kbx = tid / 16; kbx < nblocks; kbx += 2) {
      const int iqs = 2 * (tid % 16);
      const float partial = Q5Partial(blocks[kbx], row, kbx, iqs, contract);
      acc = contract ? std::fmaf(1.0F, partial, acc) : acc + partial;
    }
    lane[tid] = acc;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    float next[32];
    for (int i = 0; i < 32; ++i) {
      const float sum = lane[i] + lane[i ^ offset];
      next[i] = sum;
    }
    std::copy(next, next + 32, lane);
  }
  const float value = lane[0];
  return std::isfinite(value) ? value : 0.0F;
}

struct Score {
  int mul_match{0};
  int fma_match{0};
  int rows{0};
  int unexplained{0};
  double g_m{0};
  double m_f{0};
  double g_f{0};
};

Score ScoreQ5(const orn::TensorRef& weight, const std::int32_t* ids, int slots,
               const float* input, int k, const Q8Block* q8, int q8_blocks,
               const float* gpu) {
  Score score;
  const int nblocks = k / 256;
  Require(q8_blocks == nblocks * 8, "q8 block count");
  for (int slot = 0; slot < slots; ++slot) {
    const auto* blocks = reinterpret_cast<const block_q5_K*>(
        weight.Expert(ids[slot]));
    const Q8Block* row = q8 + static_cast<std::size_t>(slot) * q8_blocks;
    std::vector<float> f64(weight.rows, 0.0F);
    orn::cpu::MatVec(weight, ids[slot],
                     std::span<const float>(input + static_cast<std::size_t>(slot) * k,
                                            k),
                     f64);
    for (std::uint64_t row_i = 0; row_i < weight.rows; ++row_i) {
      const float g = gpu[static_cast<std::size_t>(slot) * weight.rows + row_i];
      const float mul = Q5Row(blocks + row_i * nblocks, nblocks, row, false);
      const float fma = Q5Row(blocks + row_i * nblocks, nblocks, row, true);
      const float f = f64[row_i];
      if (std::memcmp(&g, &mul, sizeof(float)) == 0)
        ++score.mul_match;
      if (std::memcmp(&g, &fma, sizeof(float)) == 0)
        ++score.fma_match;
      if (std::memcmp(&g, &mul, sizeof(float)) != 0 &&
          std::memcmp(&g, &fma, sizeof(float)) != 0)
        ++score.unexplained;
      const float m = std::memcmp(&g, &fma, 4) == 0 ? fma : mul;
      score.g_m = std::max(score.g_m, std::min(std::abs(static_cast<double>(g) - mul),
                                              std::abs(static_cast<double>(g) - fma)));
      score.m_f = std::max(score.m_f, std::abs(static_cast<double>(m) - f));
      score.g_f = std::max(score.g_f, std::abs(static_cast<double>(g) - f));
      ++score.rows;
    }
  }
  return score;
}

struct EncodeDiff {
  int blocks{0};
  int code{0};
  int scale{0};
  int aux{0};
  int pad_blocks{0};
};

EncodeDiff CompareEncode(const float* rows, int nrows, int k, const void* gpu,
                         int gpu_bytes) {
  EncodeDiff diff;
  const int padded = (k + 511) & ~511;
  diff.pad_blocks = (padded - k) / 32;
  const int blocks = nrows * padded / 32;
  Require(gpu_bytes == blocks * static_cast<int>(sizeof(Q8Block)), "q8 bytes");
  const auto* got = static_cast<const Q8Block*>(gpu);
  int index = 0;
  for (int row = 0; row < nrows; ++row) {
    for (int b = 0; b < padded / 32; ++b, ++index) {
      ++diff.blocks;
      float chunk[32] = {};
      if (b * 32 < k) {
        const int n = std::min(32, k - b * 32);
        std::copy(rows + static_cast<std::size_t>(row) * k + b * 32,
                  rows + static_cast<std::size_t>(row) * k + b * 32 + n, chunk);
      }
      float amax = 0;
      for (float v : chunk)
        amax = std::max(amax, std::fabs(v));
      std::int8_t codes[32] = {};
      float d = 0;
      if (amax != 0) {
        d = amax / 127.0F;
        for (int i = 0; i < 32; ++i)
          codes[i] = static_cast<std::int8_t>(std::roundf(chunk[i] / d));
      }
      int sumq = 0;
      for (std::int8_t c : codes)
        sumq += c;
      __half hd = __float2half(d);
      __half hs = __float2half(d * static_cast<float>(sumq));
      std::uint16_t scale = 0, aux = 0, gscale = 0, gaux = 0;
      std::memcpy(&scale, &hd, 2);
      std::memcpy(&aux, &hs, 2);
      std::memcpy(&gscale, &got[index], 2);
      std::memcpy(&gaux, reinterpret_cast<const char*>(&got[index]) + 2, 2);
      if (std::memcmp(codes, got[index].qs, 32) != 0)
        ++diff.code;
      if (scale != gscale)
        ++diff.scale;
      if (aux != gaux)
        ++diff.aux;
    }
  }
  return diff;
}

void Save(const fs::path& path, const void* data, std::size_t bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  Require(static_cast<bool>(out), path.string());
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  const char* in_dir = std::getenv("GUFO_ORNITH15_ROUTE_DIR");
  const char* out_env = std::getenv("GUFO_ORNITH15_EXPERT_DIR");
  if (!model_path || !in_dir || !fs::is_regular_file(model_path) ||
      !fs::is_directory(in_dir)) {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL and "
                 "GUFO_ORNITH15_ROUTE_DIR\n";
    return 77;
  }
  try {
    std::cout << std::setprecision(17);
    std::cout << "experiment=layer6_expert_rows frozen_input=gpu_moe_norm "
                 "frozen_ids=gpu_order\n";
    std::ifstream xin(fs::path(in_dir) / "layer6_moe_in.f32", std::ios::binary);
    std::vector<float> moe_in(kHidden);
    xin.read(reinterpret_cast<char*>(moe_in.data()), kHidden * 4);
    Require(static_cast<bool>(xin), "moe input");
    std::ifstream din(fs::path(in_dir) / "layer6_expert_down.f32", std::ios::binary);
    std::vector<float> saved_down(static_cast<std::size_t>(kUsed) * kHidden);
    din.read(reinterpret_cast<char*>(saved_down.data()),
             static_cast<std::streamsize>(saved_down.size() * 4));
    Require(static_cast<bool>(din), "saved down");

    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, error);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
    Require(weights.has_value() && !weights->mtp, error);
    Require(qfn_mmq_init(0) == 0, "qfn init");
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr, error);
    const auto& host = weights->layers.at(6);
    const auto& layer = device->layers().at(6);
    std::cout << "format gate=" << TypeName(host.ffn_gate_exps.type)
              << " up=" << TypeName(host.ffn_up_exps.type)
              << " down=" << TypeName(host.ffn_down_exps.type)
              << " shared_gate=" << TypeName(host.shexp_gate.type)
              << " shared_up=" << TypeName(host.shexp_up.type)
              << " shared_down=" << TypeName(host.shexp_down.type)
              << " shared_gate_inp=" << TypeName(host.shexp_gate_inp.type)
              << "\n";
    Require(host.ffn_gate_exps.type == GgmlType::kQ5_K &&
                host.ffn_up_exps.type == GgmlType::kQ5_K &&
                host.ffn_down_exps.type == GgmlType::kQ5_K,
            "layer 6 routed experts are not Q5_K");
    std::cout << "ids";
    for (int id : kIds)
      std::cout << " " << id;
    std::cout << "\nslot_mapping=output_slot_s_reads_expert_ids_s\n";
    std::cout << "q5k_auxiliary_consumed=0 code_sum_recomputed_by_dp4a=1\n";
    std::cout << "activation_padding_gate_up=" << ((kHidden + 511) & ~511)
              << " down=" << ((kExpert + 511) & ~511) << "\n";

    std::vector<std::int32_t> ids(kIds, kIds + kUsed);
    const std::size_t q8_in_bytes = qfn_mmq_q8_1_bytes(1, kHidden);
    const std::size_t q8_down_bytes = qfn_mmq_q8_1_bytes(kUsed, kExpert);
    std::vector<float> gate(static_cast<std::size_t>(kUsed) * kExpert);
    std::vector<float> up = gate;
    std::vector<float> swiglu = gate;
    std::vector<float> down(static_cast<std::size_t>(kUsed) * kHidden);
    std::vector<std::uint8_t> q8_in(q8_in_bytes), q8_down(q8_down_bytes);

    auto* d_x = static_cast<float*>([&] {
      void* p = nullptr;
      Require(hipMalloc(&p, kHidden * 4) == hipSuccess, "x");
      return p;
    }());
    auto* d_ids = static_cast<std::int32_t*>([&] {
      void* p = nullptr;
      Require(hipMalloc(&p, ids.size() * 4) == hipSuccess, "ids");
      return p;
    }());
    auto* d_gate = static_cast<float*>([&] {
      void* p = nullptr;
      Require(hipMalloc(&p, gate.size() * 4) == hipSuccess, "gate");
      return p;
    }());
    auto* d_up = static_cast<float*>([&] {
      void* p = nullptr;
      Require(hipMalloc(&p, up.size() * 4) == hipSuccess, "up");
      return p;
    }());
    auto* d_down = static_cast<float*>([&] {
      void* p = nullptr;
      Require(hipMalloc(&p, down.size() * 4) == hipSuccess, "down");
      return p;
    }());
    auto* d_q8_in = [&] {
      void* p = nullptr;
      Require(hipMalloc(&p, q8_in_bytes) == hipSuccess, "q8 in");
      return p;
    }();
    auto* d_q8_down = [&] {
      void* p = nullptr;
      Require(hipMalloc(&p, q8_down_bytes) == hipSuccess, "q8 down");
      return p;
    }();
    Require(hipMemcpy(d_x, moe_in.data(), kHidden * 4, hipMemcpyHostToDevice) ==
                hipSuccess,
            "upload x");
    Require(hipMemcpy(d_ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice) ==
                hipSuccess,
            "upload ids");
    const int q5 = static_cast<int>(GgmlType::kQ5_K);
    Require(qfn_mmq_quantize_q8_1(d_x, d_q8_in, 1, kHidden, nullptr) == 0,
            "quantize input");
    Require(qfn_mmq_moe_vec(q5, layer.ffn_gate_exps.data, d_x, d_ids, d_gate,
                            kExpert, kHidden, 1, kExperts, kUsed, nullptr) == 0,
            "gate");
    Require(qfn_mmq_moe_vec(q5, layer.ffn_up_exps.data, d_x, d_ids, d_up,
                            kExpert, kHidden, 1, kExperts, kUsed, nullptr) == 0,
            "up");
    Require(hipMemcpy(gate.data(), d_gate, gate.size() * 4, hipMemcpyDeviceToHost) ==
                hipSuccess,
            "download gate");
    Require(hipMemcpy(up.data(), d_up, up.size() * 4, hipMemcpyDeviceToHost) ==
                hipSuccess,
            "download up");
    fn::Swiglu(d_gate, d_up, gate.size(), nullptr);
    Require(hipMemcpy(swiglu.data(), d_gate, swiglu.size() * 4,
                      hipMemcpyDeviceToHost) == hipSuccess,
            "download swiglu");
    Require(qfn_mmq_quantize_q8_1(d_gate, d_q8_down, kUsed, kExpert, nullptr) == 0,
            "quantize swiglu");
    Require(qfn_mmq_moe_vec(q5, layer.ffn_down_exps.data, d_gate, d_ids, d_down,
                            kHidden, kExpert, kUsed, kExperts, 1, nullptr) == 0,
            "down");
    Require(hipMemcpy(down.data(), d_down, down.size() * 4, hipMemcpyDeviceToHost) ==
                hipSuccess,
            "download down");
    Require(hipMemcpy(q8_in.data(), d_q8_in, q8_in_bytes, hipMemcpyDeviceToHost) ==
                hipSuccess,
            "download q8 in");
    Require(hipMemcpy(q8_down.data(), d_q8_down, q8_down_bytes,
                      hipMemcpyDeviceToHost) == hipSuccess,
            "download q8 down");
    Require(hipDeviceSynchronize() == hipSuccess, "sync");

    const bool down_matches_capture =
        std::memcmp(down.data(), saved_down.data(), saved_down.size() * 4) == 0;
    double down_gap = 0;
    for (std::size_t i = 0; i < down.size(); ++i)
      down_gap = std::max(down_gap, std::abs(static_cast<double>(down[i]) -
                                            saved_down[i]));
    std::cout << "isolated_down_matches_captured_rows=" << down_matches_capture
              << " max_abs=" << down_gap << "\n";

    const EncodeDiff in_enc =
        CompareEncode(moe_in.data(), 1, kHidden, q8_in.data(),
                      static_cast<int>(q8_in_bytes));
    const EncodeDiff down_enc =
        CompareEncode(swiglu.data(), kUsed, kExpert, q8_down.data(),
                      static_cast<int>(q8_down_bytes));
    std::cout << "q8_input blocks=" << in_enc.blocks << " pad_blocks="
              << in_enc.pad_blocks << " code_bad=" << in_enc.code
              << " scale_bad=" << in_enc.scale << " aux_bad=" << in_enc.aux
              << "\nq8_down blocks=" << down_enc.blocks
              << " pad_blocks=" << down_enc.pad_blocks
              << " code_bad=" << down_enc.code << " scale_bad=" << down_enc.scale
              << " aux_bad=" << down_enc.aux << "\n";

    const auto* q8_in_blocks = reinterpret_cast<const Q8Block*>(q8_in.data());
    const auto* q8_down_blocks = reinterpret_cast<const Q8Block*>(q8_down.data());
    const int in_blocks = kHidden / 32;
    const int down_blocks = kExpert / 32;
    const Score gate_score =
        ScoreQ5(host.ffn_gate_exps, ids.data(), 1, moe_in.data(), kHidden,
                q8_in_blocks, in_blocks, gate.data());
    // Gate and up share the input row. MatVec for each slot uses expert ids[s],
    // while the q8 row is the single shared input. ScoreQ5 assumes one q8 row
    // per slot. Call it per slot against the same row.
    Score up_score;
    Score gate_by_slot;
    for (int slot = 0; slot < kUsed; ++slot) {
      const Score g =
          ScoreQ5(host.ffn_gate_exps, ids.data() + slot, 1, moe_in.data(),
                  kHidden, q8_in_blocks, in_blocks,
                  gate.data() + static_cast<std::size_t>(slot) * kExpert);
      const Score u =
          ScoreQ5(host.ffn_up_exps, ids.data() + slot, 1, moe_in.data(),
                  kHidden, q8_in_blocks, in_blocks,
                  up.data() + static_cast<std::size_t>(slot) * kExpert);
      gate_by_slot.mul_match += g.mul_match;
      gate_by_slot.fma_match += g.fma_match;
      gate_by_slot.unexplained += g.unexplained;
      gate_by_slot.rows += g.rows;
      gate_by_slot.g_m = std::max(gate_by_slot.g_m, g.g_m);
      gate_by_slot.m_f = std::max(gate_by_slot.m_f, g.m_f);
      gate_by_slot.g_f = std::max(gate_by_slot.g_f, g.g_f);
      up_score.mul_match += u.mul_match;
      up_score.fma_match += u.fma_match;
      up_score.unexplained += u.unexplained;
      up_score.rows += u.rows;
      up_score.g_m = std::max(up_score.g_m, u.g_m);
      up_score.m_f = std::max(up_score.m_f, u.m_f);
      up_score.g_f = std::max(up_score.g_f, u.g_f);
    }
    (void)gate_score;
    std::vector<float> swiglu_host(swiglu.size());
    double swiglu_gap = 0;
    for (std::size_t i = 0; i < swiglu.size(); ++i) {
      const float z = gate[i];
      const float silu = z / (1.0F + std::exp(-z));
      swiglu_host[i] = silu * up[i];
      swiglu_gap = std::max(
          swiglu_gap, std::abs(static_cast<double>(swiglu[i]) - swiglu_host[i]));
    }
    const Score down_score =
        ScoreQ5(host.ffn_down_exps, ids.data(), kUsed, swiglu.data(), kExpert,
                q8_down_blocks, down_blocks, down.data());
    auto report = [](const char* name, const Score& s) {
      std::cout << name << " rows=" << s.rows << " D_mul=" << s.mul_match
                << " D_fma=" << s.fma_match << " unexplained=" << s.unexplained
                << " G_M=" << s.g_m << " M_F=" << s.m_f << " G_F=" << s.g_f
                << "\n";
    };
    report("gate", gate_by_slot);
    report("up", up_score);
    report("down", down_score);
    std::cout << "swiglu_vs_libm_max_abs=" << swiglu_gap << "\n";

    const fs::path out = out_env ? fs::path(out_env)
                                 : fs::path("/tmp/ornith-expert-rows");
    fs::create_directories(out);
    Save(out / "gate.f32", gate.data(), gate.size() * 4);
    Save(out / "up.f32", up.data(), up.size() * 4);
    Save(out / "swiglu.f32", swiglu.data(), swiglu.size() * 4);
    Save(out / "down.f32", down.data(), down.size() * 4);
    Save(out / "q8_input.bin", q8_in.data(), q8_in.size());
    Save(out / "q8_down.bin", q8_down.data(), q8_down.size());
    std::cout << "fixtures=" << out.string() << "\nmeasurement_complete=1\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
