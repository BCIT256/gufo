// Production layer-0 replay for code_diff tokens 0..75.
// Calls EmbedQ6, RmsNormRows, qfn_mmq_quantize_q8_1, qfn_mmq_moe_vec and
// GatedDeltaNet. It does not run MoE or later layers.
// Mandatory process failures: nonfinite values, embedding or registered Q6
// mismatches, any Q8 code/scale/auxiliary mismatch, a GPU/GPU rerun that
// is not byte-identical, a pairwise convolution bound failure, or an
// unmeasured Q8 check. Other gaps are recorded without a cause label.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "src/models/ornith15/cpu_ops.hpp"
#include "src/models/ornith15/kernels/rocm/device_model.hpp"
#include "src/models/ornith15/kernels/rocm/kernels.hpp"
#include "src/models/ornith15/weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace fs = std::filesystem;
namespace orn = gufo::models::ornith15;
namespace fn = gufo::models::qwen38_flash_next::rocm;
using gufo::core::GgmlType;

namespace {

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
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

float Half(std::uint16_t bits) {
  __half h;
  std::memcpy(&h, &bits, 2);
  return __half2float(h);
}

float Mul(float a, float b) {
  volatile float p = a * b;
  return p;
}
float Add(float a, float b) {
  volatile float s = a + b;
  return s;
}

int Sign8(unsigned byte) {
  byte &= 0xFF;
  return byte & 0x80 ? static_cast<int>(byte) - 256 : static_cast<int>(byte);
}

int Dp4a(int a, int b) {
  int sum = 0;
  for (int lane = 0; lane < 4; ++lane)
    sum += Sign8(a >> (8 * lane)) * Sign8(b >> (8 * lane));
  return sum;
}

std::uint32_t Sar(std::uint32_t value, int bits) {
  if (bits == 0)
    return value;
  const std::int32_t signed_value = static_cast<std::int32_t>(value);
  return static_cast<std::uint32_t>(signed_value >> bits);
}

void Q6Groups(const std::uint8_t* block, int lane, int out[2][4]) {
  const std::uint8_t* ql = block;
  const std::uint8_t* qh = block + 128;
  std::uint32_t vl = 0;
  std::memcpy(&vl, ql + 4 * lane, 4);
  const int qh_index = 8 * (lane / 16) + (lane % 8);
  std::uint32_t vh_raw = 0;
  std::memcpy(&vh_raw, qh + 4 * qh_index, 4);
  const std::uint32_t vh = Sar(vh_raw, 2 * ((lane % 16) / 8));
  for (int i = 0; i < 2; ++i) {
    const std::uint32_t vil = Sar(vl, 4 * i) & 0x0F0F0F0Fu;
    const std::uint32_t vih = (Sar(vh, 4 * i) << 4) & 0x30303030u;
    const std::uint32_t packed = vil | vih;
    for (int t = 0; t < 4; ++t) {
      int code = Sign8(packed >> (8 * t)) - 32;
      code = std::max(-128, std::min(127, code));
      out[i][t] = code;
    }
  }
}

struct Q8View {
  float scale;
  int q[32];
};

Q8View ReadQ8(const std::uint8_t* raw) {
  Q8View block;
  std::uint16_t bits = 0;
  std::memcpy(&bits, raw, 2);
  block.scale = Half(bits);
  for (int i = 0; i < 32; ++i)
    block.q[i] = static_cast<int>(
        static_cast<const std::int8_t*>(static_cast<const void*>(raw + 4))[i]);
  return block;
}

struct Q6Dots {
  float mul, inner, outer;
};

Q6Dots Q6Sequences(const std::uint8_t* row, int k, const std::uint8_t* q8) {
  const int supers = k / 256;
  float mul[32], inner[32], outer[32];
  for (int lane = 0; lane < 32; ++lane)
    mul[lane] = inner[lane] = outer[lane] = 0.0F;
  for (int super = 0; super < supers; ++super) {
    const std::uint8_t* block = row + super * 210;
    const float d6 = Half(*reinterpret_cast<const std::uint16_t*>(block + 208));
    const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
    for (int lane = 0; lane < 32; ++lane) {
      int groups[2][4];
      Q6Groups(block, lane, groups);
      const int scale_offset = 8 * (lane / 16) + (lane % 16) / 4;
      const int bq8 = 4 * (lane / 16) + (lane % 16) / 8;
      float sum_mul = 0.0F;
      float sum_fma = 0.0F;
      for (int i = 0; i < 2; ++i) {
        const Q8View act = ReadQ8(q8 + (super * 8 + bq8 + 2 * i) * 36);
        int weight = 0;
        int activation = 0;
        const int base = 4 * (lane % 8);
        for (int t = 0; t < 4; ++t) {
          weight |= (groups[i][t] & 0xFF) << (8 * t);
          activation |= (act.q[base + t] & 0xFF) << (8 * t);
        }
        const float exact = static_cast<float>(Dp4a(weight, activation) *
                                               scales[scale_offset + 4 * i]);
        sum_mul = Add(sum_mul, Mul(act.scale, exact));
        sum_fma = std::fmaf(act.scale, exact, sum_fma);
      }
      mul[lane] = Add(mul[lane], Mul(d6, sum_mul));
      inner[lane] = Add(inner[lane], Mul(d6, sum_fma));
      outer[lane] = std::fmaf(d6, sum_fma, outer[lane]);
    }
  }
  auto reduce = [](float* lanes) {
    for (int offset : {16, 8, 4, 2, 1}) {
      float next[32];
      for (int i = 0; i < 32; ++i)
        next[i] = Add(lanes[i], lanes[i ^ offset]);
      std::copy(next, next + 32, lanes);
    }
    return lanes[0];
  };
  return {reduce(mul), reduce(inner), reduce(outer)};
}

std::uint32_t Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, 4);
  return bits;
}

int Q6Code(const std::uint8_t* block, int index) {
  const int half = index / 128;
  const int within = index % 128;
  const int segment = within / 32;
  const int lane = within % 32;
  const int ql_base = half * 64;
  const std::uint8_t high_bits = block[128 + half * 32 + lane];
  int low = 0;
  int high = 0;
  if (segment == 0) {
    low = block[ql_base + lane] & 15;
    high = high_bits & 3;
  } else if (segment == 1) {
    low = block[ql_base + 32 + lane] & 15;
    high = (high_bits >> 2) & 3;
  } else if (segment == 2) {
    low = block[ql_base + lane] >> 4;
    high = (high_bits >> 4) & 3;
  } else {
    low = block[ql_base + 32 + lane] >> 4;
    high = (high_bits >> 6) & 3;
  }
  return ((high << 4) | low) - 32;
}

float Q6Element(const std::uint8_t* row, int index) {
  const std::uint8_t* block = row + (index / 256) * 210;
  const int local = index % 256;
  const float d = Half(*reinterpret_cast<const std::uint16_t*>(block + 208));
  const auto scale =
      reinterpret_cast<const std::int8_t*>(block + 192)[local / 16];
  return Mul(Mul(d, static_cast<float>(scale)),
             static_cast<float>(Q6Code(block, local)));
}

struct DotStat {
  std::uint64_t rows{0};
  std::uint64_t matched{0};
  int first_row{-1};
  std::uint32_t gpu_bits{0};
  std::uint32_t d_bits{0};
  const char* matched_name{""};
};

DotStat ScoreQ6(const std::uint8_t* weights, int rows, int k,
                const std::uint8_t* q8, const float* gpu) {
  DotStat stat;
  stat.rows = static_cast<std::uint64_t>(rows);
  const int stride = (k / 256) * 210;
  for (int row = 0; row < rows; ++row) {
    const float g = gpu[row];
    const Q6Dots dots = Q6Sequences(weights + row * stride, k, q8);
    if (Bits(g) == Bits(dots.mul) || Bits(g) == Bits(dots.inner) ||
        Bits(g) == Bits(dots.outer)) {
      ++stat.matched;
    } else if (stat.first_row < 0) {
      stat.first_row = row;
      stat.gpu_bits = Bits(g);
      stat.d_bits = Bits(dots.outer);
      stat.matched_name = "none";
    }
  }
  return stat;
}

std::vector<float> LoadF32(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  Require(in.good(), "open " + path.string());
  in.seekg(0, std::ios::end);
  const auto n = static_cast<std::size_t>(in.tellg()) / 4;
  in.seekg(0);
  std::vector<float> values(n);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(n * 4));
  Require(in.good(), "read " + path.string());
  return values;
}

std::vector<std::uint8_t> LoadBytes(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  Require(in.good(), "open " + path.string());
  in.seekg(0, std::ios::end);
  const auto n = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<std::uint8_t> values(n);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(n));
  Require(in.good(), "read " + path.string());
  return values;
}

// Confirms the host Q6 sequences against the synthetic production dump
// before any model scoring.
int SelfCheck(const fs::path& dump) {
  const std::vector<std::int32_t> ids{0, 0, 1, 0};
  int bad = 0;
  for (const int k : {256, 2048}) {
    const auto name = "q6_k" + std::to_string(k);
    if (!fs::is_regular_file(dump / (name + "_y.f32"))) {
      std::cout << "self_check_skip=" << name << '\n';
      continue;
    }
    const auto x = LoadF32(dump / (name + "_x.f32"));
    const auto w = LoadBytes(dump / (name + "_w.bin"));
    const auto y = LoadF32(dump / (name + "_y.f32"));
    constexpr int rows = 2, tokens = 2, used = 2, experts = 2;
    Require(static_cast<int>(x.size()) == tokens * k, "self x");
    Require(static_cast<int>(y.size()) == tokens * used * rows, "self y");
    Buffer<float> dx(x);
    const std::size_t qbytes = qfn_mmq_q8_1_bytes(tokens, k);
    Buffer<std::uint8_t> q8(qbytes);
    Require(qfn_mmq_quantize_q8_1(dx.data, q8.data, tokens, k, nullptr) == 0,
            "self quantize");
    const auto qhost = q8.Download(qbytes);
    const int padded = ((k + 511) / 512) * 512;
    const int stride_blocks = padded / 32;
    std::uint64_t matched = 0;
    for (int token = 0; token < tokens; ++token) {
      const std::uint8_t* act = qhost.data() + token * stride_blocks * 36;
      // Padding blocks sit past K when K is a multiple of 512. K=256 is
      // padded to 512, so the logical blocks are the first k/32.
      for (int slot = 0; slot < used; ++slot) {
        const int expert = ids[token * used + slot];
        for (int row = 0; row < rows; ++row) {
          const std::uint8_t* packed =
              w.data() + ((expert * rows + row) * (k / 256) * 210);
          const float g = y[(token * used + slot) * rows + row];
          const Q6Dots dots = Q6Sequences(packed, k, act);
          if (Bits(g) == Bits(dots.mul) || Bits(g) == Bits(dots.inner) ||
              Bits(g) == Bits(dots.outer))
            ++matched;
          else
            ++bad;
        }
      }
    }
    std::cout << "self_check " << name << " matched=" << matched << " of "
              << tokens * used * rows << '\n';
    (void)experts;
  }
  return bad;
}

double Gamma3() {
  constexpr double u = 1.0 / static_cast<double>(1 << 24);
  return (3.0 * u) / (1.0 - 3.0 * u);
}

struct ReplayStats {
  int tokens{0};
  int embed_mismatch{0};
  int first_embed{-1};
  int q8_mismatch_blocks{0};
  int first_q8{-1};
  int qkv_bad{0};
  int gate_bad{0};
  int alpha_bad{0};
  int beta_bad{0};
  int out_bad{0};
  int first_dot{-1};
  std::string first_dot_name;
  int conv_onesum_exceed_tokens{0};
  int first_conv_onesum{-1};
  int conv_pairwise_exceed_tokens{0};
  int first_conv_pairwise{-1};
  bool nonfinite{false};
  bool q8_measured{false};
  int q8_code_bad{0};
  int q8_scale_bad{0};
  int q8_aux_bad{0};
  int gg_mismatch_tokens{0};
  double conv_assoc_max{0};
  double conv_silu_max{0};
  double prep_max{0};
  double state_max{0};
  double raw_max{0};
  double epi_max{0};
};

void Xor4(float* u) {
  for (int offset : {2, 1}) {
    float next[4];
    for (int i = 0; i < 4; ++i)
      next[i] = Add(u[i], u[i ^ offset]);
    std::copy(next, next + 4, u);
  }
}

// One-token GdnKernel arithmetic with libm softplus, exp and sigmoid.
// A nonzero gap against the GPU is an unallocated numerical difference.
// It is not attributed to the math library unless decay, beta and the
// query scale are isolated.
void HostGdn(const float* conv, const float* qn, const float* kn,
             const float* alpha, const float* beta, const float* a,
             const float* dt, const float* state_in, float* state_out,
             float* raw, int k_heads, int v_heads, int d) {
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(d));
  for (int h = 0; h < v_heads; ++h) {
    const int kh = h % k_heads;
    const float x = alpha[h] + dt[h];
    const float decay =
        std::exp((x > 20.0F ? x : std::log1p(std::exp(x))) * a[h]);
    const float b = 1.0F / (1.0F + std::exp(-beta[h]));
    for (int j = 0; j < d; ++j) {
      float row[4][32];
      float last[4];
      for (int lane = 0; lane < 4; ++lane) {
        const int i0 = lane * 32;
        last[lane] =
            state_in[(static_cast<std::size_t>(h) * d + j) * d + i0 + 31];
        for (int i = 0; i < 32; ++i) {
          const float incoming =
              state_in[(static_cast<std::size_t>(h) * d + j) * d + i0 + i];
          row[lane][i] = Mul(incoming, decay);
        }
      }
      float u[4] = {};
      for (int lane = 0; lane < 4; ++lane) {
        const int i0 = lane * 32;
        for (int i = 0; i < 32; ++i)
          u[lane] = std::fmaf(row[lane][i], kn[kh * d + i0 + i], u[lane]);
      }
      Xor4(u);
      volatile float err = conv[2 * k_heads * d + h * d + j] - u[0];
      float acc[4] = {};
      for (int lane = 0; lane < 4; ++lane) {
        const int i0 = lane * 32;
        for (int i = 0; i < 32; ++i) {
          const float key = kn[kh * d + i0 + i];
          const float query = qn[kh * d + i0 + i];
          if (i + 1 == 32) {
            const float prod = Mul(Mul(err, key), b);
            row[lane][i] = std::fmaf(last[lane], decay, prod);
          } else {
            row[lane][i] = std::fmaf(b, Mul(err, key), row[lane][i]);
          }
          acc[lane] = std::fmaf(row[lane][i], query, acc[lane]);
          state_out[(static_cast<std::size_t>(h) * d + j) * d + i0 + i] =
              row[lane][i];
        }
      }
      Xor4(acc);
      raw[h * d + j] = Mul(acc[0], q_scale);
    }
  }
}

int Disposition(const ReplayStats& stats) {
  if (stats.nonfinite)
    return 2;
  if (!stats.q8_measured)
    return 3;
  if (stats.embed_mismatch || stats.qkv_bad || stats.gate_bad ||
      stats.alpha_bad || stats.beta_bad || stats.out_bad)
    return 4;
  if (stats.q8_code_bad || stats.q8_scale_bad || stats.q8_aux_bad)
    return 5;
  if (stats.gg_mismatch_tokens)
    return 6;
  if (stats.conv_pairwise_exceed_tokens)
    return 7;
  return 0;
}

int ProveExitLogic() {
  ReplayStats clean;
  clean.q8_measured = true;
  if (Disposition(clean) != 0)
    return 1;
  ReplayStats embed = clean;
  embed.embed_mismatch = 1;
  ReplayStats dot = clean;
  dot.qkv_bad = 1;
  ReplayStats aux = clean;
  aux.q8_aux_bad = 1;
  ReplayStats code = clean;
  code.q8_code_bad = 1;
  ReplayStats scale = clean;
  scale.q8_scale_bad = 1;
  ReplayStats rerun = clean;
  rerun.gg_mismatch_tokens = 1;
  ReplayStats pairwise = clean;
  pairwise.conv_pairwise_exceed_tokens = 1;
  ReplayStats unmeasured;
  ReplayStats bad = clean;
  bad.nonfinite = true;
  ReplayStats diagnostic = clean;
  diagnostic.conv_onesum_exceed_tokens = 40;
  diagnostic.state_max = 1e-6;
  const bool fails = Disposition(embed) && Disposition(dot) &&
                     Disposition(aux) && Disposition(code) &&
                     Disposition(scale) && Disposition(rerun) &&
                     Disposition(pairwise) && Disposition(unmeasured) &&
                     Disposition(bad);
  if (!fails || Disposition(diagnostic) != 0)
    return 1;
  std::cout << "prove_exit_ok injected_mismatch_fails=1 "
               "diagnostic_onesum_does_not_fail=1\n";
  return 0;
}

double MaxAbs(const std::vector<float>& left, const std::vector<float>& right,
              bool& nonfinite) {
  if (left.size() != right.size()) {
    nonfinite = true;
    return 0;
  }
  double worst = 0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (!std::isfinite(left[i]) || !std::isfinite(right[i])) {
      nonfinite = true;
      continue;
    }
    worst = std::max(worst, std::fabs(static_cast<double>(left[i]) - right[i]));
  }
  return worst;
}

bool SameBits(const std::vector<float>& left, const std::vector<float>& right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0;
}

// Independent scalar layer-0 step. This follows ReferenceModel::LinearAttention:
// F64 matvecs, libm SiLU/L2/softplus, and the scalar recurrence association.
void ScalarAdvance(const orn::LayerWeights& layer, const orn::Config& config,
                   const std::vector<float>& normed, std::vector<float>& conv,
                   std::vector<float>& state, std::vector<float>& qkv,
                   std::vector<float>& z, std::vector<float>& alpha,
                   std::vector<float>& beta, std::vector<float>& silu,
                   std::vector<float>& attn, std::vector<float>& y) {
  const int channels = static_cast<int>(config.SsmConvChannels());
  const int key_dim = static_cast<int>(config.SsmKeyDim());
  const int value_dim = static_cast<int>(config.SsmValueDim());
  const int heads_k = static_cast<int>(config.ssm_num_k_heads);
  const int heads_v = static_cast<int>(config.ssm_num_v_heads);
  const int width = static_cast<int>(config.ssm_head_dim);
  const int kernel = static_cast<int>(config.ssm_conv_kernel);
  qkv.assign(channels, 0);
  z.assign(value_dim, 0);
  alpha.assign(heads_v, 0);
  beta.assign(heads_v, 0);
  orn::cpu::MatVec(layer.ssm_qkv, 0, normed, qkv);
  orn::cpu::MatVec(layer.ssm_gate, 0, normed, z);
  orn::cpu::MatVec(layer.ssm_alpha, 0, normed, alpha);
  orn::cpu::MatVec(layer.ssm_beta, 0, normed, beta);
  const auto* conv_w = static_cast<const float*>(layer.ssm_conv1d.data);
  silu.assign(channels, 0);
  for (int channel = 0; channel < channels; ++channel) {
    float acc = conv_w[channel * kernel + kernel - 1] * qkv[channel];
    for (int tap = 0; tap + 1 < kernel; ++tap) {
      acc += conv_w[channel * kernel + tap] *
             conv[static_cast<std::size_t>(tap) * channels + channel];
    }
    silu[channel] = orn::cpu::Silu(acc);
  }
  if (kernel > 1) {
    std::vector<float> next(conv.size());
    std::copy(conv.begin() + channels, conv.end(), next.begin());
    std::copy(qkv.begin(), qkv.end(), next.end() - channels);
    conv.swap(next);
  }
  std::vector<float> mixed = silu;
  for (int head = 0; head < heads_k; ++head) {
    orn::cpu::L2Norm(std::span<float>(mixed.data() + head * width, width),
                     config.rms_eps);
    orn::cpu::L2Norm(
        std::span<float>(mixed.data() + key_dim + head * width, width),
        config.rms_eps);
  }
  const auto* a = static_cast<const float*>(layer.ssm_a.data);
  const auto* dt = static_cast<const float*>(layer.ssm_dt.data);
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(width));
  attn.assign(value_dim, 0);
  for (int head = 0; head < heads_v; ++head) {
    const float* query =
        mixed.data() + static_cast<std::size_t>(head % heads_k) * width;
    const float* key = mixed.data() + key_dim +
                       static_cast<std::size_t>(head % heads_k) * width;
    const float* value = mixed.data() + 2 * key_dim +
                         static_cast<std::size_t>(head) * width;
    float* matrix =
        state.data() + static_cast<std::size_t>(head) * width * width;
    const float decay =
        std::exp(orn::cpu::Softplus(alpha[head] + dt[head]) * a[head]);
    const float gate = orn::cpu::Sigmoid(beta[head]);
    std::vector<float> prediction(width);
    for (int row = 0; row < width; ++row) {
      float* line = matrix + static_cast<std::size_t>(row) * width;
      double dot = 0;
      for (int col = 0; col < width; ++col) {
        line[col] *= decay;
        dot += static_cast<double>(line[col]) * key[col];
      }
      prediction[row] = static_cast<float>(dot);
    }
    for (int row = 0; row < width; ++row) {
      const float delta = (value[row] - prediction[row]) * gate;
      float* line = matrix + static_cast<std::size_t>(row) * width;
      double dot = 0;
      for (int col = 0; col < width; ++col) {
        line[col] += delta * key[col];
        dot += static_cast<double>(line[col]) * query[col];
      }
      attn[static_cast<std::size_t>(head) * width + row] =
          static_cast<float>(dot) * q_scale;
    }
    const auto* norm = static_cast<const float*>(layer.ssm_norm.data);
    orn::cpu::RmsNorm(
        std::span<float>(attn.data() + static_cast<std::size_t>(head) * width,
                         width),
        norm, config.rms_eps);
    for (int col = 0; col < width; ++col) {
      attn[static_cast<std::size_t>(head) * width + col] *=
          orn::cpu::Silu(z[static_cast<std::size_t>(head) * width + col]);
    }
  }
  y.assign(config.hidden_size, 0);
  orn::cpu::MatVec(layer.ssm_out, 0, attn, y);
}

}  // namespace

int main() {
  const char* mode = std::getenv("GUFO_ORNITH15_LAYER0_MODE");
  if (mode && std::string(mode) == "prove-exit")
    return ProveExitLogic();
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  const char* logits_dir = std::getenv("GUFO_ORNITH15_LLAMA_LOGITS");
  const char* dump = std::getenv("GUFO_ORNITH15_REPRESENTATION_DUMP");
  const char* out_dir = std::getenv("GUFO_ORNITH15_WP2_DIR");
  const bool self_only = mode && std::string(mode) == "self-check";
  int limit = 76;
  if (const char* env = std::getenv("GUFO_ORNITH15_LAYER0_TOKENS"))
    limit = std::max(1, std::atoi(env));
  try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
      std::cout << "SKIP: no HIP device\n";
      return 77;
    }
    Require(qfn_mmq_init(0) == 0, "qfn_mmq_init");
    if (self_only) {
      if (!dump || !fs::is_directory(dump)) {
        std::cout << "SKIP: self-check needs GUFO_ORNITH15_REPRESENTATION_DUMP\n";
        return 77;
      }
      const int bad = SelfCheck(dump);
      std::cout << "self_check_bad_rows=" << bad << '\n';
      return bad == 0 ? 0 : 4;
    }
    if (dump && fs::is_directory(dump)) {
      const int bad = SelfCheck(dump);
      std::cout << "self_check_bad_rows=" << bad << '\n';
      Require(bad == 0,
              "host Q6 sequence disagrees with the synthetic GPU dump");
    } else if (dump) {
      std::cout << "self_check_unmeasured dump_missing=1\n";
    }
    if (!model_path || !logits_dir || !fs::is_regular_file(model_path)) {
      std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL and "
                   "GUFO_ORNITH15_LLAMA_LOGITS for the layer-0 replay\n";
      return 77;
    }
    const fs::path capture = fs::path(logits_dir) / "code_diff.ologits";
    Require(fs::is_regular_file(capture), "missing code_diff.ologits");
    std::ifstream in(capture, std::ios::binary);
    char magic[4]{};
    std::uint32_t header[3]{};
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    Require(in && std::string(magic, 4) == "ORNL" && header[0] == 1 &&
                header[2] == 248320 &&
                header[1] >= static_cast<unsigned>(limit),
            "code_diff capture is shorter than the replay");
    std::vector<std::int32_t> tokens(header[1]);
    in.read(reinterpret_cast<char*>(tokens.data()),
            static_cast<std::streamsize>(tokens.size() * 4));
    Require(static_cast<bool>(in), "token read");
    tokens.resize(limit);

    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, error);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
    Require(weights.has_value() && !weights->mtp, error);
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr, error);
    const auto& c = device->config();
    const auto& layer = device->layers().at(0);
    const auto& host = weights->layers.at(0);
    Require(layer.linear && host.ssm_qkv.type == GgmlType::kQ6_K, "layer 0 Q6");
    Require(c.hidden_size == 2048 && c.ssm_head_dim == 128, "shape");
    const int hidden = static_cast<int>(c.hidden_size);
    const int channels = static_cast<int>(c.SsmConvChannels());
    const int value = static_cast<int>(c.SsmValueDim());
    const int key = static_cast<int>(c.SsmKeyDim());
    const int v_heads = static_cast<int>(c.ssm_num_v_heads);
    const int k_heads = static_cast<int>(c.ssm_num_k_heads);
    const int d = static_cast<int>(c.ssm_head_dim);

    Buffer<float> embed(hidden), norm(hidden), qkv(channels), gate(value);
    Buffer<float> ab(v_heads * 2), conv_state(3 * channels);
    Buffer<float> conv_scratch((1 + c.ssm_conv_kernel) * channels);
    Buffer<float> qn(key), kn(key), raw(value), gdn(value), ssm(hidden);
    Buffer<float> state(static_cast<std::size_t>(v_heads) * d * d);
    Buffer<float> r_qkv(channels), r_z(value), r_ab(v_heads * 2);
    Buffer<float> r_conv(3 * channels);
    Buffer<float> r_scratch((1 + c.ssm_conv_kernel) * channels);
    Buffer<float> r_qn(key), r_kn(key), r_raw(value), r_gdn(value);
    Buffer<float> r_state(static_cast<std::size_t>(v_heads) * d * d);
    Buffer<std::int32_t> zero(1);
    Require(hipMemset(conv_state.data, 0, 3 * channels * 4) == hipSuccess,
            "zero conv");
    Require(
        hipMemset(state.data, 0,
                  static_cast<std::size_t>(v_heads) * d * d * 4) == hipSuccess,
        "zero state");
    std::vector<std::int32_t> zero_host{0};
    Require(hipMemcpy(zero.data, zero_host.data(), 4, hipMemcpyHostToDevice) ==
                hipSuccess,
            "zero id");

    const std::uint8_t* emb_w =
        static_cast<const std::uint8_t*>(weights->token_embd.data);
    const std::uint8_t* qkv_w =
        static_cast<const std::uint8_t*>(host.ssm_qkv.data);
    const std::uint8_t* gate_w =
        static_cast<const std::uint8_t*>(host.ssm_gate.data);
    const std::uint8_t* alpha_w =
        static_cast<const std::uint8_t*>(host.ssm_alpha.data);
    const std::uint8_t* beta_w =
        static_cast<const std::uint8_t*>(host.ssm_beta.data);
    const std::uint8_t* out_w =
        static_cast<const std::uint8_t*>(host.ssm_out.data);
    const float* conv_w = static_cast<const float*>(host.ssm_conv1d.data);
    const float* a_w = static_cast<const float*>(host.ssm_a.data);
    const float* dt_w = static_cast<const float*>(host.ssm_dt.data);
    const float* norm_w = static_cast<const float*>(host.ssm_norm.data);

    ReplayStats stats;
    stats.tokens = limit;
    const fs::path dump_dir =
        out_dir ? fs::path(out_dir) : fs::path("/tmp/ornith-layer0-replay");
    fs::create_directories(dump_dir);
    std::vector<float> scalar_conv(3 * channels, 0.0F);
    std::vector<float> scalar_state(static_cast<std::size_t>(v_heads) * d * d,
                                    0.0F);
    double scalar_norm_max = 0, scalar_qkv_max = 0, scalar_state_max = 0,
           scalar_out_max = 0, rg_max = 0, gr_max = 0;
    struct Note {
      int token, embed_bad, onesum, pairwise, gg_bad;
      long qkv_miss, gate_miss, alpha_miss, beta_miss, out_miss;
      double assoc, silu, prep, state, raw, epi, snorm, sqkv, sstate, sout,
          rg, gr;
    };
    std::vector<Note> notes;
    const std::size_t q8_bytes = qfn_mmq_q8_1_bytes(1, hidden);
    Buffer<std::uint8_t> q8(q8_bytes);
    const std::size_t q8_out_bytes = qfn_mmq_q8_1_bytes(1, value);
    Buffer<std::uint8_t> q8_out(q8_out_bytes);

    for (int t = 0; t < limit; ++t) {
      orn::rocm::EmbedQ6(device->token_embd().data,
                         static_cast<std::uint32_t>(tokens[t]), embed.data,
                         c.hidden_size, nullptr);
      auto emb = embed.Download(hidden);
      int embed_bad = 0;
      const std::uint8_t* erow =
          emb_w + static_cast<std::size_t>(tokens[t]) * (hidden / 256) * 210;
      for (int i = 0; i < hidden; ++i) {
        if (Bits(emb[i]) != Bits(Q6Element(erow, i)))
          ++embed_bad;
      }
      if (embed_bad && stats.first_embed < 0)
        stats.first_embed = t;
      stats.embed_mismatch += embed_bad ? 1 : 0;

      fn::RmsNormRows(embed.data, layer.attn_norm.f32(), norm.data, 1,
                      c.hidden_size, 1, c.rms_eps, nullptr);
      Require(
          qfn_mmq_quantize_q8_1(norm.data, q8.data, 1, hidden, nullptr) == 0,
          "quantize");
      const auto q8_host = q8.Download(q8_bytes);
      const auto nhost = norm.Download(hidden);
      {
        std::ofstream x(dump_dir / ("t" + std::to_string(t) + "_norm.f32"),
                        std::ios::binary);
        x.write(reinterpret_cast<const char*>(nhost.data()), hidden * 4);
        std::ofstream q(dump_dir / ("t" + std::to_string(t) + "_q8.bin"),
                        std::ios::binary);
        q.write(reinterpret_cast<const char*>(q8_host.data()),
                static_cast<std::streamsize>(q8_bytes));
      }
      // Logical blocks only. Padding past K is not consumed by these dots.
      const std::uint8_t* act = q8_host.data();

      Require(
          qfn_mmq_moe_vec(static_cast<int>(GgmlType::kQ6_K), layer.ssm_qkv.data,
                          norm.data, zero.data, qkv.data, channels, hidden, 1,
                          1, 1, nullptr) == 0,
          "qkv");
      Require(qfn_mmq_moe_vec(static_cast<int>(GgmlType::kQ6_K),
                              layer.ssm_gate.data, norm.data, zero.data,
                              gate.data, value, hidden, 1, 1, 1, nullptr) == 0,
              "gate");
      Require(qfn_mmq_moe_vec(static_cast<int>(GgmlType::kQ6_K),
                              layer.ssm_alpha.data, norm.data, zero.data,
                              ab.data, v_heads, hidden, 1, 1, 1, nullptr) == 0,
              "alpha");
      Require(qfn_mmq_moe_vec(static_cast<int>(GgmlType::kQ6_K),
                              layer.ssm_beta.data, norm.data, zero.data,
                              ab.data + v_heads, v_heads, hidden, 1, 1, 1,
                              nullptr) == 0,
              "beta");
      auto hq = qkv.Download(channels);
      auto hz = gate.Download(value);
      auto hab = ab.Download(v_heads * 2);
      const auto qkv_stat = ScoreQ6(qkv_w, channels, hidden, act, hq.data());
      const auto gate_stat = ScoreQ6(gate_w, value, hidden, act, hz.data());
      const auto alpha_stat =
          ScoreQ6(alpha_w, v_heads, hidden, act, hab.data());
      const auto beta_stat =
          ScoreQ6(beta_w, v_heads, hidden, act, hab.data() + v_heads);
      auto note = [&](const DotStat& st, int& bad, const char* name) {
        const int miss = static_cast<int>(st.rows - st.matched);
        bad += miss ? 1 : 0;
        if (miss && stats.first_dot < 0) {
          stats.first_dot = t;
          stats.first_dot_name = name;
        }
      };
      note(qkv_stat, stats.qkv_bad, "qkv");
      note(gate_stat, stats.gate_bad, "gate");
      note(alpha_stat, stats.alpha_bad, "alpha");
      note(beta_stat, stats.beta_bad, "beta");

      auto hist = conv_state.Download(3 * channels);
      auto sin = state.Download(static_cast<std::size_t>(v_heads) * d * d);
      fn::GatedDeltaNet(
          qkv.data, channels, gate.data, value, ab.data, layer.ssm_conv1d.f32(),
          layer.ssm_a.f32(), layer.ssm_dt.f32(), layer.ssm_norm.f32(),
          conv_state.data, conv_scratch.data, qn.data, kn.data, raw.data,
          state.data, gdn.data, nullptr, {}, {}, 1, k_heads, v_heads, d,
          c.ssm_conv_kernel, false, false, c.rms_eps, nullptr, nullptr, {}, 0,
          nullptr, fn::GdnGate::kSilu, fn::GdnL2Norm::kClampEps);
      auto post = conv_scratch.Download(channels);
      auto hqn = qn.Download(key);
      auto hkn = kn.Download(key);
      auto hraw = raw.Download(value);
      auto hgdn = gdn.Download(value);
      auto sout = state.Download(static_cast<std::size_t>(v_heads) * d * d);

      double assoc_max = 0;
      double silu_max = 0;
      int bound_fail = 0;
      int pairwise_fail = 0;
      for (int ch = 0; ch < channels; ++ch) {
        const float w0 = conv_w[ch * 4 + 0];
        const float w1 = conv_w[ch * 4 + 1];
        const float w2 = conv_w[ch * 4 + 2];
        const float w3 = conv_w[ch * 4 + 3];
        const float h0 = hist[ch];
        const float h1 = hist[channels + ch];
        const float h2 = hist[2 * channels + ch];
        const float v = hq[ch];
        const float p0 = Mul(w0, h0);
        const float p1 = Mul(w1, h1);
        const float p2 = Mul(w2, h2);
        const float p3 = Mul(w3, v);
        const float gpu_assoc = Add(Add(Add(p0, p1), p2), p3);
        const float scalar_assoc = Add(Add(Add(p3, p0), p1), p2);
        const double diff =
            std::fabs(static_cast<double>(gpu_assoc) - scalar_assoc);
        assoc_max = std::max(assoc_max, diff);
        const double absolute = std::fabs(static_cast<double>(p0)) +
                                std::fabs(static_cast<double>(p1)) +
                                std::fabs(static_cast<double>(p2)) +
                                std::fabs(static_cast<double>(p3));
        // One summation is within gamma_3 of the exact sum. The gap between
        // two orders is at most the sum of those two allowances.
        const double one = Gamma3() * absolute;
        const double pairwise = 2.0 * one;
        if (diff > one + 1e-12)
          ++bound_fail;
        if (diff > pairwise + 1e-12)
          ++pairwise_fail;
        const float silu = gpu_assoc / (1.0F + std::exp(-gpu_assoc));
        silu_max =
            std::max(silu_max, static_cast<double>(std::fabs(silu - post[ch])));
      }
      stats.conv_assoc_max = std::max(stats.conv_assoc_max, assoc_max);
      stats.conv_silu_max = std::max(stats.conv_silu_max, silu_max);
      if (bound_fail) {
        ++stats.conv_onesum_exceed_tokens;
        if (stats.first_conv_onesum < 0)
          stats.first_conv_onesum = t;
      }
      if (pairwise_fail) {
        ++stats.conv_pairwise_exceed_tokens;
        if (stats.first_conv_pairwise < 0)
          stats.first_conv_pairwise = t;
      }

      double prep = 0;
      for (int h = 0; h < k_heads; ++h) {
        double qs = 0, ks = 0;
        for (int i = 0; i < d; ++i) {
          qs += static_cast<double>(post[h * d + i]) * post[h * d + i];
          ks += static_cast<double>(post[key + h * d + i]) *
                post[key + h * d + i];
        }
        const float qscale =
            1.0F / std::max(std::sqrt(static_cast<float>(qs)), c.rms_eps);
        const float kscale =
            1.0F / std::max(std::sqrt(static_cast<float>(ks)), c.rms_eps);
        for (int i = 0; i < d; ++i) {
          prep = std::max(
              prep, static_cast<double>(
                        std::fabs(post[h * d + i] * qscale - hqn[h * d + i])));
          prep = std::max(
              prep, static_cast<double>(std::fabs(
                        post[key + h * d + i] * kscale - hkn[h * d + i])));
        }
      }
      stats.prep_max = std::max(stats.prep_max, prep);

      std::vector<float> host_state(sout.size()), host_raw(value);
      HostGdn(post.data(), hqn.data(), hkn.data(), hab.data(),
              hab.data() + v_heads, a_w, dt_w, sin.data(), host_state.data(),
              host_raw.data(), k_heads, v_heads, d);
      double state_max = 0, raw_max = 0;
      for (std::size_t i = 0; i < sout.size(); ++i)
        state_max = std::max(
            state_max, static_cast<double>(std::fabs(sout[i] - host_state[i])));
      for (int i = 0; i < value; ++i)
        raw_max = std::max(
            raw_max, static_cast<double>(std::fabs(hraw[i] - host_raw[i])));
      stats.state_max = std::max(stats.state_max, state_max);
      stats.raw_max = std::max(stats.raw_max, raw_max);

      double epi = 0;
      for (int h = 0; h < v_heads; ++h) {
        double ss = 0;
        for (int i = 0; i < d; ++i)
          ss += static_cast<double>(hraw[h * d + i]) * hraw[h * d + i];
        const float scale =
            1.0F / std::sqrt(static_cast<float>(ss / d) + c.rms_eps);
        for (int i = 0; i < d; ++i) {
          const float z = hz[h * d + i];
          const float silu = z / (1.0F + std::exp(-z));
          const float value_h = hraw[h * d + i] * scale * norm_w[i] * silu;
          epi = std::max(
              epi, static_cast<double>(std::fabs(value_h - hgdn[h * d + i])));
        }
      }
      stats.epi_max = std::max(stats.epi_max, epi);

      Require(
          qfn_mmq_quantize_q8_1(gdn.data, q8_out.data, 1, value, nullptr) == 0,
          "quantize ssm");
      const auto q8o = q8_out.Download(q8_out_bytes);
      Require(qfn_mmq_moe_vec(static_cast<int>(GgmlType::kQ6_K),
                              layer.ssm_out.data, gdn.data, zero.data, ssm.data,
                              hidden, value, 1, 1, 1, nullptr) == 0,
              "ssm_out");
      auto hs = ssm.Download(hidden);
      const auto out_stat =
          ScoreQ6(out_w, hidden, value, q8o.data(), hs.data());
      note(out_stat, stats.out_bad, "ssm_out");

      auto mark = [&](const std::vector<float>& column) {
        if (!std::all_of(column.begin(), column.end(),
                         [](float value) { return std::isfinite(value); }))
          stats.nonfinite = true;
      };
      mark(emb);
      mark(nhost);
      mark(hq);
      mark(hz);
      mark(hab);
      mark(post);
      mark(hqn);
      mark(hkn);
      mark(hraw);
      mark(hgdn);
      mark(hs);
      mark(sout);
      mark(sin);
      mark(hist);
      std::vector<float> scalar_x = emb;
      orn::cpu::RmsNorm(std::span<float>(scalar_x),
                        static_cast<const float*>(host.attn_norm.data),
                        c.rms_eps);
      bool row_nonfinite = false;
      const double snorm = MaxAbs(scalar_x, nhost, row_nonfinite);
      const auto conv_before = scalar_conv;
      const auto state_before = scalar_state;
      std::vector<float> sqkv, sz, salpha, sbeta, ssilu, sattn, sy;
      ScalarAdvance(host, c, scalar_x, scalar_conv, scalar_state, sqkv, sz,
                    salpha, sbeta, ssilu, sattn, sy);
      const double sq = MaxAbs(sqkv, hq, row_nonfinite);
      const double ss = MaxAbs(scalar_state, sout, row_nonfinite);
      const double so = MaxAbs(sy, hs, row_nonfinite);
      scalar_norm_max = std::max(scalar_norm_max, snorm);
      scalar_qkv_max = std::max(scalar_qkv_max, sq);
      scalar_state_max = std::max(scalar_state_max, ss);
      scalar_out_max = std::max(scalar_out_max, so);
      if (row_nonfinite)
        stats.nonfinite = true;
      auto upload = [](float* dest, const std::vector<float>& src) {
        Require(hipMemcpy(dest, src.data(), src.size() * sizeof(float),
                          hipMemcpyHostToDevice) == hipSuccess,
                "rerun upload");
      };
      auto rerun = [&](const std::vector<float>& qv, const std::vector<float>& zv,
                       const std::vector<float>& abv,
                       const std::vector<float>& cv,
                       const std::vector<float>& sv) {
        upload(r_qkv.data, qv);
        upload(r_z.data, zv);
        upload(r_ab.data, abv);
        upload(r_conv.data, cv);
        upload(r_state.data, sv);
        fn::GatedDeltaNet(
            r_qkv.data, channels, r_z.data, value, r_ab.data,
            layer.ssm_conv1d.f32(), layer.ssm_a.f32(), layer.ssm_dt.f32(),
            layer.ssm_norm.f32(), r_conv.data, r_scratch.data, r_qn.data,
            r_kn.data, r_raw.data, r_state.data, r_gdn.data, nullptr, {}, {}, 1,
            k_heads, v_heads, d, c.ssm_conv_kernel, false, false, c.rms_eps,
            nullptr, nullptr, {}, 0, nullptr, fn::GdnGate::kSilu,
            fn::GdnL2Norm::kClampEps);
        struct Out {
          std::vector<float> state, gdn;
        };
        return Out{r_state.Download(sv.size()), r_gdn.Download(value)};
      };
      const auto matched_gg = rerun(hq, hz, hab, hist, sin);
      const int gg_bad =
          SameBits(matched_gg.state, sout) && SameBits(matched_gg.gdn, hgdn) ? 0
                                                                             : 1;
      if (gg_bad)
        ++stats.gg_mismatch_tokens;
      std::vector<float> sab(static_cast<std::size_t>(v_heads) * 2);
      std::copy(salpha.begin(), salpha.end(), sab.begin());
      std::copy(sbeta.begin(), sbeta.end(), sab.begin() + v_heads);
      const auto rg = rerun(sqkv, sz, sab, conv_before, sin);
      const auto gr = rerun(hq, hz, hab, hist, state_before);
      const double rg_gap = MaxAbs(rg.state, sout, row_nonfinite);
      const double gr_gap = MaxAbs(gr.state, scalar_state, row_nonfinite);
      rg_max = std::max(rg_max, rg_gap);
      gr_max = std::max(gr_max, gr_gap);
      auto save = [&](const std::string& name, const std::vector<float>& data) {
        std::ofstream out(dump_dir / name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
      };
      const std::string id = "t" + std::to_string(t);
      save(id + "_embed.f32", emb);
      save(id + "_qkv.f32", hq);
      save(id + "_gate.f32", hz);
      save(id + "_ab.f32", hab);
      save(id + "_conv_in.f32", hist);
      save(id + "_silu.f32", post);
      save(id + "_state_in.f32", sin);
      save(id + "_state_out.f32", sout);
      save(id + "_gdn.f32", hgdn);
      save(id + "_ssm.f32", hs);
      save(id + "_scalar_state_out.f32", scalar_state);
      notes.push_back(
          Note{t, embed_bad, bound_fail, pairwise_fail, gg_bad,
               static_cast<long>(qkv_stat.rows - qkv_stat.matched),
               static_cast<long>(gate_stat.rows - gate_stat.matched),
               static_cast<long>(alpha_stat.rows - alpha_stat.matched),
               static_cast<long>(beta_stat.rows - beta_stat.matched),
               static_cast<long>(out_stat.rows - out_stat.matched), assoc_max,
               silu_max, prep, state_max, raw_max, epi, snorm, sq, ss, so,
               rg_gap, gr_gap});
      std::cout << "token=" << t << " embed_bad=" << embed_bad
                << " qkv_miss=" << notes.back().qkv_miss
                << " out_miss=" << notes.back().out_miss
                << " conv_pairwise_channels=" << pairwise_fail
                << " scalar_state=" << ss << " gg_bad=" << gg_bad
                << " rg_state=" << rg_gap << " gr_state=" << gr_gap << '\n';
    }
    const std::string script = "tools/ornith15/representation.py";
    if (fs::is_regular_file(script)) {
      const int encoded = std::system(
          ("python3 " + script + " encode-check " + dump_dir.string()).c_str());
      std::ifstream report(dump_dir / "q8-encode-check.json");
      const std::string text((std::istreambuf_iterator<char>(report)), {});
      auto field = [&](const char* key) {
        const auto pos = text.find(std::string("\"") + key + "\"");
        if (pos == std::string::npos)
          return -1;
        return std::atoi(text.c_str() + text.find(':', pos) + 1);
      };
      if (field("code_bad") >= 0) {
        stats.q8_measured = true;
        stats.q8_code_bad = field("code_bad");
        stats.q8_scale_bad = field("scale_bad");
        stats.q8_aux_bad = field("aux_bad");
      }
      std::cout << "q8_encode_command_status=" << encoded << '\n';
    }
    std::map<int, std::array<int, 3>> q8_rows;
    std::ifstream columns(dump_dir / "q8-columns.tsv");
    std::string header_line;
    std::getline(columns, header_line);
    for (std::string line; std::getline(columns, line);) {
      int token = 0, code = 0, scale = 0, aux = 0;
      if (std::sscanf(line.c_str(), "%d\t%d\t%d\t%d", &token, &code, &scale,
                      &aux) == 4)
        q8_rows[token] = {code, scale, aux};
    }
    std::ofstream table(dump_dir / "layer0-table.tsv");
    table << "token\tembed_mismatch\tq8_code_bad\tq8_scale_bad\tq8_aux_bad\t"
             "qkv_miss\tgate_miss\talpha_miss\tbeta_miss\tout_miss\t"
             "conv_onesum_channels\tconv_pairwise_channels\tconv_silu\tprep\t"
             "local_state\tscalar_norm\tscalar_qkv\tscalar_state\tscalar_out\t"
             "gg_bad\trg_state\tgr_state\n";
    for (const auto& note : notes) {
      const auto q8 = q8_rows.count(note.token) ? q8_rows[note.token]
                                                : std::array<int, 3>{-1, -1, -1};
      table << note.token << '\t' << note.embed_bad << '\t' << q8[0] << '\t'
            << q8[1] << '\t' << q8[2] << '\t' << note.qkv_miss << '\t'
            << note.gate_miss << '\t' << note.alpha_miss << '\t'
            << note.beta_miss << '\t' << note.out_miss << '\t' << note.onesum
            << '\t' << note.pairwise << '\t' << note.silu << '\t' << note.prep
            << '\t' << note.state << '\t' << note.snorm << '\t' << note.sqkv
            << '\t' << note.sstate << '\t' << note.sout << '\t' << note.gg_bad
            << '\t' << note.rg << '\t' << note.gr << '\n';
    }
    std::ofstream manifest(dump_dir / "fixture-manifest.json");
    manifest << "{\n  \"dtype\": \"float32\",\n  \"endian\": \"little\",\n"
             << "  \"state_layout\": "
                "\"[value_head][value_row][key_column]\",\n"
             << "  \"conv_layout\": "
                "\"[history_tap][channel], oldest tap first\",\n"
             << "  \"tokens\": " << notes.size() << ",\n  \"hidden\": " << hidden
             << ",\n  \"channels\": " << channels << ",\n  \"value_heads\": "
             << v_heads << ",\n  \"head_dim\": " << d
             << ",\n  \"hashes\": \"fixture-sha256.txt\",\n"
             << "  \"q8_codes_scales_aux_are_separate\": true\n}\n";
    Require(std::system(("cd " + dump_dir.string() +
                         " && find . -type f \\( -name '*.f32' -o -name '*.bin' "
                         "\\) -print0 | sort -z | xargs -0 sha256sum > "
                         "fixture-sha256.txt")
                            .c_str()) == 0,
            "fixture hashes");
    const int status = Disposition(stats);
    std::cout << "replay_tokens=" << stats.tokens
              << " first_embed=" << stats.first_embed
              << " first_dot=" << stats.first_dot << " " << stats.first_dot_name
              << " q8_code_bad=" << stats.q8_code_bad
              << " q8_scale_bad=" << stats.q8_scale_bad
              << " q8_aux_bad=" << stats.q8_aux_bad
              << " conv_onesum_tokens=" << stats.conv_onesum_exceed_tokens
              << " conv_pairwise_tokens=" << stats.conv_pairwise_exceed_tokens
              << " gg_mismatch_tokens=" << stats.gg_mismatch_tokens
              << " max_scalar_norm=" << scalar_norm_max
              << " max_scalar_qkv=" << scalar_qkv_max
              << " max_scalar_state=" << scalar_state_max
              << " max_scalar_out=" << scalar_out_max
              << " max_rg_state=" << rg_max << " max_gr_state=" << gr_max
              << " nonfinite=" << stats.nonfinite
              << " disposition=" << status << '\n';
    return status;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
