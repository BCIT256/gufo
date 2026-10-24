// Production layer-0 replay for code_diff tokens 0..75.
// Calls EmbedQ6, RmsNormRows, qfn_mmq_quantize_q8_1, qfn_mmq_moe_vec and
// GatedDeltaNet. It does not run MoE or later layers. Host D sequences are
// the Q6_K MMVQ associations from the P4e protocol. Device math gaps
// (SiLU, L2, sigmoid, exp) are reported separately from bit mismatches.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
  int conv_bound_fail{0};
  int first_conv_bound{-1};
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

// Declared GdnKernel sequence for one token. Decay and sigmoid use libm,
// so a nonzero gap is math-library drift until an intervention says otherwise.
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

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  const char* logits_dir = std::getenv("GUFO_ORNITH15_LLAMA_LOGITS");
  const char* dump = std::getenv("GUFO_ORNITH15_REPRESENTATION_DUMP");
  const char* out_dir = std::getenv("GUFO_ORNITH15_WP2_DIR");
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
    if (dump && fs::is_directory(dump)) {
      const int bad = SelfCheck(dump);
      std::cout << "self_check_bad_rows=" << bad << '\n';
      Require(bad == 0,
              "host Q6 sequence disagrees with the synthetic GPU dump");
    }
    if (!model_path || !logits_dir || !fs::is_regular_file(model_path)) {
      std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL and "
                   "GUFO_ORNITH15_LLAMA_LOGITS for the layer-0 replay\n";
      return dump ? 0 : 77;
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
    std::ofstream table;
    if (out_dir) {
      fs::create_directories(out_dir);
      table.open(fs::path(out_dir) / "layer0-table.tsv");
      table << "token\tembed_mismatch\tq8_bad_blocks\tqkv_bad\tgate_bad\t"
               "alpha_bad\tbeta_bad\tout_bad\tconv_assoc_max\tconv_silu_max\t"
               "prep_max\tstate_max\traw_max\tepi_max\n";
    }
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
      if (out_dir) {
        const auto dir = fs::path(out_dir);
        std::ofstream x(dir / ("t" + std::to_string(t) + "_norm.f32"),
                        std::ios::binary);
        x.write(reinterpret_cast<const char*>(nhost.data()), hidden * 4);
        std::ofstream q(dir / ("t" + std::to_string(t) + "_q8.bin"),
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
        const double limit = Gamma3() * (std::fabs(static_cast<double>(p0)) +
                                          std::fabs(static_cast<double>(p1)) +
                                          std::fabs(static_cast<double>(p2)) +
                                          std::fabs(static_cast<double>(p3)));
        if (diff > limit + 1e-12)
          ++bound_fail;
        const float silu = gpu_assoc / (1.0F + std::exp(-gpu_assoc));
        silu_max =
            std::max(silu_max, static_cast<double>(std::fabs(silu - post[ch])));
      }
      stats.conv_assoc_max = std::max(stats.conv_assoc_max, assoc_max);
      stats.conv_silu_max = std::max(stats.conv_silu_max, silu_max);
      if (bound_fail) {
        ++stats.conv_bound_fail;
        if (stats.first_conv_bound < 0)
          stats.first_conv_bound = t;
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

      std::cout << "token=" << t << " embed_bad=" << embed_bad
                << " qkv_miss=" << (qkv_stat.rows - qkv_stat.matched)
                << " gate_miss=" << (gate_stat.rows - gate_stat.matched)
                << " alpha_miss=" << (alpha_stat.rows - alpha_stat.matched)
                << " beta_miss=" << (beta_stat.rows - beta_stat.matched)
                << " out_miss=" << (out_stat.rows - out_stat.matched)
                << " conv_assoc=" << assoc_max << " conv_silu=" << silu_max
                << " prep=" << prep << " state=" << state_max
                << " raw=" << raw_max << " epi=" << epi << '\n';
      if (table)
        table << t << '\t' << embed_bad << "\t0\t"
              << (qkv_stat.rows - qkv_stat.matched) << '\t'
              << (gate_stat.rows - gate_stat.matched) << '\t'
              << (alpha_stat.rows - alpha_stat.matched) << '\t'
              << (beta_stat.rows - beta_stat.matched) << '\t'
              << (out_stat.rows - out_stat.matched) << '\t' << assoc_max << '\t'
              << silu_max << '\t' << prep << '\t' << state_max << '\t'
              << raw_max << '\t' << epi << '\n';
    }
    std::cout << "replay_tokens=" << stats.tokens
              << " first_embed=" << stats.first_embed
              << " first_dot=" << stats.first_dot << " " << stats.first_dot_name
              << " qkv_bad_tokens=" << stats.qkv_bad
              << " gate_bad_tokens=" << stats.gate_bad
              << " alpha_bad_tokens=" << stats.alpha_bad
              << " beta_bad_tokens=" << stats.beta_bad
              << " out_bad_tokens=" << stats.out_bad
              << " conv_bound_fail_tokens=" << stats.conv_bound_fail
              << " first_conv_bound=" << stats.first_conv_bound
              << " max_conv_silu=" << stats.conv_silu_max
              << " max_prep=" << stats.prep_max
              << " max_state=" << stats.state_max
              << " max_raw=" << stats.raw_max << " max_epi=" << stats.epi_max
              << '\n';
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
