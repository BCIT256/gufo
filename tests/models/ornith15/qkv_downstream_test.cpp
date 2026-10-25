// Registered downstream measurement for code_diff positions 0..75.
// Three separate sessions: production B, production-QKV reinsertion, and
// an F64 ssm_qkv matvec on the GPU RMSNorm output. A null probe is the
// production path. This file is not a precision policy.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/ornith15/cpu_ops.hpp"
#include "src/models/ornith15/kernels/rocm/executor.hpp"
#include "src/models/ornith15/reference.hpp"

namespace orn = gufo::models::ornith15;
namespace fs = std::filesystem;
namespace {

constexpr int kTokens = 76;
constexpr int kPosition = 75;
constexpr double kClearMargin = 1.0;
constexpr double kConfidentEntropy = 0.5;
constexpr double kConfidentKl = 0.1;
constexpr double kRegisteredKl = 0.1220301624;

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

struct Capture {
  std::uint32_t vocab{0};
  std::vector<std::int32_t> tokens;
  std::vector<float> logits;
  const float* Row(std::size_t t) const { return logits.data() + t * vocab; }
};

Capture Load(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[4]{};
  std::uint32_t header[3]{};
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  Require(in && std::string(magic, 4) == "ORNL" && header[0] == 1 &&
              header[1] >= static_cast<std::uint32_t>(kTokens) &&
              header[2] == 248320,
          "invalid capture header: " + path.string());
  Capture c{header[2], std::vector<std::int32_t>(header[1]),
            std::vector<float>(std::size_t(header[1]) * header[2])};
  in.read(reinterpret_cast<char*>(c.tokens.data()),
          static_cast<std::streamsize>(c.tokens.size() * sizeof(std::int32_t)));
  in.read(reinterpret_cast<char*>(c.logits.data()),
          static_cast<std::streamsize>(c.logits.size() * sizeof(float)));
  Require(in && in.peek() == EOF, "truncated/oversized capture");
  return c;
}

std::vector<double> LogSoftmax(const float* x, std::size_t n) {
  double peak = -INFINITY;
  for (std::size_t i = 0; i < n; ++i) {
    Require(std::isfinite(x[i]), "nonfinite logits");
    peak = std::max(peak, static_cast<double>(x[i]));
  }
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += std::exp(static_cast<double>(x[i]) - peak);
  const double log_sum = peak + std::log(sum);
  std::vector<double> out(n);
  for (std::size_t i = 0; i < n; ++i)
    out[i] = x[i] - log_sum;
  return out;
}

double Kl(const std::vector<double>& p, const std::vector<double>& q) {
  double kl = 0;
  for (std::size_t i = 0; i < p.size(); ++i)
    kl += std::exp(p[i]) * (p[i] - q[i]);
  Require(std::isfinite(kl), "nonfinite KL");
  return kl;
}

double Entropy(const std::vector<double>& p) {
  double entropy = 0;
  for (double x : p)
    entropy -= std::exp(x) * x;
  return entropy;
}

std::size_t Top(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);
}

double Margin(const float* x, std::size_t n, std::size_t top) {
  float second = -INFINITY;
  for (std::size_t i = 0; i < n; ++i)
    if (i != top)
      second = std::max(second, x[i]);
  return static_cast<double>(x[top]) - second;
}

double MaxAbs(const float* left, const float* right, std::size_t n) {
  double worst = 0;
  for (std::size_t i = 0; i < n; ++i) {
    Require(std::isfinite(left[i]) && std::isfinite(right[i]), "nonfinite");
    worst = std::max(worst, std::abs(static_cast<double>(left[i]) - right[i]));
  }
  return worst;
}

bool SameBytes(const void* left, const void* right, std::size_t bytes) {
  return std::memcmp(left, right, bytes) == 0;
}

void Reinsert(void*, float* qkv, const float*, std::uint32_t channels,
              std::uint32_t, hipStream_t stream) {
  std::vector<float> host(channels);
  Require(hipMemcpyAsync(host.data(), qkv, channels * sizeof(float),
                         hipMemcpyDeviceToHost, stream) == hipSuccess,
          "reinsert download");
  Require(hipStreamSynchronize(stream) == hipSuccess, "reinsert download sync");
  Require(hipMemcpyAsync(qkv, host.data(), channels * sizeof(float),
                         hipMemcpyHostToDevice, stream) == hipSuccess,
          "reinsert upload");
  Require(hipStreamSynchronize(stream) == hipSuccess, "reinsert upload sync");
}

void SubstituteF64(void* ctx, float* qkv, const float* norm,
                   std::uint32_t channels, std::uint32_t hidden,
                   hipStream_t stream) {
  const auto* weight = static_cast<const orn::TensorRef*>(ctx);
  std::vector<float> row(hidden), y(channels, 0.0F);
  Require(hipMemcpyAsync(row.data(), norm, hidden * sizeof(float),
                         hipMemcpyDeviceToHost, stream) == hipSuccess,
          "substitute norm");
  Require(hipStreamSynchronize(stream) == hipSuccess, "substitute norm sync");
  orn::cpu::MatVec(*weight, 0, row, y);
  Require(hipMemcpyAsync(qkv, y.data(), channels * sizeof(float),
                         hipMemcpyHostToDevice, stream) == hipSuccess,
          "substitute upload");
  Require(hipStreamSynchronize(stream) == hipSuccess, "substitute upload sync");
}

struct ArmRow {
  std::vector<float> logits, conv, state, qkv_before, qkv_out;
  std::vector<std::int32_t> ids;
  std::vector<float> weights, shared;
  orn::LayerTrace residual;
};

struct ModelShape {
  std::uint32_t hidden, channels, layers, used, vocab;
  std::size_t conv, state;
};

ModelShape ShapeOf(const orn::Config& c) {
  Require(c.num_layers == 40 && c.IsLinearLayer(0) && c.vocab_size == 248320 &&
              c.hidden_size == 2048 && c.num_experts_used == 8,
          "unexpected ornith shape");
  ModelShape s;
  s.hidden = c.hidden_size;
  s.channels = static_cast<std::uint32_t>(c.SsmConvChannels());
  s.layers = c.num_layers;
  s.used = c.num_experts_used;
  s.vocab = c.vocab_size;
  s.conv = std::size_t(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  s.state = std::size_t(c.ssm_num_v_heads) * c.ssm_head_dim * c.ssm_head_dim;
  return s;
}

using EditFn = void (*)(void*, float*, const float*, std::uint32_t, std::uint32_t,
                       hipStream_t);

void RunArm(const orn::rocm::DeviceModel& device, const ModelShape& shape,
            const std::vector<std::int32_t>& tokens, const char* name,
            EditFn edit, void* ctx, std::vector<ArmRow>& rows) {
  std::string error;
  auto session = orn::rocm::Session::Create(device, kTokens, &error);
  Require(session != nullptr, std::string(name) + " session: " + error);
  std::vector<float> logits(shape.vocab), norm(shape.hidden),
      qkv_before(shape.channels), qkv_out(shape.channels), conv(shape.conv),
      state(shape.state), weights(shape.layers * shape.used),
      shared(shape.layers);
  std::vector<std::int32_t> ids(shape.layers * shape.used);
  orn::LayerTrace trace;
  rows.resize(kTokens);
  for (int t = 0; t < kTokens; ++t) {
    orn::rocm::Layer0Probe probe;
    probe.edit_qkv = edit;
    probe.edit_ctx = ctx;
    probe.norm_out = norm.data();
    probe.qkv_before = qkv_before.data();
    probe.qkv_out = qkv_out.data();
    probe.conv_out = conv.data();
    probe.state_out = state.data();
    probe.route_ids = ids.data();
    probe.route_weights = weights.data();
    probe.shared_gate = shared.data();
    Require(session->Step(tokens[t], logits, &error, &trace, &probe),
            std::string(name) + " step: " + error);
    if (edit != SubstituteF64)
      Require(SameBytes(qkv_before.data(), qkv_out.data(),
                        qkv_before.size() * sizeof(float)),
              std::string(name) + " qkv copy changed bits");
    ArmRow row;
    row.logits = logits;
    row.conv = conv;
    row.state = state;
    row.qkv_before = qkv_before;
    row.qkv_out = qkv_out;
    row.ids = ids;
    row.weights = weights;
    row.shared = shared;
    row.residual = trace;
    rows[t] = std::move(row);
    std::cout << "arm=" << name << " token=" << t << " done" << std::endl;
  }
}

struct Score {
  double ref_kl, llama_kl, entropy, margin;
  std::size_t top_ref, top_gpu, top_llama, top_sequential;
  bool clear_flip, confident_failure;
};

Score ScoreRow(const float* gpu, const float* ref, const float* llama,
               const float* sequential, std::size_t vocab) {
  const auto lp = LogSoftmax(ref, vocab);
  const auto lq = LogSoftmax(gpu, vocab);
  const auto ll = LogSoftmax(llama, vocab);
  Score s;
  s.ref_kl = Kl(lp, lq);
  s.llama_kl = Kl(ll, lq);
  s.entropy = Entropy(lp);
  s.top_ref = Top(ref, vocab);
  s.top_gpu = Top(gpu, vocab);
  s.top_llama = Top(llama, vocab);
  s.top_sequential = Top(sequential, vocab);
  s.margin = Margin(ref, vocab, s.top_ref);
  const bool margin_flip = s.top_ref != s.top_gpu && s.margin > kClearMargin;
  s.clear_flip = margin_flip && s.top_llama == s.top_ref &&
                 s.top_sequential == s.top_ref;
  s.confident_failure = s.entropy < kConfidentEntropy && s.ref_kl > kConfidentKl;
  return s;
}

const char* Direction(double value, double baseline) {
  if (value < baseline)
    return "improves";
  if (value > baseline)
    return "worsens";
  return "unchanged";
}

int IdentityCheck(const char* model_path, const char* logits_dir,
                  const char* dump_path) {
  std::cout << "mode=identity unprobed_vs_instrumented_vs_reinsert\n";
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  Require(reader != nullptr, "open: " + error);
  const auto weights =
      orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
  Require(weights.has_value() && !weights->mtp, "bind trunk: " + error);
  const auto device = orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
  Require(device != nullptr, "upload: " + error);
  const ModelShape shape = ShapeOf(device->config());
  const auto batch = Load(fs::path(logits_dir) / "code_diff.ologits");
  std::vector<std::int32_t> tokens(batch.tokens.begin(),
                                   batch.tokens.begin() + kTokens);
  std::ifstream in(dump_path, std::ios::binary);
  std::vector<float> unprobed(std::size_t(kTokens) * shape.vocab);
  in.read(reinterpret_cast<char*>(unprobed.data()),
          static_cast<std::streamsize>(unprobed.size() * sizeof(float)));
  Require(static_cast<bool>(in), "unprobed logits short");
  std::vector<ArmRow> instrumented, reinsert;
  RunArm(*device, shape, tokens, "instrumented_baseline", nullptr, nullptr,
         instrumented);
  RunArm(*device, shape, tokens, "reinsert", Reinsert, nullptr, reinsert);
  int bad_instrumented = 0, bad_reinsert = 0;
  for (int t = 0; t < kTokens; ++t) {
    const float* base = unprobed.data() + std::size_t(t) * shape.vocab;
    if (!SameBytes(base, instrumented[t].logits.data(),
                   shape.vocab * sizeof(float)))
      ++bad_instrumented;
    if (!SameBytes(base, reinsert[t].logits.data(),
                   shape.vocab * sizeof(float)))
      ++bad_reinsert;
  }
  std::cout << "unprobed_vs_instrumented_mismatch_tokens=" << bad_instrumented
            << " unprobed_vs_reinsert_mismatch_tokens=" << bad_reinsert
            << std::endl;
  if (bad_instrumented || bad_reinsert)
    return 2;
  std::cout << "identity_byte_identical=1\n";
  return 0;
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  const char* logits_dir = std::getenv("GUFO_ORNITH15_LLAMA_LOGITS");
  const char* out_env = std::getenv("GUFO_ORNITH15_WP5_DIR");
  const char* mode = std::getenv("GUFO_ORNITH15_QKV_MODE");
  if (mode && std::string(mode) == "identity") {
    const char* dump = std::getenv("GUFO_ORNITH15_UNPROBED_LOGITS");
    if (!model_path || !logits_dir || !dump || !fs::is_regular_file(model_path) ||
        !fs::is_directory(logits_dir) || !fs::is_regular_file(dump)) {
      std::cout << "SKIP: identity mode needs the model, captures and "
                   "GUFO_ORNITH15_UNPROBED_LOGITS\n";
      return 77;
    }
    try {
      std::cout << std::setprecision(17);
      return IdentityCheck(model_path, logits_dir, dump);
    } catch (const std::exception& ex) {
      std::cerr << ex.what() << '\n';
      return 1;
    }
  }
  if (!model_path || !logits_dir || !fs::is_regular_file(model_path) ||
      !fs::is_directory(logits_dir)) {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL and "
                 "GUFO_ORNITH15_LLAMA_LOGITS\n";
    return 77;
  }
  try {
    std::cout << std::setprecision(17);
    std::cout << "experiment=layer0_qkv_downstream\n"
                 "arms=production,reinsert,f64_qkv_on_gpu_norm\n"
                 "prefix=code_diff tokens 0-75 inclusive\n"
                 "substitution=f64_matvec_of_ssm_qkv_on_gpu_rms_norm\n"
                 "control_rule=reinsert_logits_and_layer0_state_byte_identical\n"
                 "prediction=no_magnitude_forecast\n"
                 "registered_position75_kl=0.1220301624 limit=0.1\n"
                 "decision=improve_worsen_or_unchanged; influence is not "
                 "correction\n";
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, "open: " + error);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
    Require(weights.has_value() && !weights->mtp, "bind trunk: " + error);
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr, "upload: " + error);
    const ModelShape shape = ShapeOf(device->config());
    const auto batch = Load(fs::path(logits_dir) / "code_diff.ologits");
    const auto sequential =
        Load(fs::path(logits_dir) / "code_diff.sequential.ologits");
    Require(batch.tokens == sequential.tokens && batch.vocab == shape.vocab,
            "capture identity");
    std::vector<std::int32_t> tokens(batch.tokens.begin(),
                                     batch.tokens.begin() + kTokens);
    std::vector<float> reference(std::size_t(kTokens) * shape.vocab);
    {
      orn::ReferenceModel ref(*weights, kTokens);
      std::vector<float> row(shape.vocab);
      for (int t = 0; t < kTokens; ++t) {
        Require(ref.Step(tokens[t], row, &error), "reference: " + error);
        std::copy(row.begin(), row.end(),
                  reference.begin() + std::size_t(t) * shape.vocab);
        std::cout << "arm=scalar_reference token=" << t << " done" << std::endl;
      }
    }
    std::vector<ArmRow> production, control, substituted;
    RunArm(*device, shape, tokens, "production", nullptr, nullptr, production);
    RunArm(*device, shape, tokens, "reinsert", Reinsert, nullptr, control);
    bool control_ok = true;
    for (int t = 0; t < kTokens && control_ok; ++t) {
      const bool logits = SameBytes(production[t].logits.data(),
                                   control[t].logits.data(),
                                   production[t].logits.size() * sizeof(float));
      const bool conv =
          SameBytes(production[t].conv.data(), control[t].conv.data(),
                    production[t].conv.size() * sizeof(float));
      const bool state =
          SameBytes(production[t].state.data(), control[t].state.data(),
                    production[t].state.size() * sizeof(float));
      const bool ids =
          SameBytes(production[t].ids.data(), control[t].ids.data(),
                    production[t].ids.size() * sizeof(std::int32_t));
      const bool weights_ok =
          SameBytes(production[t].weights.data(), control[t].weights.data(),
                    production[t].weights.size() * sizeof(float));
      const bool shared =
          SameBytes(production[t].shared.data(), control[t].shared.data(),
                    production[t].shared.size() * sizeof(float));
      bool residual = production[t].residual.size() == shape.layers &&
                      control[t].residual.size() == shape.layers;
      for (std::size_t layer = 0; residual && layer < shape.layers; ++layer)
        residual = SameBytes(production[t].residual[layer].data(),
                             control[t].residual[layer].data(),
                             shape.hidden * sizeof(float));
      if (!logits || !conv || !state || !ids || !weights_ok || !shared ||
          !residual) {
        control_ok = false;
        std::cout << "control_mismatch token=" << t << " logits=" << logits
                  << " conv=" << conv << " state=" << state << " route_ids="
                  << ids << " route_weights=" << weights_ok
                  << " shared_gate=" << shared << " residual=" << residual
                  << std::endl;
      }
    }
    std::cout << "control_byte_identical=" << control_ok << std::endl;
    if (!control_ok) {
      std::cout << "control_failed substitute_not_interpreted disposition=2\n";
      return 2;
    }
    RunArm(*device, shape, tokens, "f64_qkv_on_gpu_norm", SubstituteF64,
           const_cast<orn::TensorRef*>(&weights->layers.at(0).ssm_qkv),
           substituted);
    const fs::path out_dir =
        out_env ? fs::path(out_env) : fs::path("/tmp/ornith-wp5-qkv");
    fs::create_directories(out_dir);
    std::ofstream table(out_dir / "prefix-metrics.tsv");
    table << std::setprecision(17);
    table << "token\tarm\tref_kl\tllama_kl\tentropy\tmargin\ttop_ref\ttop_gpu\t"
             "top_llama\ttop_sequential\tclear_flip\tconfident_failure\t"
             "qkv_max_abs\tlayer0_conv_max_abs\tlayer0_state_max_abs\t"
             "route_ids_differ\troute_weights_max_abs\tshared_gate_max_abs\t"
             "residual_first_layer\tresidual_max_abs\n";
    auto emit = [&](int t, const char* arm, const ArmRow& row) {
      const Score score =
          ScoreRow(row.logits.data(),
                   reference.data() + std::size_t(t) * shape.vocab,
                   batch.Row(t), sequential.Row(t), shape.vocab);
      int route_ids = 0;
      double route_w = 0, shared = 0, residual_max = 0, conv = 0, state = 0,
             qkv = 0;
      int residual_first = -1;
      if (arm != std::string("production")) {
        qkv = MaxAbs(row.qkv_out.data(), production[t].qkv_out.data(),
                     shape.channels);
        conv = MaxAbs(row.conv.data(), production[t].conv.data(), shape.conv);
        state = MaxAbs(row.state.data(), production[t].state.data(), shape.state);
        route_ids = SameBytes(row.ids.data(), production[t].ids.data(),
                              row.ids.size() * sizeof(std::int32_t))
                        ? 0
                        : 1;
        route_w = MaxAbs(row.weights.data(), production[t].weights.data(),
                         row.weights.size());
        shared = MaxAbs(row.shared.data(), production[t].shared.data(),
                        row.shared.size());
        for (std::size_t layer = 0; layer < shape.layers; ++layer) {
          const double gap = MaxAbs(row.residual[layer].data(),
                                   production[t].residual[layer].data(),
                                   shape.hidden);
          if (gap > 0 && residual_first < 0)
            residual_first = static_cast<int>(layer);
          residual_max = std::max(residual_max, gap);
        }
      }
      table << t << '\t' << arm << '\t' << score.ref_kl << '\t' << score.llama_kl
            << '\t' << score.entropy << '\t' << score.margin << '\t'
            << score.top_ref << '\t' << score.top_gpu << '\t' << score.top_llama
            << '\t' << score.top_sequential << '\t' << score.clear_flip << '\t'
            << score.confident_failure << '\t' << qkv << '\t' << conv << '\t'
            << state << '\t' << route_ids << '\t' << route_w << '\t' << shared
            << '\t' << residual_first << '\t' << residual_max << '\n';
      return score;
    };
    std::vector<Score> base_scores, sub_scores;
    int first_route = -1, first_route_layer = -1, first_residual_token = -1,
        first_residual_layer = -1, first_logit = -1;
    double mean_base = 0, mean_sub = 0, mean_llama_base = 0, mean_llama_sub = 0;
    int new_confident = 0, lost_confident = 0, new_flips = 0, lost_flips = 0;
    for (int t = 0; t < kTokens; ++t) {
      const Score base = emit(t, "production", production[t]);
      emit(t, "reinsert", control[t]);
      const Score sub = emit(t, "f64_qkv_on_gpu_norm", substituted[t]);
      base_scores.push_back(base);
      sub_scores.push_back(sub);
      mean_base += base.ref_kl;
      mean_sub += sub.ref_kl;
      mean_llama_base += base.llama_kl;
      mean_llama_sub += sub.llama_kl;
      if (sub.confident_failure && !base.confident_failure)
        ++new_confident;
      if (base.confident_failure && !sub.confident_failure)
        ++lost_confident;
      if (sub.clear_flip && !base.clear_flip)
        ++new_flips;
      if (base.clear_flip && !sub.clear_flip)
        ++lost_flips;
      if (first_logit < 0 &&
          !SameBytes(production[t].logits.data(), substituted[t].logits.data(),
                     shape.vocab * sizeof(float)))
        first_logit = t;
      if (first_route < 0) {
        for (std::uint32_t layer = 0; layer < shape.layers; ++layer) {
          const auto n = shape.used;
          const bool ids = SameBytes(
              production[t].ids.data() + layer * n,
              substituted[t].ids.data() + layer * n, n * sizeof(std::int32_t));
          const bool w = SameBytes(production[t].weights.data() + layer * n,
                                  substituted[t].weights.data() + layer * n,
                                  n * sizeof(float));
          if (!ids || !w) {
            first_route = t;
            first_route_layer = static_cast<int>(layer);
            std::cout << "first_route_change token=" << t << " layer=" << layer
                      << " ids_differ=" << !ids << " weights_differ=" << !w
                      << " weight_max_abs="
                      << MaxAbs(production[t].weights.data() + layer * n,
                                substituted[t].weights.data() + layer * n, n)
                      << std::endl;
            break;
          }
        }
      }
      if (first_residual_token < 0) {
        for (std::uint32_t layer = 0; layer < shape.layers; ++layer) {
          if (!SameBytes(production[t].residual[layer].data(),
                         substituted[t].residual[layer].data(),
                         shape.hidden * sizeof(float))) {
            first_residual_token = t;
            first_residual_layer = static_cast<int>(layer);
            std::cout << "first_residual_change token=" << t
                      << " layer=" << layer << " max_abs="
                      << MaxAbs(production[t].residual[layer].data(),
                                substituted[t].residual[layer].data(),
                                shape.hidden)
                      << std::endl;
            break;
          }
        }
      }
    }
    const Score b75 = base_scores[kPosition];
    const Score s75 = sub_scores[kPosition];
    const bool corrects = !s75.confident_failure && new_confident == 0 &&
                          new_flips == 0 && s75.ref_kl <= kConfidentKl;
    std::cout << "position=75 entropy=" << s75.entropy
              << " margin=" << s75.margin << " top_ref=" << s75.top_ref
              << " top_gpu=" << s75.top_gpu << " top_llama=" << s75.top_llama
              << " top_sequential=" << s75.top_sequential
              << " top1_vs_scalar=" << (s75.top_ref == s75.top_gpu)
              << " top1_vs_llama=" << (s75.top_llama == s75.top_gpu)
              << " kl_ref=" << s75.ref_kl << " kl_llama=" << s75.llama_kl
              << " baseline_kl_ref=" << b75.ref_kl
              << " baseline_kl_llama=" << b75.llama_kl
              << " baseline_confident_failure=" << b75.confident_failure
              << " confident_failure=" << s75.confident_failure
              << " clear_flip=" << s75.clear_flip
              << " versus_baseline=" << Direction(s75.ref_kl, b75.ref_kl)
              << " versus_registered=" << Direction(s75.ref_kl, kRegisteredKl)
              << " limit_0.1="
              << (s75.ref_kl > kConfidentKl ? "above" : "within")
              << " corrects_failure=" << corrects << std::endl;
    std::cout << "first_logit_token=" << first_logit
              << " first_route_token=" << first_route
              << " first_route_layer=" << first_route_layer
              << " first_residual_token=" << first_residual_token
              << " first_residual_layer=" << first_residual_layer
              << " new_confident_failures=" << new_confident
              << " removed_confident_failures=" << lost_confident
              << " new_clear_flips=" << new_flips
              << " removed_clear_flips=" << lost_flips << std::endl;
    std::cout << "diagnostic_prefix_mean_kl_ref_production=" << mean_base / kTokens
              << " diagnostic_prefix_mean_kl_ref_substitute="
              << mean_sub / kTokens
              << " diagnostic_prefix_mean_kl_llama_production="
              << mean_llama_base / kTokens
              << " diagnostic_prefix_mean_kl_llama_substitute="
              << mean_llama_sub / kTokens << std::endl;
    std::cout << "measurement_complete control_passed=1 disposition=0\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
