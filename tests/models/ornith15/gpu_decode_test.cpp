// Full-vocabulary lockstep at the captured token histories.
// Acceptance is distribution-level against llama.cpp. See QUALITY.md.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/ornith15/kernels/rocm/executor.hpp"
#include "src/models/ornith15/reference.hpp"

namespace orn = gufo::models::ornith15;
namespace fs = std::filesystem;
namespace {

constexpr double kClearMargin = 1.0;
constexpr double kConfidentEntropy = 0.5;
constexpr double kConfidentKl = 0.1;
constexpr double kMeanKlRatio = 2.0;
constexpr double kOutlierRatio = 3.0;

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

bool ScalarReference() {
  const char* value = std::getenv("GUFO_ORNITH15_SCALAR_REFERENCE");
  return value != nullptr && std::string(value) == "1";
}

struct Capture {
  std::uint32_t vocab = 0;
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
  Require(static_cast<bool>(in) && std::string(magic, 4) == "ORNL" &&
              header[0] == 1 && header[1] > 0 && header[1] <= 262144 &&
              header[2] == 248320,
          "invalid capture header: " + path.string());
  Capture c;
  c.vocab = header[2];
  c.tokens.resize(header[1]);
  c.logits.resize(std::size_t(header[1]) * header[2]);
  in.read(reinterpret_cast<char*>(c.tokens.data()),
          static_cast<std::streamsize>(c.tokens.size() * sizeof(std::int32_t)));
  in.read(reinterpret_cast<char*>(c.logits.data()),
          static_cast<std::streamsize>(c.logits.size() * sizeof(float)));
  Require(static_cast<bool>(in) && in.peek() == EOF,
          "truncated/oversized capture");
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
  return x[top] - second;
}

double Ratio(double numer, double denom) {
  if (denom == 0)
    return numer == 0 ? 0 : INFINITY;
  return numer / denom;
}

struct GateStats {
  double llama_sum = 0;
  double floor_sum = 0;
  double llama_max = 0;
  std::size_t outliers = 0;
  std::size_t positions = 0;
};

bool MeanPass(const GateStats& stats) {
  return stats.llama_sum <= kMeanKlRatio * stats.floor_sum;
}

bool OutlierPass(const GateStats& stats) {
  return stats.outliers == 0;
}

void PrintGate(const GateStats& stats, double self_max) {
  const double mean_llama =
      stats.positions == 0 ? 0 : stats.llama_sum / stats.positions;
  const double mean_self =
      stats.positions == 0 ? 0 : stats.floor_sum / stats.positions;
  const bool pass = MeanPass(stats) && OutlierPass(stats);
  std::cout << "positions=" << stats.positions
            << " mean_kl_llama=" << mean_llama
            << " llama_self_mean=" << mean_self
            << " mean_ratio=" << Ratio(mean_llama, mean_self)
            << " max_kl_llama=" << stats.llama_max
            << " set_max_self_kl=" << self_max
            << " max_position_ratio=" << Ratio(stats.llama_max, self_max)
            << " outlier_failures=" << stats.outliers
            << (pass ? " PASS" : " FAIL") << std::endl;
}

void ReportTrace(const orn::LayerTrace& ref, const orn::LayerTrace& gpu) {
  std::size_t first = ref.size(), largest = 0;
  double worst = 0;
  for (std::size_t l = 0; l < ref.size(); ++l) {
    double ss = 0, delta = 0;
    for (std::size_t i = 0; i < ref[l].size(); ++i) {
      ss += static_cast<double>(ref[l][i]) * ref[l][i];
      const double d = static_cast<double>(ref[l][i]) - gpu[l][i];
      delta += d * d;
    }
    const double relative = std::sqrt(delta / std::max(ss, 1e-30));
    if ((!std::isfinite(relative) || relative > 0.1) && first == ref.size())
      first = l;
    if (relative > worst) {
      worst = relative;
      largest = l;
    }
    std::cout << "trace layer=" << l << " residual_relative_rms=" << relative
              << '\n';
  }
  if (first < ref.size())
    std::cout << "first_diverging_layer=" << first
              << " (diagnostic relative RMS > 0.1)\n";
  else
    std::cout << "first_diverging_layer=none_above_diagnostic_bound "
                 "largest_layer="
              << largest << " relative_rms=" << worst << '\n';
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  const char* logits_dir = std::getenv("GUFO_ORNITH15_LLAMA_LOGITS");
  if (model_path == nullptr || logits_dir == nullptr ||
      !fs::is_regular_file(model_path) || !fs::is_directory(logits_dir)) {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL and "
                 "GUFO_ORNITH15_LLAMA_LOGITS\n";
    return 77;
  }
  try {
    std::cout << std::setprecision(10);
    const bool scalar = ScalarReference();
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, "open: " + error);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
    Require(weights.has_value() && !weights->mtp, "bind trunk: " + error);
    std::size_t free_before = 0, free_after = 0, heap_total = 0;
    Require(hipMemGetInfo(&free_before, &heap_total) == hipSuccess,
            "GPU heap query before upload");
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr, "upload: " + error);
    Require(hipMemGetInfo(&free_after, &heap_total) == hipSuccess,
            "GPU heap query after upload");
    std::cout << "gpu_weight_resident_bytes=" << device->resident_bytes()
              << " gpu_heap_delta_bytes="
              << static_cast<std::int64_t>(free_before) -
                     static_cast<std::int64_t>(free_after)
              << std::endl;
    std::vector<std::string> prompts;
    for (const auto& entry : fs::directory_iterator(logits_dir)) {
      const auto name = entry.path().filename().string();
      if (name.ends_with(".ologits") && !name.ends_with(".sequential.ologits"))
        prompts.push_back(name.substr(0, name.size() - 8));
    }
    std::sort(prompts.begin(), prompts.end());
    Require(!prompts.empty(), "no captures");
    double self_max = 0;
    std::size_t capture_positions = 0;
    for (const auto& prompt : prompts) {
      const auto batch = Load(fs::path(logits_dir) / (prompt + ".ologits"));
      const auto sequential =
          Load(fs::path(logits_dir) / (prompt + ".sequential.ologits"));
      Require(batch.tokens == sequential.tokens &&
                  batch.vocab == weights->config.vocab_size &&
                  sequential.vocab == batch.vocab,
              "capture identity mismatch");
      capture_positions += batch.tokens.size();
      for (std::size_t t = 0; t < batch.tokens.size(); ++t)
        self_max =
            std::max(self_max, Kl(LogSoftmax(batch.Row(t), batch.vocab),
                                  LogSoftmax(sequential.Row(t), batch.vocab)));
    }
    std::cout << "capture_set prompts=" << prompts.size()
              << " positions=" << capture_positions
              << " set_max_self_kl=" << self_max
              << " mean_ratio_limit=" << kMeanKlRatio
              << " outlier_ratio_limit=" << kOutlierRatio
              << " scalar_reference=" << scalar << std::endl;

    bool ok = true;
    const auto vocab = weights->config.vocab_size;
    for (const auto& prompt : prompts) {
      const auto batch = Load(fs::path(logits_dir) / (prompt + ".ologits"));
      const auto sequential =
          Load(fs::path(logits_dir) / (prompt + ".sequential.ologits"));
      const auto n = static_cast<std::uint32_t>(batch.tokens.size());
      auto gpu = orn::rocm::Session::Create(*device, n, &error);
      Require(gpu != nullptr, "session: " + error);
      std::vector<float> actual(vocab), first_logits,
          step_logits(batch.logits.size());
      std::vector<float> expected;
      orn::LayerTrace ref_trace, gpu_trace;
      std::unique_ptr<orn::ReferenceModel> ref;
      if (scalar) {
        expected.resize(vocab);
        ref = std::make_unique<orn::ReferenceModel>(*weights, n);
      }
      GateStats stats;
      double ref_sum = 0, ref_max = 0, max_delta = 0, gpu_ms = 0;
      std::size_t ref_top1 = 0, llama_top1 = 0, clear_flips = 0;
      std::size_t unconfirmed_flips = 0, confident_failures = 0;
      bool reported = false;
      Require(!gpu->Step(-1, actual, &error) && gpu->Position() == 0,
              "invalid token accepted");
      Require(!gpu->Step(batch.tokens[0], {}, &error) && gpu->Position() == 0,
              "invalid logits accepted");
      for (std::uint32_t t = 0; t < n; ++t) {
        if (scalar)
          Require(ref->Step(batch.tokens[t], expected, &error, &ref_trace),
                  "reference: " + error);
        const auto start = std::chrono::steady_clock::now();
        Require(gpu->Step(batch.tokens[t], actual, &error,
                          scalar ? &gpu_trace : nullptr),
                "GPU: " + error);
        gpu_ms += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        if (t == 0)
          first_logits = actual;
        std::copy(actual.begin(), actual.end(),
                  step_logits.begin() + std::size_t(t) * vocab);
        const auto lq = LogSoftmax(actual.data(), vocab);
        const auto ll = LogSoftmax(batch.Row(t), vocab);
        const double lk = Kl(ll, lq);
        const double floor = Kl(ll, LogSoftmax(sequential.Row(t), vocab));
        stats.llama_sum += lk;
        stats.floor_sum += floor;
        stats.llama_max = std::max(stats.llama_max, lk);
        const bool outlier = !(lk <= kOutlierRatio * self_max);
        stats.outliers += outlier;
        ++stats.positions;
        const auto gt = Top(actual.data(), vocab);
        const auto lt = Top(batch.Row(t), vocab);
        llama_top1 += lt == gt;
        std::cout << prompt << " t=" << t << " llama_kl=" << lk
                  << " self_kl=" << floor << " top1_llama=" << (lt == gt)
                  << " outlier=" << outlier;
        if (scalar) {
          const auto lp = LogSoftmax(expected.data(), vocab);
          const double rk = Kl(lp, lq);
          const auto rt = Top(expected.data(), vocab);
          const bool margin_flip =
              rt != gt && Margin(expected.data(), vocab, rt) > kClearMargin;
          const bool flip =
              margin_flip && lt == rt && Top(sequential.Row(t), vocab) == rt;
          const bool confident = Entropy(lp) < kConfidentEntropy;
          double delta = 0;
          for (std::size_t i = 0; i < vocab; ++i)
            delta = std::max(
                delta, std::abs(static_cast<double>(expected[i]) - actual[i]));
          std::cout << " ref_kl=" << rk << " top1_ref=" << (rt == gt)
                    << " clear_flip=" << flip
                    << " unconfirmed_flip=" << (margin_flip && !flip)
                    << " confident_failure=" << (confident && rk > kConfidentKl)
                    << " max_delta_logit=" << delta;
          if ((flip || (confident && rk > kConfidentKl)) && !reported) {
            ReportTrace(ref_trace, gpu_trace);
            reported = true;
          }
          ref_sum += rk;
          ref_max = std::max(ref_max, rk);
          max_delta = std::max(max_delta, delta);
          ref_top1 += rt == gt;
          clear_flips += flip;
          unconfirmed_flips += margin_flip && !flip;
          confident_failures += confident && rk > kConfidentKl;
        }
        std::cout << std::endl;
      }
      const bool pass = MeanPass(stats) && OutlierPass(stats);
      std::cout << prompt << ": ";
      PrintGate(stats, self_max);
      if (scalar) {
        std::cout << prompt << ": diagnostic clear_flips=" << clear_flips
                  << " unconfirmed_flips=" << unconfirmed_flips
                  << " confident_failures=" << confident_failures
                  << " top1_ref=" << ref_top1 << '/' << n
                  << " top1_llama=" << llama_top1 << '/' << n
                  << " mean_kl_ref=" << ref_sum / n << " max_kl_ref=" << ref_max
                  << " max_delta_logit=" << max_delta
                  << " gpu_ms_per_token=" << gpu_ms / n << std::endl;
      } else {
        std::cout << prompt << ": top1_llama=" << llama_top1 << '/' << n
                  << " gpu_ms_per_token=" << gpu_ms / n
                  << " gpu_session_resident_bytes=" << gpu->resident_bytes()
                  << std::endl;
      }
      ok = ok && pass;
      Require(!gpu->Step(batch.tokens.back(), actual, &error) &&
                  gpu->Position() == n,
              "context limit accepted");
      auto other = orn::rocm::Session::Create(*device, n, &error);
      Require(other != nullptr &&
                  other->Step(batch.tokens[0], actual, &error) &&
                  actual == first_logits,
              "independent session changed logits");
      Require(gpu->Reset(&error) && gpu->Step(batch.tokens[0], actual, &error),
              "reset: " + error);
      Require(actual == first_logits, "reset replay changed logits");
      orn::LayerTrace replay_trace;
      Require(other->Reset(&error) &&
                  other->Step(batch.tokens[0], actual, &error, &replay_trace),
              "traced replay: " + error);
      Require(actual == first_logits, "traced replay changed logits");
      gpu.reset();
      other.reset();
    }
    Require(ok, "one or more prompts failed the distribution contract");
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
