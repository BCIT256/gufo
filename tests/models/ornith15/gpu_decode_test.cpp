// Full-vocabulary lockstep at the captured token histories, never comparing
// independently generated text. See reference_oracle_test.cpp for ORNL v1.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

struct Capture {
  std::uint32_t vocab;
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
              header[1] > 0 && header[1] <= 262144 && header[2] == 248320,
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
  return x[top] - second;
}

// Diagnostic only, never a logit acceptance tolerance. Quantized dots differ
// from the scalar oracle; identify the first substantial residual drift.
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
              << " (zero-based; diagnostic relative RMS > 0.1)\n";
  else
    std::cout
        << "first_diverging_layer=none_above_diagnostic_bound largest_layer="
        << largest << " relative_rms=" << worst << '\n';
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  const char* logits_dir = std::getenv("GUFO_ORNITH15_LLAMA_LOGITS");
  if (!model_path || !logits_dir || !fs::is_regular_file(model_path) ||
      !fs::is_directory(logits_dir)) {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL and "
                 "GUFO_ORNITH15_LLAMA_LOGITS\n";
    return 77;
  }
  try {
    std::cout << std::setprecision(10);
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
    for (const auto& prompt : prompts) {
      const auto batch = Load(fs::path(logits_dir) / (prompt + ".ologits"));
      const auto sequential =
          Load(fs::path(logits_dir) / (prompt + ".sequential.ologits"));
      const auto n = static_cast<std::uint32_t>(batch.tokens.size());
      const auto vocab = weights->config.vocab_size;
      Require(batch.tokens == sequential.tokens && batch.vocab == vocab &&
                  sequential.vocab == vocab,
              "capture identity mismatch");
      auto gpu = orn::rocm::Session::Create(*device, n, &error);
      Require(gpu != nullptr, "session: " + error);
      orn::ReferenceModel ref(*weights, n);
      std::vector<float> actual(vocab), expected(vocab), first_logits;
      orn::LayerTrace ref_trace, gpu_trace;
      double ref_sum = 0, llama_sum = 0, floor_sum = 0;
      double ref_max = 0, llama_max = 0, confident_ref_max = 0,
             confident_llama_max = 0;
      double max_delta = 0, gpu_ms = 0;
      std::size_t ref_top1 = 0, llama_top1 = 0, clear_flips = 0;
      // Invalid calls must not advance recurrent state or overwrite logits.
      Require(!gpu->Step(-1, actual, &error) && gpu->Position() == 0,
              "invalid token accepted");
      Require(!gpu->Step(batch.tokens[0], {}, &error) && gpu->Position() == 0,
              "invalid logits accepted");
      for (std::uint32_t t = 0; t < n; ++t) {
        Require(ref.Step(batch.tokens[t], expected, &error, &ref_trace),
                "reference: " + error);
        const auto start = std::chrono::steady_clock::now();
        Require(gpu->Step(batch.tokens[t], actual, &error, &gpu_trace),
                "GPU: " + error);
        gpu_ms += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        if (t == 0)
          first_logits = actual;
        const auto lp = LogSoftmax(expected.data(), vocab);
        const auto lq = LogSoftmax(actual.data(), vocab);
        const auto ll = LogSoftmax(batch.Row(t), vocab);
        const double rk = Kl(lp, lq), lk = Kl(ll, lq);
        const double floor = Kl(ll, LogSoftmax(sequential.Row(t), vocab));
        const auto rt = Top(expected.data(), vocab),
                   gt = Top(actual.data(), vocab),
                   lt = Top(batch.Row(t), vocab);
        const bool flip =
            rt != gt && Margin(expected.data(), vocab, rt) > kClearMargin;
        const bool confident = Entropy(lp) < kConfidentEntropy;
        double delta = 0;
        for (std::size_t i = 0; i < vocab; ++i)
          delta = std::max(
              delta, std::abs(static_cast<double>(expected[i]) - actual[i]));
        std::cout << prompt << " t=" << t << " ref_kl=" << rk
                  << " llama_kl=" << lk << " top1_ref=" << (rt == gt)
                  << " top1_llama=" << (lt == gt)
                  << " max_delta_logit=" << delta << std::endl;
        if (flip || (confident && rk > kConfidentKl)) {
          ReportTrace(ref_trace, gpu_trace);
          throw std::runtime_error(
              prompt + ": per-position numerical contract failed at " +
              std::to_string(t));
        }
        ref_sum += rk;
        llama_sum += lk;
        floor_sum += floor;
        ref_max = std::max(ref_max, rk);
        llama_max = std::max(llama_max, lk);
        if (confident)
          confident_ref_max = std::max(confident_ref_max, rk);
        if (Entropy(ll) < kConfidentEntropy)
          confident_llama_max = std::max(confident_llama_max, lk);
        max_delta = std::max(max_delta, delta);
        ref_top1 += rt == gt;
        llama_top1 += lt == gt;
        clear_flips += flip;
      }
      const bool pass = llama_sum <= kMeanKlRatio * floor_sum;
      std::cout << prompt << ": positions=" << n << " top1_ref=" << ref_top1
                << '/' << n << " top1_llama=" << llama_top1 << '/' << n
                << " clear_flips=" << clear_flips
                << " mean_kl_ref=" << ref_sum / n << " max_kl_ref=" << ref_max
                << " mean_kl_llama=" << llama_sum / n
                << " max_kl_llama=" << llama_max
                << " confident_max_kl_ref=" << confident_ref_max
                << " confident_max_kl_llama=" << confident_llama_max
                << " llama_batch_sequential_mean_kl=" << floor_sum / n
                << " max_delta_logit=" << max_delta
                << " gpu_ms_per_token=" << gpu_ms / n
                << " gpu_session_resident_bytes=" << gpu->resident_bytes()
                << (pass ? " PASS" : " FAIL") << std::endl;
      if (!pass) {
        ReportTrace(ref_trace, gpu_trace);
        throw std::runtime_error(
            "mean KL exceeds 2x llama batch/sequential spread");
      }
      Require(!gpu->Step(batch.tokens.back(), actual, &error) &&
                  gpu->Position() == n,
              "context limit accepted");
      // Reset and untraced replay are byte-identical; another session starts
      // clean while this one's recurrent state and KV remain live.
      auto other = orn::rocm::Session::Create(*device, n, &error);
      Require(other != nullptr && other->Step(batch.tokens[0], actual, &error),
              "independent session: " + error);
      Require(actual == first_logits, "session state leaked");
      Require(gpu->Reset(&error) && gpu->Step(batch.tokens[0], actual, &error),
              "reset: " + error);
      Require(actual == first_logits,
              "reset or optional tracing changed logits");
    }
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
