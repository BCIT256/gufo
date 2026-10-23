// Compares the scalar reference against llama.cpp's full logits for the
// prompts in tools/ornith15/prompts (see tools/ornith15/README.md):
//
//   GUFO_ORNITH15_APEX_MTP_MODEL=/path/to/Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf
//   GUFO_ORNITH15_LLAMA_LOGITS=/path/to/captures
//
// The capture directory holds NAME.ologits (one batched decode) and
// NAME.sequential.ologits (one token per decode) for every prompt. The pair
// measures llama.cpp's own numerical spread, which bounds the comparison.
// Exits 77 (skipped, never passed) unless the model and captures are supplied.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/ornith15/reference.hpp"
#include "src/models/ornith15/weights.hpp"

namespace orn = gufo::models::ornith15;
namespace fs = std::filesystem;

namespace {

// llama.cpp's CPU backend quantizes activations for its quantized dot
// products and keeps F16 KV, so its batched and sequential decodes already
// disagree; the float32 reference is held to that spread, not to equality.
// Measured on the three prompts (183 positions, APEX MTP Quality, llama.cpp
// 68d9053a CPU): llama.cpp's own KL reached 0.10 where its entropy is below
// 0.5 nats; the reference reached 0.036 there and its mean KL was at most
// 1.3x llama.cpp's own mean.
constexpr double kClearMargin = 1.0;       ///< Logits between llama's top two.
constexpr double kConfidentEntropy = 0.5;  ///< Nats.
constexpr double kConfidentKl = 0.1;
constexpr double kMeanKlRatio = 2.0;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Capture {
  std::vector<std::int32_t> tokens;
  std::uint32_t vocab{0};
  std::vector<float> logits;  ///< [tokens][vocab]

  [[nodiscard]] const float* Row(std::size_t t) const {
    return logits.data() + t * vocab;
  }
};

Capture Load(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[4] = {};
  std::uint32_t header[3] = {};
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  Require(in && std::string(magic, 4) == "ORNL" && header[0] == 1,
          "not a version-1 .ologits file: " + path.string());
  Capture c;
  c.tokens.resize(header[1]);
  c.vocab = header[2];
  c.logits.resize(static_cast<std::size_t>(header[1]) * header[2]);
  in.read(reinterpret_cast<char*>(c.tokens.data()),
          static_cast<std::streamsize>(c.tokens.size() * sizeof(std::int32_t)));
  in.read(reinterpret_cast<char*>(c.logits.data()),
          static_cast<std::streamsize>(c.logits.size() * sizeof(float)));
  Require(in && in.peek() == EOF, "truncated or oversized " + path.string());
  return c;
}

/// Log-softmax in double precision.
std::vector<double> LogSoftmax(const float* x, std::size_t n) {
  const double peak = *std::max_element(x, x + n);
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    sum += std::exp(static_cast<double>(x[i]) - peak);
  const double log_sum = peak + std::log(sum);
  std::vector<double> out(n);
  for (std::size_t i = 0; i < n; ++i)
    out[i] = static_cast<double>(x[i]) - log_sum;
  return out;
}

/// KL(p || q) in nats from log-probabilities.
double Kl(const std::vector<double>& lp, const std::vector<double>& lq) {
  double kl = 0.0;
  for (std::size_t i = 0; i < lp.size(); ++i)
    kl += std::exp(lp[i]) * (lp[i] - lq[i]);
  return kl;
}

std::size_t Argmax(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);
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
  // Per-position rows for diagnosis.
  const bool verbose = std::getenv("GUFO_ORNITH15_ORACLE_VERBOSE") != nullptr;
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, "open failed: " + error);
    const auto weights = orn::ModelWeights::Bind(*reader, {}, &error);
    Require(weights.has_value(), "bind failed: " + error);
    const std::uint32_t vocab = weights->config.vocab_size;

    std::vector<std::string> prompts;
    for (const auto& entry : fs::directory_iterator(logits_dir)) {
      const std::string name = entry.path().filename().string();
      constexpr std::string_view kSuffix = ".ologits";
      if (name.ends_with(kSuffix) && !name.ends_with(".sequential.ologits"))
        prompts.push_back(name.substr(0, name.size() - kSuffix.size()));
    }
    std::sort(prompts.begin(), prompts.end());
    Require(!prompts.empty(), "no .ologits captures in the logits directory");

    bool ok = true;
    for (const auto& prompt : prompts) {
      const fs::path dir(logits_dir);
      const Capture batch = Load(dir / (prompt + ".ologits"));
      const Capture sequential = Load(dir / (prompt + ".sequential.ologits"));
      Require(batch.vocab == vocab && sequential.vocab == vocab,
              prompt + ": vocabulary size mismatch");
      Require(batch.tokens == sequential.tokens,
              prompt + ": the paired captures saw different tokens");

      const std::size_t n = batch.tokens.size();
      orn::ReferenceModel model(*weights, static_cast<std::uint32_t>(n));
      std::vector<float> logits(vocab);
      double ref_sum = 0.0, floor_sum = 0.0, confident_max = 0.0;
      std::size_t top1 = 0, clear_flips = 0, confident = 0;
      for (std::size_t t = 0; t < n; ++t) {
        Require(model.Step(batch.tokens[t], logits, &error),
                "reference step failed: " + error);
        const float* expected = batch.Row(t);
        const auto lp = LogSoftmax(expected, vocab);
        const auto lq = LogSoftmax(logits.data(), vocab);
        const double kl = Kl(lp, lq);
        const double floor = Kl(lp, LogSoftmax(sequential.Row(t), vocab));
        Require(std::isfinite(kl) && std::isfinite(floor),
                prompt + ": non-finite comparison at position " +
                    std::to_string(t));
        double entropy = 0.0;
        for (double v : lp)
          entropy -= std::exp(v) * v;

        const std::size_t want = Argmax(expected, vocab);
        float second = -INFINITY;
        for (std::size_t i = 0; i < vocab; ++i)
          if (i != want)
            second = std::max(second, expected[i]);
        const double margin = expected[want] - second;
        const bool agrees = Argmax(logits.data(), vocab) == want;
        // A flip counts only where llama.cpp is clear and self-consistent.
        if (!agrees && margin > kClearMargin &&
            Argmax(sequential.Row(t), vocab) == want)
          ++clear_flips;
        if (entropy < kConfidentEntropy) {
          ++confident;
          confident_max = std::max(confident_max, kl);
        }
        top1 += agrees ? 1 : 0;
        ref_sum += kl;
        floor_sum += floor;
        if (verbose)
          std::cout << "  t=" << t << " token=" << batch.tokens[t]
                    << " kl=" << kl << " floor=" << floor
                    << " entropy=" << entropy << " top1=" << agrees
                    << " margin=" << margin << '\n';
      }
      const double ref_mean = ref_sum / static_cast<double>(n);
      const double floor_mean = floor_sum / static_cast<double>(n);
      const bool pass = clear_flips == 0 && confident_max <= kConfidentKl &&
                        ref_mean <= kMeanKlRatio * floor_mean;
      std::cout << prompt << ": positions=" << n << " top1=" << top1 << "/" << n
                << " clear_flips=" << clear_flips
                << " confident_max_kl=" << confident_max << " (" << confident
                << " positions) mean_kl=" << ref_mean
                << " llama_mean_kl=" << floor_mean << (pass ? " PASS" : " FAIL")
                << '\n';
      ok = ok && pass;
    }
    if (!ok) {
      std::cerr << "FAIL: reference disagrees with llama.cpp beyond its own "
                   "numerical spread\n";
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Ornith reference agrees with llama.cpp within its own "
               "numerical spread.\n";
  return 0;
}
