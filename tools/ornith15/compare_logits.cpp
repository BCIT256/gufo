// Offline diagnostic metrics for two ORNL captures with identical token IDs.
// Outputs JSON only; numerical disagreement is reported, never labelled a pass.
// first_margin_flips lacks the paired-llama corroboration used by gpu_decode.
// Usage: compare_logits FIRST.ologits SECOND.ologits
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

namespace fs = std::filesystem;
namespace {

constexpr double kClearMargin = 1.0;
constexpr double kConfidentEntropy = 0.5;
constexpr double kConfidentKl = 0.1;

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
  Require(fs::file_size(path) == 16 + std::uint64_t(header[1]) * 4 +
                                     std::uint64_t(header[1]) * header[2] * 4,
          "capture size mismatch");
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

}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 3, "usage: compare_logits FIRST.ologits SECOND.ologits");
    const auto a = Load(argv[1]), b = Load(argv[2]);
    Require(a.tokens == b.tokens && a.vocab == b.vocab, "capture identity");
    double ab = 0, ba = 0, mab = 0, mba = 0, cab = 0, cba = 0;
    std::size_t top = 0, flips = 0, fa = 0, fb = 0;
    for (std::size_t t = 0; t < a.tokens.size(); ++t) {
      const auto p = LogSoftmax(a.Row(t), a.vocab),
                 q = LogSoftmax(b.Row(t), b.vocab);
      const auto x = Kl(p, q), y = Kl(q, p);
      ab += x;
      ba += y;
      mab = std::max(mab, x);
      mba = std::max(mba, y);
      auto at = Top(a.Row(t), a.vocab), bt = Top(b.Row(t), b.vocab);
      top += at == bt;
      flips += at != bt && Margin(a.Row(t), a.vocab, at) > kClearMargin;
      if (Entropy(p) < kConfidentEntropy) {
        cab = std::max(cab, x);
        fa += x > kConfidentKl;
      }
      if (Entropy(q) < kConfidentEntropy) {
        cba = std::max(cba, y);
        fb += y > kConfidentKl;
      }
    }
    auto n = a.tokens.size();
    std::cout << std::setprecision(17) << "{\"positions\":" << n
              << ",\"top1_agreements\":" << top
              << ",\"first_margin_flips\":" << flips
              << ",\"first_second_mean_kl\":" << ab / n
              << ",\"second_first_mean_kl\":" << ba / n
              << ",\"first_second_max_kl\":" << mab
              << ",\"second_first_max_kl\":" << mba
              << ",\"first_confident_max_kl\":" << cab
              << ",\"second_confident_max_kl\":" << cba
              << ",\"first_confident_failures\":" << fa
              << ",\"second_confident_failures\":" << fb << "}\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
