#include "tests/models/ornith15/route_margin.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

std::vector<double> Softmax(const float* logits, int n) {
  double peak = logits[0];
  for (int i = 1; i < n; ++i)
    peak = std::max(peak, static_cast<double>(logits[i]));
  std::vector<double> p(n);
  double denom = 0;
  for (int i = 0; i < n; ++i) {
    p[i] = std::exp(static_cast<double>(logits[i]) - peak);
    denom += p[i];
  }
  for (double& v : p)
    v /= denom;
  return p;
}

void UnsortedTail() {
  // The true ninth value sits in the tail. The element that follows the
  // top eight in memory is not that ninth value.
  std::vector<double> p(16, 0);
  for (int i = 0; i < 8; ++i)
    p[i] = 8.0 - i;
  p[8] = 0;
  p[15] = 0.5;
  const RankBoundary boundary = RankTop(p, nullptr, 8);
  Require(boundary.eighth == 7, "eighth index");
  Require(boundary.ninth == 15, "ninth index is the tail, not position 8");
  Require(boundary.probability_margin == 0.5, "probability margin uses the tail");
  Require(boundary.ties == 0, "no ties");
}

void LogitIsNotProbability() {
  const float logits[4] = {3.0F, 0.0F, 1.0F, -10.0F};
  const auto p = Softmax(logits, 4);
  const RankBoundary boundary = RankTop(p, logits, 2);
  Require(boundary.eighth == 2, "second place");
  Require(boundary.ninth == 1, "third place");
  Require(boundary.logit_margin == 1.0, "logit margin");
  Require(boundary.probability_margin != boundary.logit_margin,
          "probability margin is a different quantity");
  const double denom = std::exp(0.0) + std::exp(-2.0) + std::exp(-3.0) +
                       std::exp(-13.0);
  const double expected =
      std::exp(-2.0) / denom - std::exp(-3.0) / denom;
  Require(std::abs(boundary.probability_margin - expected) < 1e-12,
          "hand softmax margin");
}

void Fixture(const std::string& dir) {
  auto load = [&](const char* name) {
    std::ifstream in(dir + "/" + name, std::ios::binary);
    std::vector<float> v(256);
    in.read(reinterpret_cast<char*>(v.data()), 256 * sizeof(float));
    Require(static_cast<bool>(in), name);
    return v;
  };
  const auto gpu = load("layer6_router_gpu.f32");
  const auto f64 = load("layer6_router_f64.f32");
  const auto gp = Softmax(gpu.data(), 256);
  const auto fp = Softmax(f64.data(), 256);
  const RankBoundary g = RankTop(gp, gpu.data(), 8);
  const RankBoundary f = RankTop(fp, f64.data(), 8);
  std::cout << std::setprecision(17);
  std::cout << "fixture_gpu_eighth=" << g.eighth << " ninth=" << g.ninth
            << " probability_margin=" << g.probability_margin
            << " logit_margin=" << g.logit_margin << " ties=" << g.ties
            << "\nfixture_f64_eighth=" << f.eighth << " ninth=" << f.ninth
            << " probability_margin=" << f.probability_margin
            << " logit_margin=" << f.logit_margin << " ties=" << f.ties
            << "\n";
}

}  // namespace

int main() {
  try {
    UnsortedTail();
    LogitIsNotProbability();
    std::cout << "route_margin_regression=ok\n";
    if (const char* dir = std::getenv("GUFO_ORNITH15_ROUTE_DIR"))
      Fixture(dir);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
