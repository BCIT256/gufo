// gfx1151 prefill and greedy decode rates for one Ornith session.
// Not part of ctest. Set GUFO_ORNITH15_APEX_MTP_MODEL to the pinned GGUF.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ornith15/kernels/rocm/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace orn = gufo::models::ornith15;
namespace {

constexpr std::uint32_t kContext = 4096;
constexpr int kGenerate = 128;
constexpr int kWarmSteps = 16;
constexpr int kSamples = 5;

int EnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return fallback;
  return std::atoi(value);
}

constexpr std::string_view kCodingPrompt =
    "Write a C++17 function that merges overlapping half-open integer "
    "intervals. The input is a vector of pairs [start, end). Sort a copy, "
    "then walk it once. Two intervals overlap when the next start is strictly "
    "less than the current end. Touching endpoints do not merge, so [0, 2) "
    "and [2, 5) stay separate while [0, 3) and [2, 4) become [0, 4). Return "
    "the merged vector. Do not allocate inside the loop after the sort. "
    "Reject an interval whose end is less than its start by throwing "
    "std::invalid_argument. Empty input returns an empty vector. "
    "Then add a unit test that covers empty input, a single interval, "
    "disjoint intervals, a touch at the boundary, a complete overlap, and a "
    "chain of three partial overlaps. Explain the sort comparator and why "
    "the scan is linear after sorting. Keep the explanation in plain English "
    "and show the function, the test, and the expected merged ranges.";

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

std::string ReadLine(const char* path) {
  std::ifstream in(path);
  std::string line;
  if (!std::getline(in, line))
    return "missing";
  return line;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

std::int32_t HostArgmax(const std::vector<float>& logits) {
  return static_cast<std::int32_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

std::vector<std::int32_t> Fit(const std::vector<std::uint32_t>& ids,
                              std::size_t n) {
  Require(!ids.empty(), "tokenizer produced no tokens");
  std::vector<std::int32_t> out;
  out.reserve(n);
  while (out.size() < n) {
    for (std::uint32_t id : ids) {
      out.push_back(static_cast<std::int32_t>(id));
      if (out.size() == n)
        break;
    }
  }
  return out;
}

template<typename Fn>
double Seconds(Fn&& fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

void PrintSamples(const char* name, const std::vector<double>& samples) {
  std::cout << name << "_median=" << Median(samples) << " samples=";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i != 0)
      std::cout << ',';
    std::cout << samples[i];
  }
  std::cout << '\n';
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  if (model_path == nullptr) {
    std::cerr << "set GUFO_ORNITH15_APEX_MTP_MODEL\n";
    return 1;
  }
  try {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "power_ac_online="
              << ReadLine("/sys/class/power_supply/AC0/online")
              << " battery_status="
              << ReadLine("/sys/class/power_supply/BAT0/status") << '\n';
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, error);
    auto tokenizer =
        gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, error);
    gufo::tokenization::TokenizerOptions options;
    options.add_bos = false;
    options.add_eos = false;
    options.parse_special_tokens = false;
    const auto encoded = tokenizer->Encode(kCodingPrompt, options);
    const int want_len = EnvInt("GUFO_ORNITH15_BENCH_LEN", 0);
    const auto prompt_2048 = Fit(encoded, want_len > 0 ? want_len : 2048);
    const auto prompt_261 = Fit(encoded, 261);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
    Require(weights.has_value() && !weights->mtp, error);
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr, error);
    const auto vocab = weights->config.vocab_size;
    auto session = orn::rocm::Session::Create(*device, kContext, &error);
    Require(session != nullptr, error);
    std::vector<float> last(vocab);

    const auto run = [&](const std::vector<std::int32_t>& tokens) {
      Require(session->Reset(&error), error);
      const double prefill_s = Seconds(
          [&] { Require(session->Prefill(tokens, last, &error), error); });
      std::int32_t next = HostArgmax(last);
      const double decode_s = Seconds([&] {
        for (int i = 0; i < kGenerate; ++i) {
          std::int32_t sampled = -1;
          Require(session->Step(next, {}, &error, nullptr, &sampled), error);
          next = sampled;
        }
      });
      return std::pair<double, double>{
          tokens.size() / prefill_s, static_cast<double>(kGenerate) / decode_s};
    };
    const int samples = EnvInt("GUFO_ORNITH15_BENCH_SAMPLES", kSamples);
    const int warm_steps = EnvInt("GUFO_ORNITH15_BENCH_WARM_STEPS", kWarmSteps);
    const bool decode = EnvInt("GUFO_ORNITH15_BENCH_DECODE", 1) != 0;
    const auto measure = [&](const std::vector<std::int32_t>& tokens) {
      Require(session->Reset(&error), error);
      Require(session->Prefill(tokens, last, &error), error);
      std::int32_t warm = HostArgmax(last);
      for (int i = 0; i < warm_steps; ++i) {
        std::int32_t sampled = -1;
        Require(session->Step(warm, {}, &error, nullptr, &sampled), error);
        warm = sampled;
      }
      std::vector<double> pp;
      std::vector<double> tg;
      for (int sample = 0; sample < samples; ++sample) {
        if (!decode) {
          Require(session->Reset(&error), error);
          const double prefill_s = Seconds([&] {
            Require(session->Prefill(tokens, last, &error), error);
          });
          pp.push_back(tokens.size() / prefill_s);
          tg.push_back(0);
          continue;
        }
        const auto rates = run(tokens);
        pp.push_back(rates.first);
        tg.push_back(rates.second);
      }
      std::cout << "prompt_tokens=" << tokens.size() << " context=" << kContext
                << " chunk=" << orn::rocm::Session::kMaxPrefillChunk << ' ';
      PrintSamples("pp_tok_s", pp);
      std::cout << "prompt_tokens=" << tokens.size() << ' ';
      PrintSamples("tg_tok_s", tg);
    };
    measure(prompt_2048);
    if (want_len == 0)
      measure(prompt_261);
  } catch (const std::exception& ex) {
    std::cerr << "FAIL: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}
