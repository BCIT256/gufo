// Exact snapshot restore of an Ornith GPU session.
// A continuation after SaveSnapshot must match the same continuation after
// RestoreSnapshot into another session, bit for bit, including a prefix that
// ends inside a prefill chunk and a prefix that ends on a decode Step.
// Restored full sequences also take the distribution gate. See QUALITY.md.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/ornith15/kernels/rocm/executor.hpp"

namespace orn = gufo::models::ornith15;
namespace fs = std::filesystem;
namespace {

constexpr double kMeanKlRatio = 2.0;
constexpr double kOutlierRatio = 3.0;

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
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

bool GatePass(const GateStats& stats) {
  return stats.llama_sum <= kMeanKlRatio * stats.floor_sum &&
         stats.outliers == 0;
}

struct Plan {
  const char* mode;
  std::uint32_t position;
  std::uint32_t chunk;
  bool step_origin;
};

std::vector<float> Row(std::uint32_t vocab) {
  return std::vector<float>(vocab);
}

void Reach(orn::rocm::Session& session, const std::vector<std::int32_t>& tokens,
           const Plan& plan, std::uint32_t vocab, std::vector<float>* prefix,
           std::string* error) {
  auto last = Row(vocab);
  if (plan.step_origin) {
    const std::uint32_t prefill = plan.position - 1;
    std::vector<float> all(std::size_t(prefill) * vocab);
    Require(prefill > 0 && session.Prefill(std::span<const std::int32_t>(
                                               tokens.data(), prefill),
                                           last, error, plan.chunk, all),
            "reach prefill: " + *error);
    if (prefix != nullptr)
      std::copy(all.begin(), all.end(), prefix->begin());
    Require(session.Step(tokens[prefill], last, error),
            "reach step: " + *error);
    if (prefix != nullptr)
      std::copy(last.begin(), last.end(),
                prefix->begin() + std::size_t(prefill) * vocab);
  } else {
    std::vector<float> all(std::size_t(plan.position) * vocab);
    Require(session.Prefill(
                std::span<const std::int32_t>(tokens.data(), plan.position),
                last, error, plan.chunk, all),
            "reach prefill: " + *error);
    if (prefix != nullptr)
      std::copy(all.begin(), all.end(), prefix->begin());
  }
  Require(session.Position() == plan.position, "reach position");
}

void Continue(orn::rocm::Session& session,
              const std::vector<std::int32_t>& tokens, const Plan& plan,
              std::uint32_t vocab, std::vector<float>* full,
              std::string* error) {
  const auto n = static_cast<std::uint32_t>(tokens.size());
  std::uint32_t cursor = plan.position;
  auto last = Row(vocab);
  if (plan.step_origin && cursor < n) {
    Require(session.Step(tokens[cursor], last, error),
            "continue step: " + *error);
    std::copy(last.begin(), last.end(),
              full->begin() + std::size_t(cursor) * vocab);
    ++cursor;
  }
  if (cursor < n) {
    const auto count = n - cursor;
    std::vector<float> all(std::size_t(count) * vocab);
    Require(session.Prefill(
                std::span<const std::int32_t>(tokens.data() + cursor, count),
                last, error, plan.chunk, all),
            "continue prefill: " + *error);
    std::copy(all.begin(), all.end(),
              full->begin() + std::size_t(cursor) * vocab);
  }
  Require(session.Position() == n, "continue position");
}

bool SameBits(const float* a, const float* b, std::size_t n, std::size_t* index,
              double* max_abs) {
  bool same = true;
  *index = 0;
  *max_abs = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double delta = std::abs(static_cast<double>(a[i]) - b[i]);
    if (delta > *max_abs)
      *max_abs = delta;
    if (same && std::memcmp(a + i, b + i, sizeof(float)) != 0) {
      same = false;
      *index = i;
    }
  }
  return same;
}

GateStats Score(const Capture& batch, const Capture& sequential,
                const std::vector<float>& got, double self_max) {
  GateStats stats;
  const auto vocab = batch.vocab;
  const auto n = batch.tokens.size();
  for (std::size_t t = 0; t < n; ++t) {
    const auto lq = LogSoftmax(got.data() + t * vocab, vocab);
    const auto ll = LogSoftmax(batch.Row(t), vocab);
    const double lk = Kl(ll, lq);
    const double floor = Kl(ll, LogSoftmax(sequential.Row(t), vocab));
    stats.llama_sum += lk;
    stats.floor_sum += floor;
    stats.llama_max = std::max(stats.llama_max, lk);
    stats.outliers += !(lk <= kOutlierRatio * self_max);
    ++stats.positions;
  }
  return stats;
}

void ReportPayloadDiff(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  auto word = [](const std::uint8_t* bytes, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
  };
  if (a.size() != b.size() || a.size() < 64 ||
      std::memcmp(a.data(), b.data(), 64) != 0) {
    std::cout << " payload_diff region=header bytes_a=" << a.size()
              << " bytes_b=" << b.size() << std::endl;
    return;
  }
  const std::uint32_t layers = word(a.data(), 16);
  const std::uint32_t interval = word(a.data(), 20);
  const std::size_t conv_bytes =
      std::size_t(word(a.data(), 24)) * sizeof(float);
  const std::size_t recurrent_bytes =
      std::size_t(word(a.data(), 28)) * sizeof(float);
  const std::size_t kv_bytes = std::size_t(word(a.data(), 8)) *
                               word(a.data(), 32) * sizeof(std::uint16_t);
  std::size_t cursor = 64;
  auto region = [&](const char* name, std::uint32_t layer, std::size_t bytes) {
    if (cursor + bytes > a.size()) {
      std::cout << " payload_diff region=truncated" << std::endl;
      return true;
    }
    if (std::memcmp(a.data() + cursor, b.data() + cursor, bytes) != 0) {
      std::size_t at = 0;
      while (at < bytes && a[cursor + at] == b[cursor + at])
        ++at;
      std::cout << " payload_diff region=" << name << " layer=" << layer
                << " offset=" << (cursor + at) << std::endl;
      return true;
    }
    cursor += bytes;
    return false;
  };
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    const bool linear = interval != 0 && ((layer + 1) % interval) != 0;
    if (linear) {
      if (region("conv", layer, conv_bytes) ||
          region("recurrent", layer, recurrent_bytes))
        return;
    } else if (region("K", layer, kv_bytes) || region("V", layer, kv_bytes)) {
      return;
    }
  }
  const std::size_t logit_bytes =
      std::size_t(word(a.data(), 36)) * sizeof(float);
  if (!region("logits", 0, logit_bytes))
    std::cout << " payload_diff region=none" << std::endl;
}

void PrintGate(const std::string& prompt, const Plan& plan,
               const GateStats& stats, double self_max) {
  const double mean_llama =
      stats.positions == 0 ? 0 : stats.llama_sum / stats.positions;
  const double mean_self =
      stats.positions == 0 ? 0 : stats.floor_sum / stats.positions;
  std::cout << prompt << " mode=" << plan.mode << " P=" << plan.position
            << " restored_gate positions=" << stats.positions
            << " mean_kl_llama=" << mean_llama
            << " llama_self_mean=" << mean_self
            << " mean_ratio=" << Ratio(mean_llama, mean_self)
            << " max_kl_llama=" << stats.llama_max
            << " set_max_self_kl=" << self_max
            << " outlier_failures=" << stats.outliers
            << (GatePass(stats) ? " PASS" : " FAIL") << std::endl;
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
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, "open: " + error);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
    Require(weights.has_value() && !weights->mtp, "bind trunk: " + error);
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr, "upload: " + error);
    std::vector<std::string> prompts;
    for (const auto& entry : fs::directory_iterator(logits_dir)) {
      const auto name = entry.path().filename().string();
      if (name.ends_with(".ologits") && !name.ends_with(".sequential.ologits"))
        prompts.push_back(name.substr(0, name.size() - 8));
    }
    std::sort(prompts.begin(), prompts.end());
    Require(!prompts.empty(), "no captures");
    double self_max = 0;
    for (const auto& prompt : prompts) {
      const auto batch = Load(fs::path(logits_dir) / (prompt + ".ologits"));
      const auto sequential =
          Load(fs::path(logits_dir) / (prompt + ".sequential.ologits"));
      Require(batch.tokens == sequential.tokens &&
                  batch.vocab == weights->config.vocab_size,
              "capture identity mismatch");
      for (std::size_t t = 0; t < batch.tokens.size(); ++t)
        self_max =
            std::max(self_max, Kl(LogSoftmax(batch.Row(t), batch.vocab),
                                  LogSoftmax(sequential.Row(t), batch.vocab)));
    }
    std::cout << "capture_set prompts=" << prompts.size()
              << " set_max_self_kl=" << self_max << " payload_version="
              << orn::rocm::Session::kSnapshotPayloadVersion << std::endl;

    bool ok = true;
    bool mutation_ran = false;
    for (const auto& prompt : prompts) {
      const auto batch = Load(fs::path(logits_dir) / (prompt + ".ologits"));
      const auto sequential =
          Load(fs::path(logits_dir) / (prompt + ".sequential.ologits"));
      const auto n = static_cast<std::uint32_t>(batch.tokens.size());
      const auto vocab = batch.vocab;
      std::vector<Plan> plans;
      if (n > 1)
        plans.push_back({"prefill_inside", 1, 16, false});
      if (n > 8)
        plans.push_back({"prefill_inside", 7, 16, false});
      if (n > 20)
        plans.push_back({"prefill_partial_chunk", 17, 16, false});
      if (n > 50)
        plans.push_back({"prefill_inside", 48, 64, false});
      if (n > 4) {
        const auto stepped = std::min<std::uint32_t>(16, n - 2) + 1;
        plans.push_back({"after_step", stepped, 16, true});
      }
      Require(!plans.empty(), "no snapshot positions");

      bool saw_mutation = false;
      for (const auto& plan : plans) {
        auto origin = orn::rocm::Session::Create(*device, n, &error);
        auto other = orn::rocm::Session::Create(*device, n, &error);
        Require(origin && other, "session: " + error);
        std::vector<float> prefix(std::size_t(n) * vocab);
        std::vector<float> full(std::size_t(n) * vocab);
        Reach(*origin, batch.tokens, plan, vocab, &prefix, &error);
        std::vector<std::uint8_t> payload;
        Require(origin->SaveSnapshot(&payload, &error), "snapshot: " + error);
        Require(payload.size() == origin->SnapshotPayloadBytes() &&
                    payload.size() > 64,
                "snapshot size");
        Reach(*other, batch.tokens, plan, vocab, nullptr, &error);
        std::vector<std::uint8_t> again;
        Require(other->SaveSnapshot(&again, &error),
                "second snapshot: " + error);
        const bool payloads_match = payload == again;
        Continue(*origin, batch.tokens, plan, vocab, &full, &error);
        std::copy(prefix.begin(),
                  prefix.begin() + std::size_t(plan.position) * vocab,
                  full.begin());
        auto restored = orn::rocm::Session::Create(*device, n, &error);
        Require(restored != nullptr, "fresh session: " + error);
        std::vector<float> restored_logits(vocab);
        Require(restored->RestoreSnapshot(payload, restored_logits, &error),
                "restore fresh: " + error);
        Require(restored->Position() == plan.position, "restored position");
        std::vector<float> fork(std::size_t(n) * vocab);
        Continue(*restored, batch.tokens, plan, vocab, &fork, &error);
        std::size_t index = 0;
        double max_abs = 0;
        const auto tail = std::size_t(n - plan.position) * vocab;
        const bool fresh_match =
            SameBits(full.data() + std::size_t(plan.position) * vocab,
                     fork.data() + std::size_t(plan.position) * vocab, tail,
                     &index, &max_abs);

        auto used = orn::rocm::Session::Create(*device, n, &error);
        Require(used != nullptr, "used session: " + error);
        std::vector<std::int32_t> unrelated(std::min<std::uint32_t>(8, n));
        for (std::size_t i = 0; i < unrelated.size(); ++i)
          unrelated[i] = static_cast<std::int32_t>((batch.tokens[i] + 17) %
                                                   static_cast<int>(vocab));
        auto last = Row(vocab);
        Require(used->Prefill(unrelated, last, &error),
                "unrelated prefill: " + error);
        Require(used->RestoreSnapshot(payload, restored_logits, &error),
                "restore used: " + error);
        std::vector<float> used_full(std::size_t(n) * vocab);
        Continue(*used, batch.tokens, plan, vocab, &used_full, &error);
        std::size_t used_index = 0;
        double used_abs = 0;
        const bool used_match =
            SameBits(full.data() + std::size_t(plan.position) * vocab,
                     used_full.data() + std::size_t(plan.position) * vocab,
                     tail, &used_index, &used_abs);

        Require(origin->RestoreSnapshot(payload, restored_logits, &error),
                "restore same: " + error);
        std::vector<float> same_full(std::size_t(n) * vocab);
        Continue(*origin, batch.tokens, plan, vocab, &same_full, &error);
        std::size_t same_index = 0;
        double same_abs = 0;
        const bool same_match =
            SameBits(full.data() + std::size_t(plan.position) * vocab,
                     same_full.data() + std::size_t(plan.position) * vocab,
                     tail, &same_index, &same_abs);

        std::vector<std::uint8_t> bad = payload;
        std::size_t region = 0, region_bytes = 0;
        Require(orn::rocm::Session::SnapshotRecurrentRegion(bad, 0, &region,
                                                            &region_bytes),
                "layer 0 recurrent region");
        Require(region_bytes > 0 && region + region_bytes <= bad.size(),
                "recurrent region bounds");
        bool nonzero = false;
        for (std::size_t i = 0; i < region_bytes; ++i)
          nonzero = nonzero || bad[region + i] != 0;
        std::fill(
            bad.begin() + static_cast<std::ptrdiff_t>(region),
            bad.begin() + static_cast<std::ptrdiff_t>(region + region_bytes),
            0);
        auto clean = orn::rocm::Session::Create(*device, n, &error);
        auto mutated = orn::rocm::Session::Create(*device, n, &error);
        Require(clean && mutated &&
                    clean->RestoreSnapshot(payload, restored_logits, &error) &&
                    mutated->RestoreSnapshot(bad, restored_logits, &error),
                "mutated restore: " + error);
        auto clean_row = Row(vocab);
        auto mutated_row = Row(vocab);
        const auto one = std::span<const std::int32_t>(
            batch.tokens.data() + plan.position, 1);
        if (plan.step_origin) {
          Require(clean->Step(batch.tokens[plan.position], clean_row, &error) &&
                      mutated->Step(batch.tokens[plan.position], mutated_row,
                                    &error),
                  "mutated step: " + error);
        } else {
          Require(clean->Prefill(one, clean_row, &error, plan.chunk) &&
                      mutated->Prefill(one, mutated_row, &error, plan.chunk),
                  "mutated prefill: " + error);
        }
        std::size_t mutated_index = 0;
        double mutated_abs = 0;
        const bool mutated_same = SameBits(clean_row.data(), mutated_row.data(),
                                           vocab, &mutated_index, &mutated_abs);
        const bool caught = nonzero && !mutated_same;
        saw_mutation = saw_mutation || caught;

        const std::vector<std::uint8_t> truncated(payload.begin(),
                                                  payload.end() - 1);
        auto rejected = orn::rocm::Session::Create(*device, n, &error);
        Require(rejected && rejected->Prefill(std::span<const std::int32_t>(
                                                  batch.tokens.data(), 1),
                                              last, &error),
                "reject setup: " + error);
        const auto kept = rejected->Position();
        Require(
            !rejected->RestoreSnapshot(truncated, restored_logits, &error) &&
                rejected->Position() == kept,
            "truncated snapshot modified the session");
        std::size_t attn_off = 0, attn_bytes = 0;
        Require(!orn::rocm::Session::SnapshotRecurrentRegion(
                    payload, 3, &attn_off, &attn_bytes),
                "attention layer reported a recurrent region");

        std::vector<float> restored_sequence = prefix;
        std::copy(
            fork.begin() + std::size_t(plan.position) * vocab, fork.end(),
            restored_sequence.begin() + std::size_t(plan.position) * vocab);
        const auto gate = Score(batch, sequential, restored_sequence, self_max);
        const bool gate_ok = GatePass(gate);
        PrintGate(prompt, plan, gate, self_max);
        std::cout << prompt << " mode=" << plan.mode << " P=" << plan.position
                  << " chunk=" << plan.chunk
                  << " payload_bytes=" << payload.size()
                  << " payloads_match=" << payloads_match
                  << " fresh_identical=" << fresh_match
                  << " fresh_max_abs=" << max_abs
                  << " used_identical=" << used_match
                  << " used_max_abs=" << used_abs
                  << " same_identical=" << same_match
                  << " same_max_abs=" << same_abs
                  << " mutation_nonzero=" << nonzero
                  << " mutation_caught=" << caught
                  << " mutation_max_abs=" << mutated_abs << std::endl;
        if (!fresh_match || !used_match || !same_match)
          std::cout << prompt << " mismatch fresh_index=" << index
                    << " used_index=" << used_index
                    << " same_index=" << same_index << std::endl;
        if (!payloads_match)
          ReportPayloadDiff(payload, again);
        ok = ok && payloads_match && fresh_match && used_match && same_match &&
             gate_ok;
      }
      std::cout << prompt << " mutation_caught=" << saw_mutation << std::endl;
      ok = ok && saw_mutation;
      mutation_ran = mutation_ran || saw_mutation;
    }
    Require(ok && mutation_ran, "snapshot restore was not bit-identical");
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
