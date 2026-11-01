// NextN trunk gap and a short greedy chain. A nonzero verify-versus-step gap
// is printed; it is not a failure. Non-finite logits and a trunk that moves
// when the draft block runs are failures.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
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

float MaxAbs(const float* left, const float* right, std::size_t n) {
  float gap = 0;
  for (std::size_t i = 0; i < n; ++i) {
    Require(std::isfinite(left[i]) && std::isfinite(right[i]),
            "non-finite logit");
    gap = std::max(gap, std::abs(left[i] - right[i]));
  }
  return gap;
}

std::int32_t Argmax(const std::vector<float>& logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best])
      best = i;
  }
  return static_cast<std::int32_t>(best);
}

struct Capture {
  std::vector<std::int32_t> tokens;
  std::uint32_t vocab{0};
  std::vector<float> logits;

  [[nodiscard]] const float* Row(std::size_t t) const {
    return logits.data() + t * vocab;
  }
};

Capture LoadCapture(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[4] = {};
  std::uint32_t header[3] = {};
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  Require(in && std::string(magic, 4) == "ORNL" && header[0] == 1,
          "not a version-1 .ologits file: " + path.string());
  Capture capture;
  capture.tokens.resize(header[1]);
  capture.vocab = header[2];
  capture.logits.resize(static_cast<std::size_t>(header[1]) * header[2]);
  in.read(reinterpret_cast<char*>(capture.tokens.data()),
          static_cast<std::streamsize>(capture.tokens.size() *
                                       sizeof(std::int32_t)));
  in.read(reinterpret_cast<char*>(capture.logits.data()),
          static_cast<std::streamsize>(capture.logits.size() * sizeof(float)));
  Require(in && in.peek() == EOF, "truncated or oversized " + path.string());
  return capture;
}

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

double Kl(const std::vector<double>& left, const std::vector<double>& right) {
  double kl = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i)
    kl += std::exp(left[i]) * (left[i] - right[i]);
  return kl;
}

// 1-based count of logits strictly above `token`. Vocab-sized, called a few
// times per round.
int RankOf(const std::vector<float>& logits, std::int32_t token) {
  const float mine = logits[static_cast<std::size_t>(token)];
  int higher = 0;
  for (float value : logits) {
    if (value > mine)
      ++higher;
  }
  return higher + 1;
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  if (model_path == nullptr || model_path[0] == '\0') {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL\n";
    return 77;
  }
  try {
    std::cout << std::setprecision(8);
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, "open: " + error);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{.mtp = true}, &error);
    Require(weights.has_value() && weights->mtp.has_value(),
            "bind mtp: " + error);
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr && device->has_mtp(), "upload: " + error);
    constexpr std::uint32_t kContext = 256;
    constexpr int kPrompt = 32;
    constexpr int kCompare = 8;
    constexpr int kGreedy = 12;
    constexpr int kDrafts = 3;
    auto spec = orn::rocm::Session::Create(*device, kContext, &error);
    auto plain = orn::rocm::Session::Create(*device, kContext, &error);
    Require(spec && plain, "session: " + error);
    Require(spec->EnableMtp(&error), "enable mtp: " + error);
    Require(spec->SetCanonicalGrid(16) && plain->SetCanonicalGrid(16), "grid");
    const auto vocab = weights->config.vocab_size;
    std::vector<std::int32_t> prompt(kPrompt);
    for (int i = 0; i < kPrompt; ++i)
      prompt[i] = 100 + i;
    std::vector<float> spec_logits(vocab), plain_logits(vocab);
    Require(spec->Prefill(prompt, spec_logits, &error),
            "spec prefill: " + error);
    Require(plain->Prefill(prompt, plain_logits, &error),
            "plain prefill: " + error);
    const float prefill_gap =
        MaxAbs(spec_logits.data(), plain_logits.data(), vocab);
    std::cout << "prefill_trunk_max_abs=" << prefill_gap << '\n';
    Require(prefill_gap < 1e-4F, "MTP prefill changed trunk logits");

    std::vector<std::int32_t> fed(kCompare);
    std::vector<float> stepped(static_cast<std::size_t>(kCompare) * vocab);
    for (int i = 0; i < kCompare; ++i) {
      fed[i] = 200 + i;
      Require(spec->Step(fed[i], spec_logits, &error), "spec step: " + error);
      Require(plain->Step(fed[i], plain_logits, &error),
              "plain step: " + error);
      const float step_gap =
          MaxAbs(spec_logits.data(), plain_logits.data(), vocab);
      std::cout << "step_trunk_max_abs[" << i << "]=" << step_gap << '\n';
      Require(step_gap == 0.0F, "MTP step changed trunk logits");
      std::copy(spec_logits.begin(), spec_logits.end(),
                stepped.begin() + static_cast<std::size_t>(i) * vocab);
    }

    // Each verification row must equal Step on the same token. Three passes
    // per width cover the eager body, the graph capture and its replay.
    for (const int width : {1, 2, 3, 4, 7, 8}) {
      for (int pass = 0; pass < 3; ++pass) {
        Require(spec->Reset(&error), "reset: " + error);
        Require(spec->SetCanonicalGrid(16), "grid after reset");
        Require(spec->Prefill(prompt, spec_logits, &error),
                "reprefill: " + error);
        std::vector<float> batch(static_cast<std::size_t>(width) * vocab);
        Require(spec->VerifyForward(
                    std::span<const std::int32_t>(fed.data(), width), batch,
                    &error),
                "verify: " + error);
        float verify_gap = 0;
        for (int i = 0; i < width; ++i) {
          verify_gap = std::max(
              verify_gap,
              MaxAbs(batch.data() + static_cast<std::size_t>(i) * vocab,
                     stepped.data() + static_cast<std::size_t>(i) * vocab,
                     vocab));
        }
        std::cout << "verify_vs_step_max_abs=" << verify_gap
                  << " width=" << width << " pass=" << pass << '\n';
        Require(verify_gap == 0.0F, "verification row differs from Step");
      }
    }

    // Copied tokens have no draft, so CatchUpMtp writes their NextN K/V from
    // the verified trunk hiddens. The next draft must be bit-identical to one
    // built token by token, and the trunk must be untouched.
    const auto reprefill = [&] {
      Require(spec->Reset(&error), "reset: " + error);
      Require(spec->SetCanonicalGrid(16), "grid after reset");
      Require(spec->Prefill(prompt, spec_logits, &error),
              "reprefill: " + error);
    };
    std::vector<float> draft_logits(vocab), serial_draft(vocab);
    const auto check_catch_up = [&](const char* name, int keep) {
      Require(spec->DraftToken(fed[keep], draft_logits, &error),
              "draft after catch-up: " + error);
      reprefill();
      for (int i = 0; i < keep; ++i) {
        Require(spec->DraftToken(fed[i], serial_draft, &error),
                "serial draft: " + error);
        Require(spec->VerifyDecode(fed[i], spec_logits, &error),
                "serial verify: " + error);
        spec->RewindMtp(spec->Position());
      }
      Require(spec->DraftToken(fed[keep], serial_draft, &error),
              "serial frontier draft: " + error);
      const float gap = MaxAbs(draft_logits.data(), serial_draft.data(), vocab);
      std::cout << "catch_up_draft_max_abs=" << gap << " case=" << name << '\n';
      Require(gap == 0.0F, "draft after catch-up differs");
    };
    constexpr int kCopies = 4;
    const auto fed_span = std::span<const std::int32_t>(fed);
    std::vector<float> verify_rows(static_cast<std::size_t>(kCopies) * vocab);
    for (int pass = 0; pass < 2; ++pass) {
      // All copies accepted.
      reprefill();
      const auto base = spec->Position();
      spec->SaveMtpCarry();
      Require(spec->VerifyForward(fed_span.first(kCopies), verify_rows, &error),
              "copy verify: " + error);
      Require(spec->CatchUpMtp(fed_span.first(kCopies), 0, &error),
              "catch up: " + error);
      spec->RewindMtp(base + kCopies);
      Require(spec->VerifyDecode(fed[kCopies], spec_logits, &error),
              "verify after catch-up: " + error);
      const float trunk_gap = MaxAbs(
          spec_logits.data(),
          stepped.data() + static_cast<std::size_t>(kCopies) * vocab, vocab);
      std::cout << "catch_up_trunk_max_abs=" << trunk_gap << '\n';
      Require(trunk_gap == 0.0F, "catch-up changed the trunk");
      reprefill();
      spec->SaveMtpCarry();
      Require(spec->VerifyForward(fed_span.first(kCopies), verify_rows, &error),
              "copy verify: " + error);
      Require(spec->CatchUpMtp(fed_span.first(kCopies), 0, &error),
              "catch up: " + error);
      spec->RewindMtp(base + kCopies);
      check_catch_up("all", kCopies);

      // A rejected copy rolls back first.
      reprefill();
      spec->SaveMtpCarry();
      Require(spec->VerifyForward(fed_span.first(kCopies), verify_rows, &error),
              "copy verify: " + error);
      Require(spec->RollbackMtpBatch(2, &error), "rollback: " + error);
      Require(spec->CatchUpMtp(fed_span.first(2), 0, &error),
              "catch up: " + error);
      spec->RewindMtp(base + 2);
      check_catch_up("rejected", 2);

      // One neural draft, then copies.
      reprefill();
      Require(spec->DraftToken(fed[0], serial_draft, &error),
              "mixed draft: " + error);
      Require(spec->VerifyForward(fed_span.first(kCopies), verify_rows, &error),
              "mixed verify: " + error);
      Require(spec->CatchUpMtp(fed_span.subspan(1, kCopies - 1), 1, &error),
              "mixed catch up: " + error);
      spec->RewindMtp(base + kCopies);
      check_catch_up("mixed", kCopies);
    }
    Require(!spec->CatchUpMtp(fed_span.first(kCompare), 1, &error),
            "catch-up past the verify rows must fail");

    Require(spec->Reset(&error) && plain->Reset(&error), "greedy reset");
    Require(spec->SetCanonicalGrid(16) && plain->SetCanonicalGrid(16),
            "greedy grid");
    Require(spec->Prefill(prompt, spec_logits, &error) &&
                plain->Prefill(prompt, plain_logits, &error),
            "greedy prefill: " + error);
    std::vector<std::int32_t> spec_out, plain_out;
    spec_out.reserve(kGreedy);
    plain_out.reserve(kGreedy);
    while (static_cast<int>(plain_out.size()) < kGreedy) {
      const auto token = Argmax(plain_logits);
      Require(plain->Step(token, plain_logits, &error), "plain greedy");
      plain_out.push_back(token);
    }
    int rounds = 0;
    int drafted = 0;
    int accepted = 0;
    while (static_cast<int>(spec_out.size()) < kGreedy) {
      const int room = kGreedy - static_cast<int>(spec_out.size());
      const auto anchor = Argmax(spec_logits);
      const int n_draft = std::min(kDrafts, room - 1);
      if (n_draft <= 0) {
        Require(spec->Step(anchor, spec_logits, &error), "spec tail");
        spec_out.push_back(anchor);
        continue;
      }
      std::vector<std::int32_t> chain{anchor};
      std::vector<float> draft_row(vocab);
      for (int d = 0; d < n_draft; ++d) {
        Require(spec->DraftToken(chain.back(), draft_row, &error),
                "draft: " + error);
        chain.push_back(Argmax(draft_row));
      }
      const auto base = spec->Position();
      spec->SaveMtpCarry();
      auto rollback = spec->StageSnapshot(&error);
      Require(rollback != nullptr, "rollback: " + error);
      std::vector<float> rows(chain.size() * vocab);
      Require(spec->VerifyForward(chain, rows, &error),
              "spec verify: " + error);
      std::uint32_t keep = 1;
      while (keep < chain.size()) {
        const float* row =
            rows.data() + static_cast<std::size_t>(keep - 1) * vocab;
        std::vector<float> view(row, row + vocab);
        if (Argmax(view) != chain[keep])
          break;
        ++keep;
      }
      if (keep < chain.size()) {
        Require(spec->mtp_prefix_rollback() && chain.size() <= 8,
                "rollback window");
        Require(spec->RollbackMtpBatch(keep, &error), "rollback: " + error);
      }
      if (keep == chain.size()) {
        Require(spec->DraftToken(chain.back(), draft_row, &error),
                "catch-up: " + error);
      }
      spec->RewindMtp(base + keep);
      const float* last =
          rows.data() + static_cast<std::size_t>(keep - 1) * vocab;
      std::copy(last, last + vocab, spec_logits.begin());
      ++rounds;
      drafted += n_draft;
      accepted += static_cast<int>(keep - 1);
      spec_out.insert(spec_out.end(), chain.begin(),
                      chain.begin() + static_cast<std::ptrdiff_t>(keep));
    }
    int matched = 0;
    while (matched < kGreedy && spec_out[matched] == plain_out[matched])
      ++matched;
    std::cout << "greedy_agreement=" << matched << "/" << kGreedy
              << " rounds=" << rounds << " draft_accepted=" << accepted << "/"
              << drafted << '\n';
    if (matched < kGreedy) {
      std::cout << "first_mismatch index=" << matched
                << " spec=" << spec_out[matched]
                << " plain=" << plain_out[matched] << '\n';
    }

    // Real text. Sequential ids 100+i are not a sentence the head was trained
    // on, so a zero accept count there does not locate a pairing bug.
    const auto tokenizer =
        gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, "tokenizer: " + error);
    std::string text;
    for (int i = 0; i < 8; ++i)
      text += "Merge half-open integer intervals and return them sorted. ";
    const auto encoded = tokenizer->Encode(text);
    std::vector<std::int32_t> sentence(encoded.begin(), encoded.end());
    Require(sentence.size() > 32 && sentence.size() < 160, "sentence length");
    Require(spec->Reset(&error) && plain->Reset(&error), "text reset");
    Require(spec->SetCanonicalGrid(16) && plain->SetCanonicalGrid(16),
            "text grid");
    Require(spec->Prefill(sentence, spec_logits, &error) &&
                plain->Prefill(sentence, plain_logits, &error),
            "text prefill: " + error);
    std::cout << "text_prompt_tokens=" << sentence.size() << '\n';
    constexpr int kText = 24;
    spec_out.clear();
    plain_out.clear();
    while (static_cast<int>(plain_out.size()) < kText) {
      const auto token = Argmax(plain_logits);
      Require(plain->Step(token, plain_logits, &error), "plain text");
      plain_out.push_back(token);
    }
    rounds = 0;
    drafted = 0;
    accepted = 0;
    int printed = 0;
    while (static_cast<int>(spec_out.size()) < kText) {
      const int room = kText - static_cast<int>(spec_out.size());
      const auto anchor = Argmax(spec_logits);
      const int n_draft = std::min(kDrafts, room - 1);
      if (n_draft <= 0) {
        Require(spec->Step(anchor, spec_logits, &error), "text tail");
        spec_out.push_back(anchor);
        continue;
      }
      std::vector<std::int32_t> chain{anchor};
      std::vector<float> draft_row(vocab);
      std::vector<std::vector<float>> draft_rows;
      draft_rows.reserve(static_cast<std::size_t>(n_draft));
      for (int d = 0; d < n_draft; ++d) {
        Require(spec->DraftToken(chain.back(), draft_row, &error),
                "text draft: " + error);
        draft_rows.push_back(draft_row);
        chain.push_back(Argmax(draft_row));
      }
      const auto base = spec->Position();
      spec->SaveMtpCarry();
      auto rollback = spec->StageSnapshot(&error);
      Require(rollback != nullptr, "text rollback: " + error);
      std::vector<float> rows(chain.size() * vocab);
      Require(spec->VerifyForward(chain, rows, &error),
              "text verify: " + error);
      std::uint32_t keep = 1;
      while (keep < chain.size()) {
        const float* row =
            rows.data() + static_cast<std::size_t>(keep - 1) * vocab;
        std::vector<float> view(row, row + vocab);
        if (Argmax(view) != chain[keep])
          break;
        ++keep;
      }
      if (printed < 4) {
        std::cout << "text_round " << rounds << " anchor=" << anchor;
        for (int d = 0; d < n_draft; ++d) {
          const float* row = rows.data() + static_cast<std::size_t>(d) * vocab;
          std::vector<float> view(row, row + vocab);
          const auto truth = Argmax(view);
          const int rank =
              RankOf(draft_rows[static_cast<std::size_t>(d)], truth);
          const float margin =
              draft_rows[static_cast<std::size_t>(d)]
                        [static_cast<std::size_t>(truth)] -
              draft_rows[static_cast<std::size_t>(d)]
                        [static_cast<std::size_t>(chain[d + 1])];
          std::cout << " d" << d << "=" << chain[d + 1] << " truth=" << truth
                    << " rank=" << rank << " margin=" << margin << " piece='"
                    << tokenizer->DecodeToken(
                           static_cast<gufo::tokenization::TokenId>(truth))
                    << "'";
        }
        std::cout << '\n';
        ++printed;
      }
      if (keep < chain.size()) {
        Require(spec->mtp_prefix_rollback() && chain.size() <= 8,
                "text rollback window");
        Require(spec->RollbackMtpBatch(keep, &error),
                "text rollback: " + error);
      }
      if (keep == chain.size()) {
        Require(spec->DraftToken(chain.back(), draft_row, &error),
                "text catch-up: " + error);
      }
      spec->RewindMtp(base + keep);
      const float* last =
          rows.data() + static_cast<std::size_t>(keep - 1) * vocab;
      std::copy(last, last + vocab, spec_logits.begin());
      ++rounds;
      drafted += n_draft;
      accepted += static_cast<int>(keep - 1);
      spec_out.insert(spec_out.end(), chain.begin(),
                      chain.begin() + static_cast<std::ptrdiff_t>(keep));
    }
    matched = 0;
    while (matched < kText && spec_out[matched] == plain_out[matched])
      ++matched;
    std::cout << "text_greedy_agreement=" << matched << "/" << kText
              << " rounds=" << rounds << " text_draft_accepted=" << accepted
              << "/" << drafted << '\n';
    std::cout << "text_spec='"
              << tokenizer->Decode(std::vector<gufo::tokenization::TokenId>(
                     spec_out.begin(), spec_out.end()))
              << "'\n";
    Require(matched == kText, "MTP rollback diverged from AR greedy");
    // A disconnected head accepts nothing on this sentence. The rate itself
    // is printed; this floor is only that failure.
    Require(drafted > 0 && accepted > 0, "MTP draft head accepted nothing");

    // VerifyForward rows against the capture set. Unset means this process
    // did not request the gate; ctest sets the directory. A skip is not a
    // pass of the gate.
    const char* logits_dir = std::getenv("GUFO_ORNITH15_LLAMA_LOGITS");
    if (logits_dir == nullptr || logits_dir[0] == '\0') {
      std::cout << "verify_distribution=not_requested\n";
    } else {
      Require(fs::is_directory(logits_dir),
              "GUFO_ORNITH15_LLAMA_LOGITS is not a directory");
      std::vector<std::string> prompts;
      for (const auto& entry : fs::directory_iterator(logits_dir)) {
        const auto name = entry.path().filename().string();
        if (name.ends_with(".ologits") &&
            !name.ends_with(".sequential.ologits"))
          prompts.push_back(name.substr(0, name.size() - 8));
      }
      std::sort(prompts.begin(), prompts.end());
      Require(!prompts.empty(), "no .ologits captures");
      double self_max = 0;
      for (const auto& prompt : prompts) {
        const auto batch =
            LoadCapture(fs::path(logits_dir) / (prompt + ".ologits"));
        const auto sequential = LoadCapture(fs::path(logits_dir) /
                                            (prompt + ".sequential.ologits"));
        Require(batch.tokens == sequential.tokens &&
                    batch.vocab == weights->config.vocab_size &&
                    sequential.vocab == batch.vocab &&
                    batch.tokens.size() <= kContext,
                "verify capture does not match this model");
        for (std::size_t t = 0; t < batch.tokens.size(); ++t) {
          self_max = std::max(self_max,
                              Kl(LogSoftmax(batch.Row(t), batch.vocab),
                                 LogSoftmax(sequential.Row(t), batch.vocab)));
        }
      }
      bool distribution_ok = true;
      constexpr std::uint32_t kVerifyChunk = 4;
      for (const auto& prompt : prompts) {
        const auto batch =
            LoadCapture(fs::path(logits_dir) / (prompt + ".ologits"));
        const auto sequential = LoadCapture(fs::path(logits_dir) /
                                            (prompt + ".sequential.ologits"));
        Require(spec->Reset(&error), "verify distribution reset: " + error);
        double llama_sum = 0, floor_sum = 0, llama_max = 0;
        std::size_t outliers = 0;
        std::vector<float> got(static_cast<std::size_t>(kVerifyChunk) *
                               batch.vocab);
        for (std::size_t done = 0; done < batch.tokens.size();) {
          const auto count = static_cast<std::uint32_t>(
              std::min<std::size_t>(kVerifyChunk, batch.tokens.size() - done));
          got.resize(static_cast<std::size_t>(count) * batch.vocab);
          Require(spec->VerifyForward(std::span<const std::int32_t>(
                                          batch.tokens.data() + done, count),
                                      got, &error),
                  "verify distribution forward: " + error);
          for (std::uint32_t row = 0; row < count; ++row) {
            const std::size_t t = done + row;
            const auto lq = LogSoftmax(
                got.data() + std::size_t(row) * batch.vocab, batch.vocab);
            const auto ll = LogSoftmax(batch.Row(t), batch.vocab);
            const double lk = Kl(ll, lq);
            const double floor =
                Kl(ll, LogSoftmax(sequential.Row(t), batch.vocab));
            Require(std::isfinite(lk) && std::isfinite(floor),
                    "non-finite verify KL");
            llama_sum += lk;
            floor_sum += floor;
            llama_max = std::max(llama_max, lk);
            if (!(lk <= kOutlierRatio * self_max)) {
              ++outliers;
              std::cout << "verify_outlier prompt=" << prompt << " t=" << t
                        << " llama_kl=" << lk << " self_kl=" << floor << '\n';
            }
          }
          done += count;
        }
        const bool pass =
            outliers == 0 && llama_sum <= kMeanKlRatio * floor_sum;
        const double mean_llama = llama_sum / batch.tokens.size();
        const double mean_self = floor_sum / batch.tokens.size();
        std::cout << "verify_distribution prompt=" << prompt
                  << " positions=" << batch.tokens.size()
                  << " mean_kl_llama=" << mean_llama
                  << " llama_self_mean=" << mean_self << " mean_ratio="
                  << (mean_self == 0 ? (mean_llama == 0 ? 0 : INFINITY)
                                     : mean_llama / mean_self)
                  << " max_kl_llama=" << llama_max
                  << " set_max_self_kl=" << self_max
                  << " outlier_failures=" << outliers
                  << (pass ? " PASS" : " FAIL") << '\n';
        distribution_ok = distribution_ok && pass;
      }
      Require(distribution_ok,
              "verify-launch logits failed the distribution gate");
    }
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
  return 0;
}
