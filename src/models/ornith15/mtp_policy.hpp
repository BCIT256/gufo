#ifndef GUFO_MODELS_ORNITH15_MTP_POLICY_HPP_
#define GUFO_MODELS_ORNITH15_MTP_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::models::ornith15 {

inline constexpr std::uint32_t kMaxMtpDraftTokens = 7;

// Cost of one MTP cycle relative to one autoregressive decode. Index 0 is
// the plain decode; index k is k drafts plus one decode-dot verification of
// k + 1 tokens. Measured with tools/ornith15/bench.cpp
// (GUFO_ORNITH15_BENCH_MTP) on gfx1151 at 261 and 2,052 prompt tokens, mean of
// the two.
inline constexpr float kMtpRelativeCost[8] = {
    1.00F, 1.28F, 1.64F, 2.00F, 2.43F, 2.77F, 3.20F, 3.61F,
};

struct MtpLengthState {
  std::array<float, kMaxMtpDraftTokens> successes{};
  std::array<float, kMaxMtpDraftTokens> failures{};
  std::uint32_t retry_tokens{0};
  std::uint32_t probe_depth{0};
  std::uint32_t explored_depth{0};
  std::uint32_t probe_delay{0};
  std::uint32_t failed_depths{0};
};

// Draft count from committed acceptance, never from request timings. Only
// the first rejected draft is a failure. The unverified tail and a fully
// accepted chain are censored. Zero means one autoregressive step.
class MtpLengthController {
public:
  explicit MtpLengthController(std::uint32_t limit)
      : limit_(std::clamp(limit, 1U, kMaxMtpDraftTokens)) {
    Reset();
  }

  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget) const noexcept {
    const auto cap = std::min(budget, limit_);
    if (cap == 0 || state_.retry_tokens != 0)
      return 0;
    return std::min(cap, std::max({1U, BestLength(cap), state_.probe_depth}));
  }

  void Observe(std::size_t accepted, std::size_t drafted) noexcept {
    if (drafted == 0)
      return;
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    const auto observed = accepted + (accepted < drafted ? 1 : 0);
    for (std::size_t depth = 0; depth < observed; ++depth) {
      state_.successes[depth] =
          0.75F * state_.successes[depth] + (depth < accepted ? 1.0F : 0.0F);
      state_.failures[depth] =
          0.75F * state_.failures[depth] + (depth == accepted ? 1.0F : 0.0F);
    }
    state_.probe_depth = 0;
    if (accepted == drafted && drafted < limit_) {
      if (drafted >= state_.explored_depth || state_.probe_delay == 0) {
        state_.probe_depth = std::min<std::uint32_t>(2 * drafted, limit_);
        state_.probe_delay = 16;
      } else {
        --state_.probe_delay;
      }
    }
    state_.explored_depth =
        std::max<std::uint32_t>(state_.explored_depth, observed);
    if (accepted < drafted)
      state_.failed_depths |= 1U << accepted;
    if (BestLength(limit_) == 0)
      state_.retry_tokens = 16;
  }

  void ObserveArToken() noexcept {
    if (state_.retry_tokens != 0)
      --state_.retry_tokens;
  }

  void Reset() noexcept {
    state_.successes.fill(1.5F);
    state_.failures.fill(0.5F);
    state_.retry_tokens = 0;
    state_.probe_depth = 0;
    state_.explored_depth = 0;
    state_.probe_delay = 0;
    state_.failed_depths = 0;
  }

private:
  [[nodiscard]] float Probability(unsigned depth) const noexcept {
    return depth < state_.explored_depth &&
                   (state_.failed_depths & (1U << depth)) == 0
               ? 1.0F
               : state_.successes[depth] /
                     (state_.successes[depth] + state_.failures[depth]);
  }

  [[nodiscard]] std::uint32_t BestLength(std::uint32_t cap) const noexcept {
    float survival = 1.0F;
    float expected_tokens = 1.0F;
    float best_score = 1.0F;
    std::uint32_t best = 0;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      survival *= Probability(length - 1);
      expected_tokens += survival;
      const float score = expected_tokens / kMtpRelativeCost[length];
      if (score > best_score) {
        best_score = score;
        best = length;
      }
    }
    return best;
  }

  std::uint32_t limit_;
  MtpLengthState state_{};
};

}  // namespace gufo::models::ornith15

#endif
