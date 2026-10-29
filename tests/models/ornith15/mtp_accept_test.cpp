// Sampled MTP must preserve the target distribution. The acceptance math is
// the Flash-Next header; this test is the Ornith gate on that contract.
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

namespace {

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

void CheckCertainAcceptance() {
  const std::array<float, 2> logits{40.0F, -40.0F};
  qfn::MtpProposal sure;
  sure.ids[0] = 0;
  sure.probabilities[0] = 1.0F;
  sure.size = 1;
  sure.token = 0;
  sure.probability = 1.0F;
  qfn::MtpProposal miss;
  miss.ids[0] = 1;
  miss.probabilities[0] = 1.0F;
  miss.size = 1;
  miss.token = 1;
  miss.probability = 1.0F;
  for (unsigned trial = 0; trial < 64; ++trial) {
    sampling::SamplerState accept({.temperature = 0.8F, .seed = 11 + trial});
    const auto kept = qfn::VerifyMtpProposal(logits, sure, accept);
    Require(kept.accepted && kept.token == 0,
            "unit-mass target token rejected");
    sampling::SamplerState reject({.temperature = 0.8F, .seed = 11 + trial});
    const auto corrected = qfn::VerifyMtpProposal(logits, miss, reject);
    Require(!corrected.accepted && corrected.token == 0,
            "zero-mass proposal was not corrected to the target");
  }
}

void CheckGreedyRoundTrip() {
  const std::array<float, 4> logits{0.2F, 1.5F, 1.5F, -0.4F};
  sampling::SamplerState sampler({.temperature = 0.0F, .seed = 3});
  const auto target = sampler.Sample(logits);
  Require(target == 1, "temperature 0 did not take the lowest tied argmax");
  auto matching = sampler;
  Require(qfn::VerifyDraft(logits, 1, matching,
                           [](std::int32_t) { return false; }) ==
              qfn::DraftDecision::kAccept,
          "matching greedy draft was rejected");
  auto mismatch = sampler;
  const auto rng = mismatch.rng_state();
  Require(qfn::VerifyDraft(logits, 2, mismatch,
                           [](std::int32_t) { return false; }) ==
              qfn::DraftDecision::kReject,
          "mismatched greedy draft was accepted");
  Require(mismatch.rng_state() == rng, "rejected greedy draw consumed RNG");
  Require(mismatch.Sample(logits) == target,
          "rejected greedy draft changed the next target token");
  auto stopping = sampler;
  Require(qfn::VerifyDraft(logits, 1, stopping,
                           [](std::int32_t token) { return token == 1; }) ==
              qfn::DraftDecision::kStop,
          "greedy stop token was published");
}

void CheckSampledFrequencies() {
  const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 3;
  candidates.ids = {3, 1, 4};
  candidates.logits = {1.2F, 0.1F, -0.7F};
  constexpr unsigned trials = 40000;
  constexpr double tolerance = 0.015;
  for (const float temperature : {0.7F, 1.0F}) {
    sampling::SamplerState sampler({.temperature = temperature, .seed = 73});
    const auto target = sampler.Distribution(logits);
    std::array<unsigned, 5> counts{};
    for (unsigned trial = 0; trial < trials; ++trial) {
      auto draft_rng = sampling::NextRandom(sampler.mutable_rng_state());
      const auto proposal =
          qfn::SampleMtpProposal(candidates, sampler, &draft_rng);
      const auto token =
          qfn::VerifyMtpProposal(logits, proposal, sampler).token;
      Require(token < counts.size(), "verification left the vocabulary");
      ++counts[token];
    }
    for (unsigned token = 0; token < counts.size(); ++token) {
      const double got = static_cast<double>(counts[token]) / trials;
      if (std::abs(got - target.probability(token)) >= tolerance) {
        throw std::runtime_error(
            "MTP proposal and correction distort the target distribution");
      }
    }
  }
}

}  // namespace

int main() {
  try {
    CheckCertainAcceptance();
    CheckGreedyRoundTrip();
    CheckSampledFrequencies();
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
  std::cout << "ornith15 mtp acceptance ok\n";
  return 0;
}
