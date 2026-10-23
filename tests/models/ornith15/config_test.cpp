#include "src/models/ornith15/config.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <string>

#include "tests/models/ornith15/gguf_builder.hpp"

namespace orn = gufo::models::ornith15;
namespace build = gufo::tests::ornith15;

namespace {

std::optional<orn::Config> Parse(const build::Metadata& fields,
                                 std::string* error = nullptr) {
  const auto image = build::Build(fields, {});
  std::string message;
  const auto result = orn::Config::FromGguf(*image.reader, &message);
  assert(result.has_value() == message.empty());
  if (error != nullptr)
    *error = message;
  return result;
}

void CheckInspectedArtifact() {
  const auto c = Parse(build::OrnithMetadata());
  assert(c.has_value());
  assert(c->num_layers == 40 && c->num_layers_all == 41 && c->HasMtp());
  assert(c->hidden_size == 2048 && c->context_length == 262144);
  assert(c->SsmKeyDim() == 2048 && c->SsmValueDim() == 4096);
  assert(c->SsmConvChannels() == 8192);
  assert(c->AttentionQDim() == 4096 && c->AttentionKvDim() == 512);
  assert((c->rope_sections == std::array<std::uint32_t, 4>{11, 11, 10, 0}));
  // Three DeltaNet layers, then one full-attention layer.
  for (std::uint32_t il = 0; il < c->num_layers; ++il)
    assert(c->IsLinearLayer(il) == ((il + 1) % 4 != 0));
  assert(!c->IsLinearLayer(c->num_layers));

  const auto trunk_only = Parse(build::OrnithMetadata(false));
  assert(trunk_only && trunk_only->num_layers == 40 && !trunk_only->HasMtp());
}

void CheckRejections() {
  const auto valid = build::OrnithMetadata();
  std::string error;

  auto wrong = valid;
  wrong["general.architecture"] = std::string{"qwen35"};
  assert(!Parse(wrong, &error) &&
         error == "GGUF architecture is not qwen35moe");

  // Every required key is load-bearing.
  for (const auto key : {"block_count",
                         "embedding_length",
                         "context_length",
                         "full_attention_interval",
                         "attention.head_count",
                         "attention.head_count_kv",
                         "attention.key_length",
                         "rope.dimension_count",
                         "rope.dimension_sections",
                         "ssm.conv_kernel",
                         "ssm.state_size",
                         "ssm.group_count",
                         "ssm.time_step_rank",
                         "ssm.inner_size",
                         "expert_count",
                         "expert_used_count",
                         "expert_feed_forward_length",
                         "expert_shared_feed_forward_length",
                         "attention.layer_norm_rms_epsilon",
                         "rope.freq_base"}) {
    wrong = valid;
    wrong.erase(std::string{"qwen35moe."} + key);
    assert(!Parse(wrong));
  }

  // Geometry other than the inspected artifact, including shapes that are
  // internally consistent, is rejected before any allocation.
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 45},
           {"nextn_predict_layers", 2},
           {"nextn_predict_layers", 41},
           {"embedding_length", 2560},
           {"attention.head_count", 24},
           {"attention.head_count_kv", 4},
           {"attention.key_length", 128},
           {"attention.value_length", 128},
           {"rope.dimension_count", 128},
           {"full_attention_interval", 2},
           {"ssm.time_step_rank", 48},
           {"ssm.inner_size", 6144},
           {"expert_count", 512},
           {"expert_used_count", 10},
           {"expert_used_count", 257},
           {"expert_feed_forward_length", 640},
           {"expert_shared_feed_forward_length", 640},
           {"context_length", 0}}) {
    wrong = valid;
    wrong[std::string{"qwen35moe."} + key] = value;
    assert(!Parse(wrong));
  }
  // Consistent but different DeltaNet geometry (48 x 128 value heads).
  wrong = valid;
  wrong["qwen35moe.ssm.time_step_rank"] = std::uint32_t{48};
  wrong["qwen35moe.ssm.inner_size"] = std::uint32_t{6144};
  assert(!Parse(wrong, &error) &&
         error == "unsupported Ornith-1.5-35B-A3B geometry");

  for (const auto& sections :
       {std::vector<std::uint64_t>{11, 11, 10},
        std::vector<std::uint64_t>{11, 11, 10, 1},
        std::vector<std::uint64_t>{11, 11, 11, 0},
        std::vector<std::uint64_t>{10, 11, 11, 0},
        std::vector<std::uint64_t>{0, 16, 16, 0},
        std::vector<std::uint64_t>{UINT64_MAX, 11, 10, 0}}) {
    wrong = valid;
    wrong["qwen35moe.rope.dimension_sections"] = sections;
    assert(!Parse(wrong));
  }
  for (const build::Value& value :
       {build::Value{std::uint32_t{4}}, build::Value{std::string{"4"}},
        build::Value{std::vector<std::int64_t>{-1, 11, 10, 0}},
        build::Value{std::vector<double>{11.0, 11.0, 10.0, 0.0}}}) {
    wrong = valid;
    wrong["qwen35moe.rope.dimension_sections"] = value;
    assert(!Parse(wrong));
  }

  for (const float eps : {0.0F, -1e-6F, std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
    wrong = valid;
    wrong["qwen35moe.attention.layer_norm_rms_epsilon"] = eps;
    assert(!Parse(wrong));
  }
  wrong = valid;
  wrong["qwen35moe.rope.freq_base"] = 0.0F;
  assert(!Parse(wrong));

  // A present but non-integer optional key is malformed, not absent.
  wrong = valid;
  wrong["qwen35moe.nextn_predict_layers"] = std::string{"1"};
  assert(!Parse(wrong));
}

void CheckExpertWeightScale() {
  auto fields = build::OrnithMetadata();
  for (const float neutral : {0.0F, 1.0F}) {
    fields["qwen35moe.expert_weights_scale"] = neutral;
    assert(Parse(fields));
  }
  fields["qwen35moe.expert_weights_scale"] = 2.5F;
  std::string error;
  assert(!Parse(fields, &error) &&
         error == "expert_weights_scale other than 1 is not supported");
  fields["qwen35moe.expert_weights_scale"] = std::string{"1"};
  assert(!Parse(fields));
}

}  // namespace

int main() {
  CheckInspectedArtifact();
  CheckRejections();
  CheckExpertWeightScale();
  std::cout << "Ornith metadata checks passed.\n";
}
