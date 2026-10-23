#ifndef GUFO_TESTS_MODELS_ORNITH15_GGUF_BUILDER_HPP_
#define GUFO_TESTS_MODELS_ORNITH15_GGUF_BUILDER_HPP_

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::tests::ornith15 {

using Value =
    std::variant<std::uint32_t, float, std::string, std::vector<std::uint64_t>,
                 std::vector<std::int64_t>, std::vector<double>>;
using Metadata = std::map<std::string, Value>;

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  core::GgmlType type{core::GgmlType::kF32};
};

/// Encoded size of a tensor in the formats the Ornith binder accepts, plus F16
/// for rejection cases. Payloads are zero-filled; only headers are tested.
inline std::uint64_t TensorBytes(const Tensor& t) {
  std::uint64_t block = 1;
  std::uint64_t bytes = 4;
  switch (t.type) {
    case core::GgmlType::kF32:
      break;
    case core::GgmlType::kF16:
      bytes = 2;
      break;
    case core::GgmlType::kQ8_0:
      block = 32, bytes = 34;
      break;
    case core::GgmlType::kQ5_K:
      block = 256, bytes = 176;
      break;
    case core::GgmlType::kQ6_K:
      block = 256, bytes = 210;
      break;
    case core::GgmlType::kIQ4_XS:
      block = 256, bytes = 136;
      break;
    default:
      throw std::invalid_argument("test builder has no size for this type");
  }
  std::uint64_t elements = 1;
  for (auto d : t.dims)
    elements *= d;
  return elements / block * bytes;
}

/// An in-memory GGUF v3 image and the reader parsed from it. The image must
/// outlive the reader, so both are owned together.
struct Image {
  std::vector<std::uint8_t> bytes;
  std::unique_ptr<core::GgufReader> reader;
};

inline Image Build(const Metadata& fields, const std::vector<Tensor>& tensors) {
  constexpr std::uint64_t kAlignment = 32;
  Image image;
  auto& bytes = image.bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{tensors.size()});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            pod(std::uint32_t{9});
            using E = typename T::value_type;
            pod(std::uint32_t{std::is_same_v<E, std::uint64_t>  ? 10U
                              : std::is_same_v<E, std::int64_t> ? 11U
                                                                : 12U});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(entry);
          }
        },
        value);
  }
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    text(t.name);
    pod(static_cast<std::uint32_t>(t.dims.size()));
    for (auto d : t.dims)
      pod(d);
    pod(static_cast<std::uint32_t>(t.type));
    pod(offset);
    offset += (TensorBytes(t) + kAlignment - 1) / kAlignment * kAlignment;
  }
  bytes.resize((bytes.size() + kAlignment - 1) / kAlignment * kAlignment +
               offset);
  std::string error;
  image.reader =
      core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  if (!image.reader)
    throw std::runtime_error("synthetic GGUF rejected: " + error);
  return image;
}

/// The metadata of the inspected Ornith-1.5-35B-A3B APEX MTP artifact.
inline Metadata OrnithMetadata(bool with_mtp = true) {
  Metadata fields{{"general.architecture", std::string{"qwen35moe"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", with_mtp ? 41U : 40U},
           {"embedding_length", 2048},
           {"context_length", 262144},
           {"full_attention_interval", 4},
           {"attention.head_count", 16},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 256},
           {"attention.value_length", 256},
           {"rope.dimension_count", 64},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 32},
           {"ssm.inner_size", 4096},
           {"expert_count", 256},
           {"expert_used_count", 8},
           {"expert_feed_forward_length", 512},
           {"expert_shared_feed_forward_length", 512}}) {
    fields[std::string{"qwen35moe."} + key] = value;
  }
  if (with_mtp)
    fields["qwen35moe.nextn_predict_layers"] = std::uint32_t{1};
  fields["qwen35moe.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["qwen35moe.rope.freq_base"] = 1e7F;
  fields["qwen35moe.rope.dimension_sections"] =
      std::vector<std::uint64_t>{11, 11, 10, 0};
  return fields;
}

}  // namespace gufo::tests::ornith15

#endif  // GUFO_TESTS_MODELS_ORNITH15_GGUF_BUILDER_HPP_
