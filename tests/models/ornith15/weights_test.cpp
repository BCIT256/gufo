#include "src/models/ornith15/weights.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "tests/models/ornith15/gguf_builder.hpp"

namespace orn = gufo::models::ornith15;
namespace build = gufo::tests::ornith15;
using gufo::core::GgmlType;

namespace {

// A small vocabulary keeps the embedding and head tensors a few KiB. Real
// expert stacks are hundreds of MiB, so full binds are covered by the
// external-model test; these files stop at the first layer.
constexpr std::uint64_t kVocab = 32;

std::vector<build::Tensor> Head() {
  return {{"token_embd.weight", {2048, kVocab}, GgmlType::kQ6_K},
          {"output_norm.weight", {2048}, GgmlType::kF32},
          {"output.weight", {2048, kVocab}, GgmlType::kQ6_K}};
}

std::string BindError(const std::vector<build::Tensor>& tensors,
                      orn::BindOptions options = {},
                      const build::Metadata& fields = build::OrnithMetadata()) {
  const auto image = build::Build(fields, tensors);
  std::string error;
  const auto weights = orn::ModelWeights::Bind(*image.reader, options, &error);
  assert(!weights.has_value() && !error.empty());
  return error;
}

void CheckBindOrderAndRequiredHead() {
  // With a valid head, binding proceeds into layer 0.
  assert(BindError(Head()) == "missing tensor blk.0.attn_norm.weight");

  // The output head is required; it is never tied to the embedding.
  auto tensors = Head();
  tensors.pop_back();
  assert(BindError(tensors) == "missing tensor output.weight");

  assert(BindError({}) == "missing or malformed token_embd.weight");
  assert(BindError({{"token_embd.weight", {2048}, GgmlType::kF32}}) ==
         "missing or malformed token_embd.weight");
}

void CheckFormatsAndShapes() {
  auto tensors = Head();
  tensors[0].type = GgmlType::kF16;
  assert(BindError(tensors) ==
         "tensor token_embd.weight has unsupported format F16");

  tensors = Head();
  tensors[1].type = GgmlType::kQ8_0;
  tensors[1].dims = {2048};
  assert(BindError(tensors) ==
         "tensor output_norm.weight has unsupported format Q8_0");

  tensors = Head();
  tensors[2].dims = {1024, kVocab};
  tensors[2].type = GgmlType::kQ8_0;
  assert(
      BindError(tensors) ==
      "tensor output.weight has shape [1024, 32, 1], expected [2048, 32, 1]");

  // Norms are F32 only.
  tensors = Head();
  tensors.push_back({"blk.0.attn_norm.weight", {2048}, GgmlType::kF16});
  assert(BindError(tensors) ==
         "tensor blk.0.attn_norm.weight has unsupported format F16");

  tensors = Head();
  tensors.push_back({"blk.0.attn_norm.weight", {2048, 2}, GgmlType::kF32});
  assert(BindError(tensors) ==
         "tensor blk.0.attn_norm.weight has shape [2048, 2, 1], expected "
         "[2048, 1, 1]");
}

void CheckMtpRequiresNextn() {
  assert(BindError(Head(), {.mtp = true}, build::OrnithMetadata(false)) ==
         "MTP requested but the GGUF bundles no NextN block");
  // Configuration errors surface before any tensor is examined.
  auto fields = build::OrnithMetadata();
  fields["qwen35moe.expert_count"] = std::uint32_t{512};
  assert(BindError(Head(), {}, fields) ==
         "unsupported Ornith-1.5-35B-A3B geometry");
}

void CheckRowBytes() {
  orn::TensorRef t;
  t.cols = 2048;
  t.rows = 512;
  t.experts = 256;
  const std::initializer_list<std::pair<GgmlType, std::size_t>> rows = {
      {GgmlType::kF32, 8192},    {GgmlType::kQ8_0, 2176},
      {GgmlType::kQ5_K, 1408},   {GgmlType::kQ6_K, 1680},
      {GgmlType::kIQ4_XS, 1088}, {GgmlType::kQ4_K, 0}};
  for (const auto& [type, bytes] : rows) {
    t.type = type;
    assert(t.RowBytes() == bytes);
  }
  t.type = GgmlType::kIQ4_XS;
  assert(t.SizeBytes() == 1088ULL * 512 * 256);
  t.cols = 100;
  assert(t.RowBytes() == 0);
}

}  // namespace

int main() {
  CheckBindOrderAndRequiredHead();
  CheckFormatsAndShapes();
  CheckMtpRequiresNextn();
  CheckRowBytes();
  std::cout << "Ornith weight binding checks passed.\n";
}
