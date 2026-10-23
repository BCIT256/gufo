// Binds the pinned APEX MTP Quality artifact and checks its tensor formats
// against the inspected APEX Quality recipe. Only GGUF headers are read.
//
//   GUFO_ORNITH15_APEX_MTP_MODEL=/path/to/Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf
//
// Exits 77 (skipped, never passed) when the model is not supplied.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/ornith15/weights.hpp"

namespace orn = gufo::models::ornith15;
using gufo::core::GgmlType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

/// Routed-expert format for each trunk layer in the APEX Quality recipe.
GgmlType RecipeExpertType(std::uint32_t layer) {
  if (layer < 5 || layer >= 35)
    return GgmlType::kQ6_K;
  if (layer < 10 || layer >= 30)
    return GgmlType::kQ5_K;
  return GgmlType::kIQ4_XS;
}

/// The reader's own payload size must match the binder's block geometry.
void CheckSize(const gufo::core::GgufReader& reader, const orn::TensorRef& t) {
  if (t.empty())
    return;
  const auto* info = reader.FindTensor(t.name);
  Require(info != nullptr && info->size_bytes == t.SizeBytes(),
          "encoded size disagrees with the reader for " + std::string(t.name));
}

void CheckLayer(const gufo::core::GgufReader& reader,
                const orn::LayerWeights& l, GgmlType experts, GgmlType dense,
                GgmlType shared, const std::string& where) {
  for (const auto* t :
       {&l.attn_norm, &l.post_attention_norm, &l.router, &l.shexp_gate_inp})
    Require(!t->empty() && t->type == GgmlType::kF32, where + ": F32 tensor");
  for (const auto* t : {&l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps})
    Require(t->type == experts && t->experts == 256,
            where + ": routed expert format " + std::string(t->name));
  for (const auto* t : {&l.shexp_gate, &l.shexp_up, &l.shexp_down})
    Require(t->type == shared, where + ": shared expert format");
  const std::vector<const orn::TensorRef*> projections =
      l.linear ? std::vector<const orn::TensorRef*>{&l.ssm_qkv, &l.ssm_gate,
                                                    &l.ssm_alpha, &l.ssm_beta,
                                                    &l.ssm_out}
               : std::vector<const orn::TensorRef*>{&l.attn_q, &l.attn_k,
                                                    &l.attn_v, &l.attn_out};
  for (const auto* t : projections)
    Require(!t->empty() && t->type == dense,
            where + ": projection format " + std::string(t->name));
  Require(l.linear == l.attn_q.empty() && l.linear != l.ssm_qkv.empty(),
          where + ": layer kind");
  for (const auto* t : {&l.attn_norm,
                        &l.post_attention_norm,
                        &l.ssm_qkv,
                        &l.ssm_gate,
                        &l.ssm_conv1d,
                        &l.ssm_alpha,
                        &l.ssm_beta,
                        &l.ssm_dt,
                        &l.ssm_a,
                        &l.ssm_norm,
                        &l.ssm_out,
                        &l.attn_q,
                        &l.attn_k,
                        &l.attn_v,
                        &l.attn_out,
                        &l.attn_q_norm,
                        &l.attn_k_norm,
                        &l.router,
                        &l.ffn_gate_exps,
                        &l.ffn_up_exps,
                        &l.ffn_down_exps,
                        &l.shexp_gate_inp,
                        &l.shexp_gate,
                        &l.shexp_up,
                        &l.shexp_down,
                        &l.nextn_enorm,
                        &l.nextn_hnorm,
                        &l.nextn_eh_proj,
                        &l.nextn_shared_head_norm})
    CheckSize(reader, *t);
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  if (path == nullptr || !std::filesystem::is_regular_file(path)) {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL to the APEX MTP "
                 "Quality GGUF\n";
    return 77;
  }
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Require(reader != nullptr, "open failed: " + error);

    const auto ar = orn::ModelWeights::Bind(*reader, {}, &error);
    Require(ar.has_value(), "AR bind failed: " + error);
    Require(!ar->mtp.has_value(), "AR bind must leave the NextN block unbound");
    const auto& c = ar->config;
    Require(c.num_layers == 40 && c.HasMtp() && c.vocab_size == 248320,
            "trunk configuration");
    Require(ar->token_embd.type == GgmlType::kQ6_K &&
                ar->output.type == GgmlType::kQ6_K &&
                ar->output.data != ar->token_embd.data,
            "embedding and untied output head");
    CheckSize(*reader, ar->token_embd);
    CheckSize(*reader, ar->output);
    Require(ar->layers.size() == 40, "layer count");
    for (std::uint32_t il = 0; il < 40; ++il)
      CheckLayer(*reader, ar->layers[il], RecipeExpertType(il), GgmlType::kQ6_K,
                 GgmlType::kQ8_0, "blk." + std::to_string(il));

    const auto mtp = orn::ModelWeights::Bind(*reader, {.mtp = true}, &error);
    Require(mtp.has_value() && mtp->mtp.has_value(), "MTP bind: " + error);
    const auto& head = *mtp->mtp;
    CheckLayer(*reader, head, GgmlType::kQ8_0, GgmlType::kQ8_0, GgmlType::kQ8_0,
               "blk.40");
    Require(head.nextn_eh_proj.type == GgmlType::kQ8_0 &&
                head.nextn_eh_proj.cols == 4096 && !head.nextn_enorm.empty() &&
                !head.nextn_hnorm.empty() &&
                !head.nextn_shared_head_norm.empty(),
            "NextN fusion tensors");
    // The MTP bind sees the same trunk tensors as the AR bind.
    for (std::uint32_t il = 0; il < 40; ++il)
      Require(mtp->layers[il].ffn_down_exps.data ==
                  ar->layers[il].ffn_down_exps.data,
              "trunk binding differs between AR and MTP");
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Ornith APEX MTP artifact binding checks passed.\n";
  return 0;
}
