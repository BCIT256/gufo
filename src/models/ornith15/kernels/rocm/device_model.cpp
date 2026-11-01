#include "src/models/ornith15/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <stdexcept>
#include <utility>

#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::ornith15::rocm {

DeviceModel::~DeviceModel() {
  for (void* p : allocations_)
    (void)hipFree(p);
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& weights,
                                                 const core::GgufReader& reader,
                                                 std::string* error) {
  // Composition derived from Flash-Next at 53b0f5b7; its kernel libraries
  // remain linked, not copied. BindOptions{} leaves blk.40 untouched.
  auto model = std::unique_ptr<DeviceModel>(new DeviceModel);
  try {
    if (weights.layers.size() != weights.config.num_layers)
      throw std::runtime_error("GPU upload requires one trunk layer per block");
    if (weights.mtp && weights.mtp->linear)
      throw std::runtime_error("the NextN block is a full-attention layer");
    if (qfn_mmq_init(0))
      throw std::runtime_error("HIP/MMQ initialization failed");
    model->config_ = weights.config;
    model->layers_.resize(weights.layers.size());
    auto stager = hip::WeightUpload::Create(reader.GetMappedRegions(), error);
    if (!stager)
      return nullptr;
    constexpr std::size_t tail = 4096;
    const auto visit = [&](auto&& copy) {
      copy(weights.token_embd, model->token_embd_);
      copy(weights.output_norm, model->output_norm_);
      copy(weights.output, model->output_);
      for (std::size_t i = 0; i < weights.layers.size(); ++i) {
        const auto& w = weights.layers[i];
        auto& d = model->layers_[i];
        d.linear = w.linear;
        for (const auto& [src, dst] :
             {std::pair{&w.attn_norm, &d.attn_norm},
              {&w.post_attention_norm, &d.post_attention_norm},
              {&w.ssm_qkv, &d.ssm_qkv},
              {&w.ssm_gate, &d.ssm_gate},
              {&w.ssm_conv1d, &d.ssm_conv1d},
              {&w.ssm_alpha, &d.ssm_alpha},
              {&w.ssm_beta, &d.ssm_beta},
              {&w.ssm_dt, &d.ssm_dt},
              {&w.ssm_a, &d.ssm_a},
              {&w.ssm_norm, &d.ssm_norm},
              {&w.ssm_out, &d.ssm_out},
              {&w.attn_q, &d.attn_q},
              {&w.attn_k, &d.attn_k},
              {&w.attn_v, &d.attn_v},
              {&w.attn_out, &d.attn_out},
              {&w.attn_q_norm, &d.attn_q_norm},
              {&w.attn_k_norm, &d.attn_k_norm},
              {&w.router, &d.router},
              {&w.ffn_gate_exps, &d.ffn_gate_exps},
              {&w.ffn_up_exps, &d.ffn_up_exps},
              {&w.ffn_down_exps, &d.ffn_down_exps},
              {&w.shexp_gate_inp, &d.shexp_gate_inp},
              {&w.shexp_gate, &d.shexp_gate},
              {&w.shexp_up, &d.shexp_up},
              {&w.shexp_down, &d.shexp_down},
              {&w.nextn_enorm, &d.nextn_enorm},
              {&w.nextn_hnorm, &d.nextn_hnorm},
              {&w.nextn_eh_proj, &d.nextn_eh_proj},
              {&w.nextn_shared_head_norm, &d.nextn_shared_head_norm}})
          if (!src->empty())
            copy(*src, *dst);
      }
      if (weights.mtp) {
        model->mtp_ = DeviceLayer{};
        model->mtp_->linear = false;
        const auto& w = *weights.mtp;
        auto& d = *model->mtp_;
        for (const auto& [src, dst] :
             {std::pair{&w.attn_norm, &d.attn_norm},
              {&w.post_attention_norm, &d.post_attention_norm},
              {&w.attn_q, &d.attn_q},
              {&w.attn_k, &d.attn_k},
              {&w.attn_v, &d.attn_v},
              {&w.attn_out, &d.attn_out},
              {&w.attn_q_norm, &d.attn_q_norm},
              {&w.attn_k_norm, &d.attn_k_norm},
              {&w.router, &d.router},
              {&w.ffn_gate_exps, &d.ffn_gate_exps},
              {&w.ffn_up_exps, &d.ffn_up_exps},
              {&w.ffn_down_exps, &d.ffn_down_exps},
              {&w.shexp_gate_inp, &d.shexp_gate_inp},
              {&w.shexp_gate, &d.shexp_gate},
              {&w.shexp_up, &d.shexp_up},
              {&w.shexp_down, &d.shexp_down},
              {&w.nextn_enorm, &d.nextn_enorm},
              {&w.nextn_hnorm, &d.nextn_hnorm},
              {&w.nextn_eh_proj, &d.nextn_eh_proj},
              {&w.nextn_shared_head_norm, &d.nextn_shared_head_norm}})
          if (!src->empty())
            copy(*src, *dst);
      }
    };
    std::size_t total = 0;
    visit([&](const TensorRef& t, DeviceTensor&) {
      total += t.SizeBytes() + tail;
    });
    const hip::WeightAllocator allocator(total);
    visit([&](const TensorRef& t, DeviceTensor& d) {
      const auto size = t.SizeBytes();
      void* ptr = nullptr;
      if (allocator.Allocate(&ptr, size + tail) != hipSuccess)
        throw std::runtime_error("allocation failed: " + std::string(t.name));
      model->allocations_.push_back(ptr);
      model->bytes_ += size + tail;
      if (!stager->Copy(t.shard, t.file_offset, size, ptr, error) ||
          hipMemset(static_cast<char*>(ptr) + size, 0, tail) != hipSuccess)
        throw std::runtime_error("upload failed: " + std::string(t.name));
      d = {ptr, t.type, static_cast<std::uint32_t>(t.cols),
           static_cast<std::uint32_t>(t.rows),
           static_cast<std::uint32_t>(t.experts)};
    });
    if (!stager->Finish(error))
      return nullptr;
    return model;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::ornith15::rocm
