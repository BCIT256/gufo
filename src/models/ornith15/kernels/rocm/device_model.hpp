#ifndef GUFO_MODELS_ORNITH15_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_ORNITH15_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/models/ornith15/weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

namespace gufo::models::ornith15::rocm {

using DeviceTensor = qwen38_flash_next::rocm::DeviceTensor;

struct DeviceLayer {
  bool linear{false};
  DeviceTensor attn_norm, post_attention_norm;
  DeviceTensor ssm_qkv, ssm_gate, ssm_conv1d, ssm_alpha, ssm_beta, ssm_dt,
      ssm_a, ssm_norm, ssm_out;
  DeviceTensor attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm;
  DeviceTensor router, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
      shexp_gate_inp, shexp_gate, shexp_up, shexp_down;
  // NextN block only. Empty on trunk layers.
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_eh_proj, nextn_shared_head_norm;
};

/// Trunk weights, plus the bundled NextN block when the bind requested it.
/// Encodings stay the GGUF originals. No narrowing, repacking, or shared
/// mutable state. An autoregressive bind uploads the trunk alone.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;
  static std::unique_ptr<DeviceModel> Upload(const ModelWeights& weights,
                                             const core::GgufReader& reader,
                                             std::string* error = nullptr);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output_norm() const noexcept { return output_norm_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  [[nodiscard]] bool has_mtp() const noexcept { return mtp_.has_value(); }
  const DeviceLayer& mtp() const { return *mtp_; }
  std::size_t resident_bytes() const noexcept { return bytes_; }

private:
  DeviceModel() = default;
  Config config_;
  DeviceTensor token_embd_, output_norm_, output_;
  std::vector<DeviceLayer> layers_;
  std::optional<DeviceLayer> mtp_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

}  // namespace gufo::models::ornith15::rocm

#endif
