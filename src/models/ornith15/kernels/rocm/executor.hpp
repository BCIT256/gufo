#ifndef GUFO_MODELS_ORNITH15_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_ORNITH15_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/ornith15/kernels/rocm/device_model.hpp"
#include "src/models/ornith15/trace.hpp"

namespace gufo::models::ornith15::rocm {

#ifdef GUFO_ORNITH15_TEST_PROBE
/// Test-only capture. The production library is built without this type, and
/// its Step signature cannot replace layer-0 QKV. `edit_qkv` may overwrite
/// that projection before GatedDeltaNet; leaving it null only copies.
struct Layer0Probe {
  void (*edit_qkv)(void* ctx, float* qkv, const float* norm,
                   std::uint32_t channels, std::uint32_t hidden,
                   hipStream_t stream) = nullptr;
  void* edit_ctx{nullptr};
  float* norm_out{nullptr};
  float* qkv_before{nullptr};
  float* qkv_out{nullptr};
  float* conv_out{nullptr};
  float* state_out{nullptr};
  std::int32_t* route_ids{nullptr};
  float* route_weights{nullptr};
  float* shared_gate{nullptr};
  int detail_layer{-1};
  float* detail_moe_in{nullptr};
  float* detail_router{nullptr};
  float* detail_expert_down{nullptr};
  float* detail_shared{nullptr};
  float* detail_moe_out{nullptr};
};
#endif

/// One eager single-token AR session. The DeviceModel must outlive it.
/// Each session owns its stream, scratch, KV and recurrent/conv state.
/// No graphs, batching, speculative decode or chunked prefill.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  static std::unique_ptr<Session> Create(const DeviceModel& model,
                                         std::uint32_t max_context,
                                         std::string* error = nullptr);
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::string* error = nullptr,
                          LayerTrace* trace = nullptr
#ifdef GUFO_ORNITH15_TEST_PROBE
                          ,
                          Layer0Probe* probe = nullptr
#endif
  );
  [[nodiscard]] bool Reset(std::string* error = nullptr);
  std::uint32_t Position() const noexcept { return position_; }
  std::size_t resident_bytes() const noexcept { return bytes_; }

private:
  Session() = default;
  void Dense(const DeviceTensor& w, const float* x, float* out);
  void Quantized(const DeviceTensor& w, const float* x, const std::int32_t* ids,
                 float* out, std::uint32_t rows, std::uint32_t used);
  void LinearAttention(const DeviceLayer& l, std::size_t layer
#ifdef GUFO_ORNITH15_TEST_PROBE
                       ,
                       Layer0Probe* probe
#endif
  );
  void Attention(const DeviceLayer& l, std::size_t layer);
  void Moe(const DeviceLayer& l
#ifdef GUFO_ORNITH15_TEST_PROBE
           ,
           std::size_t layer, Layer0Probe* probe
#endif
  );
  void* Allocate(std::size_t bytes);
  float* Floats(std::size_t count);

  const DeviceModel* model_{nullptr};
  hipStream_t stream_{nullptr};
  std::uint32_t max_context_{0}, position_{0};
  bool failed_{false};
  std::size_t bytes_{0};
  std::vector<void*> allocations_;
  struct State {
    float* conv{nullptr};
    float* recurrent{nullptr};
    __half* k{nullptr};
    __half* v{nullptr};
  };
  std::vector<State> state_;
  // A single row of each intermediate, sized exclusively from Config.
  std::int32_t* zero_id_{nullptr};
  std::uint32_t* device_pos_{nullptr};
  float *res_{nullptr}, *xn_{nullptr}, *block_out_{nullptr}, *logits_{nullptr};
  float *qkv_{nullptr}, *z_{nullptr}, *ab_{nullptr}, *conv_{nullptr},
      *qn_{nullptr}, *kn_{nullptr}, *raw_{nullptr}, *gdn_out_{nullptr};
  float *qg_{nullptr}, *q_{nullptr}, *gate_{nullptr}, *k_{nullptr},
      *v_{nullptr}, *ctx_{nullptr};
  float *router_{nullptr}, *route_weights_{nullptr}, *expert_gate_{nullptr},
      *expert_up_{nullptr}, *expert_down_{nullptr}, *shared_gate_{nullptr},
      *shared_up_{nullptr}, *shared_out_{nullptr}, *shared_inp_{nullptr};
  std::int32_t* ids_{nullptr};
};

}  // namespace gufo::models::ornith15::rocm

#endif
