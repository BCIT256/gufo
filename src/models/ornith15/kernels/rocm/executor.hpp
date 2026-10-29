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

/// One autoregressive session. The DeviceModel must outlive it. Each session
/// owns its stream, scratch, KV and recurrent/conv state. Prefill runs the
/// prompt in chunks. A plain decode may replay a HIP graph. Device failure
/// requires Reset.
class Session {
public:
  static constexpr std::uint32_t kMaxPrefillChunk = 512;

  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  static std::unique_ptr<Session> Create(const DeviceModel& model,
                                         std::uint32_t max_context,
                                         std::string* error = nullptr);
  /// Advances one token. Writes `logits` when it holds one vocabulary row.
  /// When `sampled` is set, writes the device argmax (lowest index on ties);
  /// an empty `logits` span then skips the vocabulary download. Tracing runs
  /// only when `trace` is set. A plain step may replay a captured graph.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::string* error = nullptr,
                          LayerTrace* trace = nullptr,
                          std::int32_t* sampled = nullptr);
  /// Runs `tokens` in chunks of at most `chunk_tokens` (maximum
  /// kMaxPrefillChunk). `last_logits` receives the final position. When
  /// `all_logits` has one vocabulary row per token, every position is
  /// returned. Conv, recurrent and KV state carry across chunks. Invalid
  /// arguments return false and leave state unchanged.
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens,
                             std::span<float> last_logits,
                             std::string* error = nullptr,
                             std::uint32_t chunk_tokens = kMaxPrefillChunk,
                             std::span<float> all_logits = {});
  [[nodiscard]] bool Reset(std::string* error = nullptr);
  std::uint32_t Position() const noexcept { return position_; }
  std::size_t resident_bytes() const noexcept { return bytes_; }

  /// Host payload version of SaveSnapshot / RestoreSnapshot.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 1;
  /// Bytes SaveSnapshot writes at the current position. KV is the valid
  /// rows only; recurrent and convolution state are whole layers.
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const noexcept;
  /// Device-to-host copy of everything Step and Prefill read: position,
  /// gated-attention KV below that position, every DeltaNet recurrent and
  /// short-convolution state, and the logits. A failed session returns false
  /// and leaves `payload` unchanged.
  [[nodiscard]] bool SaveSnapshot(std::vector<std::uint8_t>* payload,
                                  std::string* error = nullptr) const;
  /// Host-to-device restore into this session, fresh or already used.
  /// `logits` receives the restored row when it holds one vocabulary row;
  /// an empty span leaves that download to the caller. An invalid payload
  /// returns false and does not modify this session.
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::span<float> logits,
                                     std::string* error = nullptr);
  /// Offset and length of one linear layer's recurrent matrix inside a
  /// version-1 payload. False when the payload is not that version or the
  /// layer is not DeltaNet.
  [[nodiscard]] static bool SnapshotRecurrentRegion(
      std::span<const std::uint8_t> payload, std::uint32_t layer,
      std::size_t* offset, std::size_t* bytes);

private:
  Session() = default;
  void Dense(const DeviceTensor& w, const float* x, float* out);
  void Project(const DeviceTensor& w, const float* x, float* out,
               std::uint32_t n);
  void Quantized(const DeviceTensor& w, const float* x, const std::int32_t* ids,
                 float* out, std::uint32_t rows, std::uint32_t used);
  void MoeRaw(const DeviceTensor& w, const float* x, const std::int32_t* ids,
              float* out, std::uint32_t n_tokens, std::uint32_t used);
  void LinearAttention(const DeviceLayer& l, std::size_t layer,
                       std::uint32_t n);
  void Attention(const DeviceLayer& l, std::size_t layer, std::uint32_t n);
  void Moe(const DeviceLayer& l, std::uint32_t n);
  void RunChunk(std::uint32_t n, LayerTrace* trace);
  void DecodeBody(LayerTrace* trace);
  void ProjectLastLogits(std::uint32_t n);
  void ProjectAllLogits(std::uint32_t n);
  void TryCapture();
  void NoteGraphFailure();
  void EnsureLogitsBatch();
  void* Allocate(std::size_t bytes);
  float* Floats(std::size_t count);

  const DeviceModel* model_{nullptr};
  hipStream_t stream_{nullptr};
  hipGraphExec_t graph_exec_{nullptr};
  std::uint32_t max_context_{0}, position_{0};
  bool failed_{false};
  bool warmed_{false};
  bool graph_failed_{false};
  bool graph_noted_{false};
  std::size_t bytes_{0};
  std::vector<void*> allocations_;
  struct State {
    float* conv{nullptr};
    float* recurrent{nullptr};
    __half* k{nullptr};
    __half* v{nullptr};
  };
  std::vector<State> state_;
  // Scratch is sized for kMaxPrefillChunk. Decode uses the first row.
  std::int32_t* zero_ids_{nullptr};
  std::int32_t* zero_id_{nullptr};
  std::int32_t* token_ids_{nullptr};
  std::uint32_t* device_pos_{nullptr};
  float *res_{nullptr}, *xn_{nullptr}, *block_out_{nullptr}, *logits_{nullptr};
  float* logits_batch_{nullptr};
  float *qkv_{nullptr}, *z_{nullptr}, *ab_{nullptr}, *alpha_{nullptr},
      *beta_{nullptr}, *conv_{nullptr}, *qn_{nullptr}, *kn_{nullptr},
      *raw_{nullptr}, *gdn_out_{nullptr};
  float *qg_{nullptr}, *q_{nullptr}, *gate_{nullptr}, *k_{nullptr},
      *v_{nullptr}, *ctx_{nullptr};
  float *router_{nullptr}, *route_weights_{nullptr}, *expert_gate_{nullptr},
      *expert_up_{nullptr}, *expert_down_{nullptr}, *shared_gate_{nullptr},
      *shared_up_{nullptr}, *shared_out_{nullptr}, *shared_inp_{nullptr};
  std::int32_t* ids_{nullptr};
  void* argmax_scratch_{nullptr};
  std::int32_t* sampled_dev_{nullptr};
};

}  // namespace gufo::models::ornith15::rocm

#endif
