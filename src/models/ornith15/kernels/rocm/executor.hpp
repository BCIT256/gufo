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

/// Private device image of one session frontier. The host may keep prefilling
/// after StageSnapshot; ExportStagedSnapshot reads this image, not the live
/// buffers. The image outlives the session that produced it.
struct StagedSnapshot {
  std::uint32_t position = 0;
  /// Identity of the append-only KV history this image was taken from.
  std::uint64_t lineage = 0;
  std::size_t bytes = 0;
  void* device = nullptr;
  hipEvent_t ready = nullptr;
  alignas(64) std::uint8_t header_bytes[64]{};

  ~StagedSnapshot();
  StagedSnapshot(const StagedSnapshot&) = delete;
  StagedSnapshot& operator=(const StagedSnapshot&) = delete;
  StagedSnapshot() = default;
};

/// Copies a staged image to a host payload on a private stream once the image
/// is ready. Needs no session, so it may run on any thread while the session
/// that staged the image keeps working.
[[nodiscard]] bool ExportStagedImage(const StagedSnapshot& staged,
                                     std::vector<std::uint8_t>* payload,
                                     std::string* error = nullptr);

/// One autoregressive session. The DeviceModel must outlive it. Each session
/// owns its stream, scratch, KV and recurrent/conv state. Prefill runs the
/// prompt in chunks. A plain decode may replay a HIP graph. Device failure
/// requires Reset.
class Session {
public:
  /// 4096 covers a 2,052-token prompt in one launch. A 2,048 cap leaves a
  /// short tail that reads every routed expert again.
  static constexpr std::uint32_t kMaxPrefillChunk = 4096;

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
  ///
  /// With a canonical grid, the call must already sit on a cell boundary.
  /// Up to the chunk size, the call is one launch, its short final cell
  /// included; a longer call splits on cell boundaries and syncs once at the
  /// end. One-token launches use the batched kernels, not the decode dots. A
  /// zero grid keeps the caller's chunk size and syncs every chunk. A token's
  /// result follows the state at the start of its absolute cell, that cell's
  /// tokens, and earlier KV.
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens,
                             std::span<float> last_logits,
                             std::string* error = nullptr,
                             std::uint32_t chunk_tokens = kMaxPrefillChunk,
                             std::span<float> all_logits = {});
  /// Prefill in one launch that also freezes the state at absolute position
  /// `boundary` into a staged image, as StageSnapshot would after a prefill
  /// that stopped there. Requires a grid session without MTP and
  /// Position() < boundary < Position() + tokens.size() <= Position() +
  /// kMaxPrefillChunk. The live session continues to the end of `tokens`.
  /// When no device memory is left for the image, the prefill still runs and
  /// `*staged` is null.
  [[nodiscard]] bool PrefillThrough(std::span<const std::int32_t> tokens,
                                    std::span<float> last_logits,
                                    std::uint32_t boundary,
                                    std::shared_ptr<StagedSnapshot>* staged,
                                    std::string* error = nullptr);
  /// Enables absolute prefill cells. `grid` 0 disables them. The session
  /// must still be at position 0, and `grid` cannot exceed kMaxPrefillChunk.
  [[nodiscard]] bool SetCanonicalGrid(std::uint32_t grid);
  [[nodiscard]] std::uint32_t CanonicalGrid() const noexcept {
    return canonical_grid_;
  }
  /// Test hook. When set, batched Q6_K and Q5_K use the historic
  /// column-dependent product, so a wide call no longer matches 16-token
  /// cells. Production leaves this off. Not a runtime option.
  void SetChunkDependentReductionForTest(bool enable) {
    chunk_dependent_reduction_ = enable;
  }
  /// Test hook. When set, a one-token grid launch uses the decode dots, so it
  /// no longer matches that token inside a wider launch. Not a runtime option.
  void SetLoneTokenDecodeDotsForTest(bool enable) {
    lone_token_decode_dots_ = enable;
  }
  /// Allocates the NextN cache. The model must have been uploaded with the
  /// bundled block, and the session must still be at position 0. Autoregressive
  /// sessions never call this.
  [[nodiscard]] bool EnableMtp(std::string* error = nullptr);
  [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_enabled_; }
  /// One MTP step at the draft frontier. `token` is the token just chosen;
  /// the hidden paired with it is the trunk post-norm when the chain is cold
  /// and the previous MTP head-norm when it is not. Writes draft logits.
  [[nodiscard]] bool DraftToken(std::int32_t token, std::span<float> logits,
                                std::string* error = nullptr);
  /// Queues that step on a private stream and scratch so a trunk decode can
  /// run beside it. FinishDraft downloads the logits. DraftToken is both.
  [[nodiscard]] bool BeginDraft(std::int32_t token,
                                std::string* error = nullptr);
  [[nodiscard]] bool FinishDraft(std::span<float> logits,
                                 std::string* error = nullptr);
  /// Trunk forward of a decode-time verification batch. Not grid-aligned.
  /// Does not write the NextN cache (the draft already did). Updates the
  /// carried trunk hidden from the last token. A batch of 2 to 8 tokens
  /// also stores DeltaNet prefixes for RollbackMtpBatch.
  [[nodiscard]] bool VerifyForward(std::span<const std::int32_t> tokens,
                                   std::span<float> all_logits,
                                   std::string* error = nullptr);
  /// One decode-dot trunk step. Same numerics as Step, including its captured
  /// graph. Refreshes the carried trunk hidden and does not run NextN.
  [[nodiscard]] bool VerifyDecode(std::int32_t token, std::span<float> logits,
                                  std::string* error = nullptr);
  /// DeltaNet prefixes are available for a rejected batch of at most 8 tokens.
  [[nodiscard]] bool mtp_prefix_rollback() const noexcept {
    return gdn_rollback_ready_;
  }
  /// Restores DeltaNet, the convolution window, and the carried trunk hidden
  /// to the first `keep` tokens of the verification batch just run. KV above
  /// the rewound position is left in place and is not read.
  [[nodiscard]] bool RollbackMtpBatch(std::uint32_t keep,
                                      std::string* error = nullptr);
  /// Drops speculative NextN positions above `position` and the draft chain.
  /// Trunk state is unchanged. `position` is the trunk length to keep.
  void RewindMtp(std::uint32_t position) noexcept;
  /// Device copy of the carried trunk hidden, so a rejected batch can replay
  /// from the same seed the draft used.
  void SaveMtpCarry();
  void RestoreMtpCarry();
  [[nodiscard]] bool Reset(std::string* error = nullptr);
  std::uint32_t Position() const noexcept { return position_; }
  std::size_t resident_bytes() const noexcept { return bytes_; }

  /// Host payload version of SaveSnapshot / RestoreSnapshot.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 1;
  /// Bytes SaveSnapshot writes at the current position. KV is the valid
  /// rows only; recurrent and convolution state are whole layers.
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const noexcept;
  /// Payload size of a snapshot at `position`.
  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      std::uint32_t position) const noexcept;
  /// Device-to-host copy of everything Step and Prefill read: position,
  /// gated-attention KV below that position, every DeltaNet recurrent and
  /// short-convolution state, and the logits. A failed session returns false
  /// and leaves `payload` unchanged.
  [[nodiscard]] bool SaveSnapshot(std::vector<std::uint8_t>* payload,
                                  std::string* error = nullptr) const;
  /// Copies the current frontier into private device storage on the session
  /// stream and returns without a host sync. Null when the session cannot
  /// stage. Prefill and Step may then proceed; they are ordered after the copy.
  [[nodiscard]] std::shared_ptr<StagedSnapshot> StageSnapshot(
      std::string* error = nullptr);
  /// Host payload of a staged image. Bit-identical to SaveSnapshot at the
  /// staged position, including after the live session has moved on.
  [[nodiscard]] bool ExportStagedSnapshot(
      const std::shared_ptr<StagedSnapshot>& staged,
      std::vector<std::uint8_t>* payload, std::string* error = nullptr) const;
  /// Device-to-device restore from a staged image. Skips KV rows that are
  /// already the live prefix of this same history.
  [[nodiscard]] bool RestoreStagedSnapshot(const StagedSnapshot& staged,
                                           std::span<float> logits,
                                           std::string* error = nullptr);
  /// True when the most recent RestoreStagedSnapshot left KV in place.
  [[nodiscard]] bool LastRestoreSkippedKv() const noexcept {
    return last_restore_skipped_kv_;
  }
  /// Append-only KV history identity. A staged restore skips KV only when
  /// the destination still carries this history through the staged position.
  [[nodiscard]] std::uint64_t KvLineage() const noexcept { return lineage_; }
  [[nodiscard]] std::uint32_t KvHighWater() const noexcept {
    return kv_high_water_;
  }
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
  struct State {
    float* conv{nullptr};
    float* recurrent{nullptr};
    __half* k{nullptr};
    __half* v{nullptr};
  };
  // Per-prefix DeltaNet images for a verification batch of at most 8 tokens.
  // Row t is the state after token t. Only linear layers are allocated.
  struct GdnRollback {
    float* state[7]{};
    float* conv[7]{};
  };

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
  void Attention(const DeviceLayer& l, State& state, std::uint32_t n);
  /// Post-norm trunk rows into `h_rows_`, then one NextN launch. `token_ids_`
  /// already holds the tokens. Requires `mtp_position_ == position_`.
  void MtpConsume(std::uint32_t n);
  void MtpRun(std::uint32_t n, std::uint32_t start_pos, const float* hidden,
              bool write_head);
  void CaptureTrunkHidden(std::uint32_t n);
  void Moe(const DeviceLayer& l, std::uint32_t n);
  void RunChunk(std::uint32_t n, LayerTrace* trace);
  void DecodeBody(LayerTrace* trace);
  void ProjectLastLogits(std::uint32_t n);
  void ProjectAllLogits(std::uint32_t n);
  void TryCapture();
  void TryCaptureVerify(std::uint32_t n);
  void NoteGraphFailure();
  /// Points the drafter at its private scratch, then swaps back. The decode
  /// graph keeps the addresses it captured, so the two must not share them.
  void SwapDraftScratch() noexcept;
  void EnsureLogitsBatch();
  void* Allocate(std::size_t bytes);
  float* Floats(std::size_t count);
  /// `into_base` copies state into `base`; otherwise copies `base` into state.
  /// `skip_kv` leaves gated-attention K/V untouched. `base` is the payload
  /// address, including the 64-byte header the caller fills separately.
  void TransferPayload(std::uint8_t* base, bool into_base, hipMemcpyKind kind,
                       hipStream_t stream, bool skip_kv,
                       std::uint32_t position) const;

  const DeviceModel* model_{nullptr};
  hipStream_t stream_{nullptr};
  mutable hipStream_t copy_stream_{nullptr};
  hipGraphExec_t graph_exec_{nullptr};
  // One captured trunk forward per verification width. Index 0 is unused.
  hipGraphExec_t verify_graph_[9]{};
  bool verify_warmed_[9]{};
  bool verify_graph_failed_[9]{};
  std::uint32_t max_context_{0}, position_{0};
  std::uint32_t canonical_grid_{0};
  bool chunk_dependent_reduction_{false};
  /// Set while a grid prefill runs: n == 1 keeps the batched kernels.
  bool grid_rows_{false};
  bool lone_token_decode_dots_{false};
  /// Set during PrefillThrough: linear layers write their boundary state
  /// into this image. Offsets are per layer; zero for attention layers.
  struct InPassImage {
    std::uint8_t* base = nullptr;
    std::uint32_t tokens = 0;
    std::vector<std::size_t> conv;
    std::vector<std::size_t> recurrent;
  } in_pass_;
  std::uint64_t lineage_{0};
  std::uint32_t kv_high_water_{0};
  bool last_restore_skipped_kv_{false};
  bool failed_{false};
  bool warmed_{false};
  bool graph_failed_{false};
  bool graph_noted_{false};
  std::size_t bytes_{0};
  std::vector<void*> allocations_;
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
  bool mtp_enabled_{false};
  bool mtp_suspend_{false};
  bool mtp_chain_{false};
  bool gdn_rollback_ready_{false};
  bool gdn_rollback_{false};
  std::uint32_t mtp_position_{0};
  std::uint32_t mtp_batch_tokens_{0};
  std::vector<GdnRollback> gdn_saved_;
  State mtp_state_{};
  float *h_rows_{nullptr}, *h_in_{nullptr}, *h_carry_{nullptr},
      *h_carry_saved_{nullptr}, *chain_h_{nullptr};
  float *e_buf_{nullptr}, *e_norm_{nullptr}, *h_norm_{nullptr},
      *concat_{nullptr};
  // Private n=1 scratch. The decode graph owns the shared buffers above.
  hipStream_t mtp_stream_{nullptr};
  bool mtp_draft_inflight_{false};
  std::int32_t* mtp_token_{nullptr};
  std::uint32_t* mtp_pos_{nullptr};
  float *mtp_res_{nullptr}, *mtp_xn_{nullptr}, *mtp_block_{nullptr},
      *mtp_logits_{nullptr}, *mtp_hidden_{nullptr};
  float *mtp_qg_{nullptr}, *mtp_q_{nullptr}, *mtp_gate_{nullptr},
      *mtp_k_{nullptr}, *mtp_v_{nullptr}, *mtp_ctx_{nullptr};
  float *mtp_router_{nullptr}, *mtp_route_w_{nullptr};
  std::int32_t* mtp_ids_{nullptr};
  float *mtp_expert_gate_{nullptr}, *mtp_expert_up_{nullptr},
      *mtp_expert_down_{nullptr}, *mtp_shared_gate_{nullptr},
      *mtp_shared_up_{nullptr}, *mtp_shared_out_{nullptr},
      *mtp_shared_inp_{nullptr};
};

}  // namespace gufo::models::ornith15::rocm

#endif
