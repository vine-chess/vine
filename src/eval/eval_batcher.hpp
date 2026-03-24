#ifndef VINE_EVAL_BATCHER_H
#define VINE_EVAL_BATCHER_H

#include "../chess/move_gen.hpp"
#include "../util/types.hpp"
#include <array>
#include <condition_variable>
#include <mutex>
#include <span>
#include <thread>

namespace network {

class PolicyQueue {
  public:
    static constexpr u32 BATCH_SIZE = 1;

    // Thread-facing copy of the information it needs to retrieve the result of its evaluation request
    struct PolicySlot {
        // The index into the processed batch results
        u32 board_idx = 0;
        // Span over all move indices, filled by an individual thread
        std::span<u16> move_indices;
        // Span over all moving piece types, filled by an individual thread
        std::span<PieceType> piece_types;
    };

    // Structure containing the logit results of an individual request in a batch
    struct BatchResult {
        std::array<f32, MAX_MOVES> logits{};
    };

    PolicyQueue();
    PolicyQueue(const PolicyQueue &) = delete;
    PolicyQueue &operator=(const PolicyQueue &) = delete;
    ~PolicyQueue();

    // Spawn the "GPU" worker/polling thread
    void start();
    // Stop the "GPU" worker/polling thread
    void stop();

    // Reserves a slot in the current batch for processing (may block until a slot is available)
    [[nodiscard]] PolicySlot reserve_policy_slot(const BoardState &state, u32 len);

    // Called by a thread after it has filled its slot with the move indices to be evaluated
    void mark_ready(const PolicySlot &slot);

    // Blocks a thread until the "GPU" has processed the current batch
    [[nodiscard]] BatchResult wait_for_result(const PolicySlot &slot);

    // Called by a thread after it has consumed the results of its evaluations
    void mark_consumed(const PolicySlot &slot);

  private:
    void gpu_loop();

    void process_batch();

    void reset_slots();

    // Internal structure to track the state of an evaluation request
    struct InternalSlot {
        BoardState board_state;
        u32 move_count = 0;
        bool ready = false;
        bool consumed = false;
    };

    enum class Phase {
        FILLING,    // The queue still has slots to be filled with
        PROCESSING, // The "GPU" has begun processing each evaluation (unused for now)
        COMPLETED   // The "GPU" has finished evaluating all requests of the current batch
    };

    // The state of the current batch
    Phase phase_ = Phase::FILLING;
    // The evaluated results of the most recent batch
    std::array<BatchResult, BATCH_SIZE> batch_results_{};
    // The move information that will be passed to inference (must be dense and kept separate from the slot structure)
    std::array<std::array<u16, MAX_MOVES>, BATCH_SIZE> move_indices_;
    std::array<std::array<PieceType, MAX_MOVES>, BATCH_SIZE> move_piece_types_;
    // Structure that holds information about each evaluation request
    std::array<InternalSlot, BATCH_SIZE> slots_{};
    // Information about the current slots
    u32 reserved_slots_ = 0;
    u32 ready_slots_ = 0;
    u32 completed_slots_ = 0;
    u32 consumed_slots_ = 0;
    // The worker/"GPU" thread
    std::thread gpu_thread_;
    // Mutex to ensure only one thread can modify the queue's state at a time
    std::mutex state_mutex_;
    // Condition variable to the "GPU" that the batch can be processed now
    std::condition_variable cv_producer_;
    // Condition variable to the threads that the batch results can be consumed now
    std::condition_variable cv_consumers_;
    // Boolean to signal to the "GPU" to exit
    bool stop_requested_ = false;
};

class GlobalPolicyQueue {
  public:
    static GlobalPolicyQueue &get();

    [[nodiscard]] PolicyQueue &queue();

  private:
    GlobalPolicyQueue() = default;

    PolicyQueue queue_;
};

} // namespace network

#endif // VINE_EVAL_BATCHER_H