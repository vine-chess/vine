#ifndef GPU_QUEUE_HPP
#define GPU_QUEUE_HPP

#include "policy_network.hpp"
#include "value_network.hpp"

#include "../chess/move_gen.hpp"
#include "../util/types.hpp"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace network {

class GpuValueQueue {
  public:
    struct ValueSlot {
        u32 board_idx = 0;
    };

    struct ValueResult {
        f64 value = 0.0;
    };

    explicit GpuValueQueue(u32 batch_size = 1) : batch_size_(batch_size), batch_results_(batch_size), slots_(batch_size) {}

    GpuValueQueue(const GpuValueQueue &) = delete;
    GpuValueQueue &operator=(const GpuValueQueue &) = delete;

    ~GpuValueQueue() {
        stop();
    }

    void set_batch_size(u32 batch_size) {
        batch_size = std::max<u32>(1, batch_size);
        const bool restart = gpu_thread_.joinable();
        if (restart) {
            stop();
        }

        {
            std::scoped_lock lock(state_mutex_);
            batch_size_ = batch_size;
            batch_results_.assign(batch_size_, {});
            slots_.assign(batch_size_, {});
            reset_slots();
            stop_requested_ = false;
        }

        if (restart) {
            start();
        }
    }

    [[nodiscard]] u32 batch_size() const {
        return batch_size_;
    }

    void start() {
        std::scoped_lock lock(state_mutex_);

        if (gpu_thread_.joinable()) {
            return;
        }

        stop_requested_ = false;
        gpu_thread_ = std::thread(&GpuValueQueue::gpu_loop, this);
    }

    void stop() {
        {
            std::scoped_lock lock(state_mutex_);
            stop_requested_ = true;
            cv_producer_.notify_all();
            cv_consumers_.notify_all();
        }

        if (gpu_thread_.joinable()) {
            gpu_thread_.join();
        }
    }

    [[nodiscard]] ValueSlot reserve_value_slot(const BoardState &state) {
        std::unique_lock lock(state_mutex_);
        cv_consumers_.wait(lock, [&] { return stop_requested_ || (phase_ == Phase::FILLING && reserved_slots_ < batch_size_); });

        const u32 board_idx = reserved_slots_++;
        slots_[board_idx].board_state = state;
        slots_[board_idx].ready = false;
        slots_[board_idx].consumed = false;
        return {board_idx};
    }

    void mark_ready(const ValueSlot &slot) {
        std::scoped_lock lock(state_mutex_);

        slots_[slot.board_idx].ready = true;
        ++ready_slots_;

        if (ready_slots_ == batch_size_) {
            cv_producer_.notify_one();
        }
    }

    [[nodiscard]] ValueResult wait_for_result(const ValueSlot &slot) {
        std::unique_lock lock(state_mutex_);
        cv_consumers_.wait(lock, [&] { return stop_requested_ || phase_ == Phase::COMPLETED; });

        return batch_results_[slot.board_idx];
    }

    void mark_consumed(const ValueSlot &slot) {
        std::scoped_lock lock(state_mutex_);

        slots_[slot.board_idx].consumed = true;
        ++consumed_slots_;

        if (consumed_slots_ == completed_slots_) {
            reset_slots();
            cv_producer_.notify_one();
            cv_consumers_.notify_all();
        }
    }

  private:
    struct InternalSlot {
        BoardState board_state;
        bool ready = false;
        bool consumed = false;
    };

    enum class Phase { FILLING, PROCESSING, COMPLETED };

    void reset_slots() {
        reserved_slots_ = 0;
        ready_slots_ = 0;
        completed_slots_ = 0;
        consumed_slots_ = 0;
        phase_ = Phase::FILLING;
        std::fill(slots_.begin(), slots_.end(), InternalSlot{});
    }

    void gpu_loop() {
        std::unique_lock lock(state_mutex_);

        while (!stop_requested_) {
            cv_producer_.wait(lock, [&] { return stop_requested_ || ready_slots_ == batch_size_; });

            if (stop_requested_) {
                break;
            }

            phase_ = Phase::PROCESSING;
            process_batch();
            phase_ = Phase::COMPLETED;
            cv_consumers_.notify_all();
            cv_producer_.wait(lock, [&] { return stop_requested_ || phase_ == Phase::FILLING; });
        }
    }

    void process_batch() {
        std::vector<value::CudaBoardInput> inputs(batch_size_);
        std::vector<f32> outputs(batch_size_, 0.0f);

        for (u32 batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
            const auto &state = slots_[batch_idx].board_state;
            auto &input = inputs[batch_idx];
            std::copy(state.piece_type_on_sq.begin(), state.piece_type_on_sq.end(), std::begin(input.pieces));
            input.side_to_move = static_cast<u8>(state.side_to_move);
        }

        value::evaluate_many(inputs.data(), outputs.data(), batch_size_);

        for (u32 batch_idx = 0; batch_idx < batch_size_; ++batch_idx, ++completed_slots_) {
            batch_results_[batch_idx].value = outputs[batch_idx];
        }
    }

    u32 batch_size_ = 1;
    Phase phase_ = Phase::FILLING;
    std::vector<ValueResult> batch_results_;
    std::vector<InternalSlot> slots_;
    u32 reserved_slots_ = 0;
    u32 ready_slots_ = 0;
    u32 completed_slots_ = 0;
    u32 consumed_slots_ = 0;
    std::thread gpu_thread_;
    std::mutex state_mutex_;
    std::condition_variable cv_producer_;
    std::condition_variable cv_consumers_;
    bool stop_requested_ = false;
};

class GlobalGpuValueQueue {
  public:
    static GlobalGpuValueQueue &get() {
        static GlobalGpuValueQueue instance;
        return instance;
    }

    [[nodiscard]] GpuValueQueue &queue() {
        return queue_;
    }

  private:
    GlobalGpuValueQueue() = default;
    GpuValueQueue queue_;
};

class GpuPolicyQueue {
  public:
    struct PolicySlot {
        u32 board_idx = 0;
        std::span<u16> move_indices;
    };

    struct PolicyResult {
        std::array<f32, MAX_MOVES> logits{};
    };

    explicit GpuPolicyQueue(u32 batch_size = 1)
        : batch_size_(batch_size), batch_results_(batch_size), move_indices_(batch_size), slots_(batch_size) {}

    GpuPolicyQueue(const GpuPolicyQueue &) = delete;
    GpuPolicyQueue &operator=(const GpuPolicyQueue &) = delete;

    ~GpuPolicyQueue() {
        stop();
    }

    void set_batch_size(u32 batch_size) {
        batch_size = std::max<u32>(1, batch_size);
        const bool restart = gpu_thread_.joinable();
        if (restart) {
            stop();
        }

        {
            std::scoped_lock lock(state_mutex_);
            batch_size_ = batch_size;
            batch_results_.assign(batch_size_, {});
            move_indices_.assign(batch_size_, {});
            slots_.assign(batch_size_, {});
            reset_slots();
            stop_requested_ = false;
        }

        if (restart) {
            start();
        }
    }

    [[nodiscard]] u32 batch_size() const {
        return batch_size_;
    }

    void start() {
        std::scoped_lock lock(state_mutex_);

        if (gpu_thread_.joinable()) {
            return;
        }

        stop_requested_ = false;
        gpu_thread_ = std::thread(&GpuPolicyQueue::gpu_loop, this);
    }

    void stop() {
        {
            std::scoped_lock lock(state_mutex_);
            stop_requested_ = true;
            cv_producer_.notify_all();
            cv_consumers_.notify_all();
        }

        if (gpu_thread_.joinable()) {
            gpu_thread_.join();
        }
    }

    [[nodiscard]] PolicySlot reserve_policy_slot(const BoardState &state, u32 len) {
        std::unique_lock lock(state_mutex_);
        cv_consumers_.wait(lock, [&] { return stop_requested_ || (phase_ == Phase::FILLING && reserved_slots_ < batch_size_); });

        const u32 board_idx = reserved_slots_++;
        slots_[board_idx].board_state = state;
        slots_[board_idx].move_count = std::min<u32>(len, MAX_MOVES);
        slots_[board_idx].ready = false;
        slots_[board_idx].consumed = false;

        return {board_idx, std::span(move_indices_[board_idx].data(), slots_[board_idx].move_count)};
    }

    void mark_ready(const PolicySlot &slot) {
        std::scoped_lock lock(state_mutex_);

        slots_[slot.board_idx].ready = true;
        ++ready_slots_;

        if (ready_slots_ == batch_size_) {
            cv_producer_.notify_one();
        }
    }

    [[nodiscard]] PolicyResult wait_for_result(const PolicySlot &slot) {
        std::unique_lock lock(state_mutex_);
        cv_consumers_.wait(lock, [&] { return stop_requested_ || phase_ == Phase::COMPLETED; });

        return batch_results_[slot.board_idx];
    }

    void mark_consumed(const PolicySlot &slot) {
        std::scoped_lock lock(state_mutex_);

        slots_[slot.board_idx].consumed = true;
        ++consumed_slots_;

        if (consumed_slots_ == completed_slots_) {
            reset_slots();
            cv_producer_.notify_one();
            cv_consumers_.notify_all();
        }
    }

  private:
    struct InternalSlot {
        BoardState board_state;
        u32 move_count = 0;
        bool ready = false;
        bool consumed = false;
    };

    enum class Phase { FILLING, PROCESSING, COMPLETED };

    void reset_slots() {
        reserved_slots_ = 0;
        ready_slots_ = 0;
        completed_slots_ = 0;
        consumed_slots_ = 0;
        phase_ = Phase::FILLING;
        std::fill(slots_.begin(), slots_.end(), InternalSlot{});
        std::fill(move_indices_.begin(), move_indices_.end(), std::array<u16, MAX_MOVES>{});
    }

    void gpu_loop() {
        std::unique_lock lock(state_mutex_);

        while (!stop_requested_) {
            cv_producer_.wait(lock, [&] { return stop_requested_ || ready_slots_ == batch_size_; });

            if (stop_requested_) {
                break;
            }

            phase_ = Phase::PROCESSING;
            process_batch();
            phase_ = Phase::COMPLETED;
            cv_consumers_.notify_all();
            cv_producer_.wait(lock, [&] { return stop_requested_ || phase_ == Phase::FILLING; });
        }
    }

    void process_batch() {
        std::vector<policy::CudaPolicyInput> inputs(batch_size_);
        usize total_move_count = 0;
        for (u32 batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
            total_move_count += slots_[batch_idx].move_count;
        }

        std::vector<u16> flat_move_indices(total_move_count);
        std::vector<f32> flat_outputs(total_move_count, 0.0f);

        usize move_offset = 0;
        for (u32 batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
            const auto &state = slots_[batch_idx].board_state;
            const auto &slot = slots_[batch_idx];
            auto &input = inputs[batch_idx];
            std::copy(state.piece_type_on_sq.begin(), state.piece_type_on_sq.end(), std::begin(input.pieces));
            input.move_offset = static_cast<u32>(move_offset);
            input.move_count = static_cast<u8>(slot.move_count);
            input.side_to_move = static_cast<u8>(state.side_to_move);

            std::copy_n(move_indices_[batch_idx].begin(), slot.move_count, flat_move_indices.begin() + move_offset);
            move_offset += slot.move_count;
        }

        policy::evaluate_many(inputs.data(), flat_move_indices.data(), flat_outputs.data(), batch_size_);

        move_offset = 0;
        for (u32 batch_idx = 0; batch_idx < batch_size_; ++batch_idx, ++completed_slots_) {
            auto &result = batch_results_[batch_idx];
            result.logits.fill(0.0f);
            std::copy_n(flat_outputs.begin() + move_offset, slots_[batch_idx].move_count, result.logits.begin());
            move_offset += slots_[batch_idx].move_count;
        }
    }

    u32 batch_size_ = 1;
    Phase phase_ = Phase::FILLING;
    std::vector<PolicyResult> batch_results_;
    std::vector<std::array<u16, MAX_MOVES>> move_indices_;
    std::vector<InternalSlot> slots_;
    u32 reserved_slots_ = 0;
    u32 ready_slots_ = 0;
    u32 completed_slots_ = 0;
    u32 consumed_slots_ = 0;
    std::thread gpu_thread_;
    std::mutex state_mutex_;
    std::condition_variable cv_producer_;
    std::condition_variable cv_consumers_;
    bool stop_requested_ = false;
};

class GlobalGpuPolicyQueue {
  public:
    static GlobalGpuPolicyQueue &get() {
        static GlobalGpuPolicyQueue instance;
        return instance;
    }

    [[nodiscard]] GpuPolicyQueue &queue() {
        return queue_;
    }

  private:
    GlobalGpuPolicyQueue() = default;
    GpuPolicyQueue queue_;
};

} // namespace network

#endif // GPU_QUEUE_HPP
