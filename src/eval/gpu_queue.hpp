#ifndef GPU_QUEUE_HPP
#define GPU_QUEUE_HPP

#include "policy_network.hpp"
#include "value_network.hpp"

#include "../chess/move_gen.hpp"
#include "../util/ring_queue.hpp"
#include "../util/types.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace network {

class GpuValueQueue {
  public:
    using ValueSlot = u32;
    using ValueResult = f64;

    explicit GpuValueQueue(u32 batch_size = 1) : batch_size_(batch_size) {
        reset_storage();
    }

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
            reset_storage();
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
        stop_requested_.store(true, std::memory_order_relaxed);

        if (gpu_thread_.joinable()) {
            gpu_thread_.join();
        }
    }

    [[nodiscard]] ValueSlot reserve_value_slot(const BoardState &state) {
        u32 board_idx = 0;
        while (!stop_requested_.load(std::memory_order_relaxed) && !free_slots_.try_pop(board_idx)) {
            std::this_thread::yield();
        }

        slots_[board_idx].board_state = state;
        slots_[board_idx].done.store(false, std::memory_order_relaxed);
        return board_idx;
    }

    void mark_ready(const ValueSlot &slot) {
        while (!stop_requested_.load(std::memory_order_relaxed) && !ready_slots_.try_push(slot)) {
            std::this_thread::yield();
        }
    }

    [[nodiscard]] ValueResult wait_for_result(const ValueSlot &slot) {
        while (!stop_requested_.load(std::memory_order_relaxed) &&
               !slots_[slot].done.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        return batch_results_[slot];
    }

    void mark_consumed(const ValueSlot &slot) {
        while (!stop_requested_.load(std::memory_order_relaxed) && !free_slots_.try_push(slot)) {
            std::this_thread::yield();
        }
    }

  private:
    struct InternalSlot {
        BoardState board_state;
        std::atomic<bool> done{false};
    };

    void reset_storage() {
        batch_results_ = std::make_unique<ValueResult[]>(batch_size_);
        slots_ = std::make_unique<InternalSlot[]>(batch_size_);
        ready_indices_ = std::make_unique<u32[]>(batch_size_);
        free_slots_.reset(batch_size_);
        ready_slots_.reset(batch_size_);
        for (u32 board_idx = 0; board_idx < batch_size_; ++board_idx) {
            (void) free_slots_.try_push(board_idx);
        }
    }

    void gpu_loop() {
        while (!stop_requested_.load(std::memory_order_relaxed)) {
            u32 first_idx = 0;
            if (!ready_slots_.try_pop(first_idx)) {
                std::this_thread::yield();
                continue;
            }

            usize count = 1;
            ready_indices_[0] = first_idx;
            while (count < batch_size_ && ready_slots_.try_pop(ready_indices_[count])) {
                ++count;
            }

            process_batch(std::span(ready_indices_.get(), count));
        }
    }

    void process_batch(std::span<const u32> ready_indices) {
        std::vector<value::CudaBoardInput> inputs(ready_indices.size());
        std::vector<f32> outputs(ready_indices.size(), 0.0f);

        for (usize batch_idx = 0; batch_idx < ready_indices.size(); ++batch_idx) {
            const auto &state = slots_[ready_indices[batch_idx]].board_state;
            auto &input = inputs[batch_idx];
            input.pieces.compress(state.piece_type_on_sq);
            input.side_to_move = static_cast<u8>(state.side_to_move);
        }

        value::evaluate_many(inputs.data(), outputs.data(), static_cast<u32>(ready_indices.size()));

        for (usize batch_idx = 0; batch_idx < ready_indices.size(); ++batch_idx) {
            batch_results_[ready_indices[batch_idx]] = outputs[batch_idx];
            slots_[ready_indices[batch_idx]].done.store(true, std::memory_order_release);
        }
    }

    u32 batch_size_ = 1;
    std::unique_ptr<ValueResult[]> batch_results_;
    std::unique_ptr<InternalSlot[]> slots_;
    std::unique_ptr<u32[]> ready_indices_;
    util::RingQueue<u32> free_slots_;
    util::RingQueue<u32> ready_slots_;
    std::thread gpu_thread_;
    std::mutex state_mutex_;
    std::atomic<bool> stop_requested_ = false;
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
        : batch_size_(batch_size) {
        reset_storage();
    }

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
            reset_storage();
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
        stop_requested_.store(true, std::memory_order_relaxed);

        if (gpu_thread_.joinable()) {
            gpu_thread_.join();
        }
    }

    [[nodiscard]] PolicySlot reserve_policy_slot(const BoardState &state, u32 len) {
        u32 board_idx = 0;
        while (!stop_requested_.load(std::memory_order_relaxed) && !free_slots_.try_pop(board_idx)) {
            std::this_thread::yield();
        }

        slots_[board_idx].board_state = state;
        slots_[board_idx].move_count = std::min<u32>(len, MAX_MOVES);
        slots_[board_idx].done.store(false, std::memory_order_relaxed);

        return {board_idx, std::span(move_indices_[board_idx].data(), slots_[board_idx].move_count)};
    }

    void mark_ready(const PolicySlot &slot) {
        while (!stop_requested_.load(std::memory_order_relaxed) && !ready_slots_.try_push(slot.board_idx)) {
            std::this_thread::yield();
        }
    }

    [[nodiscard]] PolicyResult wait_for_result(const PolicySlot &slot) {
        while (!stop_requested_.load(std::memory_order_relaxed) &&
               !slots_[slot.board_idx].done.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        return batch_results_[slot.board_idx];
    }

    void mark_consumed(const PolicySlot &slot) {
        while (!stop_requested_.load(std::memory_order_relaxed) && !free_slots_.try_push(slot.board_idx)) {
            std::this_thread::yield();
        }
    }

  private:
    struct InternalSlot {
        BoardState board_state;
        u32 move_count = 0;
        std::atomic<bool> done{false};
    };

    void reset_storage() {
        batch_results_ = std::make_unique<PolicyResult[]>(batch_size_);
        move_indices_ = std::make_unique<std::array<u16, MAX_MOVES>[]>(batch_size_);
        slots_ = std::make_unique<InternalSlot[]>(batch_size_);
        ready_indices_ = std::make_unique<u32[]>(batch_size_);
        free_slots_.reset(batch_size_);
        ready_slots_.reset(batch_size_);
        for (u32 board_idx = 0; board_idx < batch_size_; ++board_idx) {
            (void) free_slots_.try_push(board_idx);
        }
    }

    void gpu_loop() {
        while (!stop_requested_.load(std::memory_order_relaxed)) {
            u32 first_idx = 0;
            if (!ready_slots_.try_pop(first_idx)) {
                std::this_thread::yield();
                continue;
            }

            usize count = 1;
            ready_indices_[0] = first_idx;
            while (count < batch_size_ && ready_slots_.try_pop(ready_indices_[count])) {
                ++count;
            }

            process_batch(std::span(ready_indices_.get(), count));
        }
    }

    void process_batch(std::span<const u32> ready_indices) {
        std::vector<policy::CudaPolicyInput> inputs(ready_indices.size());
        usize total_move_count = 0;
        for (const auto board_idx : ready_indices) {
            total_move_count += slots_[board_idx].move_count;
        }

        std::vector<u16> flat_move_indices(total_move_count);
        std::vector<f32> flat_outputs(total_move_count, 0.0f);

        usize move_offset = 0;
        for (usize batch_idx = 0; batch_idx < ready_indices.size(); ++batch_idx) {
            const auto board_idx = ready_indices[batch_idx];
            const auto &state = slots_[board_idx].board_state;
            const auto &slot = slots_[board_idx];
            auto &input = inputs[batch_idx];
            input.pieces.compress(state.piece_type_on_sq);
            input.move_offset = static_cast<u32>(move_offset);
            input.move_count = static_cast<u8>(slot.move_count);
            input.side_to_move = static_cast<u8>(state.side_to_move);

            std::copy_n(move_indices_[board_idx].begin(), slot.move_count, flat_move_indices.begin() + move_offset);
            move_offset += slot.move_count;
        }

        policy::evaluate_many(inputs.data(), flat_move_indices.data(), flat_outputs.data(),
                              static_cast<u32>(ready_indices.size()));

        move_offset = 0;
        for (usize batch_idx = 0; batch_idx < ready_indices.size(); ++batch_idx) {
            const auto board_idx = ready_indices[batch_idx];
            auto &result = batch_results_[board_idx];
            result.logits.fill(0.0f);
            std::copy_n(flat_outputs.begin() + move_offset, slots_[board_idx].move_count, result.logits.begin());
            move_offset += slots_[board_idx].move_count;
            slots_[board_idx].done.store(true, std::memory_order_release);
        }
    }

    u32 batch_size_ = 1;
    std::unique_ptr<PolicyResult[]> batch_results_;
    std::unique_ptr<std::array<u16, MAX_MOVES>[]> move_indices_;
    std::unique_ptr<InternalSlot[]> slots_;
    std::unique_ptr<u32[]> ready_indices_;
    util::RingQueue<u32> free_slots_;
    util::RingQueue<u32> ready_slots_;
    std::thread gpu_thread_;
    std::mutex state_mutex_;
    std::atomic<bool> stop_requested_ = false;
};

} // namespace network

#endif // GPU_QUEUE_HPP
