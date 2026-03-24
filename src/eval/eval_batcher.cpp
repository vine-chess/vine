#include "eval_batcher.hpp"

namespace network {

PolicyQueue::PolicyQueue() = default;

PolicyQueue::~PolicyQueue() {
    stop();
}

void PolicyQueue::start() {
    std::scoped_lock lock(state_mutex_);

    if (gpu_thread_.joinable()) {
        return;
    }

    stop_requested_ = false;
    gpu_thread_ = std::thread(&PolicyQueue::gpu_loop, this);
}

void PolicyQueue::stop() {
    {
        std::scoped_lock lock(mutex_);
        stop_requested_ = true;
        cv_producer_.notify_all();
        cv_consumers_.notify_all();
    }

    if (gpu_thread_.joinable()) {
        gpu_thread_.join();
    }
}

PolicyQueue::PolicySlot PolicyQueue::reserve_policy_slot(u32 len) {
    std::unique_lock lock(state_mutex_);

    if (len > MAX_MOVES) {
        throw std::runtime_error("too many moves for one policy request");
    }

    // Wait until either the "GPU" stops or we have an available slot to reserve
    cv_consumers_.wait(lock,
                       [&] { return stop_requested_ || (phase_ == Phase::Filling && reserved_slots_ < kBatchSize); });

    if (stop_requested_) {
        throw std::runtime_error("policy queue stopped");
    }

    const u32 board_idx = reserved_slots_++;
    slots_[board_idx].move_count = len;
    slots_[board_idx].ready = false;
    slots_[board_idx].consumed = false;

    return PolicySlot{generation_, board_idx, std::span<u16>(&slots_[board_idx].move_indices, len)};
}

void PolicyQueue::mark_ready(const PolicySlot &slot) {
    std::scoped_lock lock(state_mutex_);

    if (slot.generation != generation_) {
        throw std::runtime_error("mark_ready called with stale generation");
    }

    if (slot.board_idx >= reserved_slots_) {
        throw std::runtime_error("invalid board index in mark_ready");
    }

    if (slots_[slot.board_idx].ready) {
        throw std::runtime_error("slot marked ready twice");
    }

    slots_[slot.board_idx].ready = true;
    ++ready_slots_;

    // Notify the "GPU" if all slots are now ready
    if (ready_slots_ == kBatchSize) {
        cv_producer_.notify_one();
    }
}

PolicyQueue::BatchResult PolicyQueue::wait_for_result(const PolicySlot &slot) {
    std::unique_lock lock(state_mutex_);

    // Wait for the current batch to be processed before consuming this slot's result
    cv_consumers_.wait(lock, [&] {
        return stop_requested_ || (phase_ == Phase::Completed && completed_generation_ == slot.generation);
    });

    if (stop_requested_) {
        throw std::runtime_error("policy queue stopped");
    }

    return batch_results_[slot.board_idx];
}

void PolicyQueue::mark_consumed(const PolicySlot &slot) {
    std::scoped_lock lock(mutex_);

    if (phase_ != Phase::Completed || completed_generation_ != slot.generation) {
        throw std::runtime_error("mark_consumed called for non-completed generation");
    }

    if (slot.board_idx >= completed_slots_) {
        throw std::runtime_error("invalid board index in mark_consumed");
    }

    if (slots_[slot.board_idx].consumed) {
        throw std::runtime_error("slot consumed twice");
    }

    slots_[slot.board_idx].consumed = true;
    ++consumed_slots_;

    if (consumed_slots_ == completed_boards_) {
        reset_slots();
        // Notify the "GPU" that all batch results have been consumed, so that it may begin waiting for the next batch
        cv_producer_.notify_one();
        // Notify to all threads that all batch results have been consumed, so that they may reserve another slot
        cv_consumers_.notify_all();
    }
}

void PolicyQueue::reset_slots() {
    // Reset all slot information for the next batch
    ++generation_;
    reserved_slots_ = 0;
    ready_slots_ = 0;
    completed_slots_ = 0;
    consumed_slots_ = 0;
    phase_ = Phase::Filling;

    for (auto &slot : slots_) {
        slot = {};
    }
}

void PolicyQueue::gpu_loop() {
    std::unique_lock lock(state_mutex_);

    while (!stop_requested_) {
        // wait and process batches
    }
}

} // namespace network