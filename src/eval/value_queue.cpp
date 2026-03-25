#include "value_queue.hpp"
#include "value_network.hpp"

namespace network {

ValueQueue::ValueQueue() = default;

ValueQueue::~ValueQueue() {
    stop();
}

void ValueQueue::start() {
    std::scoped_lock lock(state_mutex_);

    if (gpu_thread_.joinable()) {
        return;
    }

    stop_requested_ = false;
    gpu_thread_ = std::thread(&ValueQueue::gpu_loop, this);
}

void ValueQueue::stop() {
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

ValueQueue::ValueSlot ValueQueue::reserve_value_slot(const BoardState &state) {
    std::unique_lock lock(state_mutex_);

    // Wait until either the "GPU" stops or we have an available slot to reserve
    cv_consumers_.wait(lock,
                       [&] { return stop_requested_ || (phase_ == Phase::FILLING && reserved_slots_ < BATCH_SIZE); });

    if (stop_requested_) {
        throw std::runtime_error("value queue stopped");
    }

    const u32 board_idx = reserved_slots_++;
    slots_[board_idx].board_state = state;
    slots_[board_idx].ready = false;
    slots_[board_idx].consumed = false;

    return {board_idx};
}

void ValueQueue::mark_ready(const ValueSlot &slot) {
    std::scoped_lock lock(state_mutex_);

    if (slot.board_idx >= reserved_slots_) {
        throw std::runtime_error("invalid board index in mark_ready");
    }

    if (slots_[slot.board_idx].ready) {
        throw std::runtime_error("slot marked ready twice");
    }

    slots_[slot.board_idx].ready = true;
    ++ready_slots_;

    // Notify the "GPU" if all slots are now ready
    if (ready_slots_ == BATCH_SIZE) {
        cv_producer_.notify_one();
    }
}

ValueQueue::ValueResult ValueQueue::wait_for_result(const ValueSlot &slot) {
    std::unique_lock lock(state_mutex_);

    // Wait for the current batch to be processed before consuming this slot's result
    cv_consumers_.wait(lock, [&] { return stop_requested_ || phase_ == Phase::COMPLETED; });

    if (stop_requested_) {
        throw std::runtime_error("value queue stopped");
    }

    return batch_results_[slot.board_idx];
}

void ValueQueue::mark_consumed(const ValueSlot &slot) {
    std::scoped_lock lock(state_mutex_);

    if (phase_ != Phase::COMPLETED) {
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

    if (consumed_slots_ == completed_slots_) {
        reset_slots();
        // Notify the "GPU" that all batch results have been consumed, so that it may begin waiting for the next batch
        cv_producer_.notify_one();
        // Notify to all threads that all batch results have been consumed, so that they may reserve another slot
        cv_consumers_.notify_all();
    }
}

void ValueQueue::reset_slots() {
    // Reset all slot information for the next batch
    reserved_slots_ = 0;
    ready_slots_ = 0;
    completed_slots_ = 0;
    consumed_slots_ = 0;
    phase_ = Phase::FILLING;
    slots_ = {};
}

void ValueQueue::gpu_loop() {
    std::unique_lock lock(state_mutex_);

    while (!stop_requested_) {
        // Wait until a stop request, or we have all slots ready to be evaluated
        cv_producer_.wait(lock, [&] { return stop_requested_ || ready_slots_ == BATCH_SIZE; });

        if (stop_requested_) {
            break;
        }

        phase_ = Phase::PROCESSING;
        process_batch();
        phase_ = Phase::COMPLETED;

        // Notify all threads that this batch has been completed
        cv_consumers_.notify_all();

        // Wait for this batch to be consumed before waiting for all ready slots
        cv_producer_.wait(lock, [&] { return stop_requested_ || phase_ == Phase::FILLING; });
    }
}

void ValueQueue::process_batch() {
    for (u32 batch_idx = 0; batch_idx < BATCH_SIZE; ++batch_idx, ++completed_slots_) {
        const auto &slot = slots_[batch_idx];
        auto &result = batch_results_[batch_idx];
        result.value = network::value::evaluate(slot.board_state);
    }
}

GlobalValueQueue &GlobalValueQueue::get() {
    static GlobalValueQueue instance;
    return instance;
}

ValueQueue &GlobalValueQueue::queue() {
    return queue_;
}

} // namespace network