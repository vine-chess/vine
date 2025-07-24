#ifndef STATIC_VECTOR_HPP
#define STATIC_VECTOR_HPP

#include "assert.hpp"
#include "types.hpp"

#include <array>
#include <memory>
#include <type_traits>

namespace util {

template <typename T, usize max_size>
class StaticVector {
  public:
    using value_type = T;
    using iterator = T *;
    using const_iterator = const T *;
    using size_type = usize;

    StaticVector() : size_(0) {}

    ~StaticVector() {
        clear();
    }

    explicit StaticVector(usize size) : size_(size) {
        std::uninitialized_value_construct_n(data(), size);
    }

    iterator push_back(const T &value) {
        vine_assert(size_ < max_size);
        T *res = std::construct_at(data() + size_, value);
        size_++;
        return res;
    }

    iterator push_back(T &&value) {
        return emplace_back(std::move(value));
    }

    template <typename... Args>
    iterator emplace_back(Args &&...args) {
        vine_assert(size_ < max_size);
        T *res = std::construct_at(data() + size_, std::forward<Args>(args)...);
        size_++;
        return res;
    }

    void push_back_conditional(const T &value, bool condition) {
        static_assert(std::is_trivially_destructible_v<T>, "T has to be a completely trivial type for this");
        data()[size_] = value;
        size_ += condition;
    }

    T pop_back() {
        T &last = back();
        const T res = std::move(last);
        std::destroy_at(&last);
        size_--;
        return res;
    }

    void clear() {
        std::destroy_n(data(), size_);
        size_ = 0;
    }

    [[nodiscard]] usize size() const {
        return size_;
    }

    [[nodiscard]] constexpr usize capacity() const {
        return max_size;
    }

    [[nodiscard]] bool empty() const {
        return size_ == 0;
    }

    [[nodiscard]] T *data() {
        return reinterpret_cast<T *>(data_.data());
    }

    [[nodiscard]] const T *data() const {
        return reinterpret_cast<const T *>(data_.data());
    }

    [[nodiscard]] iterator begin() {
        return data();
    }

    [[nodiscard]] const_iterator begin() const {
        return data();
    }

    [[nodiscard]] const_iterator cbegin() const {
        return begin();
    }

    [[nodiscard]] iterator end() {
        return begin() + size_;
    }

    [[nodiscard]] const_iterator end() const {
        return begin() + size_;
    }

    [[nodiscard]] const_iterator cend() const {
        return end();
    }

    T &back() {
        vine_assert(size_ > 0);
        return data()[size_ - 1];
    }

    const T &back() const {
        vine_assert(size_ > 0);
        return data()[size_ - 1];
    }

    [[nodiscard]] T &operator[](usize index) {
        vine_assert(index < size_);
        return data()[index];
    }

    [[nodiscard]] const T &operator[](usize index) const {
        vine_assert(index < size_);
        return data()[index];
    }

  private:
    alignas(T) std::array<std::byte, max_size * sizeof(T)> data_;
    usize size_;
};

} // namespace util

#endif // STATIC_VECTOR_HPP
