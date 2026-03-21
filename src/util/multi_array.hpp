#ifndef MULTI_ARRAY_HPP
#define MULTI_ARRAY_HPP

#include <array>
#include <span>
#include <type_traits>

namespace util {

template <typename T, std::size_t Size, std::size_t... Sizes>
class MultiArray;

namespace detail {

template <typename T, std::size_t Size, std::size_t... Sizes>
struct MultiArrayHelper {
    using ChildType = MultiArray<T, Sizes...>;
};

template <typename T, std::size_t Size>
struct MultiArrayHelper<T, Size> {
    using ChildType = T;
};

} // namespace detail

// MultiArray is a generic N-dimensional array.
// The template parameters (Size and Sizes) encode the dimensions of the array.
template <typename T, std::size_t Size, std::size_t... Sizes>
class MultiArray {
    using ChildType = typename detail::MultiArrayHelper<T, Size, Sizes...>::ChildType;
    using ArrayType = std::array<ChildType, Size>;
    ArrayType data_;

  public:
    static_assert(std::is_standard_layout_v<ChildType>);
    static_assert(std::is_trivially_copyable_v<ChildType>);

    using value_type = typename ArrayType::value_type;
    using size_type = typename ArrayType::size_type;
    using difference_type = typename ArrayType::difference_type;
    using reference = typename ArrayType::reference;
    using const_reference = typename ArrayType::const_reference;
    using pointer = typename ArrayType::pointer;
    using const_pointer = typename ArrayType::const_pointer;
    using iterator = typename ArrayType::iterator;
    using const_iterator = typename ArrayType::const_iterator;
    using reverse_iterator = typename ArrayType::reverse_iterator;
    using const_reverse_iterator = typename ArrayType::const_reverse_iterator;

    constexpr auto &at(size_type index) noexcept {
        return data_.at(index);
    }
    constexpr const auto &at(size_type index) const noexcept {
        return data_.at(index);
    }

    constexpr auto &operator[](size_type index) noexcept {
        return data_[index];
    }
    constexpr const auto &operator[](size_type index) const noexcept {
        return data_[index];
    }

    constexpr auto &front() noexcept {
        return data_.front();
    }
    constexpr const auto &front() const noexcept {
        return data_.front();
    }
    constexpr auto &back() noexcept {
        return data_.back();
    }
    constexpr const auto &back() const noexcept {
        return data_.back();
    }

    auto *data() {
        return data_.data();
    }
    const auto *data() const {
        return data_.data();
    }

    constexpr auto begin() noexcept {
        return data_.begin();
    }
    constexpr auto end() noexcept {
        return data_.end();
    }
    constexpr auto begin() const noexcept {
        return data_.begin();
    }
    constexpr auto end() const noexcept {
        return data_.end();
    }
    constexpr auto cbegin() const noexcept {
        return data_.cbegin();
    }
    constexpr auto cend() const noexcept {
        return data_.cend();
    }

    constexpr auto rbegin() noexcept {
        return data_.rbegin();
    }
    constexpr auto rend() noexcept {
        return data_.rend();
    }
    constexpr auto rbegin() const noexcept {
        return data_.rbegin();
    }
    constexpr auto rend() const noexcept {
        return data_.rend();
    }
    constexpr auto crbegin() const noexcept {
        return data_.crbegin();
    }
    constexpr auto crend() const noexcept {
        return data_.crend();
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return data_.empty();
    }
    constexpr size_type size() const noexcept {
        return data_.size();
    }
    constexpr size_type max_size() const noexcept {
        return data_.max_size();
    }

    [[nodiscard]] static consteval size_type flat_size() noexcept {
        return Size * (Sizes * ... * 1);
    }

    [[nodiscard]] constexpr T *flat_data() noexcept {
        return reinterpret_cast<T *>(data_.data());
    }

    [[nodiscard]] constexpr const T *flat_data() const noexcept {
        return reinterpret_cast<const T *>(data_.data());
    }

    [[nodiscard]] constexpr auto flat_span() noexcept {
        return std::span<T, flat_size()>(flat_data(), flat_size());
    }

    [[nodiscard]] constexpr auto flat_span() const noexcept {
        return std::span<const T, flat_size()>(flat_data(), flat_size());
    }

    constexpr MultiArray<T, Size, Sizes...> &operator=(const MultiArray<T, Size, Sizes...> &other) = default;

    template <bool NoExtraDimension = sizeof...(Sizes) == 0,
              typename = typename std::enable_if_t<NoExtraDimension, bool>>
    constexpr MultiArray<T, Size, Sizes...> &operator=(const std::array<T, Size> &other) noexcept {
        data_ = other;
        return *this;
    }

    template <bool NoExtraDimension = sizeof...(Sizes) == 0,
              typename = typename std::enable_if_t<NoExtraDimension, bool>>
    constexpr std::array<T, Size> &as_array() noexcept {
        return data_;
    }

    template <bool NoExtraDimension = sizeof...(Sizes) == 0,
              typename = typename std::enable_if_t<NoExtraDimension, bool>>
    constexpr const std::array<T, Size> &as_array() const noexcept {
        return data_;
    }

    template <typename U>
    void fill(const U &v) {
        for (auto &ele : data_) {
            if constexpr (sizeof...(Sizes) == 0)
                ele = v;
            else
                ele.fill(v);
        }
    }

    constexpr void swap(MultiArray<T, Size, Sizes...> &other) noexcept {
        data_.swap(other.data_);
    }
};

} // namespace util

#endif // MULTI_ARRAY_HPP
