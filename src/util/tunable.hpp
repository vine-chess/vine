#ifndef TUNEABLE_HPP
#define TUNEABLE_HPP

#include "types.hpp"
#include <string_view>
#include <vector>

namespace util {

#define SPSA

#ifdef SPSA
#define TUNABLE(name, value, min, max) inline Tunable<decltype(value)> name(#name, value, min, max, (max - min) / 20)

constexpr f64 LEARNING_RATE = 0.002;

template <typename T>
class Tunable {
  public:
    explicit Tunable(std::string_view name, T value, T min, T max, f64 step)
        : name_(name), value_(value), min_(min), max_(max) {
        std::string type;
        if constexpr (std::is_same_v<T, i32>) {
            type = "int";
        } else if constexpr (std::is_same_v<T, f32>) {
            type = "float";
        } else {
            throw std::invalid_argument("Tunable: invalid type");
        }
        std::cout << name_ << ", " << type << ", " << value_ << ",  " << min << ", " << max << ", " << step << ", "
                  << LEARNING_RATE << std::endl;
        tunables.emplace_back(this);
    }

    constexpr operator T() const {
        return value_;
    }

    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return name_;
    }

    [[nodiscard]] constexpr T value() const noexcept {
        return value_;
    }

    void set_value(T value) {
        value_ = value;
    }

    [[nodiscard]] constexpr T min() const noexcept {
        return min_;
    }

    [[nodiscard]] constexpr T max() const noexcept {
        return max_;
    }

    static std::vector<Tunable *> tunables;

  private:
    std::string_view name_;
    T value_, min_, max_;
};
#else
#define TUNABLE(name, value, min, max, disabled) static constexpr auto name = value
#endif

} // namespace util

#endif // TUNEABLE_HPP