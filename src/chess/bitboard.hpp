#pragma once

#include "../util/types.hpp"

#include <bit>

class BitBoard {
public:
	constexpr BitBoard() : raw_{} {}
	constexpr BitBoard(u64 bb) : raw_{bb} {}
	
	[[nodiscard]] constexpr u8 get_lsb() const {
		return std::countr_zero(raw_);
	}

	constexpr void clear_lsb() {
		raw_ &= raw_ - 1;
	}

	[[nodiscard]] constexpr bool empty() {
		return raw_ == 0;
	}

	[[nodiscard]] constexpr BitBoard operator+(BitBoard other) const {
		return BitBoard{raw_ + other.raw_};
	}

	constexpr BitBoard& operator+=(BitBoard other) {
		raw_ += other.raw_;
		return *this;
	}
	
	[[nodiscard]] constexpr BitBoard operator-(BitBoard other) const {
		return BitBoard{raw_ - other.raw_};
	}

	constexpr BitBoard& operator-=(BitBoard other) {
		raw_ -= other.raw_;
		return *this;
	}
	
	[[nodiscard]] constexpr BitBoard operator|(BitBoard other) const {
		return BitBoard{raw_ | other.raw_};
	}

	constexpr BitBoard& operator|=(BitBoard other) {
		raw_ |= other.raw_;
		return *this;
	}
	
	[[nodiscard]] constexpr BitBoard operator<<(BitBoard other) const {
		return BitBoard{raw_ << other.raw_};
	}

	constexpr BitBoard& operator<<=(BitBoard other) {
		raw_ <<= other.raw_;
		return *this;
	}
	
	[[nodiscard]] constexpr BitBoard operator>>(BitBoard other) const {
		return BitBoard{raw_ >> other.raw_};
	}

	constexpr BitBoard& operator>>=(BitBoard other) {
		raw_ >>= other.raw_;
		return *this;
	}
	
	class Iterator {

	};

private:
	u64 raw_;
};

class Iterator {

	public:
	BitBoard state_;

	constexpr Iterator(BitBoard bb) : state_{bb} {}
		
	constexpr Square operator*() const {
		return Square{state_.get_lsb()};
	}
};
