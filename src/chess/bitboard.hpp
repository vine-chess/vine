#pragma once

#include "../util/types.hpp"

#include <bit>

class BitBoard {
public:
	constexpr BitBoard() : m_raw{} {}
	constexpr BitBoard(u64 bb) : m_raw{bb} {}
	
	[[nodiscard]] constexpr u8 get_lsb() const {
		return std::countr_zero(m_raw);
	}

	constexpr void clear_lsb() {
		m_raw &= m_raw - 1;
	}

	[[nodiscard]] constexpr bool empty() {
		return m_raw == 0;
	}

	[[nodiscard]] constexpr BitBoard operator+(BitBoard other) const {
		return BitBoard{m_raw + other.m_raw};
	}

	constexpr BitBoard& operator+=(BitBoard other) {
		m_raw += other.m_raw;
		return *this;
	}
	
	[[nodiscard]] constexpr BitBoard operator-(BitBoard other) const {
		return BitBoard{m_raw - other.m_raw};
	}

	constexpr BitBoard& operator-=(BitBoard other) {
		m_raw -= other.m_raw;
		return *this;
	}
	
	[[nodiscard]] constexpr BitBoard operator|(BitBoard other) const {
		return BitBoard{m_raw | other.m_raw};
	}

	constexpr BitBoard& operator|=(BitBoard other) {
		m_raw |= other.m_raw;
		return *this;
	}
	
	[[nodiscard]] constexpr BitBoard operator<<(BitBoard other) const {
		return BitBoard{m_raw << other.m_raw};
	}

	constexpr BitBoard& operator<<=(BitBoard other) {
		m_raw <<= other.m_raw;
		return *this;
	}
	
	[[nodiscard]] constexpr BitBoard operator>>(BitBoard other) const {
		return BitBoard{m_raw >> other.m_raw};
	}

	constexpr BitBoard& operator>>=(BitBoard other) {
		m_raw >>= other.m_raw;
		return *this;
	}

private:
	u64 m_raw;
};

