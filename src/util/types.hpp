#pragma once

#include <cstdint>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

class Square {
public:
	[[nodiscard]] constexpr bool is_valid() {
		return m_raw < 64;
	}
private:
	u8 m_raw;
};

