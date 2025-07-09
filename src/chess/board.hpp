#pragma once

#include "../util/types.hpp"
#include "bitboard.hpp"

#include <array>

class Board {
public:
	u64 perft(int depth);
private: 
	std::array<BitBoard, 6> pieces_;
};
