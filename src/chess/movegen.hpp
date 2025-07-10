#ifndef MOVEGEN_HPP
#define MOVEGEN_HPP

#include "bitboard.hpp"
#include <array>

constexpr static auto KNIGHT_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        const auto sq =  Bitboard(1ull << i);
        const auto forward_back = sq.shift_masked<2, 0>() |  sq.shift_masked<-2, 0>();
        const auto left_right = sq.shift_masked<0, 2>() |  sq.shift_masked<0, -2>();
        res[i] = 
            forward_back.shift_masked<0, 1>() | forward_back.shift_masked<0, -1>() | 
            left_right.shift_masked<1, 0>() | left_right.shift_masked<-1, 0>();  
    }
    return res;
}();

constexpr static auto KING_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        res[i] = 1ull << i;
        res[i] |= res[i].shift_masked<1, 0>();
        res[i] |= res[i].shift_masked<-1, 0>();
        res[i] |= res[i].shift_masked<0, 1>();
        res[i] |= res[i].shift_masked<0, -1>();
        res[i] ^= 1ull << i;
    }
    return res;
}();

#endif // MOVEGEN_HPP
