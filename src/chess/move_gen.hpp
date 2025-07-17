#ifndef MOVE_GEN_HPP
#define MOVE_GEN_HPP

#include "../util/static_vector.hpp"
#include "bitboard.hpp"
#include "board_state.hpp"
#include <array>

using MoveList = util::StaticVector<Move, 218>;

constexpr static auto UP = Bitboard::UP;
constexpr static auto DOWN = Bitboard::DOWN;
constexpr static auto LEFT = Bitboard::LEFT;
constexpr static auto RIGHT = Bitboard::RIGHT;

constexpr static auto KNIGHT_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        const auto sq = Bitboard(Square(i));
        const auto forward_back = sq.shift<UP * 2, 0>() | sq.shift<DOWN * 2, 0>();
        const auto left_right = sq.shift<0, LEFT * 2>() | sq.shift<0, RIGHT * 2>();
        res[i] = forward_back.shift<0, LEFT>() | forward_back.shift<0, RIGHT>() | left_right.shift<UP, 0>() |
                 left_right.shift<DOWN, 0>();
    }
    return res;
}();

constexpr static auto KING_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        res[i] = Bitboard(Square(i));
        res[i] |= res[i].shift<UP, 0>();
        res[i] |= res[i].shift<DOWN, 0>();
        res[i] |= res[i].shift<0, RIGHT>();
        res[i] |= res[i].shift<0, LEFT>();
        res[i] ^= Bitboard(Square(i));
    }
    return res;
}();

/// [SQ][COLOR]
constexpr static auto PAWN_ATTACKS = []() {
    std::array<std::array<Bitboard, 2>, 64> res;
    for (int i = 0; i < 64; ++i) {
        res[i][0] = Bitboard(Square(i)).shift<UP, LEFT>() | Bitboard(Square(i)).shift<UP, RIGHT>();
        res[i][1] = Bitboard(Square(i)).shift<DOWN, LEFT>() | Bitboard(Square(i)).shift<DOWN, RIGHT>();
    }
    return res;
}();

constexpr static auto BISHOP_RAYS = []() {
    std::array<Bitboard, 65> res;
    for (int i = 0; i < 65; ++i) {
        res[i] = Bitboard::get_ray<UP, LEFT>(Square(i)) | Bitboard::get_ray<UP, RIGHT>(Square(i)) |
                 Bitboard::get_ray<DOWN, LEFT>(Square(i)) | Bitboard::get_ray<DOWN, RIGHT>(Square(i));
    }
    return res;
}();

constexpr static auto ROOK_RAYS = []() {
    std::array<Bitboard, 65> res;
    for (int i = 0; i < 65; ++i) {
        res[i] = Bitboard::get_ray<UP, 0>(i) | Bitboard::get_ray<0, RIGHT>(Square(i)) |
                 Bitboard::get_ray<0, LEFT>(Square(i)) | Bitboard::get_ray<DOWN, 0>(Square(i));
    }
    return res;
}();

constexpr static auto QUEEN_RAYS = []() {
    std::array<Bitboard, 65> res;
    for (int i = 0; i < 65; ++i) {
        res[i] = BISHOP_RAYS[i] | ROOK_RAYS[i];
    }
    return res;
}();

constexpr static auto ROOK_RAY_BETWEEN = []() {
    std::array<std::array<Bitboard, 65>, 65> res;
    for (int i = 0; i < 65; ++i) {
        for (int j = 0; j < 65; ++j) {
            res[i][j] = Bitboard::get_ray<UP, 0>(i) & Bitboard::get_ray<DOWN, 0>(j) |
                        Bitboard::get_ray<UP, 0>(j) & Bitboard::get_ray<DOWN, 0>(i) |
                        Bitboard::get_ray<0, LEFT>(i) & Bitboard::get_ray<0, RIGHT>(j) |
                        Bitboard::get_ray<0, LEFT>(j) & Bitboard::get_ray<0, RIGHT>(i);
        }
    }
    return res;
}();

constexpr static auto BISHOP_RAY_BETWEEN = []() {
    std::array<std::array<Bitboard, 65>, 65> res;
    for (int i = 0; i < 65; ++i) {
        for (int j = 0; j < 65; ++j) {
            res[i][j] = Bitboard::get_ray<UP, LEFT>(i) & Bitboard::get_ray<DOWN, RIGHT>(j) |
                        Bitboard::get_ray<UP, LEFT>(j) & Bitboard::get_ray<DOWN, RIGHT>(i) |
                        Bitboard::get_ray<DOWN, LEFT>(i) & Bitboard::get_ray<UP, RIGHT>(j) |
                        Bitboard::get_ray<DOWN, LEFT>(j) & Bitboard::get_ray<UP, RIGHT>(i);
        }
    }
    return res;
}();

constexpr static auto RAY_BETWEEN = []() {
    std::array<std::array<Bitboard, 65>, 65> res;
    for (int i = 0; i < 65; ++i) {
        for (int j = 0; j < 65; ++j) {
            res[i][j] = ROOK_RAY_BETWEEN[i][j] | BISHOP_RAY_BETWEEN[i][j];
        }
    }
    return res;
}();

void pawn_moves(const BoardState &state, MoveList &move_list, Bitboard allowed_destinations = Bitboard::ALL_SET);
void knight_moves(const BoardState &state, MoveList &move_list, Bitboard allowed_destionations = Bitboard::ALL_SET);
void slider_moves(const BoardState &state, MoveList &move_list, Bitboard allowed_destionations = Bitboard::ALL_SET);
void king_moves(const BoardState &state, MoveList &move_list, Bitboard allowed_destionations = Bitboard::ALL_SET);
void generate_moves(const BoardState &state, MoveList &move_list);

#endif // MOVE_GEN_HPP
