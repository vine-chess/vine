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

void pawn_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations = Bitboard::ALL_SET);
void knight_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destionations = Bitboard::ALL_SET);
void slider_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destionations = Bitboard::ALL_SET);
void king_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destionations = Bitboard::ALL_SET);
void generate_moves(const BoardState &board, MoveList &move_list);

#endif // MOVE_GEN_HPP
