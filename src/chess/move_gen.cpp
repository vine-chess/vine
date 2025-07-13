#include "move_gen.hpp"

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

constexpr static auto BISHOP_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        res[i] = Bitboard::get_ray<UP, LEFT>(Square(i)) | Bitboard::get_ray<UP, RIGHT>(Square(i)) |
                 Bitboard::get_ray<DOWN, LEFT>(Square(i)) | Bitboard::get_ray<DOWN, RIGHT>(Square(i));
    }
    return res;
}();

constexpr static auto ROOK_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        res[i] = Bitboard::get_ray<UP, 0>(Square(i)) | Bitboard::get_ray<0, RIGHT>(Square(i)) |
                 Bitboard::get_ray<0, LEFT>(Square(i)) | Bitboard::get_ray<DOWN, 0>(Square(i));
    }
    return res;
}();

constexpr static auto QUEEN_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        res[i] = BISHOP_MOVES[i] | ROOK_MOVES[i];
    }
    return res;
}();

inline Bitboard compute_bishop_attacks(Square sq, Bitboard occ) {
    auto up_left = Bitboard::get_ray_precomputed<UP, LEFT>(sq);
    auto up_right = Bitboard::get_ray_precomputed<UP, RIGHT>(sq);
    auto down_left = Bitboard::get_ray_precomputed<DOWN, LEFT>(sq);
    auto down_right = Bitboard::get_ray_precomputed<DOWN, RIGHT>(sq);
    up_left &= ~Bitboard::get_ray_precomputed<UP, LEFT>((up_left & occ).lsb());
    up_right &= ~Bitboard::get_ray_precomputed<UP, RIGHT>((up_right & occ).lsb());
    down_left &= ~Bitboard::get_ray_precomputed<DOWN, LEFT>((down_left & occ).msb());
    down_right &= ~Bitboard::get_ray_precomputed<DOWN, RIGHT>((down_right & occ).msb());
    return up_left | up_right | down_left | down_right;
}

inline Bitboard compute_rook_attacks(Square sq, Bitboard occ) {
    auto up = Bitboard::get_ray_precomputed<UP, 0>(sq);
    auto right = Bitboard::get_ray_precomputed<0, RIGHT>(sq);
    auto left = Bitboard::get_ray_precomputed<0, LEFT>(sq);
    auto down = Bitboard::get_ray_precomputed<DOWN, 0>(sq);
    up &= ~Bitboard::get_ray_precomputed<UP, 0>((up & occ).lsb());
    right &= ~Bitboard::get_ray_precomputed<0, RIGHT>((right & occ).lsb());
    left &= ~Bitboard::get_ray_precomputed<0, LEFT>((left & occ).msb());
    down &= ~Bitboard::get_ray_precomputed<DOWN, 0>((down & occ).msb());
    return up | right | left | down;
}

inline void pawn_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    const auto forward = board.side_to_move == Color::WHITE ? 1 : -1;

    const auto occ = board.occupancy();

    const auto pawns = board.pawns(board.side_to_move);

    const Bitboard allowed_double_push_rank =
        Bitboard::rank_mask(board.side_to_move == Color::WHITE ? Rank::FOURTH : Rank::FIFTH);
    const Bitboard promo_ranks = Bitboard::rank_mask(Rank::FIRST) | Bitboard::rank_mask(Rank::EIGHTH);

    const auto one_forward = pawns.rotl(8 * forward) & allowed_destinations & ~occ;
    const auto two_forward = one_forward.rotl(8 * forward) & allowed_destinations & ~occ & allowed_double_push_rank;
    const auto left_captures = one_forward.shift<0, LEFT>() & board.occupancy(~board.side_to_move);
    const auto right_captures = one_forward.shift<0, RIGHT>() & board.occupancy(~board.side_to_move);

    for (auto sq : one_forward & ~promo_ranks) {
        move_list.emplace_back(sq - forward * 8, sq);
    }

    for (auto sq : two_forward) {
        move_list.emplace_back(sq - forward * 16, sq);
    }

    for (auto sq : (one_forward & promo_ranks)) {
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_KNIGHT);
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_BISHOP);
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_ROOK);
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_QUEEN);
    }

    for (auto sq : left_captures & ~promo_ranks) {
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::CAPTURE_BIT);
    }

    for (auto sq : right_captures & ~promo_ranks) {
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::CAPTURE_BIT);
    }

    for (auto sq : (left_captures & promo_ranks)) {
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_KNIGHT_CAPTURE);
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_BISHOP_CAPTURE);
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_ROOK_CAPTURE);
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_QUEEN_CAPTURE);
    }
    
    for (auto sq : (right_captures & promo_ranks)) {
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_KNIGHT_CAPTURE);
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_BISHOP_CAPTURE);
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_ROOK_CAPTURE);
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_QUEEN_CAPTURE);
    }
}

inline void knight_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    const auto occ = board.occupancy();
    const auto us = board.occupancy(board.side_to_move);
    const auto them = board.occupancy(~board.side_to_move);
    for (auto from : board.knights(board.side_to_move)) {
        const auto legal = KNIGHT_MOVES[from] & allowed_destinations;
        for (auto to : legal & ~us) {
            move_list.emplace_back(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
        }
    }
}

inline void slider_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    const auto occ = board.occupancy();
    const auto us = board.occupancy(board.side_to_move);
    const auto them = board.occupancy(~board.side_to_move);

    for (auto from : board.queens(board.side_to_move) | board.bishops(board.side_to_move)) {
        const auto legal = compute_bishop_attacks(from, occ) & allowed_destinations;
        for (auto to : legal & ~us) {
            move_list.emplace_back(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
        }
    }

    for (auto from : board.queens(board.side_to_move) | board.rooks(board.side_to_move)) {
        const auto legal = compute_rook_attacks(from, occ) & allowed_destinations;
        for (auto to : legal & ~us) {
            move_list.emplace_back(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
        }
    }
}

inline void king_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    const auto occ = board.occupancy();
    const auto us = board.occupancy(board.side_to_move);
    const auto them = board.occupancy(~board.side_to_move);
    const auto king = board.king(board.side_to_move).lsb();
    const auto legal = KING_MOVES[king] & allowed_destinations;

    for (auto to : legal & ~us) {
        move_list.emplace_back(king, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
    }
}

inline void generate_moves(const BoardState &board, MoveList &move_list) {
    pawn_moves(board, move_list);
    knight_moves(board, move_list);
    slider_moves(board, move_list);
    king_moves(board, move_list);
}