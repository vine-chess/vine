#include "move_gen.hpp"
#include "magics.hpp"

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

void pawn_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
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

void knight_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
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

void slider_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    const auto occ = board.occupancy();
    const auto us = board.occupancy(board.side_to_move);
    const auto them = board.occupancy(~board.side_to_move);

    for (auto from : board.queens(board.side_to_move) | board.bishops(board.side_to_move)) {
        const auto legal = BISHOP_ATTACKS[from][get_bishop_attack_idx(from, occ)] & allowed_destinations;
        for (auto to : legal & ~us) {
            move_list.emplace_back(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
        }
    }

    for (auto from : board.queens(board.side_to_move) | board.rooks(board.side_to_move)) {
        const auto legal = ROOK_ATTACKS[from][get_rook_attack_idx(from, occ)] & allowed_destinations;
        for (auto to : legal & ~us) {
            move_list.emplace_back(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
        }
    }
}

void king_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    const auto occ = board.occupancy();
    const auto us = board.occupancy(board.side_to_move);
    const auto them = board.occupancy(~board.side_to_move);
    const auto king = board.king(board.side_to_move).lsb();
    const auto legal = KING_MOVES[king] & allowed_destinations;

    for (auto to : legal & ~us) {
        move_list.emplace_back(king, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
    }
}

void generate_moves(const BoardState &board, MoveList &move_list) {
    pawn_moves(board, move_list);
    knight_moves(board, move_list);
    slider_moves(board, move_list);
    king_moves(board, move_list);
}