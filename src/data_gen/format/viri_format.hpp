#ifndef DATAGEN_VIRI_FORMAT_HPP
#define DATAGEN_VIRI_FORMAT_HPP

#include "../../chess/board.hpp"
#include "../../util/static_vector.hpp"
#include "../../util/types.hpp"

#include "writer.hpp"

#include <array>
#include <iostream>
#include <ostream>
#include <vector>

#define assert_defaulted(x) assert(x == std::decay_t<decltype(x)>{});

namespace datagen {

struct __attribute__((packed)) MarlinPackedBoard {
    // occupancy: LittleEndian(u64),
    // pieces: [16]u8 align(8),
    // stm_ep_square: u8,
    // halfmove_clock: u8,
    // fullmove_number: LittleEndian(u16),
    // eval: LittleEndian(i16),
    // wdl: u8,
    // extra: u8,
    Bitboard occ{};
    std::array<u8, 16> pieces{};
    u8 stm_ep_square{};
    u8 halfmove_clock{};
    u16 fullmove_number{};
    [[maybe_unused]] i16 score{};
    u8 wdl{};
    [[maybe_unused]] u8 extra{};

    MarlinPackedBoard() = default;

    explicit MarlinPackedBoard(BoardState const &board)
        : occ(board.occupancy()), halfmove_clock(board.fifty_moves_clock), fullmove_number(1), pieces() {

        assert_defaulted(pieces);
        for (usize i = 0; const auto sq : occ) {
            const auto idx = i / 2;
            const auto low_nibble = i % 2 == 0;
            const auto side = board.get_piece_color(sq);
            const auto col_flag = side == Color::BLACK ? 1 << 3 : 0;

            const auto starting_rank = side == Color::WHITE ? Rank::FIRST : Rank::EIGHTH;
            auto pt = static_cast<u8>(board.get_piece_type(sq)) - PieceType::PAWN;

            if (sq.rank() == starting_rank) {
                if (board.castle_rights.can_kingside_castle(side) &&
                        board.castle_rights.kingside_rook_file(side) == sq.file() ||
                    board.castle_rights.can_queenside_castle(side) &&
                        board.castle_rights.queenside_rook_file(side) == sq.file()) {
                    pt = 6;
                }
            }

            vine_assert(pt < 7);

            pieces[idx] |= (col_flag | pt) << (low_nibble ? 0 : 4);
            vine_assert((pieces[idx] & 0b111) < 7);
            vine_assert((pieces[idx] >> 4 & 0b111) < 7);
            ++i;
        }
        stm_ep_square = (board.side_to_move == Color::BLACK ? 0x80 : 0x0) | board.en_passant_sq;
    }
};

class ViriformatWriter {
  public:
    explicit ViriformatWriter(std::ostream &out);

    void push_board_state(const BoardState &state);
    void push_move(Move best_move, f64 root_q, const BoardState &state);
    void write_with_result(double game_result);

    [[nodiscard]] u16 to_monty_move(Move move, const BoardState &state) const;

  private:
    BoardState initial_state_;
    MarlinPackedBoard compressed_board_;
    std::vector<std::pair<u16, i16>> move_score_pairs_;
    Writer writer_;
};

} // namespace datagen

#endif // DATAGEN_VIRI_FORMAT_HPP
