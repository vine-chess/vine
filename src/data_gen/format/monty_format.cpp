#include "monty_format.hpp"
#include "../../util/math.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <ranges>

namespace datagen {

void MontyFormatWriter::put_u8(u8 v) {
    out_.put(static_cast<char>(v));
}

void MontyFormatWriter::put_u16(u16 v) {
    out_.write(reinterpret_cast<const char *>(&v), 2);
}

void MontyFormatWriter::put_u64(u64 v) {
    out_.write(reinterpret_cast<const char *>(&v), 8);
}

MontyFormatWriter::MontyFormatWriter(std::ostream &out) : out_(out) {}

void MontyFormatWriter::push_board_state(const BoardState &state) {
    initial_state_ = state;
    moves_.clear();
    const std::array<u64, 8> raw = {static_cast<u64>(state.occupancy(Color::WHITE)),
                                    static_cast<u64>(state.occupancy(Color::BLACK)),
                                    static_cast<u64>(state.pawns()),
                                    static_cast<u64>(state.knights()),
                                    static_cast<u64>(state.bishops()),
                                    static_cast<u64>(state.rooks()),
                                    static_cast<u64>(state.queens()),
                                    static_cast<u64>(state.kings())};
    compressed_board_.bbs = {raw[1], raw[5] ^ raw[6] ^ raw[7], raw[3] ^ raw[4] ^ raw[7], raw[2] ^ raw[4] ^ raw[6]};
    compressed_board_.side_to_move = state.side_to_move;
    compressed_board_.en_passant_square = state.en_passant_sq.is_valid() ? static_cast<u8>(state.en_passant_sq) : 0;
    compressed_board_.castle_rights = state.castle_rights.to_monty_mask();
    compressed_board_.fifty_moves_clock = state.fifty_moves_clock;
    compressed_board_.full_move_count = 1; // TODO: full move clock
}

void MontyFormatWriter::push_move(Move best_move, f64 root_q, f64 static_eval, const VisitsDistribution &visit_dist,
                                  const BoardState &state) {

    moves_.push_back({to_monty_move(best_move, state), root_q, util::math::sigmoid(static_eval), visit_dist});
}

void MontyFormatWriter::write_with_result(f64 result) {
    // Initial starting position
    out_.write(reinterpret_cast<const char *>(&compressed_board_), sizeof(compressed_board_));

    auto valid_file_or_zero = [](File f, File if_wrong) { return f == File::NO_FILE ? if_wrong : static_cast<u8>(f); };

    // Initial rook files
    put_u8(valid_file_or_zero(initial_state_.castle_rights.queenside_rook_file(Color::WHITE), File::A));
    put_u8(valid_file_or_zero(initial_state_.castle_rights.kingside_rook_file(Color::WHITE), File::H));
    put_u8(valid_file_or_zero(initial_state_.castle_rights.queenside_rook_file(Color::BLACK), File::A));
    put_u8(valid_file_or_zero(initial_state_.castle_rights.kingside_rook_file(Color::BLACK), File::H));

    // Game outcome
    put_u8(static_cast<u8>(result * 2.0));

    for (auto move_data : moves_) {
        // Put the root score and the best move at root for each position
        put_u16(move_data.best_move);
        put_u16(static_cast<u16>(move_data.root_q * std::numeric_limits<u16>::max()));
        put_u16(static_cast<u16>(move_data.static_eval * std::numeric_limits<u16>::max()));

        // Sort by the move value
        std::sort(move_data.visits.begin(), move_data.visits.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        u8 count = static_cast<u8>(move_data.visits.size());
        put_u8(count);

        if (count > 0) {
            u32 max_visits = 0;
            for (const auto &[_, visits] : move_data.visits) {
                max_visits = std::max(max_visits, visits);
            }

            for (const auto &[_, visits] : move_data.visits) {
                put_u8(static_cast<u8>(visits * 255.0 / max_visits));
            }
        }
    }

    put_u16(0); // Move::null() as a sentinel
    out_.flush();
}

u16 MontyFormatWriter::to_monty_move(Move move, const BoardState &state) const {
    static constexpr u16 FLAG_QUIET = 0, FLAG_DBL_PUSH = 1, FLAG_CAP = 4, FLAG_ENP = 5, FLAG_KS = 2, FLAG_QS = 3,
                         FLAG_NPR = 8, FLAG_BPR = 9, FLAG_RPR = 10, FLAG_QPR = 11, FLAG_NPC = 12, FLAG_BPC = 13,
                         FLAG_RPC = 14, FLAG_QPC = 15;

    const u16 from = static_cast<u16>(move.from());
    u16 to = static_cast<u16>(move.to());

    u16 flag = FLAG_QUIET;
    if (move.is_castling()) {
        flag = (to > from) ? FLAG_KS : FLAG_QS;
        to = move.king_castling_to();
    } else if (move.is_ep()) {
        flag = FLAG_ENP;
    } else if (move.is_promo()) {
        const bool cap = move.is_capture();
        switch (move.promo_type()) {
        case PieceType::KNIGHT:
            flag = cap ? FLAG_NPC : FLAG_NPR;
            break;
        case PieceType::BISHOP:
            flag = cap ? FLAG_BPC : FLAG_BPR;
            break;
        case PieceType::ROOK:
            flag = cap ? FLAG_RPC : FLAG_RPR;
            break;
        case PieceType::QUEEN:
            flag = cap ? FLAG_QPC : FLAG_QPR;
            break;
        default:
            break;
        }
    } else if (move.is_capture()) {
        flag = FLAG_CAP;
    } else if (state.get_piece_type(from) == PieceType::PAWN && (from ^ to) == 16) {
        flag = FLAG_DBL_PUSH;
    }

    return static_cast<u16>((from << 10) | (to << 4) | flag);
}

} // namespace datagen
