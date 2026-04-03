#ifndef DATAGEN_VIRI_FORMAT_CPP
#define DATAGEN_VIRI_FORMAT_CPP

#include "../../util/math.hpp"
#include "../../util/types.hpp"

#include "viri_format.hpp"
#include "writer.hpp"

#include <algorithm>
#include <ostream>
#include <vector>

namespace datagen {

u16 to_viri_move(Move move) {
    constexpr u16 promo_flag_bits = 0b1100;
    constexpr u16 ep_flag_bits = 0b0100;
    constexpr u16 castle_flag_bits = 0b1000;

    u16 res = 0;

    if (move.is_castling()) {
        res |= castle_flag_bits;
    } else if (move.is_ep()) {
        res |= ep_flag_bits;
    } else if (move.is_promo()) {
        res |= promo_flag_bits;
        res |= move.promo_type() - PieceType::KNIGHT;
    }

    res = res << 6 | static_cast<u16>(move.to());
    res = res << 6 | static_cast<u16>(move.from());

    return res;
}

ViriformatWriter::ViriformatWriter(std::ostream &out) : writer_(out) {}

void ViriformatWriter::push_board_state(const BoardState &state) {
    move_score_pairs_.clear();
    initial_state_ = state;
    compressed_board_ = MarlinPackedBoard(state);
}

void ViriformatWriter::push_move(Move best_move, f64 root_q, const BoardState &state) {
    if (state.side_to_move == Color::BLACK) {
        root_q = 1.0 - root_q;
    }
    const auto score = 400 * util::math::inverse_sigmoid(root_q);

    move_score_pairs_.push_back({
        to_viri_move(best_move),
        static_cast<i16>(std::clamp<f64>(score, -32767, 32767)),
    });
}

void ViriformatWriter::write_with_result(double game_result) {
    compressed_board_.wdl = static_cast<u8>(game_result * 2.0);

    writer_.put_raw_bytes(compressed_board_);
    for (const auto [move, score] : move_score_pairs_) {
        writer_.put_u16(move);
        writer_.put_u16(static_cast<u16>(score));
    }
    writer_.put_raw_bytes(std::array<u8, 4>{});
    writer_.flush();
}

} // namespace datagen
#endif // DATAGEN_VIRI_FORMAT_CPP
