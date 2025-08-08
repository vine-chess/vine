#ifndef DATAGEN_MONTY_FORMAT_HPP
#define DATAGEN_MONTY_FORMAT_HPP

#include "../../chess/board.hpp"
#include "../../chess/move_gen.hpp"
#include "../../util/static_vector.hpp"
#include "../../util/types.hpp"
#include <array>
#include <ostream>
#include <vector>

namespace datagen {

using VisitsDistribution = std::vector<std::pair<u16, u32>>;

struct __attribute__((packed)) MontyFormatCompressedBoard {
    std::array<u64, 4> bbs{};
    Color side_to_move{};
    Square en_passant_square{};
    u8 castle_rights{};
    u8 fifty_moves_clock{};
    u16 full_move_count{};
};

struct MontyFormatMoveData {
    u16 best_move{};
    f64 root_q{};
    VisitsDistribution visits{};
};

class MontyFormatWriter {
  public:
    explicit MontyFormatWriter(std::ostream &out);

    void push_board_state(const BoardState &state);
    void push_move(Move best_move, f64 root_q, const VisitsDistribution &visit_dist, const BoardState &state);
    void write_with_result(double game_result);

    [[nodiscard]] u16 to_monty_move(Move move, const BoardState &state) const;

  private:
    void put_u8(u8 v);
    void put_u16(u16 v);
    void put_u64(u64 v);

    BoardState initial_state_;
    MontyFormatCompressedBoard compressed_board_;
    std::vector<MontyFormatMoveData> moves_;
    std::vector<u8> buffer_;
    std::ostream &out_;
};

} // namespace datagen

#endif