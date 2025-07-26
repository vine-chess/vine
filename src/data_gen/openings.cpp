#include "openings.hpp"
#include "../chess/move_gen.hpp"

namespace datagen {

std::vector<BoardState> generate_openings(usize count, usize seed, usize random_moves) {
    std::vector<BoardState> openings;
    rng::seed_generator(seed);

    const auto pick_random_move_count = [&]() { return random_moves; };

    Board board(STARTPOS_FEN);
    for (usize num_generated = 0; num_generated < count;) {
        board.undo_n_moves(board.history().size() - 1);

        const auto random_move_count = pick_random_move_count();

        auto generated_successfully = true;
        for (usize i = 0; i <= random_move_count; ++i) {
            MoveList moves;
            generate_moves(board.state(), moves);
            if (moves.empty() || board.is_fifty_move_draw() || board.has_threefold_repetition()) {
                generated_successfully = false;
                break;
            }

            // Generated enough moves and didn't reach a terminal node
            if (i == random_move_count) {
                break;
            }

            util::StaticVector<std::pair<Move, i32>, 218> scored_moves;
            for (const auto move : moves) {
                scored_moves.push_back({move, rng::next_u64(-100, 100)});
            }

            const auto move = std::max_element(std::begin(scored_moves), std::end(scored_moves),
                                               [](auto &lhs, auto &rhs) { return lhs.second < rhs.second; });
            board.make_move(move->first);
        }

        if (generated_successfully) {
            openings.emplace_back(board.state());
            ++num_generated;
        }
    }

    return openings;
}

} // namespace datagen