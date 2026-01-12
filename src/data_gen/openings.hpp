#ifndef OPENINGS_HPP
#define OPENINGS_HPP

#include "../chess/board.hpp"
#include "../util/random.hpp"

#include "../search/game_tree.hpp"
#include <optional>
#include <span>
#include <string>

namespace datagen {

search::NodeIndex pick_node_temperature(search::GameTree const &tree, f64 temperature, bool downsample_captures,
                           // how close the score of the move has to be
                           // to the highest scored move (in centipawns)
                           std::optional<i32> score_diff = std::nullopt,
                           // how close the number of visits has to be
                           // relative to the scored visit move
                           // if it has more visits that is accepted
                           std::optional<f64> visits_limit = std::nullopt);

BoardState generate_opening(std::span<const std::string> opening_fens, usize random_moves,
                            f64 initial_temperature = 1.25, f64 gamma = 0.9);

} // namespace datagen

#endif // OPENINGS_HPP
