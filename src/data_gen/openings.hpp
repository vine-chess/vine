#ifndef OPENINGS_HPP
#define OPENINGS_HPP

#include "../chess/board.hpp"
#include <string_view>

namespace datagen {

BoardState generate_opening(usize random_moves, std::string_view initial_fen = STARTPOS_FEN);

}

#endif // OPENINGS_HPP
