#ifndef OPENINGS_HPP
#define OPENINGS_HPP

#include "../chess/board.hpp"
#include "../util/random.hpp"
#include "opening_book.hpp"
#include <span>
#include <string>

namespace datagen {

BoardState generate_opening(const std::unique_ptr<OpeningBook>& book, usize random_moves, f64 initial_temperature = 1.25,
                            f64 gamma = 0.9);

}

#endif // OPENINGS_HPP
