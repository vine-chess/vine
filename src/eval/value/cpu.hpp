#ifndef EVAL_VALUE_CPU_HPP
#define EVAL_VALUE_CPU_HPP

#include "../../chess/board_state.hpp"
#include "shared.hpp"

#ifdef MIXER_VALUE_NETWORK
#include "mixer/cpu.hpp"
#else
#include "dense/cpu.hpp"
#endif

namespace network::value {

constexpr i16 EVAL_SCALE = 400;

f64 evaluate(const BoardState &state);

} // namespace network::value

#endif // EVAL_VALUE_CPU_HPP
