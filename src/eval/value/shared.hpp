#ifndef EVAL_VALUE_SHARED_HPP
#define EVAL_VALUE_SHARED_HPP

#include "../../util/types.hpp"
#include "../board.hpp"

#if !defined(DENSE_VALUE_NETWORK) && !defined(MIXER_VALUE_NETWORK)
#define MIXER_VALUE_NETWORK
#define MIXER_VALUE_WMMA
#endif

#if defined(DENSE_VALUE_NETWORK) && defined(MIXER_VALUE_NETWORK)
#error cant be both!
#endif

#ifdef MIXER_VALUE_NETWORK
#include "mixer/shared.hpp"
#else
#include "dense/shared.hpp"
#endif

namespace network::value {

constexpr i16 QA = 255;

struct CudaBoardInput {
    cuda_common::CompressedMailbox pieces{};
    u8 side_to_move = 0;
};

bool cuda_available();
void evaluate_many(const CudaBoardInput *inputs, f32 *outputs, usize count);

} // namespace network::value

#endif // EVAL_VALUE_SHARED_HPP
