#ifndef EVAL_VALUE_MIXER_CPU_HPP
#define EVAL_VALUE_MIXER_CPU_HPP

#include "../../../util/multi_array.hpp"
#include "shared.hpp"

namespace network::value {

struct MixerLayer {
    util::MultiArray<f32, MIXER_D1, MIXER_INNER1> wl_up;
#if MIXER_DO_DOWN_PROJ
    util::MultiArray<f32, MIXER_INNER1, MIXER_D1> wl_down;
#endif
    util::MultiArray<f32, MIXER_INNER2, MIXER_D2> wr_up;
#if MIXER_DO_DOWN_PROJ
    util::MultiArray<f32, MIXER_D2, MIXER_INNER2> wr_down;
#endif
};

struct alignas(64) ValueNetwork {
    util::MultiArray<i16, MIXER_FEATURE_COUNT, MIXER_SIZE> ft_weights;
    util::MultiArray<i16, MIXER_SIZE> ft_biases;
    util::MultiArray<MixerLayer, MIXER_NUM_LAYERS> layers;
    util::MultiArray<f32, MIXER_SIZE> value_weights;
    f32 value_bias = 0.0f;
};

} // namespace network::value

#endif // EVAL_VALUE_MIXER_CPU_HPP
