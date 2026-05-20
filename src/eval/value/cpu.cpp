#include "cpu.hpp"

#ifdef MIXER_VALUE_NETWORK
#include "mixer/cpu_inference.hpp"
#else
#include "dense/cpu_inference.hpp"
#endif
