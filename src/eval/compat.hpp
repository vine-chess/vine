#ifndef EVAL_COMPAT_HPP
#define EVAL_COMPAT_HPP

#ifdef __CUDACC__
#define VINE_HOST __host__
#define VINE_DEVICE __device__
#else
#define VINE_HOST
#define VINE_DEVICE
#endif
#define VINE_HOST_DEVICE VINE_HOST VINE_DEVICE

#endif // EVAL_COMPAT_HPP
