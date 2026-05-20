#include "../board_cuda.cuh"
#include "gpu.cuh"

#include "../../chess/board_state.hpp"

#ifdef MIXER_VALUE_NETWORK
#include "mixer/gpu.cuh"
#else
#include "dense/gpu.cuh"
#endif

namespace network::value {

namespace {

cuda_common::DeviceCached<CudaValueNetwork> &cached_value_network() {
    static cuda_common::DeviceCached<CudaValueNetwork> cached_network;
    return cached_network;
}

} // namespace

bool cuda_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

CudaExecutor::CudaExecutor() {
    cuda_common::cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    device_network_ =
        cached_value_network().get_or_init([](auto &host_network) { export_cuda_network(host_network); });
}

CudaExecutor::~CudaExecutor() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void CudaExecutor::reserve(usize count) {
    host_inputs_.reserve(count);
    host_outputs_.reserve(count);
    device_inputs_.reserve(count);
    device_outputs_.reserve(count);
}

CudaBoardInput *CudaExecutor::inputs() {
    return host_inputs_.data();
}

f32 *CudaExecutor::outputs() {
    return host_outputs_.data();
}

void CudaExecutor::launch(usize count) {
    if (count == 0) {
        return;
    }

    reserve(count);

    const i32 count_i32 = static_cast<i32>(count);
    cuda_common::cuda_check(cudaMemcpyAsync(device_inputs_.data(), host_inputs_.data(), count * sizeof(CudaBoardInput),
                                            cudaMemcpyHostToDevice, stream_));

    const i32 block_count = static_cast<i32>((count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    evaluate_kernel<<<block_count, THREADS_PER_BLOCK, 0, stream_>>>(device_inputs_.data(), device_outputs_.data(),
                                                                    count_i32, device_network_);
    cuda_common::cuda_check(cudaGetLastError());

    cuda_common::cuda_check(cudaMemcpyAsync(host_outputs_.data(), device_outputs_.data(), count * sizeof(f32),
                                            cudaMemcpyDeviceToHost, stream_));
}

void CudaExecutor::wait() {
    cuda_common::cuda_check(cudaStreamSynchronize(stream_));
}

void evaluate_many(const CudaBoardInput *inputs, f32 *outputs, usize count) {
    if (count == 0) {
        return;
    }

    static thread_local CudaExecutor executor;
    executor.reserve(count);
    for (usize i = 0; i < count; ++i) {
        executor.inputs()[i] = inputs[i];
    }
    executor.launch(count);
    executor.wait();
    for (usize i = 0; i < count; ++i) {
        outputs[i] = executor.outputs()[i];
    }
}

} // namespace network::value
