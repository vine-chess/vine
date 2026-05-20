#ifndef EVAL_VALUE_GPU_CUH
#define EVAL_VALUE_GPU_CUH

#include "../cuda_utils.cuh"
#include "shared.hpp"

namespace network::value {

class CudaExecutor {
  public:
    CudaExecutor();
    ~CudaExecutor();

    void reserve(usize count);

    [[nodiscard]] CudaBoardInput *inputs();
    [[nodiscard]] f32 *outputs();

    void launch(usize count);
    void wait();

  private:
    cuda_common::PinnedArray<CudaBoardInput> host_inputs_;
    cuda_common::PinnedArray<f32> host_outputs_;
    cuda_common::DeviceBuffer<CudaBoardInput> device_inputs_;
    cuda_common::DeviceBuffer<f32> device_outputs_;
    cudaStream_t stream_ = nullptr;
    CudaValueNetwork *device_network_ = nullptr;
};

} // namespace network::value

#endif // EVAL_VALUE_GPU_CUH
