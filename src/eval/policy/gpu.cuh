#ifndef EVAL_POLICY_GPU_CUH
#define EVAL_POLICY_GPU_CUH

#include "../cuda_utils.cuh"
#include "shared.hpp"

namespace network::policy {

class CudaExecutor {
  public:
    CudaExecutor();
    ~CudaExecutor();

    void reserve(usize position_count, usize total_move_count);

    [[nodiscard]] CudaPolicyInput *inputs();
    [[nodiscard]] u16 *move_indices();
    [[nodiscard]] f32 *outputs();

    void launch(usize position_count, usize total_move_count);
    void wait();

  private:
    cuda_common::PinnedArray<CudaPolicyInput> host_inputs_;
    cuda_common::PinnedArray<u16> host_move_indices_;
    cuda_common::PinnedArray<f32> host_outputs_;
    cuda_common::DeviceBuffer<CudaPolicyInput> device_inputs_;
    cuda_common::DeviceBuffer<u16> device_move_indices_;
    cuda_common::DeviceBuffer<f32> device_outputs_;
    cudaStream_t stream_ = nullptr;
    CudaPolicyNetwork *device_network_ = nullptr;
};

} // namespace network::policy

#endif // EVAL_POLICY_GPU_CUH
