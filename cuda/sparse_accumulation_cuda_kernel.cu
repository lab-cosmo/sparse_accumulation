#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
using namespace torch::indexing;

std::vector<torch::Tensor> sparse_accumulation_cuda_forward(
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    int output_size,
    torch::Tensor idx_1,
    torch::Tensor idx_2,
    torch::Tensor multipliers)
    {
  auto output = torch::zeros({X1.sizes()[0], X1.sizes()[1], output_size}, torch::kF32);  
  auto X1_c = torch::zeros_like(X1);

  const auto batch_size = 2;
  const auto state_size = 1;

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(output.type(), "sparse_accumulation_forward_cuda", ([&] {
    sparse_accumulation_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        output.data<scalar_t>(),
        X1.data<scalar_t>()
        );
  }));

  return {output};
}


template <typename scalar_t>
__global__ void sparse_accumulation_cuda_forward_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ X1) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    output[index] = X1[column];
  }
}


std::vector<torch::Tensor> sparse_accumulation_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    torch::Tensor idx_1,
    torch::Tensor idx_2, 
    torch::Tensor multipliers)
    {
    auto d_X1 = torch::zeros_like(X1);
    auto d_X2 = torch::zeros_like(X2); 

    const auto batch_size = 2;
    const auto state_size = 1;

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(d_X1.type(), "sparse_accumulation_backward_cuda", ([&] {
      sparse_accumulation_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        X1.data<scalar_t>(),
        X2.data<scalar_t>(),
        d_X1.data<scalar_t>(),
        d_X2.data<scalar_t>()
        );
    }));




}


template <typename scalar_t>
__global__ void sparse_accumulation_cuda_forward_kernel(
    const scalar_t* __restrict__ X1,
    const scalar_t* __restrict__ X2,
    scalar_t* __restrict__ d_X1,
    scalar_t* __restrict__ d_X2
    ) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    d_X1[index] = X1[column];
    d_X2[index] = X2[column];
  }
