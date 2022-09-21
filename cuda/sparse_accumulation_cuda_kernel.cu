#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
using namespace torch::indexing;

template <typename scalar_t>
__global__ void sparse_accumulation_cuda_forward_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ X1,
    const int nx,
    const int ny ) {
    
    int i = threadIdx.x + blockDim.x*blockIdx.x ;
    int j = threadIdx.y + blockDim.y*blockIdx.y ;

    if (i<nx && j<ny) {
        int pos = nx*j + i;
        output[pos] = X1[pos];
    };
  // const int index = blockIdx.x * blockDim.x + threadIdx.x;
  // //printf("hello I am blockIdx %d, blockDim %d, threadIdx %d \n",blockIdx.x , blockDim.x , threadIdx.x);
  // if (index < n) {
  //   //printf("hello inside1 loop I am blockIdx %d, blockDim %d, threadIdx %d \n",blockIdx.x , blockDim.x , threadIdx.x);
  //   output[index] = X1[index];
  // }
}

template <typename scalar_t>
__global__ void sparse_accumulation_cuda_backward_kernel(
    const scalar_t* __restrict__ X1,
    const scalar_t* __restrict__ X2,
    scalar_t* __restrict__ d_X1,
    scalar_t* __restrict__ d_X2
    ) {
  const int state_size = 100 ; 
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    d_X1[index] = X1[column];
    d_X2[index] = X2[column];
  }
    }


std::vector<torch::Tensor> sparse_accumulation_cuda_forward(
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    int output_size,
    torch::Tensor idx_1,
    torch::Tensor idx_2,
    torch::Tensor multipliers)
    {
  //auto output = torch::zeros({X1.sizes()[0], X1.sizes()[1], output_size}, torch::kF32); 
  //auto output = torch::zeros({X1.sizes()[0]}, torch::kF32);  
  auto output = torch::zeros_like(X1);

  //const auto batch_size = 10;
  const auto batch_sizex = output.sizes()[0];
  const auto batch_sizey = output.sizes()[1];
  //const auto state_size = 1;

  auto nx = batch_sizex ; 
  auto ny = batch_sizey ; 
  auto threads = 124;
  //const dim3 blocks((n+threads-1)/threads, batch_size);
  //auto blocks = (n+threads-1)/threads;

  //AT_DISPATCH_FLOATING_TYPES(output.type(), "sparse_accumulation_forward_cuda", ([&] {
  //  sparse_accumulation_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
  //      output.data<scalar_t>(),
  //      X1.data<scalar_t>(),
  //      n1,
  //      n2,
  //      );
  //}));

  auto find_num_blocks = [](int x, int bdim) {return (x+bdim-1)/bdim;};
  dim3 block_dim(16, 16);
  int nbx = find_num_blocks(nx, block_dim.x);
  int nby = find_num_blocks(ny, block_dim.y);
  dim3 grid_dim(nbx, nby);

  AT_DISPATCH_FLOATING_TYPES(output.type(), "sparse_accumulation_forward_cuda", ([&] {
  sparse_accumulation_cuda_forward_kernel<scalar_t><<<grid_dim, block_dim>>>(
      output.data<scalar_t>(),
      X1.data<scalar_t>(),
      nx,
      ny
      );
  }));

  return {output};
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
    return {d_X1, d_X2};

}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sparse_accumulation_gpu_forward(
      torch::Tensor X1,
      torch::Tensor X2,
      torch::Tensor idx_output,
      int64_t output_size,
      torch::Tensor idx_1,
      torch::Tensor idx_2,
      torch::Tensor multipliers){

  CHECK_INPUT(X1);
  CHECK_INPUT(X2);
  CHECK_INPUT(idx_output);
  //CHECK_INPUT(output_size);
  CHECK_INPUT(idx_1);
  CHECK_INPUT(idx_2);
  CHECK_INPUT(multipliers);

  return sparse_accumulation_cuda_forward(X1,X2,idx_output,output_size,idx_1,idx_2,multipliers);
}

std::vector<torch::Tensor> sparse_accumulation_gpu_backward(
  torch::Tensor d_output,
  torch::Tensor X1,
  torch::Tensor X2,
  torch::Tensor idx_output,
  torch::Tensor idx_1,
  torch::Tensor idx_2, 
  torch::Tensor multipliers
    ) {
  CHECK_INPUT(d_output);
  CHECK_INPUT(X1);
  CHECK_INPUT(X2);
  CHECK_INPUT(idx_output);
  CHECK_INPUT(idx_1);
  CHECK_INPUT(idx_2 );
  CHECK_INPUT(multipliers);

  return sparse_accumulation_cuda_backward(d_output,X1,X2,idx_output,idx_1,idx_2,multipliers);
}

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("forward", &sparse_accumulation_gpu_forward,  "Sparse Accumulation forward (CUDA)");
//  m.def("backward", &sparse_accumulation_gpu_forward, "Sparse Accumulation backward (CUDA)");
//}

TORCH_LIBRARY(sparse_accumulation_cuda, m) {
    m.def("forward", sparse_accumulation_gpu_forward);
    m.def("backward", sparse_accumulation_gpu_backward);
}