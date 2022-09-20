#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> sparse_accumulation_cuda_forward(torch::Tensor X1,
      torch::Tensor X2,
      torch::Tensor idx_output,
      int output_size,
      torch::Tensor idx_1,
      torch::Tensor idx_2,
      torch::Tensor multipliers);

std::vector<torch::Tensor> sparse_accumulation_cuda_backward(
      torch::Tensor d_output,
      torch::Tensor X1,
      torch::Tensor X2,
      torch::Tensor idx_output,
      torch::Tensor idx_1,
      torch::Tensor idx_2, 
      torch::Tensor multipliers
);
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sparse_accumulation_gpu_forward(
    torch::Tensor X1,
       torch::Tensor X2,
       torch::Tensor idx_output,
       int output_size,
       torch::Tensor idx_1,
       torch::Tensor idx_2,
       torch::Tensor multipliers){

  CHECK_INPUT(X1);
  CHECK_INPUT(X2);
  CHECK_INPUT(idx_output);
  CHECK_INPUT(output_size);
  CHECK_INPUT(idx_1);
  CHECK_INPUT(idx_2);
  CHECK_INPUT(multiplier);

  return sparse_accumulation_cuda_forward(X1,X2,idx_output,output_size,idx_1,idx_2,multipliers);
}

std::vector<torch::Tensor> sparse_accumulation_gpu_forward(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_accumulation_gpu_forward,  "Sparse Accumulation forward (CUDA)");
  m.def("backward", &sparse_accumulation_gpu_forward, "Sparse Accumulation backward (CUDA)");
}