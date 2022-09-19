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
torch::Tensor multipliers);
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
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(old_h);
  CHECK_INPUT(old_cell);

  return sparse_accumulation_cuda_forward(X1,X2,idx_output,output_size,idx_1,idx_2,multipliers);
}

std::vector<torch::Tensor> sparse_accumulation_gpu_forward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);

  return sparse_accumulation_cuda_backward(X1,X2,idx_output,output_size,idx_1,idx_2,multipliers);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_accumulation_gpu_forward,  "Sparse Accumulation forward (CUDA)");
  m.def("backward", &sparse_accumulation_gpu_forward, "Sparse Accumulation backward (CUDA)");
}