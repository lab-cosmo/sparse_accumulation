#include <torch/extension.h>

#include <iostream>

torch::Tensor sparse_accumulation_forward(torch::Tensor X1,
                                  torch::Tensor X2,
                                  torch::Tensor idx_output,
                                  int output_size,
                                  torch::Tensor idx_1,
                                  torch::Tensor idx_2){

    auto output = torch::zeros((X1.sizes()[0], X1.sizes()[1], output_size), torch::kF32);
    return output; 
}

std::vector<torch::Tensor> sparse_accumulation_backward(torch::Tensor X1,
                                  torch::Tensor X2,
                                  torch::Tensor idx_output,
                                  int output_size,
                                  torch::Tensor idx_1,
                                  torch::Tensor idx_2){
    
    auto d_X1 = torch::zeros_like(X1);
    auto d_X2 = torch::zeros_like(X2);
    
    return {d_X1, d_X2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_accumulation_forward, "sparse accumulation forward");
  m.def("backward", &sparse_accumulation_backward, "sparse accumulation backward");
}


