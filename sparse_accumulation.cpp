#include <torch/extension.h>

#include <iostream>
using namespace torch::indexing;

torch::Tensor sparse_accumulation_forward(torch::Tensor X1,
                                  torch::Tensor X2,
                                  torch::Tensor idx_output,
                                  int output_size,
                                  torch::Tensor idx_1,
                                  torch::Tensor idx_2,
                                  torch::Tensor multipliers){
    
    
    auto X1_a = X1.accessor<float, 3>();
    auto X2_a = X2.accessor<float, 3>();
    auto multipliers_a = multipliers.accessor<float, 1>();    
    
    auto output = torch::zeros({X1.sizes()[0], X1.sizes()[1], output_size}, torch::kF32);    
    auto output_a = output.accessor<float, 3>();
    
    auto idx_1_a = idx_1.accessor<long, 1>();
    auto idx_2_a = idx_2.accessor<long, 1>();
    auto idx_output_a = idx_output.accessor<long, 1>();
    
    for (int index_first = 0; index_first < output.size(0); ++index_first){
        for (int index_second = 0; index_second < output.size(1); ++index_second) {
            for (int index = 0; index < idx_output_a.size(0); ++index) {                
                auto first = X1_a[index_first][index_second][idx_1_a[index]];
                auto second = X2_a[index_first][index_second][idx_2_a[index]];
                auto third = multipliers_a[index];
                
                auto contribution = first * second * third;                
                output_a[index_first][index_second][idx_output_a[index]] += contribution;
            }
        }
    }
    
    return output; 
}

std::vector<torch::Tensor> sparse_accumulation_backward(torch::Tensor d_output,
                                                        torch::Tensor X1,
                                                        torch::Tensor X2,
                                                        torch::Tensor idx_output,
                                                        torch::Tensor idx_1,
                                                        torch::Tensor idx_2, 
                                                        torch::Tensor multipliers){
    
    auto d_X1 = torch::zeros_like(X1);
    auto d_X2 = torch::zeros_like(X2);
    
    return {d_X1, d_X2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_accumulation_forward, "sparse accumulation forward");
  m.def("backward", &sparse_accumulation_backward, "sparse accumulation backward");
}


