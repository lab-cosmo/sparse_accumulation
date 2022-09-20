#include <torch/extension.h>

#include <iostream>
using namespace torch::indexing;

torch::Tensor sparse_accumulation_active_dim_first_forward(torch::Tensor X1,
                                  torch::Tensor X2,
                                  torch::Tensor idx_output,
                                  int output_size,
                                  torch::Tensor idx_1,
                                  torch::Tensor idx_2,
                                  torch::Tensor multipliers){
    
    
    auto X1_a = X1.accessor<float, 3>();
    auto X2_a = X2.accessor<float, 3>();
    auto multipliers_a = multipliers.accessor<float, 1>();    
    
    auto output = torch::zeros({output_size, X1.sizes()[1], X1.sizes()[2]}, torch::kF32);    
    auto output_a = output.accessor<float, 3>();
    
    auto idx_1_a = idx_1.accessor<long, 1>();
    auto idx_2_a = idx_2.accessor<long, 1>();
    auto idx_output_a = idx_output.accessor<long, 1>();
    
            
    for (int index_first = 0; index_first < output.size(1); ++index_first){
        for (int index_second = 0; index_second < output.size(2); ++index_second) {
            for (int index = 0; index < idx_output_a.size(0); ++index) {     
                auto first = X1_a[idx_1_a[index]][index_first][index_second];
                auto second = X2_a[idx_2_a[index]][index_first][index_second];               
                auto third = multipliers_a[index];  
                auto contribution = first * second * third;                
                output_a[idx_output_a[index]][index_first][index_second] += contribution;
            }
        }
    }
    
    return output; 
}

std::vector<torch::Tensor> sparse_accumulation_active_dim_first_backward(torch::Tensor d_output,
                                                        torch::Tensor X1,
                                                        torch::Tensor X2,
                                                        torch::Tensor idx_output,
                                                        torch::Tensor idx_1,
                                                        torch::Tensor idx_2, 
                                                        torch::Tensor multipliers){
    
    
    auto X1_a = X1.accessor<float, 3>();
    auto X2_a = X2.accessor<float, 3>();
    auto multipliers_a = multipliers.accessor<float, 1>();    
    
 
    auto d_output_a = d_output.accessor<float, 3>();
    
    auto idx_1_a = idx_1.accessor<long, 1>();
    auto idx_2_a = idx_2.accessor<long, 1>();
    auto idx_output_a = idx_output.accessor<long, 1>();
    
    auto d_X1 = torch::zeros_like(X1);
    auto d_X2 = torch::zeros_like(X2);
    
    auto d_X1_a = d_X1.accessor<float, 3>();
    auto d_X2_a = d_X2.accessor<float, 3>();
    
                
    for (int index_first = 0; index_first < d_output_a.size(1); ++index_first){
        for (int index_second = 0; index_second < d_output_a.size(2); ++index_second) {
            for (int index = 0; index < idx_output_a.size(0); ++index) { 
                auto from_X1 = X1_a[idx_1_a[index]][index_first][index_second];
                auto from_X2 = X2_a[idx_2_a[index]][index_first][index_second];
                auto multiplier = multipliers_a[index];   
                auto grad = d_output_a[idx_output_a[index]][index_first][index_second];
                
                auto to_X1 = multiplier * grad * from_X2;
                auto to_X2 = multiplier * grad * from_X1;
                
                d_X1_a[idx_1_a[index]][index_first][index_second] += to_X1;
                d_X2_a[idx_2_a[index]][index_first][index_second] += to_X2;
            }
        }
    }
    
    return {d_X1, d_X2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_accumulation_active_dim_first_forward, "sparse accumulation active dim first forward");
  m.def("backward", &sparse_accumulation_active_dim_first_backward, "sparse accumulation active dim first backward");
}


