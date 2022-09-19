#include <torch/extension.h>

#include <iostream>

torch::Tensor sparse_accumulation(torch::Tensor X1,
                                  torch::Tensor X2,
                                  torch::Tensor idx_output,
                                  int output_size,
                                  torch::Tensor idx_1,
                                  torch::Tensor idx_2){

    auto result = torch::zeros((X1.sizes()[0], X1.sizes()[1], output_size), torch::kF32);
    return result; 
}