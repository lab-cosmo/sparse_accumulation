#include <torch/extension.h>

#include <iostream>
using namespace torch::indexing;


std::vector<torch::Tensor> sparse_accumulation_active_dim_first_contiguous_backward(torch::Tensor d_output,
                                                        torch::Tensor X1,
                                                        torch::Tensor X2,
                                                        torch::Tensor idx_output,
                                                        torch::Tensor idx_1,
                                                        torch::Tensor idx_2, 
                                                        torch::Tensor multipliers){
    
    auto d_X1 = torch::zeros_like(X1);
    auto d_X2 = torch::zeros_like(X2);
    
    float* d_X1_ptr = d_X1.data_ptr<float>();
    float* d_X2_ptr = d_X2.data_ptr<float>();
    float* d_output_ptr = d_output.data_ptr<float>();
    
    float* X1_ptr = X1.data_ptr<float>();
    float* X2_ptr = X2.data_ptr<float>();
   
    float* multipliers_ptr = multipliers.data_ptr<float>();
    long* idx_1_ptr = idx_1.data_ptr<long>();
    long* idx_2_ptr = idx_2.data_ptr<long>();
    long* idx_output_ptr = idx_output.data_ptr<long>();
    
    long active_size = idx_output.sizes()[0];
    long first_size = X1.sizes()[1];
    long second_size = X1.sizes()[2];
    long inner_size = first_size * second_size;
    
    for (int index = 0; index < active_size; ++index) {
        long shift_active_x1 = idx_1_ptr[index] * inner_size;
        long shift_active_x2 = idx_2_ptr[index] * inner_size;
        long shift_active_output = idx_output_ptr[index] * inner_size;
        float multiplier = multipliers_ptr[index];
        float grad;
        long shift_local = 0;
        for (int index_first = 0; index_first < first_size; ++index_first) {
            for (int index_second = 0; index_second < second_size; ++index_second) { 
                grad = d_output_ptr[shift_active_output + shift_local] * multiplier;               
                d_X1_ptr[shift_active_x1 + shift_local] += grad * X2_ptr[shift_active_x2 + shift_local];
                d_X2_ptr[shift_active_x2 + shift_local] += grad * X1_ptr[shift_active_x1 + shift_local];
                ++shift_local;                    
            }
        }
    }
    
    return {d_X1, d_X2};    
}

torch::Tensor sparse_accumulation_active_dim_first_contiguous_forward(torch::Tensor X1,
                                  torch::Tensor X2,
                                  torch::Tensor idx_output,
                                  int output_size,
                                  torch::Tensor idx_1,
                                  torch::Tensor idx_2,
                                  torch::Tensor multipliers){
    
    auto output = torch::zeros({output_size, X1.sizes()[1], X1.sizes()[2]}, torch::kF32);    
    float* X1_ptr = X1.data_ptr<float>();
    float* X2_ptr = X2.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* multipliers_ptr = multipliers.data_ptr<float>();
    long* idx_1_ptr = idx_1.data_ptr<long>();
    long* idx_2_ptr = idx_2.data_ptr<long>();
    long* idx_output_ptr = idx_output.data_ptr<long>();
    
    long active_size = idx_output.sizes()[0];
    long first_size = X1.sizes()[1];
    long second_size = X1.sizes()[2];
    long inner_size = first_size * second_size;
    
    for (int index = 0; index < active_size; ++index) {
        long shift_active_x1 = idx_1_ptr[index] * inner_size;
        long shift_active_x2 = idx_2_ptr[index] * inner_size;
        long shift_active_output = idx_output_ptr[index] * inner_size;
        float third = multipliers_ptr[index];
        
        long shift_local = 0;
        for (int index_first = 0; index_first < first_size; ++index_first) {
            for (int index_second = 0; index_second < second_size; ++index_second) {                 
                output_ptr[shift_active_output + shift_local] += X1_ptr[shift_active_x1 + shift_local] * X2_ptr[shift_active_x2 + shift_local] * third;
                ++shift_local;                    
            }
        }
    }
    
    return output;    
}
    

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
    
    for (int index = 0; index < idx_output_a.size(0); ++index) {            
        for (int index_first = 0; index_first < output.size(1); ++index_first){
            for (int index_second = 0; index_second < output.size(2); ++index_second) {             
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
    
    for (int index = 0; index < idx_output_a.size(0); ++index) {             
        for (int index_first = 0; index_first < d_output_a.size(1); ++index_first){
            for (int index_second = 0; index_second < d_output_a.size(2); ++index_second) {            
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
  m.def("forward_contiguous", &sparse_accumulation_active_dim_first_contiguous_forward, "sparse accumulation active dim first contiguous forward");
  m.def("backward", &sparse_accumulation_active_dim_first_backward, "sparse accumulation active dim first backward");
  m.def("backward_contiguous", &sparse_accumulation_active_dim_first_contiguous_backward, "sparse accumulation active dim first contiguous backward");
}


