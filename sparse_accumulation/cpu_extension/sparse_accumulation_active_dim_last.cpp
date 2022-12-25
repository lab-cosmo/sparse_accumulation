#include <torch/extension.h>

#include <iostream>
#include <omp.h>
using namespace torch::indexing;


template<typename scalar_t>
void _sparse_accumulation_active_dim_last_contiguous_backward(
    torch::Tensor d_X1,
    torch::Tensor d_X2,
    torch::Tensor d_output,
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    torch::Tensor idx_1,
    torch::Tensor idx_2, 
    torch::Tensor multipliers) {
    
    scalar_t* d_X1_ptr = d_X1.data_ptr<scalar_t>();
    scalar_t* d_X2_ptr = d_X2.data_ptr<scalar_t>();
    scalar_t* d_output_ptr = d_output.data_ptr<scalar_t>();
    
    scalar_t* X1_ptr = X1.data_ptr<scalar_t>();
    scalar_t* X2_ptr = X2.data_ptr<scalar_t>();
   
    scalar_t* multipliers_ptr = multipliers.data_ptr<scalar_t>();
    long* idx_1_ptr = idx_1.data_ptr<long>();
    long* idx_2_ptr = idx_2.data_ptr<long>();
    long* idx_output_ptr = idx_output.data_ptr<long>();
    
    long active_size = idx_output.sizes()[0];
    long first_size = X1.sizes()[0];
    long second_size = X1.sizes()[1];        
    
    long output_active_dim = d_output.sizes()[2];
    long X1_active_dim = X1.sizes()[2];
    long X2_active_dim = X2.sizes()[2];    

    #pragma omp parallel for
    for (int index_first = 0; index_first < first_size; ++index_first){
        // #pragma omp parallel for  // This makes little difference
        for (int index_second = 0; index_second < second_size; ++index_second) {
            long shift_number = index_first * second_size + index_second;
            long shift_output = shift_number * output_active_dim;
            long shift_X1 = shift_number * X1_active_dim;
            long shift_X2 = shift_number * X2_active_dim;
            for (int index = 0; index < active_size; ++index) {
                scalar_t grad = d_output_ptr[shift_output + idx_output_ptr[index]] * multipliers_ptr[index];
                d_X1_ptr[shift_X1 + idx_1_ptr[index]] += grad * X2_ptr[shift_X2 + idx_2_ptr[index]];
                d_X2_ptr[shift_X2 + idx_2_ptr[index]] += grad * X1_ptr[shift_X1 + idx_1_ptr[index]];
            }
            shift_output += output_active_dim;
            shift_X1 += X1_active_dim;
            shift_X2 += X2_active_dim; 
        }
    }  

}


template<typename scalar_t>
void _sparse_accumulation_active_dim_last_contiguous_forward(
    torch::Tensor output,
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    int output_size,
    torch::Tensor idx_1,
    torch::Tensor idx_2,
    torch::Tensor multipliers) {
       
    scalar_t* X1_ptr = X1.data_ptr<scalar_t>();
    scalar_t* X2_ptr = X2.data_ptr<scalar_t>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();
    scalar_t* multipliers_ptr = multipliers.data_ptr<scalar_t>();
    long* idx_1_ptr = idx_1.data_ptr<long>();
    long* idx_2_ptr = idx_2.data_ptr<long>();
    long* idx_output_ptr = idx_output.data_ptr<long>();
    
    long active_size = idx_output.sizes()[0];
    long first_size = X1.sizes()[0];
    long second_size = X1.sizes()[1];
    
    long output_active_dim = output_size;
    long X1_active_dim = X1.sizes()[2];
    long X2_active_dim = X2.sizes()[2];
    
    #pragma omp parallel for
    for (int index_first = 0; index_first < first_size; ++index_first){
        // #pragma omp parallel for  // This makes little difference
        for (int index_second = 0; index_second < second_size; ++index_second) {
            long shift_number = index_first * second_size + index_second;
            long shift_output = shift_number * output_active_dim;
            long shift_X1 = shift_number * X1_active_dim;
            long shift_X2 = shift_number * X2_active_dim;
            for (int index = 0; index < active_size; ++index) { 
                output_ptr[shift_output + idx_output_ptr[index]] += multipliers_ptr[index] * X1_ptr[shift_X1 + idx_1_ptr[index]] * X2_ptr[shift_X2 + idx_2_ptr[index]];                                             
            } 
        }
    }
        
}


template<typename scalar_t>
void _sparse_accumulation_active_dim_last_forward(
    torch::Tensor output,
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    int output_size,
    torch::Tensor idx_1,
    torch::Tensor idx_2,
    torch::Tensor multipliers) {
    
    
    auto X1_a = X1.accessor<scalar_t, 3>();
    auto X2_a = X2.accessor<scalar_t, 3>();
    auto multipliers_a = multipliers.accessor<scalar_t, 1>();    
       
    auto output_a = output.accessor<scalar_t, 3>();
    
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
    
}


template<typename scalar_t>
void _sparse_accumulation_active_dim_last_backward(
    torch::Tensor d_X1,
    torch::Tensor d_X2,  
    torch::Tensor d_output,
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    torch::Tensor idx_1,
    torch::Tensor idx_2, 
    torch::Tensor multipliers) {
    
    
    auto X1_a = X1.accessor<scalar_t, 3>();
    auto X2_a = X2.accessor<scalar_t, 3>();
    auto multipliers_a = multipliers.accessor<scalar_t, 1>();    
    
    auto d_output_a = d_output.accessor<scalar_t, 3>();
    
    auto idx_1_a = idx_1.accessor<long, 1>();
    auto idx_2_a = idx_2.accessor<long, 1>();
    auto idx_output_a = idx_output.accessor<long, 1>();
    
    auto d_X1_a = d_X1.accessor<scalar_t, 3>();
    auto d_X2_a = d_X2.accessor<scalar_t, 3>();
    
    for (int index_first = 0; index_first < d_output_a.size(0); ++index_first){
        for (int index_second = 0; index_second < d_output_a.size(1); ++index_second) {
            for (int index = 0; index < idx_output_a.size(0); ++index) {                
                auto from_X1 = X1_a[index_first][index_second][idx_1_a[index]];
                auto from_X2 = X2_a[index_first][index_second][idx_2_a[index]];
                auto multiplier = multipliers_a[index];
                auto grad = d_output_a[index_first][index_second][idx_output_a[index]];
                
                auto to_X1 = multiplier * grad * from_X2;
                auto to_X2 = multiplier * grad * from_X1;
                
                d_X1_a[index_first][index_second][idx_1_a[index]] += to_X1;
                d_X2_a[index_first][index_second][idx_2_a[index]] += to_X2;
            }
        }
    }
    
}


std::vector<torch::Tensor> sparse_accumulation_active_dim_last_contiguous_backward(
    torch::Tensor d_output,
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    torch::Tensor idx_1,
    torch::Tensor idx_2, 
    torch::Tensor multipliers
) {

    auto d_X1 = torch::zeros_like(X1);
    auto d_X2 = torch::zeros_like(X2);

    AT_DISPATCH_FLOATING_TYPES(X1.type(), "sparse_accumulation_active_dim_last_contiguous_backward", ([&] {
        _sparse_accumulation_active_dim_last_contiguous_backward<scalar_t>(
            d_X1,
            d_X2,
            d_output,
            X1,
            X2,
            idx_output,
            idx_1,
            idx_2, 
            multipliers
        );
    }));

    return {d_X1, d_X2};
}


torch::Tensor sparse_accumulation_active_dim_last_contiguous_forward(
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    int output_size,
    torch::Tensor idx_1,
    torch::Tensor idx_2,
    torch::Tensor multipliers
) {

    auto output = torch::zeros({X1.sizes()[0], X1.sizes()[1], output_size}, X1.options());

    AT_DISPATCH_FLOATING_TYPES(X1.type(), "sparse_accumulation_active_dim_last_contiguous_forward", ([&] {
        _sparse_accumulation_active_dim_last_contiguous_forward<scalar_t>(
            output,
            X1,
            X2,
            idx_output,
            output_size,
            idx_1,
            idx_2,
            multipliers
        );
    }));

    return output;
}


std::vector<torch::Tensor> sparse_accumulation_active_dim_last_backward(
    torch::Tensor d_output,
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    torch::Tensor idx_1,
    torch::Tensor idx_2, 
    torch::Tensor multipliers
) {

    auto d_X1 = torch::zeros_like(X1);
    auto d_X2 = torch::zeros_like(X2);

    AT_DISPATCH_FLOATING_TYPES(X1.type(), "sparse_accumulation_active_dim_last_backward", ([&] {
        _sparse_accumulation_active_dim_last_backward<scalar_t>(
            d_X1,
            d_X2,
            d_output,
            X1,
            X2,
            idx_output,
            idx_1,
            idx_2, 
            multipliers
        );
    }));

    return {d_X1, d_X2};
}


torch::Tensor sparse_accumulation_active_dim_last_forward(
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    int output_size,
    torch::Tensor idx_1,
    torch::Tensor idx_2,
    torch::Tensor multipliers
) {

    auto output = torch::zeros({X1.sizes()[0], X1.sizes()[1], output_size}, X1.options());

    AT_DISPATCH_FLOATING_TYPES(X1.type(), "sparse_accumulation_active_dim_last_forward", ([&] {
        _sparse_accumulation_active_dim_last_forward<scalar_t>(
            output,
            X1,
            X2,
            idx_output,
            output_size,
            idx_1,
            idx_2,
            multipliers
        );
    }));

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_accumulation_active_dim_last_forward, "sparse accumulation active dim last forward");
  m.def("forward_contiguous", &sparse_accumulation_active_dim_last_contiguous_forward, "sparse accumulation active dim last contiguous forward");
  m.def("backward", &sparse_accumulation_active_dim_last_backward, "sparse accumulation active dim last backward");
  m.def("backward_contiguous", &sparse_accumulation_active_dim_last_contiguous_backward, "sparse accumulation active dim last contiguous backward");
}


