#include <torch/extension.h>

#include <iostream>
#include <omp.h>
using namespace torch::indexing;


template<typename scalar_t>
void _sparse_accumulation_active_dim_middle_contiguous_backward(
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
    long second_size = X1.sizes()[2];
    
    long inner_size = second_size * d_output.sizes()[1];
    
    #pragma omp parallel for
    for (int index_first = 0; index_first < first_size; ++index_first) {
        for (int index = 0; index < active_size; ++index) {
            long shift_first = index_first * inner_size;

            long shift_output = idx_output_ptr[index] * second_size + shift_first;
            long shift_X1 = idx_1_ptr[index] * second_size + shift_first;
            long shift_X2 = idx_2_ptr[index] * second_size + shift_first;
            
            scalar_t multiplier = multipliers_ptr[index];

            #pragma omp parallel for
            for (int index_second = 0; index_second < second_size; ++index_second) {
                
                scalar_t grad = d_output_ptr[shift_output + index_second] * multiplier;               
                d_X1_ptr[shift_X1 + index_second] += grad * X2_ptr[shift_X2 + index_second];
                d_X2_ptr[shift_X2 + index_second] += grad * X1_ptr[shift_X1 + index_second];
            }
        }
    } 

}


template<typename scalar_t>
void _sparse_accumulation_active_dim_middle_contiguous_forward(
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
    long second_size = X1.sizes()[2];
    
    long inner_size = second_size * output_size;
    
    #pragma omp parallel for
    for (int index_first = 0; index_first < first_size; ++index_first) {
        for (int index = 0; index < active_size; ++index) {
            long shift_first = index_first * inner_size;

            long shift_output = idx_output_ptr[index] * second_size + shift_first;
            long shift_X1 = idx_1_ptr[index] * second_size + shift_first;
            long shift_X2 = idx_2_ptr[index] * second_size + shift_first;
            
            scalar_t multiplier = multipliers_ptr[index];
            #pragma omp parallel for
            for (int index_second = 0; index_second < second_size; ++index_second) { 
                output_ptr[shift_output + index_second] += X1_ptr[shift_X1 + index_second] * X2_ptr[shift_X2 + index_second] * multiplier;
            }
        }
    }    
        
}


std::vector<torch::Tensor> sparse_accumulation_active_dim_middle_contiguous_backward(
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

    AT_DISPATCH_FLOATING_TYPES(X1.type(), "sparse_accumulation_active_dim_middle_contiguous_backward", ([&] {
        _sparse_accumulation_active_dim_middle_contiguous_backward<scalar_t>(
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


torch::Tensor sparse_accumulation_active_dim_middle_contiguous_forward(
    torch::Tensor X1,
    torch::Tensor X2,
    torch::Tensor idx_output,
    int output_size,
    torch::Tensor idx_1,
    torch::Tensor idx_2,
    torch::Tensor multipliers
) {

    auto output = torch::zeros({X1.sizes()[0], output_size, X1.sizes()[2]}, X1.options());

    AT_DISPATCH_FLOATING_TYPES(X1.type(), "sparse_accumulation_active_dim_middle_contiguous_forward", ([&] {
        _sparse_accumulation_active_dim_middle_contiguous_forward<scalar_t>(
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
  m.def("forward_contiguous", &sparse_accumulation_active_dim_middle_contiguous_forward, "sparse accumulation active dim middle contiguous forward"); 
  m.def("backward_contiguous", &sparse_accumulation_active_dim_middle_contiguous_backward, "sparse accumulation active dim middle contiguous backward");
}


