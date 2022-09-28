#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
using namespace torch::indexing;

template <typename scalar_t >
__global__ void sparse_accumulation_cuda_forward_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ X1,
    const scalar_t* __restrict__ X2,
    const int64_t* __restrict__ idx_output,
    const int64_t* __restrict__ idx_1,
    const int64_t* __restrict__ idx_2,
    const scalar_t* __restrict__ multipliers,
    const int output_size,
    const int X1_third_size,
    const int X2_third_size,
    const int nx,
    const int ny,
    const int nz) {
    
    int i = threadIdx.x + blockDim.x * blockIdx.x ;
    int j = threadIdx.y + blockDim.y * blockIdx.y ;
    //int z = threadIdx.z + blockDim.z * blockIdx.z ;

    //if (i<nx && j<ny && z<nz) {
    //    int pos = nx*ny*z + nx*j + i;
    //    output[pos] = X1[pos];
    //};

    if (i<nx && j<ny) {
      //printf("in kernel i %d  j %d\n",i,j) ;
      for (int z = 0 ; z < nz ; ++z){
        int z_output = idx_output[z];
        int z_X1 = idx_1[z] ;
        int z_X2 = idx_2[z] ;
        //int pos = nx*ny*z + nx*j + i;
        //int pos_X1 = nx*ny*z_X1 + nx*j + i ;
        //int pos_X2 = nx*ny*z_X2 + nx*j + i ;
        //int pos_output = nx*ny*z_output + nx*j+  i ;

        int pos_X1 = z_X1 + j*X1_third_size + i*ny*X1_third_size ;
        int pos_output = z_output+ j*output_size+  i*output_size*ny ;
        int pos_X2 = z_X2 + j*X2_third_size + i*ny*X2_third_size ;
        //printf("z_output %d \n",z_output) ;
        //printf("z_X1 %d \n",z_X1);
        //printf("z_X2 %d \n",z_X2);
        //printf("pos_X1 %d \n",pos_X1);
        //printf("pos_x2 %d \n",pos_X1);
       //printf("pos_output %d  X1 %f  X2 %f  z %d  multipliers[z] %f \n",pos_output,X1[pos_X1],X2[pos_X2],z,multipliers[z]);
        //printf("pos_output %d \n X1 %f \n X2 %f \n",pos_output,X1[pos_X1],X2[pos_X2]);
        //printf("X1 %f \n",X1[pos_X1]);
        //printf("X2 %f \n",X2[pos_X2]);
        //printf(" i use 2 \n multipliers %f \n",multipliers[z]);
        output[pos_output] += X1[pos_X1]*X2[pos_X2]*multipliers[z];
        //__syncthreads();

        //printf("pos_output %d \n",pos_output);
        //printf("z %d \n",z);
      };
      //output[pos_output] += 1; //multipliers[z];
    };
    //for (int index_first = 0; index_first < output.size(0); ++index_first){
    //    for (int index_second = 0; index_second < output.size(1); ++index_second) {
    //        for (int index = 0; index < idx_output_a.size(0); ++index) {                
    //            auto first = X1_a[index_first][index_second][idx_1_a[index]];
    //            auto second = X2_a[index_first][index_second][idx_2_a[index]];
    //            auto third = multipliers_a[index];
    //            auto contribution = first * second * third;                
    //            output_a[index_first][index_second][idx_output_a[index]] += contribution;
    //        }
    //    }
    //}
  // const int index = blockIdx.x * blockDim.x + threadIdx.x;
  // //printf("hello I am blockIdx %d, blockDim %d, threadIdx %d \n",blockIdx.x , blockDim.x , threadIdx.x);
  // if (index < n) {
  //   //printf("hello inside1 loop I am blockIdx %d, blockDim %d, threadIdx %d \n",blockIdx.x , blockDim.x , threadIdx.x);
  //   output[index] = X1[index];
  // }
}

template <typename scalar_t >
__global__ void sparse_accumulation_cuda_forward_kernel_grpwrites(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ X1,
    const scalar_t* __restrict__ X2,
    const int64_t* __restrict__ idx_output,
    const int64_t* __restrict__ idx_1,
    const int64_t* __restrict__ idx_2,
    const scalar_t* __restrict__ multipliers,
    const int output_size,
    const int X1_third_size,
    const int X2_third_size,
    const int nx,
    const int ny,
    const int nz) {
    
    int zid = blockIdx.x/nx;
    int colid = blockIdx.x%nx;

    auto ltid = threadIdx.x;

    extern __shared__ float buffer[];
    float* opcols = buffer;
    float* muls = opcols + ny;
    int* idx_1s = reinterpret_cast<int*>(muls + nz);
    int* idx_2s = reinterpret_cast<int*>(idx_1s + nz);
    int* idx_outputs = reinterpret_cast<int*>(idx_2s + nz);

    int loopcount = nz/blockDim.x + 1;

    for (int i = 0; i < loopcount; i++)
    {
      auto index = blockDim.x*i + ltid;
      if (index < nz)
      {
        muls[index] = multipliers[index];
        idx_1s[index] = idx_1[index];
        idx_2s[index] = idx_2[index];
        idx_outputs[index] = idx_output[index];
      }
    }

    loopcount = ny/blockDim.x + 1;
    // initialize shared memory with zero
    for (int i = 0; i < loopcount; i++)
    {
      auto index = blockDim.x*i + ltid;
      if (index < ny)
      {
        opcols[index] = 0.0;
      }
    }

    // loop over the multipliers
    for (int opz = 0; opz < nz; opz++)
    {
      if (idx_output[opz] == zid)
      {
        //printf("multiplier = %f \n", muls[opz]);
        for (int i = 0; i < loopcount; i++)
        {
        // compute over the column elements
        auto index = blockDim.x*i + ltid;
        if (index < ny)
          {
            // calculate global memory indices in terms of "colid", "index", "zid"
            int X1_pos = idx_1s[opz] + index*X1_third_size + colid*ny*X1_third_size;
            int X2_pos = idx_2s[opz] + index*X2_third_size + colid*ny*X2_third_size;

            opcols[index] += X1[X1_pos]*X2[X2_pos]*muls[opz]; // up to 50% time taken in this global memory read
          }
        }
      }
    }

    // copy output data to global memory
    for (int i = 0; i < loopcount; i++)
    {
      // compute over the column elements
      auto index = blockDim.x*i + ltid;
      if (index < ny)
        {
          // calculate global memory indices for the output array in terms of "colid", "index", "zid"
          int op_index = zid + index*output_size + colid*ny*output_size;
          output[op_index] = opcols[index];
        }
    }
}

template <typename scalar_t>
__global__ void sparse_accumulation_cuda_backward_kernel(
    scalar_t* __restrict__ d_X1,
    scalar_t* __restrict__ d_X2,
    const scalar_t* __restrict__ d_output,
    const scalar_t* __restrict__ X1,
    const scalar_t* __restrict__ X2,
    const int64_t* __restrict__ idx_output,
    const int64_t* __restrict__ idx_1,
    const int64_t* __restrict__ idx_2,
    const scalar_t* __restrict__ multipliers,
    const int output_size,
    const int X1_third_size,
    const int X2_third_size,
    const int nx,
    const int ny,
    const int nz
    ) {
    int i = threadIdx.x + blockDim.x * blockIdx.x ;
    int j = threadIdx.y + blockDim.y * blockIdx.y ;

    if (i<nx && j<ny) {
      for (auto z = 0 ; z < nz ; ++z){
        int z_output = idx_output[z];
        int z_X1 = idx_1[z] ;
        int z_X2 = idx_2[z] ;

        int pos_X1 = z_X1 + j*X1_third_size + i*ny*X1_third_size ;
        int pos_output = z_output+ j*output_size+  i*output_size*ny ;
        int pos_X2 = z_X2 + j*X2_third_size + i*ny*X2_third_size ;
        auto grad_multi = d_output[pos_output] * multipliers[z];
        d_X1[pos_X1] += grad_multi*X2[pos_X2]; 
        d_X2[pos_X2] += grad_multi*X1[pos_X1];
      };
    };
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
  //auto output = torch::zeros_like(X1);
  auto output = torch::zeros({X1.sizes()[0], X1.sizes()[1], output_size}, 
            torch::TensorOptions()
            .dtype(X1.dtype())
            .device(X1.device())); 

  auto X1_third_size = X1.sizes()[2]; 
  auto X2_third_size = X2.sizes()[2]; 
  const auto batch_sizex = output.sizes()[0];
  const auto batch_sizey = output.sizes()[1];
  const auto batch_sizez = idx_output.sizes()[0];

  auto nx = batch_sizex ; 
  auto ny = batch_sizey ; 
  auto nz = batch_sizez ; 
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
  int nbz = find_num_blocks(nz, block_dim.z);
  dim3 grid_dim(nbx, nby);

  int num_threads = 128;
  int num_blocks  = output_size*nx;
  

  AT_DISPATCH_FLOATING_TYPES(output.type(), "sparse_accumulation_forward_cuda", ([&] {
  // sparse_accumulation_cuda_forward_kernel<scalar_t><<<grid_dim, block_dim>>>(
  sparse_accumulation_cuda_forward_kernel_grpwrites<scalar_t><<<num_blocks, num_threads, 16384>>>(
  //sparse_accumulation_cuda_forward_kernel_grpwrites<scalar_t><<<5, 1, 16384>>>(
      output.data<scalar_t>(),
      X1.data<scalar_t>(),
      X2.data<scalar_t>(),
      idx_output.data<int64_t>(),
      idx_1.data<int64_t>(),
      idx_2.data<int64_t>(),
      multipliers.data<scalar_t>(),
      output_size,
      X1_third_size,
      X2_third_size,
      nx,
      ny,
      nz
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

    auto X1_third_size = X1.sizes()[2]; 
    auto X2_third_size = X2.sizes()[2]; 
    const auto nx = d_output.sizes()[0]    ;
    const auto ny = d_output.sizes()[1]    ;
    const auto output_size = d_output.sizes()[2] ;
    const auto nz = idx_output.sizes()[0];

    auto find_num_blocks = [](int x, int bdim) {return (x+bdim-1)/bdim;};
    dim3 block_dim(16, 16);
    int nbx = find_num_blocks(nx, block_dim.x);
    int nby = find_num_blocks(ny, block_dim.y);
    dim3 grid_dim(nbx, nby);



    AT_DISPATCH_FLOATING_TYPES(X1.type(), "sparse_accumulation_backward_cuda", ([&] {
      sparse_accumulation_cuda_backward_kernel<scalar_t><<<grid_dim, block_dim>>>(
        d_X1.data<scalar_t>(),
        d_X2.data<scalar_t>(),
        d_output.data<scalar_t>(),
        X1.data<scalar_t>(),
        X2.data<scalar_t>(),
        idx_output.data<int64_t>(),
        idx_1.data<int64_t>(),
        idx_2.data<int64_t>(),
        multipliers.data<scalar_t>(),
        output_size,
        X1_third_size,
        X2_third_size,
        nx,
        ny,
        nz
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