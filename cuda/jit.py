from torch.utils.cpp_extension import load
sparse_accumulation_cuda = load(
    'sparse_accumulation_cuda', ['sparse_accumulation_cuda.cpp', 'sparse_accumulation_cuda_kernel.cu'], verbose=True)
help(sparse_accumulation_cuda)