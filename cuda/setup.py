from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='sparse_accumulation_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'sparse_accumulation_cuda'    , 
            ['sparse_accumulation_cuda.cpp',
            'sparse_accumulation_cuda_kernel.cu',])
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})