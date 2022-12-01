from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

ext_cuda = cpp_extension.CUDAExtension("sparse_accumulation_cuda",
                         ["cuda_optimized/sparse_accumulation_cuda_kernel2D.cu"])

ext_first = cpp_extension.CppExtension('sparse_accumulation_active_dim_first_cpp', ['cpu_kernel/sparse_accumulation_active_dim_first.cpp'])

ext_middle = cpp_extension.CppExtension('sparse_accumulation_active_dim_middle_cpp', ['cpu_kernel/sparse_accumulation_active_dim_middle.cpp'])

ext_last = cpp_extension.CppExtension('sparse_accumulation_active_dim_last_cpp', ['cpu_kernel/sparse_accumulation_active_dim_last.cpp'])

setup(name='sparse_accumulation',
      packages = find_packages(),
      ext_modules=[ext_first, ext_middle, ext_last, ext_cuda],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

