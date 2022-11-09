from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sparse_accumulation_active_dim_middle_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_accumulation_active_dim_middle_cpp', ['cpu_kernel/sparse_accumulation_active_dim_middle.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})


setup(name='sparse_accumulation_active_dim_last_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_accumulation_active_dim_last_cpp', ['cpu_kernel/sparse_accumulation_active_dim_last.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='sparse_accumulation_active_dim_first_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_accumulation_active_dim_first_cpp', ['cpu_kernel/sparse_accumulation_active_dim_first.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})