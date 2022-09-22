from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sparse_accumulation_active_dim_middle_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_accumulation_active_dim_middle_cpp', ['sparse_accumulation_active_dim_middle.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})


setup(name='sparse_accumulation_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_accumulation_cpp', ['sparse_accumulation.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='sparse_accumulation_active_dim_first_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_accumulation_active_dim_first_cpp', ['sparse_accumulation_active_dim_first.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})