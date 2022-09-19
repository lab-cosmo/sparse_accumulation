from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sparse_accumulation_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_accumulation_cpp', ['sparse_accumulation.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})