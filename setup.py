import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f if not line.strip().startswith("#")
    ]

ext_first_active_dim = [
      CppExtension(
   name='sparse_accumulation_active_dim_first_cpp',
   sources=['cpu_kernel/sparse_accumulation_active_dim_first.cpp'],
   libraries=["torch"],
   language='c++')
]

setup(name='sparse_accumulation_active_dim_first_cpp',
      ext_modules= ext_first_active_dim,
      cmdclass={'build_ext': BuildExtension})

ext_middle_active_dim = [
      CppExtension(
   name='sparse_accumulation_active_dim_middle_cpp',
   sources=['cpu_kernel/sparse_accumulation_active_dim_middle.cpp'],
   libraries=["torch"],
   language='c++')
]

setup(name='sparse_accumulation_active_dim_middle_cpp',
      ext_modules= ext_middle_active_dim,
      cmdclass={'build_ext': BuildExtension})

ext_last_active_dim = [
      CppExtension(
   name='sparse_accumulation_active_dim_last_cpp',
   sources=['cpu_kernel/sparse_accumulation_active_dim_last.cpp'],
   libraries=["torch"],
   language='c++')
]

setup(name='sparse_accumulation_active_dim_last_cpp',
      ext_modules= ext_middle_active_dim,
      cmdclass={'build_ext': BuildExtension})