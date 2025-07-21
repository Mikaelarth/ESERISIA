"""
Setup script for ESERISIA AI Ultra-Fast Extensions
=================================================

Compiles the world's fastest AI kernels for maximum performance.
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils.cpp_extension import CUDAExtension, CppExtension
import os
import platform

# CUDA availability check
def is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# Compiler flags for maximum optimization
def get_compiler_flags():
    flags = [
        '-O3',           # Maximum optimization
        '-march=native', # CPU-specific optimizations
        '-ffast-math',   # Fast math operations
        '-funroll-loops',# Loop unrolling
        '-DWITH_CUDA' if is_cuda_available() else '',
    ]
    
    # Windows-specific flags
    if platform.system() == 'Windows':
        flags.extend(['/O2', '/fp:fast', '/arch:AVX2'])
    
    return [f for f in flags if f]  # Remove empty strings

# CUDA compiler flags
def get_cuda_flags():
    if not is_cuda_available():
        return []
    
    # Get CUDA architecture
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        arch = f"sm_{capability[0]}{capability[1]}"
    else:
        arch = "sm_80"  # Default to Ampere
    
    return [
        f'-arch={arch}',
        '-O3',
        '--use_fast_math',
        '--expt-relaxed-constexpr',
        '-Xcompiler', '-fPIC',
        '-DWITH_CUDA',
        '-DCUDA_HAS_FP16=1',
        '--expt-extended-lambda',
        '--compiler-options', '-fno-strict-aliasing',
    ]

# Include directories
include_dirs = [
    pybind11.get_include(),
    torch.utils.cpp_extension.include_paths()[0],
]

# Library directories
library_dirs = []
libraries = []

if is_cuda_available():
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME:
        include_dirs.extend([
            os.path.join(CUDA_HOME, 'include'),
        ])
        library_dirs.extend([
            os.path.join(CUDA_HOME, 'lib64'),
            os.path.join(CUDA_HOME, 'lib'),
        ])
        libraries.extend(['cudart', 'cublas', 'curand', 'cudnn'])

# Source files
cpp_sources = [
    'eserisia/extensions/cuda_kernels.cpp',
]

cuda_sources = []
if is_cuda_available():
    cuda_sources = [
        'eserisia/extensions/cuda_implementations.cu',
    ]

# Create extension
if is_cuda_available():
    extension = CUDAExtension(
        name='eserisia_cuda_ext',
        sources=cpp_sources + cuda_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            'cxx': get_compiler_flags(),
            'nvcc': get_cuda_flags()
        },
        extra_link_args=['-lcudart', '-lcublas', '-lcurand'] if platform.system() != 'Windows' else [],
    )
else:
    extension = CppExtension(
        name='eserisia_cpu_ext', 
        sources=cpp_sources,
        include_dirs=include_dirs,
        extra_compile_args=get_compiler_flags(),
    )

# Setup configuration
if __name__ == "__main__":
    setup(
        name="eserisia-ai-extensions",
        ext_modules=[extension],
        cmdclass={'build_ext': build_ext},
        zip_safe=False,
        python_requires=">=3.11",
        install_requires=[
            "torch>=2.3.0",
            "pybind11>=2.10.0",
            "numpy>=1.24.0",
        ],
        extras_require={
            "cuda": [
                "cupy-cuda12x>=13.1.0",
                "nvidia-ml-py>=12.535.0",
            ]
        }
    )
