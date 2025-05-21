import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")
        print(f"get_extensions")

    assert CUDA_HOME is not None

    define_macros = [
        ("WITH_CUDA", None),
    ]
    extra_compile_args = {
        "cxx": [
            "-std=c++17",
            "-O2",
        ],
        "nvcc": [
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--extended-lambda",
            "-D_ENABLE_EXTENDED_ALIGNED_STORAGE",
            "-std=c++17",
            "--ptxas-options=-O2",
            "--ptxas-options=-allow-expensive-optimizations=true",
            "--threads",
            "64",
            "--ptxas-options=-v",
            "-gencode=arch=compute_80,code=sm_80",     # Ampere
            "-gencode=arch=compute_86,code=sm_86",     # Ampere
            "-gencode=arch=compute_89,code=sm_89",     # Hopper
            "-gencode=arch=compute_89,code=compute_89" # Hopper
        ]
    }
    extra_link_args = []

    if debug_mode:
        print(f"extra_compile_args: {extra_compile_args}")

    extensions_dir = os.path.join("fast_kernel", "csrc")
    include_dirs = [extensions_dir]
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    if debug_mode:
        print(f"extensions_dir: {extensions_dir}")
        print(f"extensions_dir: {os.path.join(extensions_dir, '**', '*.cpp')}")
        print(f"cpp sources: {sources}")

    cuda_sources = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)
    sources += cuda_sources
    if debug_mode:
        print(f"cuda sources: {cuda_sources}")
        print(f"final sources: {sources}")

    # cutlass
    cutlass_root = os.path.join("third_party", "cutlass")
    cutlass_include = os.path.join(cutlass_root, "include")
    if debug_mode:
        print(f"cutlass root: {cutlass_root}")
    if not os.path.exists(cutlass_root) or not os.path.exists(
            cutlass_include):
        raise RuntimeError("Cannot find cutlass. Please run "
                            "`git submodule update --init --recursive`.")
    include_dirs.append(cutlass_include)
    cutlass_tools_util_include = os.path.join(cutlass_root, "tools", "util", "include")
    include_dirs.append(cutlass_tools_util_include)

    # cudnn
    try:
        from nvidia import cudnn
    except ImportError:
        cudnn = None
    if cudnn is not None:
        if debug_mode:
            print(f"Using cuDNN from {cudnn.__file__}")
        cudnn_dir = os.path.dirname(cudnn.__file__)
        include_dirs.append(os.path.join(cudnn_dir, "include"))

    ext_modules = [
        CUDAExtension(
            "fast_kernel._C",
            sources,
            include_dirs=[os.path.abspath(p) for p in include_dirs],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules

setup(
    name="fast-kernel",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
