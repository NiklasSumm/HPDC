from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup
import os
import subprocess


def find_cuda():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        return cuda_home

    # Try common CUDA installation paths
    common_paths = [
        "/usr/local/cuda",
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    raise RuntimeError("CUDA installation not found.")


# Find CUDA installation
cuda_home = find_cuda()

print(f"Found CUDA at: {cuda_home}")

# CUDA include and library paths
cuda_include = os.path.join(cuda_home, "include")
cuda_lib = os.path.join(cuda_home, "lib64")


# Compile CUDA source to object file
def compile_cuda():
    """Compile CUDA source file to object file"""
    nvcc_cmd = [
        "nvcc",
        "-c",
        "bitonic_sort.cu",
        "-o",
        "bitonic.o",
        "-std=c++20",
        "--compiler-options",
        "-O3,-fPIC",
        "-I",
        cuda_include,
    ]

    print("Compiling CUDA source...")
    print(" ".join(nvcc_cmd))

    result = subprocess.run(nvcc_cmd)
    if result.returncode != 0:
        raise RuntimeError("Failed to compile CUDA source")

    return "bitonic.o"


class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Compile CUDA source first
        cuda_obj = compile_cuda()

        # Add CUDA object file to each extension
        for ext in self.extensions:
            ext.extra_objects.append(cuda_obj)

        # Call parent build
        super().build_extensions()


# Define the extension module
ext_modules = [
    Pybind11Extension(
        "bitonic_sort_cuda",
        [
            "bitonic_sort_wrapper.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            cuda_include,
        ],
        libraries=["cudart"],
        library_dirs=[cuda_lib],
        runtime_library_dirs=[cuda_lib],
        language="c++",
        cxx_std=20,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="bitonic_sort_cuda",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.6",
)
