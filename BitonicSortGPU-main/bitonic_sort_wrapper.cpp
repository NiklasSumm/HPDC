#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#include <vector>

// Forward declarations of CUDA functions
extern "C" void bitonicSortSharedMemory(float* data, int n);
extern "C" void bitonicSortCPU(float* data, int n);

namespace py = pybind11;

/**
 * Python wrapper for GPU bitonic sort with shared memory
 */
py::array_t<float> bitonic_sort_gpu_shared(py::array_t<float> input) {
  auto buf = input.request();
  float* ptr = static_cast<float*>(buf.ptr);
  bitonicSortSharedMemory(ptr, buf.shape[0]);
  return input;
}

/**
 * Python wrapper for CPU bitonic sort (for comparison)
 */

py::array_t<float> bitonic_sort_cpu(py::array_t<float> input) {
  auto buf = input.request();
  float* ptr = static_cast<float*>(buf.ptr);
  bitonicSortCPU(ptr, buf.shape[0]);
  return input;
}

PYBIND11_MODULE(bitonic_sort_cuda, m) {
  m.doc() = "Bitonic Sort CUDA implementation with Python bindings";

  m.def("bitonic_sort_gpu_shared", &bitonic_sort_gpu_shared,
        "Sort array using GPU bitonic sort with shared memory",
        py::arg("input"));

  m.def("bitonic_sort_cpu", &bitonic_sort_cpu,
        "Sort array using CPU implementation", py::arg("input"));
}
