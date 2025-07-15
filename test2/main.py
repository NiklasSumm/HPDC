import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Try to import the compiled CUDA module
try:
    import bitonic_sort_cuda

    CUDA_AVAILABLE = True
    print("CUDA module loaded successfully!")
except ImportError as e:
    print(f"Warning: Could not import CUDA module: {e}")
    CUDA_AVAILABLE = False


def generate_random_array(size, seed=128):
    np.random.seed(seed)
    return np.random.uniform(0.0, 1000.0, size).astype(np.float32)


def validate_sort(original, sorted_array, reference_sorted):
    if not np.allclose(sorted_array, reference_sorted, rtol=1e-4):
        print("ERROR: Sorted array does not match reference!")
        print(f"First 10 elements of result: {sorted_array[:10]}")
        print(f"First 10 elements of reference: {reference_sorted[:10]}")
        return False
    if not np.all(sorted_array[:-1] <= sorted_array[1:]):
        print("ERROR: Sorted array is not monotonically increasing!")
        return False
    return True


def benchmark(func, data, repeats=10):
    # Warm-up
    _ = func(data.copy())
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = func(data.copy())
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / repeats * 1000.0
    return avg_ms


def numpy_sort(data):
    return np.sort(data)


def cpu_sort(data):
    return bitonic_sort_cuda.std_sort_cpu(data)


def gpu_sort(data):
    return bitonic_sort_cuda.bitonic_sort_gpu_shared(data)


def run_performance_comparison():
    sizes = [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
        4194304,
        8388608,
        16777216,
        33554432,
        #33554432*2,
        #33554432*4,
    ]

    print(f"{'N':>8}   {'NumPy (ms)':>10}   {'CPU (ms)':>10}   {'GPU (ms)':>10}")
    print("--------  ----------  ----------  ----------")

    results = {
        "sizes": sizes,
        "numpy": [],
        "cpu": [],
        "gpu": [],
    }

    for n in sizes:
        print(f"► Test N = {n}")
        a = generate_random_array(n)

        # Referenz mit NumPy
        ref = numpy_sort(a)
        t_np = benchmark(numpy_sort, a)
        results["numpy"].append(t_np)

        # CPU (std::sort) baseline
        t_cpu = benchmark(cpu_sort, a) if CUDA_AVAILABLE else float("nan")
        res_cpu = cpu_sort(a.copy()) if CUDA_AVAILABLE else None
        if CUDA_AVAILABLE and not validate_sort(a, res_cpu, ref):
            sys.exit(1)
        results["cpu"].append(t_cpu)

        # GPU-Bitonic Sort
        if CUDA_AVAILABLE:
            t_gpu = benchmark(gpu_sort, a)
            res_gpu = gpu_sort(a.copy())
            if not validate_sort(a, res_gpu, ref):
                sys.exit(1)
        else:
            t_gpu = float("nan")
        results["gpu"].append(t_gpu)

        print(f"{n:8d}   {t_np:10.3f}   {t_cpu:10.3f}   {t_gpu:10.3f}")

    return results


def plot_results(results):
    time_plot_name = "comparison_plot"
    x = results["sizes"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="tight")

    # --- First axes: Time comparison ---
    axes[0].plot(x, results["numpy"], "b-o", label="numpy.sort")
    axes[0].plot(x, results["cpu"], "m-d", label="std::sort")
    if CUDA_AVAILABLE:
        axes[0].plot(x, results["gpu"], "g-^", label="GPU")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Array size [-]")
    axes[0].set_ylabel("Average time [ms]")
    axes[0].set_title("Execution time")
    axes[0].legend()
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # --- Second axes: Speedup comparison ---
    # Compute speedups
    # For both ratios: (reference_time / fast_time)
    speedup_gpu_vs_cpu = np.array(results["cpu"]) / np.array(results["gpu"])
    speedup_gpu_vs_numpy = np.array(results["numpy"]) / np.array(results["gpu"])

    if CUDA_AVAILABLE:
        axes[1].plot(x, speedup_gpu_vs_cpu, "r-o", label="GPU vs std::sort")
        axes[1].plot(x, speedup_gpu_vs_numpy, "b-d", label="GPU vs numpy.sort")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Array size [-]")
    axes[1].set_ylabel("Speedup [-]")
    axes[1].set_title("GPU Speedup")
    axes[1].legend()
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.savefig(f"{time_plot_name}.png", dpi=600)
    print(f"\nPlot saved as {time_plot_name}.png")


if __name__ == "__main__":
    print("=== Bitonic Sort Benchmark ===")
    res = run_performance_comparison()
    plot_results(res)
