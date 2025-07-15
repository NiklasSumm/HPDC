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
    return bitonic_sort_cuda.bitonic_sort_cpu(data)


def gpu_sort(data):
    return bitonic_sort_cuda.bitonic_sort_gpu_shared(data)


def run_performance_comparison():
    sizes = [
        # 128,
        # 512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        # 262144,
        # 524288,
        # 1048576,
        # 2097152,
        # 4194304,
        # 8388608,
        # 16777216,
        # 33554432,
        # 67108864,
        # 134217728,
        # 268435456,
        # 536870912,
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

        # CPU-Bitonic Sort
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
    plt.figure(figsize=(10, 6))
    x = results["sizes"]

    plt.plot(x, results["numpy"], "b-o", label="NumPy sort")
    plt.plot(x, results["cpu"], "m-d", label="CPU bitonic")
    if CUDA_AVAILABLE:
        plt.plot(x, results["gpu"], "g-^", label="GPU bitonic")

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Array size N")
    plt.ylabel("Average time (ms)")
    plt.title("Sort Performance Comparison")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("comparison_plot.png", dpi=300)
    print("\nPlot gespeichert unter comparison_plot.png")


if __name__ == "__main__":
    print("=== Bitonic Sort Benchmark ===")
    res = run_performance_comparison()
    plot_results(res)
