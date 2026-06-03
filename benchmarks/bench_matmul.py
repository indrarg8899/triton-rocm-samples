"""
Matrix Multiplication Benchmark — Triton vs PyTorch on AMD ROCm GPUs.

Measures throughput (GFLOPS) and latency across different matrix sizes.
"""

import torch
import time
import sys
sys.path.insert(0, "..")
from kernels.matmul import matmul


def benchmark_fn(fn, warmup=10, rep=100):
    """Benchmark a function with warmup and repetition."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / rep


def compute_gflops(M, N, K, time_s):
    """Compute GFLOPS for matrix multiply."""
    flops = 2 * M * N * K
    return flops / time_s / 1e9


def main():
    print("=" * 70)
    print("Matrix Multiplication Benchmark — Triton vs PyTorch (ROCm)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return

    device = "cuda"
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"{'Size':>12} {'PyTorch (ms)':>14} {'PyTorch GFLOPS':>16} {'Triton (ms)':>14} {'Triton GFLOPS':>16} {'Speedup':>8}")
    print("-" * 84)

    for M, N, K in sizes:
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        B = torch.randn(K, N, device=device, dtype=torch.float16)

        # PyTorch benchmark
        pytorch_fn = lambda: torch.matmul(A, B)
        pytorch_time = benchmark_fn(pytorch_fn)
        pytorch_gflops = compute_gflops(M, N, K, pytorch_time)

        # Triton benchmark
        triton_fn = lambda: matmul(A, B)
        triton_time = benchmark_fn(triton_fn)
        triton_gflops = compute_gflops(M, N, K, triton_time)

        speedup = pytorch_time / triton_time
        print(f"{M}x{N}x{K} {pytorch_time*1000:>12.2f}ms {pytorch_gflops:>14.1f} {triton_time*1000:>12.2f}ms {triton_gflops:>14.1f} {speedup:>7.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
