"""
ROCm Comparison Benchmark — Comprehensive kernel performance comparison.

Runs all Triton kernels and compares against PyTorch equivalents,
reporting throughput metrics and generating a summary.
"""

import torch
import time
import json
import sys
sys.path.insert(0, "..")

from kernels.matmul import matmul
from kernels.softmax import online_softmax
from kernels.layernorm import fused_layernorm
from kernels.rmsnorm import rmsnorm
from kernels.gelu import gelu
from kernels.reduction import reduce_sum


def benchmark_fn(fn, warmup=5, rep=50):
    """Benchmark a function with warmup and repetition."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep


def main():
    print("=" * 70)
    print("Comprehensive Triton Kernel Benchmark — AMD ROCm")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return

    device = "cuda"
    results = []

    # 1. MatMul
    print("\n[1/6] Matrix Multiplication (4096x4096)")
    A = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    B = torch.randn(4096, 4096, device=device, dtype=torch.float16)

    pt_time = benchmark_fn(lambda: torch.matmul(A, B))
    tr_time = benchmark_fn(lambda: matmul(A, B))
    results.append({"kernel": "MatMul 4096x4096", "pytorch_ms": pt_time*1000, "triton_ms": tr_time*1000, "speedup": pt_time/tr_time})
    print(f"  PyTorch: {pt_time*1000:.2f}ms | Triton: {tr_time*1000:.2f}ms | Speedup: {pt_time/tr_time:.2f}x")

    # 2. Softmax
    print("\n[2/6] Online Softmax (4096)")
    x_soft = torch.randn(4096, 4096, device=device, dtype=torch.float32)

    pt_time = benchmark_fn(lambda: torch.softmax(x_soft, dim=-1))
    tr_time = benchmark_fn(lambda: online_softmax(x_soft))
    results.append({"kernel": "Softmax 4096x4096", "pytorch_ms": pt_time*1000, "triton_ms": tr_time*1000, "speedup": pt_time/tr_time})
    print(f"  PyTorch: {pt_time*1000:.2f}ms | Triton: {tr_time*1000:.2f}ms | Speedup: {pt_time/tr_time:.2f}x")

    # 3. LayerNorm
    print("\n[3/6] Fused LayerNorm (4096)")
    x_ln = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    w_ln = torch.randn(4096, device=device, dtype=torch.float32)
    b_ln = torch.randn(4096, device=device, dtype=torch.float32)

    pt_time = benchmark_fn(lambda: torch.nn.functional.layer_norm(x_ln, (4096,), weight=w_ln, bias=b_ln))
    tr_time = benchmark_fn(lambda: fused_layernorm(x_ln, w_ln, b_ln))
    results.append({"kernel": "LayerNorm 4096x4096", "pytorch_ms": pt_time*1000, "triton_ms": tr_time*1000, "speedup": pt_time/tr_time})
    print(f"  PyTorch: {pt_time*1000:.2f}ms | Triton: {tr_time*1000:.2f}ms | Speedup: {pt_time/tr_time:.2f}x")

    # 4. RMSNorm
    print("\n[4/6] RMSNorm (4096)")
    w_rms = torch.randn(4096, device=device, dtype=torch.float32)

    tr_time = benchmark_fn(lambda: rmsnorm(x_ln, w_rms))
    results.append({"kernel": "RMSNorm 4096x4096", "pytorch_ms": None, "triton_ms": tr_time*1000, "speedup": None})
    print(f"  Triton: {tr_time*1000:.2f}ms")

    # 5. GELU
    print("\n[5/6] GELU (1M elements)")
    x_gelu = torch.randn(1024 * 1024, device=device, dtype=torch.float32)

    pt_time = benchmark_fn(lambda: torch.nn.functional.gelu(x_gelu))
    tr_time = benchmark_fn(lambda: gelu(x_gelu))
    results.append({"kernel": "GELU 1M", "pytorch_ms": pt_time*1000, "triton_ms": tr_time*1000, "speedup": pt_time/tr_time})
    print(f"  PyTorch: {pt_time*1000:.2f}ms | Triton: {tr_time*1000:.2f}ms | Speedup: {pt_time/tr_time:.2f}x")

    # 6. Reduction
    print("\n[6/6] Parallel Sum Reduction (4M elements)")
    x_red = torch.randn(4 * 1024 * 1024, device=device, dtype=torch.float32)

    pt_time = benchmark_fn(lambda: x_red.sum())
    tr_time = benchmark_fn(lambda: reduce_sum(x_red))
    results.append({"kernel": "Sum Reduce 4M", "pytorch_ms": pt_time*1000, "triton_ms": tr_time*1000, "speedup": pt_time/tr_time})
    print(f"  PyTorch: {pt_time*1000:.2f}ms | Triton: {tr_time*1000:.2f}ms | Speedup: {pt_time/tr_time:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Kernel':<25} {'PyTorch (ms)':>14} {'Triton (ms)':>14} {'Speedup':>8}")
    print("-" * 64)
    for r in results:
        pt = f"{r['pytorch_ms']:.2f}" if r['pytorch_ms'] else "N/A"
        sp = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        print(f"{r['kernel']:<25} {pt:>14} {r['triton_ms']:>12.2f} {sp:>8}")

    # Save results
    with open("rocm_benchmark_results.json", "w") as f:
        json.dump({
            "device": torch.cuda.get_device_name(0),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to rocm_benchmark_results.json")


if __name__ == "__main__":
    main()
