"""
Flash Attention Benchmark — Triton vs PyTorch on AMD ROCm GPUs.

Measures throughput and memory usage for different sequence lengths.
"""

import torch
import time
import sys
sys.path.insert(0, "..")
from kernels.flash_attention import flash_attention


def benchmark_fn(fn, warmup=5, rep=50):
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


def main():
    print("=" * 70)
    print("Flash Attention Benchmark — Triton vs PyTorch (ROCm)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return

    device = "cuda"
    B, H, D = 2, 8, 64
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Config: B={B}, H={H}, D={D}, causal=True")
    print(f"\n{'Seq Len':>10} {'PyTorch (ms)':>14} {'Triton (ms)':>14} {'Speedup':>8} {'TFLOPS (Triton)':>16}")
    print("-" * 66)

    for S in seq_lengths:
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        # PyTorch SDPA
        pytorch_fn = lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        pytorch_time = benchmark_fn(pytorch_fn)

        # Triton Flash Attention
        triton_fn = lambda: flash_attention(q, k, v)
        triton_time = benchmark_fn(triton_fn)

        # TFLOPS: 2*B*H*S^2*D for attention forward
        tflops = 2 * B * H * S * S * D / triton_time / 1e12
        speedup = pytorch_time / triton_time

        print(f"{S:>8} {pytorch_time*1000:>12.2f}ms {triton_time*1000:>12.2f}ms {speedup:>7.2f}x {tflops:>14.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
