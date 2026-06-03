"""
RMSNorm — Fused Triton kernel for AMD ROCm GPUs.

Root Mean Square Normalization as used in LLaMA / Mistral / Gemma models.
Computes: y = x / sqrt(mean(x^2) + eps) * gamma

More efficient than LayerNorm (no mean subtraction, no bias).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    out_ptr, input_ptr, weight_ptr,
    stride, n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma

    Each program handles one row.
    """
    row = tl.program_id(0)
    row_start = input_ptr + row * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    x = tl.load(row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rms = tl.sqrt(mean_sq + eps)

    # Normalize
    x_norm = x / rms

    # Load weight (gamma only, no bias in RMSNorm)
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)

    # Scale by gamma
    out = x_norm * w

    # Write output
    tl.store(out_ptr + row * stride + col_offsets, out, mask=mask)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Root Mean Square Normalization.

    Args:
        x: (M, N) input tensor
        weight: (N,) gamma parameter
        eps: numerical stability epsilon
    Returns:
        output: (M, N) normalized tensor
    """
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)

    rmsnorm_kernel[triton.cdiv(M, 1)](
        out, x, weight,
        x.stride(0), N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


def run_test():
    """Quick correctness test against manual RMSNorm."""
    M, N = 128, 1024
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    weight = torch.randn(N, device="cuda", dtype=torch.float32)

    out_triton = rmsnorm(x, weight)

    # Reference implementation
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
    out_ref = (x / rms) * weight

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"RMSNorm {M}x{N} — Max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"RMSNorm accuracy check failed: {max_diff}"
    print("PASS")
    return out_triton


if __name__ == "__main__":
    run_test()
