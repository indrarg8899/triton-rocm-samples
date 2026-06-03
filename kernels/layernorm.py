"""
Fused LayerNorm — Triton kernel for AMD ROCm GPUs.

Single-pass fused layer normalization: computes mean and variance,
then normalizes and applies affine transform (gamma, beta) in one kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    out_ptr, input_ptr, weight_ptr, bias_ptr,
    stride, n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

    Each program handles one row. Two-pass within program:
    Pass 1: compute mean and variance
    Pass 2: normalize and write output
    """
    row = tl.program_id(0)
    row_start = input_ptr + row * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    x = tl.load(row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Pass 1: mean
    mean = tl.sum(x, axis=0) / n_cols

    # Pass 1: variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * inv_std

    # Load weight and bias
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Affine transform
    out = x_norm * w + b

    # Write output
    out_store = out.to(input_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride + col_offsets, out_store, mask=mask)


def fused_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused layer normalization.

    Args:
        x: (M, N) input tensor
        weight: (N,) gamma parameter
        bias: (N,) beta parameter
        eps: numerical stability epsilon
    Returns:
        output: (M, N) normalized tensor
    """
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)

    layernorm_kernel[triton.cdiv(M, 1)](
        out, x, weight, bias,
        x.stride(0), N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


def run_test():
    """Quick correctness test against PyTorch LayerNorm."""
    M, N = 128, 1024
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    weight = torch.randn(N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    out_triton = fused_layernorm(x, weight, bias)
    out_ref = torch.nn.functional.layer_norm(x, (N,), weight=weight, bias=bias)

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"Fused LayerNorm {M}x{N} — Max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-4, f"LayerNorm accuracy check failed: {max_diff}"
    print("PASS")
    return out_triton


if __name__ == "__main__":
    run_test()
