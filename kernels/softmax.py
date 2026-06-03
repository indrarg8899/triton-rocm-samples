"""
Online Softmax — Triton kernel for AMD ROCm GPUs.

Numerically stable softmax using the online (single-pass) algorithm.
Avoids materializing intermediate exp values in global memory.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online softmax along last dimension.

    Each program handles one row. Uses online max tracking for
    numerical stability — no two-pass needed.
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * n_cols

    # Compute offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Online max and sum tracking
    running_max = tl.full([], value=float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([], dtype=tl.float32)

    # Pass 1: online max and exp sum
    for block_start in range(0, n_cols, BLOCK_SIZE):
        mask = col_offsets + block_start < n_cols
        x = tl.load(row_start_ptr + col_offsets + block_start, mask=mask, other=float("-inf"))
        x_f32 = x.to(tl.float32)

        # Online update
        new_max = tl.maximum(running_max, tl.max(x_f32, axis=0))
        running_sum = tl.exp(running_max - new_max) * running_sum + tl.sum(tl.exp(x_f32 - new_max), axis=0)
        running_max = new_max

    # Pass 2: write normalized values
    for block_start in range(0, n_cols, BLOCK_SIZE):
        mask = col_offsets + block_start < n_cols
        x = tl.load(row_start_ptr + col_offsets + block_start, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        softmax_out = tl.exp(x_f32 - running_max) / running_sum
        tl.store(
            output_ptr + row_idx * n_cols + col_offsets + block_start,
            softmax_out,
            mask=mask,
        )


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax along last dimension using online algorithm.

    Args:
        x: (M, N) input tensor
    Returns:
        output: (M, N) softmax output
    """
    M, N = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)

    softmax_kernel[triton.cdiv(M, 1)](
        output, x,
        n_cols=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return output


def run_test():
    """Quick correctness test against PyTorch softmax."""
    M, N = 128, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)

    out_triton = online_softmax(x)
    out_ref = torch.softmax(x, dim=-1)

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"Online Softmax {M}x{N} — Max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Softmax accuracy check failed: {max_diff}"
    print("PASS")
    return out_triton


if __name__ == "__main__":
    run_test()
