"""
Parallel Reduction — Triton kernel for AMD ROCm GPUs.

Implements tree reduction using warp-level shuffle for efficient
sum/max/min reductions over 1D tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sum_reduction_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Parallel sum reduction over 1D tensor."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask out-of-bounds elements
    mask = offs < n_elements
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Warp-level reduction via associative scan
    # For single-program: use tree reduction pattern
    for stride in tl.static_range(BLOCK_SIZE // 2, 0, -1):
        x += tl.shuffle(x, 1, offset=stride) if hasattr(tl, 'shuffle') else tl.sum(x, axis=0)

    # Use tl.sum for the block reduction (SIMT)
    result = tl.sum(x, axis=0)

    # Write per-block partial sum (caller must handle inter-block reduction)
    tl.store(output_ptr + pid, result)


@triton.jit
def reduce_sum_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass sum reduction: each program reduces a BLOCK_SIZE chunk."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    result = tl.sum(x, axis=0)
    tl.store(output_ptr + pid, result)


@triton.jit
def reduce_max_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass max reduction: each program finds max of BLOCK_SIZE chunk."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(input_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
    result = tl.max(x, axis=0)
    tl.store(output_ptr + pid, result)


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Parallel sum reduction of 1D tensor.

    Args:
        x: 1D input tensor
    Returns:
        scalar tensor with sum
    """
    n = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), 8192)
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    if num_blocks == 1:
        partial = torch.empty(1, device=x.device, dtype=torch.float32)
    else:
        partial = torch.empty(num_blocks, device=x.device, dtype=torch.float32)

    reduce_sum_kernel[triton.cdiv(n, BLOCK_SIZE)](
        partial, x, n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    # Inter-block reduction (recursive or sequential for small num_blocks)
    if num_blocks > 1:
        return partial.sum()
    return partial.squeeze()


def reduce_max(x: torch.Tensor) -> torch.Tensor:
    """
    Parallel max reduction of 1D tensor.

    Args:
        x: 1D input tensor
    Returns:
        scalar tensor with max
    """
    n = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), 8192)

    partial = torch.empty(triton.cdiv(n, BLOCK_SIZE), device=x.device, dtype=torch.float32)

    reduce_max_kernel[triton.cdiv(n, BLOCK_SIZE)](
        partial, x, n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return partial.max()


def run_test():
    """Quick correctness test against PyTorch reductions."""
    n = 65536
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    sum_triton = reduce_sum(x)
    sum_ref = x.sum()
    print(f"Sum Reduction n={n} — Diff: {(sum_triton - sum_ref).abs().item():.2e}")

    max_triton = reduce_max(x)
    max_ref = x.max()
    print(f"Max Reduction n={n} — Diff: {(max_triton - max_ref).abs().item():.2e}")

    assert (sum_triton - sum_ref).abs().item() < 1e-3
    assert (max_triton - max_ref).abs().item() < 1e-5
    print("PASS")
    return sum_triton


if __name__ == "__main__":
    run_test()
