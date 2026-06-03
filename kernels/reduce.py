"""Parallel reduction kernels (sum, mean, max, argmax) for AMD GPUs."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def reduce_sum_kernel(
    input_ptr, output_ptr,
    n_elements,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Parallel sum reduction along last axis."""
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * stride_row

    _sum = tl.zeros([], dtype=tl.float32)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for i in range(0, tl.cdiv(n_elements, BLOCK_SIZE)):
        mask = (i * BLOCK_SIZE + col_offsets) < n_elements
        x = tl.load(row_start + i * BLOCK_SIZE + col_offsets, mask=mask, other=0.0)
        _sum += tl.sum(x, axis=0)

    tl.store(output_ptr + row_idx, _sum)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def reduce_max_kernel(
    input_ptr, output_ptr,
    n_elements,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Parallel max reduction along last axis."""
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * stride_row

    _max = tl.full([], value=-float("inf"), dtype=tl.float32)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for i in range(0, tl.cdiv(n_elements, BLOCK_SIZE)):
        mask = (i * BLOCK_SIZE + col_offsets) < n_elements
        x = tl.load(row_start + i * BLOCK_SIZE + col_offsets, mask=mask, other=-float("inf"))
        _max = tl.maximum(_max, tl.max(x, axis=0))

    tl.store(output_ptr + row_idx, _max)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def segmented_reduce_kernel(
    input_ptr, segment_ids_ptr, output_ptr, n_segments,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Segmented sum reduction by segment IDs."""
    seg_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    seg_sum = tl.zeros([], dtype=tl.float32)
    for i in range(0, tl.cdiv(n_elements, BLOCK_SIZE)):
        idx = i * BLOCK_SIZE + col_offsets
        mask = idx < n_elements
        sids = tl.load(segment_ids_ptr + idx, mask=mask, other=-1)
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        seg_mask = sids == seg_id
        seg_sum += tl.sum(tl.where(seg_mask, x, 0.0), axis=0)

    tl.store(output_ptr + seg_id, seg_sum)


def reduce_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim != -1:
        x = x.transpose(dim, -1).contiguous()
    orig_shape = x.shape[:-1]
    n_elements = x.shape[-1]
    x_flat = x.reshape(-1, n_elements).contiguous()
    n_rows = x_flat.shape[0]

    out = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    grid = (n_rows,)
    reduce_sum_kernel[grid](x_flat, out, n_elements, x_flat.stride(0))
    return out.reshape(orig_shape)


def reduce_max(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim != -1:
        x = x.transpose(dim, -1).contiguous()
    orig_shape = x.shape[:-1]
    n_elements = x.shape[-1]
    x_flat = x.reshape(-1, n_elements).contiguous()
    n_rows = x_flat.shape[0]

    out = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    grid = (n_rows,)
    reduce_max_kernel[grid](x_flat, out, n_elements, x_flat.stride(0))
    return out.reshape(orig_shape)


if __name__ == "__main__":
    x = torch.randn(16_000_000, device="cuda", dtype=torch.float16)
    result = reduce_sum(x)
    ref = x.sum()
    error = abs(result.item() - ref.item())
    print(f"Reduce sum 16M: result={result.item():.2f}, ref={ref.item():.2f}, error={error:.4f}")
