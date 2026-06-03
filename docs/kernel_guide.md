# Kernel Development Guide

This guide explains how each Triton kernel is structured and how to write new ones for ROCm/AMD GPUs.

## Kernel Architecture

All kernels follow the Triton programming model:

1. **Grid**: Defines how many program instances launch
2. **Program**: Each program processes a block of data
3. **Block**: Within each program, `tl.arange(0, BLOCK_SIZE)` creates a vector of lanes

## Shared Memory Tiling

The matmul kernel uses software-managed tiling:

```
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + k * stride_ak)  # Load tile from A
    b = tl.load(b_ptr + k * stride_bk)  # Load tile from B
    acc += tl.dot(a, b)                  # Accumulate in registers
```

Key optimizations:
- **Multibuffering**: Prefetch next tile while computing current
- **Autotuning**: Select optimal BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps
- **Register blocking**: Keep accumulator in FP32 for precision

## Online Algorithms

### Online Softmax
Single-pass softmax tracking max and sum simultaneously:

```python
new_max = max(running_max, max(x))
running_sum = exp(running_max - new_max) * running_sum + sum(exp(x - new_max))
running_max = new_max
```

### Flash Attention
IO-aware attention with online softmax:
- Tile Q, K, V in blocks
- For each Q block, iterate over K, V blocks
- Accumulate with online softmax (no full attention matrix in memory)

## Fused Kernels

### LayerNorm
Two-pass within program:
1. Compute mean and variance
2. Normalize and apply affine transform

### RMSNorm
Single-pass (no mean subtraction needed):
```
rms = sqrt(mean(x^2) + eps)
y = x / rms * gamma
```

### GELU
Elementwise fused activation:
```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

## Writing New Kernels

### Template

```python
@triton.jit
def my_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(input_ptr + offs, mask=mask, other=0.0)

    # Compute...

    tl.store(output_ptr + offs, result, mask=mask)
```

### Best Practices

1. **Use `tl.constexpr`** for block sizes and compile-time constants
2. **Mask out-of-bounds** with `tl.load(..., mask=..., other=...)`
3. **Autotune** with `@triton.autotune` over multiple configs
4. **Fuse operations** to avoid global memory round-trips
5. **Use FP32 accumulators** for precision, FP16 for compute
6. **Profile with `triton.profiler`** to find memory/compute bottlenecks

### Autotuning Guide

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=4, num_warps=4),
    ],
    key=["M", "N"],  # Tune per unique (M, N) pair
)
```

- **num_warps**: Number of warps per program (2, 4, 8)
- **num_stages**: Software pipeline depth (1-5)
- **BLOCK sizes**: Choose powers of 2, balanced with register usage

## ROCm-Specific Notes

- Triton compiles to HIP/AMDGPU via the ROCm backend
- Shared memory on MI300X: 256KB per CU
- Wavefront size: 64 (vs 32 on NVIDIA)
- `tl.libdevice` calls map to ROCm device intrinsics
- Memory hierarchy: Global → LDS (shared) → Registers
