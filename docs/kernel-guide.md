# Kernel Writing Guide

## Triton Basics

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK) + pid * BLOCK
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * 2, mask=mask)
```

## Key Concepts

1. **Program IDs**: Each thread block gets `tl.program_id()`
2. **Block sizes**: Use `tl.constexpr` for compile-time constants
3. **Masking**: Always mask bounds with `tl.load(..., mask=mask, other=0.0)`
4. **Dot product**: `tl.dot(a, b)` for matrix ops (uses MFMA on AMD)
5. **Autotune**: `@triton.autotune` for finding optimal block sizes

## AMD-Specific Tips

- Use `num_warps=4` or `num_warps=8` (AMD wavefront size is 64)
- `num_stages` controls software pipelining (2-3 is typical)
- MFMA instructions used automatically for `tl.dot` on CDNA
