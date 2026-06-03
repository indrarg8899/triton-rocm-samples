# Triton on ROCm — Guide for AMD GPU Development

## Overview

Triton is an open-source language for programming GPUs at a high level. On AMD GPUs, Triton compiles to HIP code via the ROCm backend, enabling performance-portable GPU kernels.

## Supported Hardware

| GPU | Architecture | Memory | Status |
|-----|-------------|--------|--------|
| MI300X | CDNA 3 | 192 GB HBM3 | Full support |
| MI250X | CDNA 2 | 128 GB HBM2e | Full support |
| MI210 | CDNA 2 | 64 GB HBM2e | Full support |
| W7900 | RDNA 3 | 48 GB GDDR6 | Partial support |
| W7800 | RDNA 3 | 32 GB GDDR6 | Partial support |

## Installation

### Prerequisites

```bash
# ROCm 6.0+
sudo apt install rocm-dev rocm-hip-runtime

# Verify ROCm
rocminfo | head -20

# Set GPU target (for MI300X)
export ROCM_TARGET_ARCH=gfx942
```

### Install Triton for ROCm

```bash
# Option 1: Pre-built wheel (recommended)
pip install triton==2.1.0

# Option 2: Build from source
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -e .
```

### Verify Installation

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")
out = torch.empty(1024, device="cuda")

add_kernel[triton.cdiv(1024, 256)](x, y, out, 1024, BLOCK=256)
print("Triton on ROCm: OK")
```

## Key Differences from NVIDIA

| Feature | NVIDIA (CUDA) | AMD (ROCm) |
|---------|--------------|------------|
| Wavefront size | 32 | 64 |
| Shared memory | 48-228 KB | 64-256 KB |
| Global memory | GDDR/HBM | HBM2e/HBM3 |
| Device calls | `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| Kernel launch | `<<<grid, block>>>` | `hipLaunchKernelGGL` |

### Implications for Triton Kernels

1. **Wavefront size**: `num_warps` parameter — AMD uses wave64, so 2 warps = 128 threads
2. **Shared memory**: Larger on MI300X — can use bigger tile sizes
3. **Register pressure**: MI300X has 256 registers per thread — more headroom
4. **Memory bandwidth**: MI300X HBM3: 5.3 TB/s — use larger block sizes

## Performance Optimization

### Memory Hierarchy

```
Global Memory (HBM3)
    ↓ Load/Store
Shared Memory (LDS, 256KB)
    ↓ Register file
Registers (256 per thread)
    ↓
Compute Units (110 CUs on MI300X)
```

### Optimization Strategies

1. **Maximize shared memory reuse**: Larger tiles = fewer global loads
2. **Coalesce memory access**: Contiguous access patterns
3. **Use vectorized loads**: `tl.load(ptr, mask=...)` loads multiple elements
4. **Overlap compute and memory**: Software pipelining with `num_stages`
5. **Minimize register spills**: Tune BLOCK sizes to stay within register budget

### Autotuning for AMD

```python
@triton.autotune(
    configs=[
        # More aggressive tiling for MI300X
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_stages=4, num_warps=16),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        # Conservative for smaller GPUs
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
    ],
    key=["M", "N"],
)
```

## Debugging

### ROCm Profiling

```bash
# RocProf for kernel timing
rocprof --stats python my_kernel.py

# Trace mode
rocprof --trace python my_kernel.py
```

### Triton Debugging

```python
# Enable Triton debug mode
import os
os.environ["TRITON_DEBUG"] = "1"

# Print IR
@triton.jit
def debug_kernel(...):
    tl.static_print("Compiling with BLOCK_SIZE =", BLOCK_SIZE)
```

## Resources

- [Triton Documentation](https://triton-lang.org/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD GPU Architecture Guide](https://www.amd.com/en/technologies/cdna)
- [Triton GitHub](https://github.com/triton-lang/triton)
