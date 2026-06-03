# Autotuning

## How It Works

Triton's `@triton.autotune` decorator tests different kernel configurations:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    ],
    key=["M", "N"],  # tune when these change
)
@triton.jit
def my_kernel(...):
    pass
```

## Key Parameters

- `num_warps`: 4, 8, 16 (AMD has 64-wide wavefronts)
- `num_stages`: 1-3 (software pipelining depth)
- Block sizes: multiples of 64 for AMD alignment

## Benchmarking

```bash
python -m benchmarks.run_all --kernel matmul --sizes 1024,2048,4096
```

The autotuner caches results in `~/.triton/cache/`.
