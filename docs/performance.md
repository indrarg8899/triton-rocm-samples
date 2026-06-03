# Performance Tips

## Memory Coalescing

Ensure consecutive threads access consecutive memory addresses. Use `tl.arange(0, BLOCK)` for natural coalescing.

## Avoid Bank Conflicts

Add padding to shared memory arrays. Pad first dimension by 1 for 2D arrays.

## Utilize MFMA Instructions

`tl.dot` on CDNA GPUs uses Matrix FMA instructions. Provide float16/bfloat16 inputs for best performance.

## Register Pressure

Keep `num_warps` low (4-8) to reduce register pressure. Higher warps = fewer registers per thread.

## Profiling

```bash
rocprof python kernel.py
# Checkoccupancy, VGPR usage, memory throughput
```

## Common Pitfalls

- Using Python `if` instead of `tl.where` (causes divergence)
- Loading without masks (crashes on boundary)
- Too large BLOCK_SIZE (exceeds shared memory)
