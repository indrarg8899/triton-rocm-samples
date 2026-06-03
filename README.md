# Triton ROCm Samples

<p align="center">
  <img src="https://img.shields.io/badge/ROCm-6.0-red?logo=amd" alt="ROCm">
  <img src="https://img.shields.io/badge/Triton-2.1-blue" alt="Triton">
  <img src="https://img.shields.io/badge/GPU-MI300X-orange" alt="MI300X">
  <img src="https://img.shields.io/badge/Python-3.10%2B-green?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</p>

<p align="center">
  <b>Production-grade Triton kernel samples optimized for AMD ROCm GPUs (MI300X, MI250X, W7900)</b>
</p>

---

## Features

- **8 hand-optimized Triton kernels** targeting ROCm HIP backend
- **Shared memory tiling** for matrix multiplication
- **Flash Attention v2** with online softmax
- **Fused operator kernels** (LayerNorm, RMSNorm, GELU, Softmax)
- **Parallel reduction** with warp-level primitives
- **2D convolution** with implicit GEMM
- **Benchmarking suite** comparing Triton vs PyTorch on AMD GPUs
- **Zero PyTorch dependency** — pure Triton where possible
- **Fully tested** with pytest

## Kernel Library

| Kernel | File | Description |
|--------|------|-------------|
| Tiled MatMul | `kernels/matmul.py` | Shared-memory tiled matrix multiply with autotuning |
| Flash Attention v2 | `kernels/flash_attention.py` | IO-aware attention with causal masking |
| Online Softmax | `kernels/softmax.py` | Numerically stable online softmax |
| Parallel Reduction | `kernels/reduction.py` | Warp-shuffle reduction primitives |
| Fused LayerNorm | `kernels/layernorm.py` | Fused layer normalization + affine transform |
| GELU Activation | `kernels/gelu.py` | Fused GELU with approximate tanh |
| 2D Convolution | `kernels/conv2d.py` | Implicit GEMM 2D convolution |
| RMSNorm | `kernels/rmsnorm.py` | Root mean square normalization |

## Quick Start

```bash
# Install
git clone https://github.com/indrarg8899/triton-rocm-samples.git
cd triton-rocm-samples
pip install -e .

# Run a kernel
python -c "from kernels.matmul import matmul_kernel; matmul_kernel.run_test()"

# Run benchmarks
python benchmarks/bench_matmul.py

# Run tests
pytest tests/
```

## Requirements

- AMD GPU with ROCm 6.0+ (MI300X recommended)
- Python 3.10+
- Triton 2.1+
- PyTorch (for benchmarks only)

## Benchmarks (MI300X)

| Kernel | PyTorch (ms) | Triton (ms) | Speedup |
|--------|-------------|-------------|---------|
| MatMul 4096x4096 | 2.84 | 2.12 | 1.34x |
| Flash Attention 2048 | 3.21 | 1.87 | 1.72x |
| Softmax 4096 | 0.42 | 0.28 | 1.50x |
| LayerNorm 4096 | 0.31 | 0.19 | 1.63x |

## Project Structure

```
triton-rocm-samples/
├── kernels/              # Triton kernel implementations
│   ├── matmul.py
│   ├── flash_attention.py
│   ├── softmax.py
│   ├── reduction.py
│   ├── layernorm.py
│   ├── gelu.py
│   ├── conv2d.py
│   └── rmsnorm.py
├── benchmarks/           # Performance benchmarks
│   ├── bench_matmul.py
│   ├── bench_attention.py
│   └── compare_rocm.py
├── tests/                # Unit tests
│   └── test_kernels.py
├── docs/                 # Documentation
│   ├── kernel_guide.md
│   └── triton_rocm.md
├── configs/
│   └── default.yml
├── setup.py
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-kernel`)
3. Add kernel + test + benchmark
4. Run `pytest tests/`
5. Submit PR

## License

MIT — see [LICENSE](LICENSE)
