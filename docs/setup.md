# Triton on ROCm Setup

## Prerequisites

- ROCm 6.0+ with HIP runtime
- Python 3.10+
- PyTorch with ROCm support

## Install

```bash
pip install triton-rocm-samples
```

## Build Triton from Source (optional)

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
git checkout v2.3.1
LLVM_ENABLE_RUNTIMES=off pip install -e python/
```

## Environment

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCM_HOME=/opt/rocm
```

## Verify

```python
import triton
import triton.language as tl
print(f"Triton {triton.__version__} on ROCm")
```
