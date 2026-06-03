"""
GELU Activation — Fused Triton kernel for AMD ROCm GPUs.

Fused GELU using the tanh approximation:
  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Fused kernel avoids materializing intermediate tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU activation with tanh approximation.

    Each program processes BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU tanh approximation
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654  # sqrt(2/pi)
    COEFF: tl.constexpr = 0.044715

    x_cubed = x * x * x
    inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed)
    tanh_inner = tl.libdevice.tanh(inner)  # Use libdevice for accuracy on ROCm
    out = 0.5 * x * (1.0 + tanh_inner)

    tl.store(output_ptr + offs, out, mask=mask)


@triton.jit
def gelu_exact_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Exact GELU using erf: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    Slightly slower but more accurate.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    SQRT_2_INV: tl.constexpr = 0.7071067811865476  # 1/sqrt(2)
    erf_val = tl.libdevice.erf(x * SQRT_2_INV)
    out = 0.5 * x * (1.0 + erf_val)

    tl.store(output_ptr + offs, out, mask=mask)


def gelu(x: torch.Tensor, exact: bool = False) -> torch.Tensor:
    """
    Fused GELU activation.

    Args:
        x: Input tensor (flat or multi-dimensional)
        exact: If True, use exact erf-based GELU
    Returns:
        output: GELU(x) same shape as input
    """
    n = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = min(triton.next_power_of_2(n), 65536)
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    if exact:
        gelu_exact_kernel[num_blocks](
            output, x, n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
    else:
        gelu_kernel[num_blocks](
            output, x, n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    return output


def run_test():
    """Quick correctness test against PyTorch GELU."""
    n = 4096
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    out_triton = gelu(x, exact=True)
    out_ref = torch.nn.functional.gelu(x)

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"GELU (exact) n={n} — Max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"GELU accuracy check failed: {max_diff}"
    print("PASS")
    return out_triton


if __name__ == "__main__":
    run_test()
