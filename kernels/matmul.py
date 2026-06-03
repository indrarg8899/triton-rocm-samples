"""
Tiled Matrix Multiplication with Shared Memory — Triton for ROCm/AMD GPUs.

Uses block-level tiling with shared memory to maximize memory reuse.
Supports arbitrary matrix sizes with automatic padding.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=5, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matmul: C = A @ B with shared-memory style tiling."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = 1  # single group for now
    group_size = num_pid_m * num_pid_n // num_pid_in_group

    pid_m = (pid * num_pid_in_group + tl.arange(0, num_pid_in_group)) // num_pid_n
    pid_n = (pid * num_pid_in_group + tl.arange(0, num_pid_in_group)) % num_pid_n

    # Accumulator
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Pointers to first block of A and B
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptr = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptr = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Mask
        a_mask = (offs_am[:, None] < M) & ((k * BLOCK_K + offs_k[None, :]) < K)
        b_mask = ((k * BLOCK_K + offs_k[:, None]) < K) & (offs_bn[None, :] < N)

        # Load blocks (software pipelining friendly)
        a = tl.load(a_ptr, mask=a_mask, other=0.0)
        b = tl.load(b_ptr, mask=b_mask, other=0.0)

        # Accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk

    # Write back
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptr = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptr, accumulator, mask=c_mask)


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiply C = A @ B using Triton tiled kernel.

    Args:
        A: (M, K) tensor, contiguous
        B: (K, N) tensor, contiguous
    Returns:
        C: (M, N) tensor
    """
    assert A.is_contiguous(), "A must be contiguous"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[0], f"Incompatible shapes: {A.shape} @ {B.shape}"

    M, K = A.shape
    K2, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


def run_test():
    """Quick correctness test against PyTorch."""
    M, N, K = 512, 512, 512
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    C_triton = matmul(A, B)
    C_ref = torch.matmul(A, B)

    cos_sim = torch.nn.functional.cosine_similarity(
        C_triton.flatten(), C_ref.flatten(), dim=0
    )
    print(f"MatMul {M}x{N}x{K} — Cosine similarity: {cos_sim.item():.6f}")
    assert cos_sim.item() > 0.99, f"MatMul accuracy check failed: {cos_sim.item()}"
    print("PASS")
    return C_triton


if __name__ == "__main__":
    run_test()
