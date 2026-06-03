"""
Flash Attention v2 in Triton for AMD ROCm GPUs.

Implements the IO-aware attention algorithm from Dao et al. (2023) with
online softmax for memory-efficient attention computation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_forward_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    seqlen: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    scale: tl.constexpr,
):
    """Flash Attention forward pass with online softmax."""
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    # Block offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)

    # Pointers
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh) + offs_m[:, None] * stride_qd + offs_d[None, :]
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh) + offs_n[None, :] * stride_kd + offs_d[:, None]
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh) + offs_n[:, None] * stride_vd + offs_d[None, :]

    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0).to(tl.float32)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # Loop over K, V blocks (causal: only attend to positions <= current)
    for start_n in range(0, seqlen, BLOCK_N):
        offs_n_cur = start_n + offs_n

        # Load K, V blocks
        k = tl.load(k_ptrs, mask=offs_n_cur[None, :] < seqlen, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=offs_n_cur[:, None] < seqlen, other=0.0).to(tl.float32)

        # QK^T
        qk = tl.dot(q, k) * scale

        # Causal mask
        causal_mask = offs_m[:, None] >= offs_n_cur[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(tl.max(qk, axis=1) - m_new)

        # Update accumulator
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p, v)

        m_i = m_new
        l_i = l_new

        # Advance pointers
        k_ptrs += BLOCK_N * stride_kd
        v_ptrs += BLOCK_N * stride_vd

    # Normalize
    acc = acc / l_i[:, None]

    # Write output
    o_ptrs = O + (off_b * stride_ob + off_h * stride_oh) + offs_m[:, None] * stride_od + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < seqlen)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_m: int = 128,
    block_n: int = 128,
) -> torch.Tensor:
    """
    Flash Attention v2 forward pass.

    Args:
        q: (batch, n_heads, seqlen, head_dim)
        k: (batch, n_heads, seqlen, head_dim)
        v: (batch, n_heads, seqlen, head_dim)
        causal: Whether to apply causal masking
        block_m, block_n: Block sizes for tiling
    Returns:
        o: (batch, n_heads, seqlen, head_dim)
    """
    batch, n_heads, seqlen, head_dim = q.shape
    assert k.shape == q.shape and v.shape == q.shape
    assert head_dim in {16, 32, 64, 128, 256}

    scale = head_dim ** -0.5
    o = torch.empty_like(q)

    grid = (triton.cdiv(seqlen, block_m), n_heads, batch)

    _flash_attn_forward_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        n_heads=n_heads,
        head_dim=head_dim,
        seqlen=seqlen,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        scale=scale,
    )
    return o


def run_test():
    """Quick correctness test against PyTorch scaled dot-product attention."""
    B, H, S, D = 2, 8, 512, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    o_triton = flash_attention(q, k, v)

    # Reference: PyTorch SDPA with causal mask
    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    cos_sim = torch.nn.functional.cosine_similarity(
        o_triton.flatten().float(), o_ref.flatten().float(), dim=0
    )
    print(f"Flash Attention {B}x{H}x{S}x{D} — Cosine similarity: {cos_sim.item():.6f}")
    assert cos_sim.item() > 0.98, f"Flash Attention accuracy check failed: {cos_sim.item()}"
    print("PASS")
    return o_triton


if __name__ == "__main__":
    run_test()
