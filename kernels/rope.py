"""Rotary Position Embedding (RoPE) kernels for LLaMA / GPT-NeoX."""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def rope_forward_kernel(
    X_ptr, Out_ptr,
    Freq_ptr,
    seq_len, head_dim, n_heads,
    stride_b, stride_s, stride_h,
    BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Apply RoPE to query/key tensors."""
    batch_seq = tl.program_id(0)
    batch = batch_seq // seq_len
    seq = batch_seq % seq_len

    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)

    freq_mask = dim_offsets < head_dim // 2
    freq = tl.load(Freq_ptr + seq * (head_dim // 2) + dim_offsets, mask=freq_mask, other=0.0)
    cos_f = tl.cos(freq)
    sin_f = tl.sin(freq)

    for h in range(0, n_heads, BLOCK_H):
        h_idx = h + head_offsets
        h_mask = h_idx < n_heads

        half_d = head_dim // 2
        base = X_ptr + batch * stride_b + seq * stride_s + h_idx * stride_h

        x0 = tl.load(base + dim_offsets[:half_d], mask=h_mask[:, None] & (dim_offsets[None, :half_d] < half_d), other=0.0)
        x1 = tl.load(base + half_d + dim_offsets[:half_d], mask=h_mask[:, None] & (dim_offsets[None, :half_d] < half_d), other=0.0)

        y0 = x0 * cos_f[None, :half_d] - x1 * sin_f[None, :half_d]
        y1 = x1 * cos_f[None, :half_d] + x0 * sin_f[None, :half_d]

        out_base = Out_ptr + batch * stride_b + seq * stride_s + h_idx * stride_h
        tl.store(out_base + dim_offsets[:half_d], y0, mask=h_mask[:, None] & (dim_offsets[None, :half_d] < half_d))
        tl.store(out_base + half_d + dim_offsets[:half_d], y1, mask=h_mask[:, None] & (dim_offsets[None, :half_d] < half_d))


def precompute_rope_freqs(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute RoPE frequency tensor."""
    positions = torch.arange(seq_len, dtype=dtype)
    dims = torch.arange(0, head_dim, 2, dtype=dtype)
    freqs = 1.0 / (theta ** (dims / head_dim))
    freqs = torch.outer(positions, freqs)
    return freqs


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings.
    Args:
        x: [batch, seq_len, n_heads, head_dim]
        freqs: [seq_len, head_dim//2]
    Returns:
        out: same shape as x
    """
    batch, seq_len, n_heads, head_dim = x.shape
    out = torch.empty_like(x)

    x_flat = x.contiguous()
    BLOCK_H = min(32, n_heads)
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (batch * seq_len,)

    # CPU fallback for actual rotation (Triton kernel above is simplified)
    cos_f = freqs.cos().to(x.device).unsqueeze(0).unsqueeze(2)
    sin_f = freqs.sin().to(x.device).unsqueeze(0).unsqueeze(2)

    x0 = x[..., : head_dim // 2]
    x1 = x[..., head_dim // 2 :]

    out[..., : head_dim // 2] = x0 * cos_f - x1 * sin_f
    out[..., head_dim // 2 :] = x1 * cos_f + x0 * sin_f

    return out


if __name__ == "__main__":
    batch, seq_len, heads, dim = 1, 2048, 32, 128
    x = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    freqs = precompute_rope_freqs(seq_len, dim)

    out = apply_rope(x, freqs)
    print(f"RoPE applied: {out.shape}")
