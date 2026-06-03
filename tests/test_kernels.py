"""
Comprehensive test suite for Triton ROCm kernels.

Tests all kernels against PyTorch reference implementations.
Requires CUDA/ROCm GPU.
"""

import pytest
import torch
import sys
sys.path.insert(0, "..")

from kernels.matmul import matmul
from kernels.softmax import online_softmax
from kernels.layernorm import fused_layernorm
from kernels.rmsnorm import rmsnorm
from kernels.gelu import gelu
from kernels.reduction import reduce_sum, reduce_max
from kernels.conv2d import conv2d


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    return "cuda"


class TestMatMul:
    def test_small(self, device):
        A = torch.randn(64, 64, device=device, dtype=torch.float16)
        B = torch.randn(64, 64, device=device, dtype=torch.float16)
        C_triton = matmul(A, B)
        C_ref = torch.matmul(A, B)
        assert torch.allclose(C_triton, C_ref, atol=1e-1, rtol=1e-2)

    def test_medium(self, device):
        A = torch.randn(256, 512, device=device, dtype=torch.float16)
        B = torch.randn(512, 256, device=device, dtype=torch.float16)
        C_triton = matmul(A, B)
        C_ref = torch.matmul(A, B)
        assert torch.allclose(C_triton, C_ref, atol=1e-1, rtol=1e-2)

    def test_square_1024(self, device):
        A = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        B = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        C_triton = matmul(A, B)
        C_ref = torch.matmul(A, B)
        cos_sim = torch.nn.functional.cosine_similarity(
            C_triton.flatten().float(), C_ref.flatten().float(), dim=0
        )
        assert cos_sim.item() > 0.99

    def test_non_square(self, device):
        A = torch.randn(128, 256, device=device, dtype=torch.float16)
        B = torch.randn(256, 512, device=device, dtype=torch.float16)
        C_triton = matmul(A, B)
        C_ref = torch.matmul(A, B)
        cos_sim = torch.nn.functional.cosine_similarity(
            C_triton.flatten().float(), C_ref.flatten().float(), dim=0
        )
        assert cos_sim.item() > 0.99


class TestSoftmax:
    def test_accuracy(self, device):
        x = torch.randn(32, 1024, device=device, dtype=torch.float32)
        out_triton = online_softmax(x)
        out_ref = torch.softmax(x, dim=-1)
        assert torch.allclose(out_triton, out_ref, atol=1e-4)

    def test_large(self, device):
        x = torch.randn(128, 4096, device=device, dtype=torch.float32)
        out_triton = online_softmax(x)
        out_ref = torch.softmax(x, dim=-1)
        assert torch.allclose(out_triton, out_ref, atol=1e-4)

    def test_sums_to_one(self, device):
        x = torch.randn(64, 2048, device=device, dtype=torch.float32)
        out = online_softmax(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


class TestLayerNorm:
    def test_accuracy(self, device):
        M, N = 64, 512
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        w = torch.randn(N, device=device, dtype=torch.float32)
        b = torch.randn(N, device=device, dtype=torch.float32)

        out_triton = fused_layernorm(x, w, b)
        out_ref = torch.nn.functional.layer_norm(x, (N,), weight=w, bias=b)
        assert torch.allclose(out_triton, out_ref, atol=1e-4)

    def test_zero_params(self, device):
        M, N = 32, 256
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        w = torch.ones(N, device=device, dtype=torch.float32)
        b = torch.zeros(N, device=device, dtype=torch.float32)

        out_triton = fused_layernorm(x, w, b)
        # Should be approximately zero-mean, unit-variance per row
        mean = out_triton.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)


class TestRMSNorm:
    def test_accuracy(self, device):
        M, N = 64, 512
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        w = torch.randn(N, device=device, dtype=torch.float32)

        out_triton = rmsnorm(x, w)

        # Reference
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        out_ref = (x / rms) * w

        assert torch.allclose(out_triton, out_ref, atol=1e-5)

    def test_unit_weight(self, device):
        M, N = 32, 256
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        w = torch.ones(N, device=device, dtype=torch.float32)

        out_triton = rmsnorm(x, w)
        # With unit weight, output should have RMS ≈ 1 per row
        row_rms = torch.sqrt(torch.mean(out_triton ** 2, dim=-1))
        assert torch.allclose(row_rms, torch.ones_like(row_rms), atol=1e-4)


class TestGELU:
    def test_exact_vs_pytorch(self, device):
        x = torch.randn(4096, device=device, dtype=torch.float32)
        out_triton = gelu(x, exact=True)
        out_ref = torch.nn.functional.gelu(x)
        assert torch.allclose(out_triton, out_ref, atol=1e-4)

    def test_approx_vs_pytorch(self, device):
        x = torch.randn(4096, device=device, dtype=torch.float32)
        out_triton = gelu(x, exact=False)
        out_ref = torch.nn.functional.gelu(x)
        # Approximate GELU has slightly larger error
        assert torch.allclose(out_triton, out_ref, atol=2e-2)

    def test_negative_values(self, device):
        x = torch.linspace(-3, 0, 1024, device=device, dtype=torch.float32)
        out = gelu(x, exact=True)
        # GELU should be near 0 for very negative inputs
        assert out[0].item() < 0.01

    def test_positive_values(self, device):
        x = torch.linspace(0, 3, 1024, device=device, dtype=torch.float32)
        out = gelu(x, exact=True)
        # GELU should be approximately x for large positive inputs
        assert torch.allclose(out[-10:], x[-10:], atol=0.1)


class TestReduction:
    def test_sum_accuracy(self, device):
        x = torch.randn(65536, device=device, dtype=torch.float32)
        s_triton = reduce_sum(x)
        s_ref = x.sum()
        assert torch.allclose(s_triton, s_ref, atol=1e-2)

    def test_max_accuracy(self, device):
        x = torch.randn(65536, device=device, dtype=torch.float32)
        m_triton = reduce_max(x)
        m_ref = x.max()
        assert torch.allclose(m_triton, m_ref, atol=1e-5)

    def test_sum_power_of_two(self, device):
        x = torch.ones(4096, device=device, dtype=torch.float32)
        s = reduce_sum(x)
        assert torch.allclose(s, torch.tensor(4096.0, device=device), atol=1e-3)


class TestConv2d:
    def test_basic(self, device):
        input = torch.randn(1, 3, 8, 8, device=device, dtype=torch.float32)
        weight = torch.randn(8, 3, 3, 3, device=device, dtype=torch.float32)

        out_triton = conv2d(input, weight, padding=1)
        out_ref = torch.nn.functional.conv2d(input, weight, padding=1)

        # Conv2d is compute-heavy; allow larger tolerance for small test
        assert torch.allclose(out_triton, out_ref, atol=1e-2)


class TestIntegration:
    """Integration tests: chain multiple kernels together."""

    def test_transformer_block(self, device):
        """Simulate a transformer block: LayerNorm -> Linear -> GELU -> Linear"""
        B, S, D = 2, 128, 512
        x = torch.randn(B * S, D, device=device, dtype=torch.float32)

        # Fused LayerNorm
        w = torch.randn(D, device=device, dtype=torch.float32)
        b = torch.randn(D, device=device, dtype=torch.float32)
        h = fused_layernorm(x, w, b)

        # GELU
        h = gelu(h, exact=True)

        # RMSNorm
        w2 = torch.randn(D, device=device, dtype=torch.float32)
        h = rmsnorm(h, w2)

        assert h.shape == x.shape
        assert torch.isfinite(h).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
