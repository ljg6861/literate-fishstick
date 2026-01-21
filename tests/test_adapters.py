"""
Unit tests for PEFT adapters (LoRA, DoRA, PiSSA)
"""

import pytest
import torch
import torch.nn as nn
import math

from efficient_training.adapters import (
    LoRALinear,
    DoRALinear,
    pissa_init,
    apply_pissa_init,
    inject_lora_adapters,
    merge_lora_weights,
    get_lora_params,
    get_lora_param_groups,
    count_lora_parameters,
)


class TestLoRALinear:
    """Tests for LoRALinear adapter."""
    
    def test_init_shapes(self):
        """Test that LoRA matrices have correct shapes."""
        base = nn.Linear(64, 128)
        lora = LoRALinear(base, rank=8, alpha=16.0)
        
        assert lora.lora_A.shape == (8, 64)
        assert lora.lora_B.shape == (128, 8)
    
    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        base = nn.Linear(64, 128)
        lora = LoRALinear(base, rank=8)
        
        x = torch.randn(4, 32, 64)
        out = lora(x)
        
        assert out.shape == (4, 32, 128)
    
    def test_zero_init_matches_base(self):
        """Test that with B=0, output matches base layer (plus scaling)."""
        base = nn.Linear(64, 128, bias=False)
        lora = LoRALinear(base, rank=8, alpha=16.0, dropout=0.0)
        
        # B is zero-initialized, so LoRA contribution should be 0
        x = torch.randn(4, 64)
        
        base_out = base(x)
        lora_out = lora(x)
        
        torch.testing.assert_close(base_out, lora_out, atol=1e-5, rtol=1e-5)
    
    def test_base_frozen(self):
        """Test that base layer weights are frozen."""
        base = nn.Linear(64, 128)
        lora = LoRALinear(base, rank=8)
        
        assert not lora.base_layer.weight.requires_grad
        if lora.base_layer.bias is not None:
            assert not lora.base_layer.bias.requires_grad
        
        # LoRA params should be trainable
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad
    
    def test_merge_weights(self):
        """Test that merge_weights produces equivalent linear layer."""
        base = nn.Linear(64, 128, bias=True)
        lora = LoRALinear(base, rank=8, alpha=16.0)
        
        # Set non-zero LoRA weights
        with torch.no_grad():
            lora.lora_A.normal_()
            lora.lora_B.normal_()
        
        x = torch.randn(4, 64)
        lora_out = lora(x)
        
        merged = lora.merge_weights()
        merged_out = merged(x)
        
        torch.testing.assert_close(lora_out, merged_out, atol=1e-4, rtol=1e-4)
    
    def test_rslora_scaling(self):
        """Test rank-stabilized LoRA uses sqrt scaling."""
        base = nn.Linear(64, 128)
        lora_std = LoRALinear(base, rank=16, alpha=32.0, use_rslora=False)
        lora_rs = LoRALinear(nn.Linear(64, 128), rank=16, alpha=32.0, use_rslora=True)
        
        # Standard: alpha/rank = 32/16 = 2
        assert lora_std.scaling == 2.0
        # RS-LoRA: alpha/sqrt(rank) = 32/4 = 8
        assert lora_rs.scaling == 8.0


class TestDoRALinear:
    """Tests for DoRALinear adapter."""
    
    def test_init_shapes(self):
        """Test that DoRA components have correct shapes."""
        base = nn.Linear(64, 128)
        dora = DoRALinear(base, rank=8, alpha=16.0)
        
        assert dora.lora_A.shape == (8, 64)
        assert dora.lora_B.shape == (128, 8)
        assert dora.magnitude.shape == (128,)
    
    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        base = nn.Linear(64, 128)
        dora = DoRALinear(base, rank=8)
        
        x = torch.randn(4, 32, 64)
        out = dora(x)
        
        assert out.shape == (4, 32, 128)
    
    def test_magnitude_init(self):
        """Test that magnitude is initialized from base weight norms."""
        base = nn.Linear(64, 128, bias=False)
        expected_norms = base.weight.norm(dim=1)
        
        dora = DoRALinear(base, rank=8)
        
        torch.testing.assert_close(dora.magnitude, expected_norms, atol=1e-5, rtol=1e-5)
    
    def test_base_frozen(self):
        """Test that base layer weights are frozen."""
        base = nn.Linear(64, 128)
        dora = DoRALinear(base, rank=8)
        
        assert not dora.base_layer.weight.requires_grad
        assert dora.lora_A.requires_grad
        assert dora.lora_B.requires_grad
        assert dora.magnitude.requires_grad
    
    def test_merge_weights(self):
        """Test that merge_weights produces equivalent linear layer."""
        base = nn.Linear(64, 128, bias=True)
        dora = DoRALinear(base, rank=8, alpha=16.0)
        
        # Set non-zero LoRA weights
        with torch.no_grad():
            dora.lora_A.normal_()
            dora.lora_B.normal_()
        
        x = torch.randn(4, 64)
        dora_out = dora(x)
        
        merged = dora.merge_weights()
        merged_out = merged(x)
        
        torch.testing.assert_close(dora_out, merged_out, atol=1e-5, rtol=1e-5)


class TestPiSSA:
    """Tests for PiSSA initialization."""
    
    def test_pissa_shapes(self):
        """Test that PiSSA returns correct shapes."""
        weight = torch.randn(128, 64)
        residual, A, B = pissa_init(weight, rank=8)
        
        assert residual.shape == weight.shape
        assert A.shape == (8, 64)
        assert B.shape == (128, 8)
    
    def test_pissa_reconstruction(self):
        """Test that PiSSA produces valid low-rank approximation."""
        weight = torch.randn(128, 64)
        residual, A, B = pissa_init(weight, rank=8)
        
        # Reconstruction: residual + B @ A should equal original weight
        reconstructed = residual + B @ A
        
        torch.testing.assert_close(reconstructed, weight, atol=1e-4, rtol=1e-4)
    
    def test_pissa_orthogonality(self):
        """Test that PiSSA initialization has good conditioning."""
        weight = torch.randn(128, 64)
        residual, A, B = pissa_init(weight, rank=8)
        
        # Check that A and B have reasonable norms (not degenerate)
        assert A.norm() > 0.1
        assert B.norm() > 0.1
    
    def test_apply_pissa_to_lora(self):
        """Test applying PiSSA to an existing LoRA layer."""
        base = nn.Linear(64, 128, bias=False)
        lora = LoRALinear(base, rank=8, alpha=16.0)
        
        original_base_weight = base.weight.clone()
        
        apply_pissa_init(lora)
        
        # Base weight should now be different (residual)
        assert not torch.allclose(lora.base_layer.weight, original_base_weight)
        
        # But A and B should reconstruct close to original
        reconstructed = lora.base_layer.weight + lora.lora_B @ lora.lora_A
        torch.testing.assert_close(reconstructed, original_base_weight, atol=1e-4, rtol=1e-4)


class TestInjection:
    """Tests for adapter injection utilities."""
    
    def test_inject_lora_adapters(self):
        """Test injecting LoRA into a simple model."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        
        adapted = inject_lora_adapters(model, rank=8, alpha=16.0)
        
        # Should have adapted 2 linear layers
        assert len(adapted) == 2
        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[2], LoRALinear)
    
    def test_inject_dora_adapters(self):
        """Test injecting DoRA into a model."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        
        adapted = inject_lora_adapters(model, rank=8, use_dora=True)
        
        assert isinstance(model[0], DoRALinear)
        assert isinstance(model[2], DoRALinear)
    
    def test_inject_with_target_modules(self):
        """Test injecting only specific modules."""
        model = nn.ModuleDict({
            "encoder": nn.Linear(64, 128),
            "decoder": nn.Linear(128, 64),
        })
        
        adapted = inject_lora_adapters(
            model, 
            target_modules={"encoder"}, 
            rank=8
        )
        
        assert len(adapted) == 1
        assert isinstance(model.encoder, LoRALinear)
        assert isinstance(model.decoder, nn.Linear)  # Not adapted
    
    def test_merge_all_adapters(self):
        """Test merging all adapters back to base."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        
        inject_lora_adapters(model, rank=8)
        
        # Set non-zero weights
        for m in model.modules():
            if isinstance(m, LoRALinear):
                with torch.no_grad():
                    m.lora_A.normal_()
                    m.lora_B.normal_()
        
        x = torch.randn(4, 64)
        pre_merge_out = model(x)
        
        merge_lora_weights(model)
        
        # Should now be standard Linear layers
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[2], nn.Linear)
        assert not isinstance(model[0], LoRALinear)
        
        post_merge_out = model(x)
        torch.testing.assert_close(pre_merge_out, post_merge_out, atol=1e-3, rtol=1e-3)


class TestParameterUtils:
    """Tests for parameter utility functions."""
    
    def test_get_lora_params(self):
        """Test extracting LoRA parameters."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 32),
        )
        inject_lora_adapters(model, rank=8)
        
        params = get_lora_params(model)
        
        # 2 layers * 2 params (A, B) = 4 parameters
        assert len(params) == 4
    
    def test_get_lora_param_groups(self):
        """Test getting parameter groups for LoRA+."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 32),
        )
        inject_lora_adapters(model, rank=8)
        
        groups = get_lora_param_groups(model, lr_A=1e-4, lr_B=2e-4)
        
        assert len(groups) == 2
        assert groups[0]["name"] == "lora_A"
        assert groups[0]["lr"] == 1e-4
        assert groups[1]["name"] == "lora_B"
        assert groups[1]["lr"] == 2e-4
    
    def test_count_lora_parameters(self):
        """Test counting trainable vs total parameters."""
        model = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.Linear(128, 32, bias=False),
        )
        
        # Before LoRA: all params trainable
        trainable_before, total_before = count_lora_parameters(model)
        assert trainable_before == total_before
        
        inject_lora_adapters(model, rank=8)
        
        trainable_after, total_after = count_lora_parameters(model)
        
        # Total should be same (base) + LoRA params
        # Trainable should be much less (only LoRA)
        assert trainable_after < trainable_before
        assert total_after > total_before


class TestGPUSupport:
    """Tests for GPU support (skipped if no GPU)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_lora_on_gpu(self):
        """Test LoRA works on GPU."""
        base = nn.Linear(64, 128).cuda()
        lora = LoRALinear(base, rank=8).cuda()
        
        x = torch.randn(4, 64).cuda()
        out = lora(x)
        
        assert out.device.type == "cuda"
        assert out.shape == (4, 128)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dora_on_gpu(self):
        """Test DoRA works on GPU."""
        base = nn.Linear(64, 128).cuda()
        dora = DoRALinear(base, rank=8).cuda()
        
        x = torch.randn(4, 64).cuda()
        out = dora(x)
        
        assert out.device.type == "cuda"
        assert out.shape == (4, 128)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bfloat16_support(self):
        """Test LoRA supports bfloat16."""
        base = nn.Linear(64, 128).cuda().to(torch.bfloat16)
        lora = LoRALinear(base, rank=8)
        lora.lora_A.data = lora.lora_A.data.cuda().to(torch.bfloat16)
        lora.lora_B.data = lora.lora_B.data.cuda().to(torch.bfloat16)
        
        x = torch.randn(4, 64).cuda().to(torch.bfloat16)
        out = lora(x)
        
        assert out.dtype == torch.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
