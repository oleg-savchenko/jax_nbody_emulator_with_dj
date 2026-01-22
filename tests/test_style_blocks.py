"""
Tests for style_blocks.py module.

High-level blocks composed of styled layers for building neural networks.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.style_blocks import (
    StyleResampleBlock3D, StyleResNetBlock3D
)


class TestStyleResampleBlock3D:
    """Test StyleResampleBlock3D"""
    
    def test_upsampling_block(self):
        """Test block with upsampling layers"""
        key = random.PRNGKey(42)
        
        # Block: Upsample -> Activation -> Upsample
        block = StyleResampleBlock3D(
            seq='UAU',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Two upsamplings: (4,8,8) -> (8,16,16) -> (16,32,32)
        expected_shape = (1, 32, 16, 32, 32)
        assert output.shape == expected_shape
    
    def test_downsampling_block(self):
        """Test block with downsampling layers"""
        key = random.PRNGKey(42)
        
        # Block: Downsample -> Activation
        block = StyleResampleBlock3D(
            seq='DA',
            style_size=2,
            in_chan=32,
            out_chan=64
        )
        
        x = random.normal(key, (1, 32, 8, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # One downsampling: (8,16,16) -> (4,8,8)
        expected_shape = (1, 64, 4, 8, 8)
        assert output.shape == expected_shape
    
    def test_mixed_resampling(self):
        """Test block with both up and down sampling"""
        key = random.PRNGKey(42)
        
        # Block: Down -> Act -> Up -> Act
        block = StyleResampleBlock3D(
            seq='DAUA',
            style_size=2,
            in_chan=16,
            out_chan=16
        )
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 16, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Down then up: (8,16,16) -> (4,8,8) -> (8,16,16)
        expected_shape = (1, 16, *spatial_shape)
        assert output.shape == expected_shape
    
    def test_single_upsample(self):
        """Test block with single upsample operation"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3D(
            seq='U',
            style_size=2,
            in_chan=32,
            out_chan=16
        )
        
        x = random.normal(key, (1, 32, 4, 4, 4))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Single upsample: (4,4,4) -> (8,8,8)
        expected_shape = (1, 16, 8, 8, 8)
        assert output.shape == expected_shape
    
    def test_single_downsample(self):
        """Test block with single downsample operation"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3D(
            seq='D',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Single downsample: (8,8,8) -> (4,4,4)
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_multiple_activations(self):
        """Test block with multiple activation layers"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3D(
            seq='UAAUA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 4, 4))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Two upsamplings: (4,4,4) -> (8,8,8) -> (16,16,16)
        expected_shape = (1, 32, 16, 16, 16)
        assert output.shape == expected_shape
    
    def test_invalid_layer_type(self):
        """Test that invalid layer type raises error during initialization"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3D(
            seq='UXA',  # X is invalid
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        with pytest.raises(ValueError, match='Layer type "X" not supported'):
            params = block.init(key, x, s)
    
    def test_style_conditioning_effect(self):
        """Test that different style vectors produce different outputs"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3D(
            seq='UA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s1 = jnp.array([[0.3, 1.0]])
        s2 = jnp.array([[0.5, 1.5]])
        
        params = block.init(key, x, s1)
        
        output1 = block.apply(params, x, s1)
        output2 = block.apply(params, x, s2)
        
        assert not jnp.allclose(output1, output2)
    
    def test_custom_style_size(self):
        """Test block with custom style size"""
        key = random.PRNGKey(42)
        
        style_size = 8
        block = StyleResampleBlock3D(
            seq='DA',
            style_size=style_size,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = random.normal(key, (1, style_size))
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_batched_input(self):
        """Test block with batched input"""
        key = random.PRNGKey(42)
        batch_size = 4
        
        block = StyleResampleBlock3D(
            seq='UA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (batch_size, 16, 4, 8, 8))
        s = random.normal(key, (batch_size, 2))
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        expected_shape = (batch_size, 32, 8, 16, 16)
        assert output.shape == expected_shape


class TestStyleResNetBlock3D:
    """Test StyleResNetBlock3D"""
    
    def test_basic_resnet_block(self):
        """Test basic ResNet block: Conv -> Act -> Conv"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Two 3x3x3 convs crop 2 voxels per side: (8,8,8) -> (4,4,4)
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_resnet_with_final_activation(self):
        """Test ResNet block with final activation after residual"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CACA',  # Final A applies after residual addition
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_single_conv_resnet(self):
        """Test ResNet block with single convolution"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # One conv crops 1 voxel per side: (8,8,8) -> (6,6,6)
        expected_shape = (1, 32, 6, 6, 6)
        assert output.shape == expected_shape
    
    def test_skip_connection_cropping(self):
        """Test that skip connection is cropped correctly"""
        key = random.PRNGKey(42)
        
        # Three convolutions should crop 3 voxels per side from skip
        block = StyleResNetBlock3D(
            seq='CACAC',
            style_size=2,
            in_chan=16,
            out_chan=16  # Same channels for easier testing
        )
        
        x = random.normal(key, (1, 16, 10, 10, 10))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Three convs: (10,10,10) -> (4,4,4)
        # Skip is cropped by 3 per side: (10,10,10) -> (4,4,4)
        expected_shape = (1, 16, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_residual_identity_channels(self):
        """Test that residual connection adds both paths"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CA',
            style_size=2,
            in_chan=16,
            out_chan=16
        )
        
        # Use non-zero input to test residual addition
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Output should be non-zero and have correct shape
        assert output.shape == (1, 16, 6, 6, 6)
        assert not jnp.allclose(output, 0.0)
        
        # Test that changing input changes output (residual is working)
        x2 = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        output2 = block.apply(params, x2, s)
        
        # Different inputs should give different outputs
        assert not jnp.allclose(output, output2)
    
    def test_residual_channel_projection(self):
        """Test that skip connection projects channels correctly"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=64  # Different output channels
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Skip should project from 16 to 64 channels
        expected_shape = (1, 64, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_invalid_layer_type(self):
        """Test that invalid layer type raises error during initialization"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CXA',  # X is invalid
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        # Error is raised during init
        with pytest.raises(ValueError, match='Layer type "X" not supported'):
            params = block.init(key, x, s)
    
    def test_style_conditioning_effect(self):
        """Test that different style vectors produce different outputs"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s1 = jnp.array([[0.3, 1.0]])
        s2 = jnp.array([[0.5, 1.5]])
        
        params = block.init(key, x, s1)
        
        output1 = block.apply(params, x, s1)
        output2 = block.apply(params, x, s2)
        
        assert not jnp.allclose(output1, output2)
    
    def test_batched_input(self):
        """Test block with batched input"""
        key = random.PRNGKey(42)
        batch_size = 4
        
        block = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (batch_size, 16, 8, 8, 8))
        s = random.normal(key, (batch_size, 2))
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        expected_shape = (batch_size, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_deep_resnet_block(self):
        """Test ResNet block with many convolutions"""
        key = random.PRNGKey(42)
        
        # Five convolutions
        block = StyleResNetBlock3D(
            seq='CACACACA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 12, 12, 12))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Four convs crop 4 voxels per side: (12,12,12) -> (4,4,4)
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape


class TestBlockIntegration:
    """Test integration between different blocks"""
    
    def test_encoder_decoder_pipeline(self):
        """Test typical encoder-decoder architecture"""
        key = random.PRNGKey(42)
        
        # Encoder: ResNet -> Downsample
        encoder = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        downsample = StyleResampleBlock3D(
            seq='DA',
            style_size=2,
            in_chan=32,
            out_chan=64
        )
        
        # Decoder: Upsample -> ResNet
        upsample = StyleResampleBlock3D(
            seq='UA',
            style_size=2,
            in_chan=64,
            out_chan=32
        )
        
        decoder = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=32,
            out_chan=16
        )
        
        x = random.normal(key, (1, 16, 16, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        # Encode
        enc_params = encoder.init(key, x, s)
        x1 = encoder.apply(enc_params, x, s)  # (16,16,16) -> (12,12,12)
        
        down_params = downsample.init(random.fold_in(key, 1), x1, s)
        x2 = downsample.apply(down_params, x1, s)  # (12,12,12) -> (6,6,6)
        
        # Decode
        up_params = upsample.init(random.fold_in(key, 2), x2, s)
        x3 = upsample.apply(up_params, x2, s)  # (6,6,6) -> (12,12,12)
        
        dec_params = decoder.init(random.fold_in(key, 3), x3, s)
        x4 = decoder.apply(dec_params, x3, s)  # (12,12,12) -> (8,8,8)
        
        assert x4.shape == (1, 16, 8, 8, 8)
    
    def test_unet_skip_connection_pattern(self):
        """Test U-Net style skip connection pattern"""
        key = random.PRNGKey(42)
        
        # Encoder path
        enc1 = StyleResNetBlock3D(seq='CA', style_size=2, in_chan=16, out_chan=32)
        down1 = StyleResampleBlock3D(seq='D', style_size=2, in_chan=32, out_chan=64)
        
        # Decoder path
        up1 = StyleResampleBlock3D(seq='U', style_size=2, in_chan=64, out_chan=32)
        dec1 = StyleResNetBlock3D(seq='CA', style_size=2, in_chan=32, out_chan=16)
        
        x = random.normal(key, (1, 16, 10, 10, 10))
        s = jnp.array([[0.3, 1.0]])
        
        # Encode
        enc1_params = enc1.init(key, x, s)
        e1 = enc1.apply(enc1_params, x, s)  # (10,10,10) -> (8,8,8)
        
        down1_params = down1.init(random.fold_in(key, 1), e1, s)
        d1 = down1.apply(down1_params, e1, s)  # (8,8,8) -> (4,4,4)
        
        # Decode
        up1_params = up1.init(random.fold_in(key, 2), d1, s)
        u1 = up1.apply(up1_params, d1, s)  # (4,4,4) -> (8,8,8)
        
        dec1_params = dec1.init(random.fold_in(key, 3), u1, s)
        output = dec1.apply(dec1_params, u1, s)  # (8,8,8) -> (6,6,6)
        
        assert output.shape == (1, 16, 6, 6, 6)


class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation_resample(self):
        """Test that resample block works with JIT"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3D(
            seq='UA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        
        jit_apply = jax.jit(block.apply)
        output = jit_apply(params, x, s)
        
        assert output.shape == (1, 32, 8, 16, 16)
        assert jnp.all(jnp.isfinite(output))
    
    def test_jit_compilation_resnet(self):
        """Test that resnet block works with JIT"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        
        jit_apply = jax.jit(block.apply)
        output = jit_apply(params, x, s)
        
        assert output.shape == (1, 32, 4, 4, 4)
        assert jnp.all(jnp.isfinite(output))
    
    def test_gradient_computation_resample(self):
        """Test gradient computation through resample block"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3D(
            seq='UA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        
        def loss_fn(params):
            output = block.apply(params, x, s)
            return jnp.mean(output**2)
        
        grad = jax.grad(loss_fn)(params)
        
        # Check gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    
    def test_gradient_computation_resnet(self):
        """Test gradient computation through resnet block"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        
        def loss_fn(params):
            output = block.apply(params, x, s)
            return jnp.mean(output**2)
        
        grad = jax.grad(loss_fn)(params)
        
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)


class TestNumericalStability:
    """Test numerical stability of blocks"""
    
    def test_small_input_resample(self):
        """Test resample block with small input values"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3D(
            seq='UA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8)) * 1e-6
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_small_input_resnet(self):
        """Test resnet block with small input values"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e-6
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_large_style_values(self):
        """Test blocks with large style values"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[100.0, 100.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        assert jnp.all(jnp.isfinite(output))
