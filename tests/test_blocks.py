"""
Tests for blocks.py module.

High-level blocks composed of layers for building neural networks.
These are standard blocks without style conditioning, used when style
parameters have been premodulated into the input.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.blocks import (
    ResampleBlock3D, ResNetBlock3D
)


class TestResampleBlock3D:
    """Test ResampleBlock3D"""
    
    def test_upsampling_block(self):
        """Test block with upsampling layers"""
        key = random.PRNGKey(42)
        
        # Block: Upsample -> Activation -> Upsample
        block = ResampleBlock3D(
            seq='UAU',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Two upsamplings: (4,8,8) -> (8,16,16) -> (16,32,32)
        expected_shape = (1, 32, 16, 32, 32)
        assert output.shape == expected_shape
    
    def test_downsampling_block(self):
        """Test block with downsampling layers"""
        key = random.PRNGKey(42)
        
        # Block: Downsample -> Activation
        block = ResampleBlock3D(
            seq='DA',
            in_chan=32,
            out_chan=64
        )
        
        x = random.normal(key, (1, 32, 8, 16, 16))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # One downsampling: (8,16,16) -> (4,8,8)
        expected_shape = (1, 64, 4, 8, 8)
        assert output.shape == expected_shape
    
    def test_mixed_resampling(self):
        """Test block with both up and down sampling"""
        key = random.PRNGKey(42)
        
        # Block: Down -> Act -> Up -> Act
        block = ResampleBlock3D(
            seq='DAUA',
            in_chan=16,
            out_chan=16
        )
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 16, *spatial_shape))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Down then up: (8,16,16) -> (4,8,8) -> (8,16,16)
        expected_shape = (1, 16, *spatial_shape)
        assert output.shape == expected_shape
    
    def test_single_upsample(self):
        """Test block with single upsample operation"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3D(
            seq='U',
            in_chan=32,
            out_chan=16
        )
        
        x = random.normal(key, (1, 32, 4, 4, 4))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Single upsample: (4,4,4) -> (8,8,8)
        expected_shape = (1, 16, 8, 8, 8)
        assert output.shape == expected_shape
    
    def test_single_downsample(self):
        """Test block with single downsample operation"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3D(
            seq='D',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Single downsample: (8,8,8) -> (4,4,4)
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_multiple_activations(self):
        """Test block with multiple activation layers"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3D(
            seq='UAAUA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 4, 4))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Two upsamplings: (4,4,4) -> (8,8,8) -> (16,16,16)
        expected_shape = (1, 32, 16, 16, 16)
        assert output.shape == expected_shape
    
    def test_invalid_layer_type(self):
        """Test that invalid layer type raises error during initialization"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3D(
            seq='UXA',  # X is invalid
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        with pytest.raises(ValueError, match='Layer type "X" not supported'):
            params = block.init(key, x)
    
    def test_dtype_parameter(self):
        """Test that dtype parameter is respected"""
        key = random.PRNGKey(42)
        
        block_fp16 = ResampleBlock3D(
            seq='UA',
            in_chan=16,
            out_chan=32,
            dtype=jnp.float16
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8)).astype(jnp.float16)
        
        params = block_fp16.init(key, x)
        
        # Check that parameters are in correct dtype
        first_param = jax.tree_util.tree_leaves(params)[0]
        assert first_param.dtype == jnp.float16
    
    def test_batched_input(self):
        """Test block with batched input"""
        key = random.PRNGKey(42)
        batch_size = 4
        
        block = ResampleBlock3D(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (batch_size, 16, 4, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        expected_shape = (batch_size, 32, 8, 16, 16)
        assert output.shape == expected_shape
    
    def test_activation_effect(self):
        """Test that activation affects output"""
        key = random.PRNGKey(42)
        
        block_no_act = ResampleBlock3D(seq='D', in_chan=16, out_chan=32)
        block_with_act = ResampleBlock3D(seq='DA', in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params_no_act = block_no_act.init(key, x)
        params_with_act = block_with_act.init(key, x)
        
        output_no_act = block_no_act.apply(params_no_act, x)
        output_with_act = block_with_act.apply(params_with_act, x)
        
        # Outputs should be different due to activation
        assert not jnp.allclose(output_no_act, output_with_act)


class TestResNetBlock3D:
    """Test ResNetBlock3D"""
    
    def test_basic_resnet_block(self):
        """Test basic ResNet block: Conv -> Act -> Conv"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Two 3x3x3 convs crop 2 voxels per side: (8,8,8) -> (4,4,4)
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_resnet_with_final_activation(self):
        """Test ResNet block with final activation after residual"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CACA',  # Final A applies after residual addition
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_single_conv_resnet(self):
        """Test ResNet block with single convolution"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # One conv crops 1 voxel per side: (8,8,8) -> (6,6,6)
        expected_shape = (1, 32, 6, 6, 6)
        assert output.shape == expected_shape
    
    def test_skip_connection_cropping(self):
        """Test that skip connection is cropped correctly"""
        key = random.PRNGKey(42)
        
        # Three convolutions should crop 3 voxels per side from skip
        block = ResNetBlock3D(
            seq='CACAC',
            in_chan=16,
            out_chan=16  # Same channels for easier testing
        )
        
        x = random.normal(key, (1, 16, 10, 10, 10))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Three convs: (10,10,10) -> (4,4,4)
        # Skip is cropped by 3 per side: (10,10,10) -> (4,4,4)
        expected_shape = (1, 16, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_residual_identity_channels(self):
        """Test that residual connection adds both paths"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CA',
            in_chan=16,
            out_chan=16
        )
        
        # Use non-zero input to test residual addition
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Output should be non-zero and have correct shape
        assert output.shape == (1, 16, 6, 6, 6)
        assert not jnp.allclose(output, 0.0)
        
        # Test that changing input changes output (residual is working)
        x2 = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        output2 = block.apply(params, x2)
        
        # Different inputs should give different outputs
        assert not jnp.allclose(output, output2)
    
    def test_residual_channel_projection(self):
        """Test that skip connection projects channels correctly"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=64  # Different output channels
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Skip should project from 16 to 64 channels
        expected_shape = (1, 64, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_invalid_layer_type(self):
        """Test that invalid layer type raises error during initialization"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CXA',  # X is invalid
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        # Error is raised during init
        with pytest.raises(ValueError, match='Layer type "X" not supported'):
            params = block.init(key, x)
    
    def test_dtype_parameter(self):
        """Test that dtype parameter is respected"""
        key = random.PRNGKey(42)
        
        block_fp16 = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=32,
            dtype=jnp.float16
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8)).astype(jnp.float16)
        
        params = block_fp16.init(key, x)
        
        first_param = jax.tree_util.tree_leaves(params)[0]
        assert first_param.dtype == jnp.float16
    
    def test_batched_input(self):
        """Test block with batched input"""
        key = random.PRNGKey(42)
        batch_size = 4
        
        block = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (batch_size, 16, 8, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        expected_shape = (batch_size, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_deep_resnet_block(self):
        """Test ResNet block with many convolutions"""
        key = random.PRNGKey(42)
        
        # Four convolutions
        block = ResNetBlock3D(
            seq='CACACACA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 12, 12, 12))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Four convs crop 4 voxels per side: (12,12,12) -> (4,4,4)
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape


class TestBlockIntegration:
    """Test integration between different blocks"""
    
    def test_encoder_decoder_pipeline(self):
        """Test typical encoder-decoder architecture"""
        key = random.PRNGKey(42)
        
        # Encoder: ResNet -> Downsample
        encoder = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        downsample = ResampleBlock3D(
            seq='DA',
            in_chan=32,
            out_chan=64
        )
        
        # Decoder: Upsample -> ResNet
        upsample = ResampleBlock3D(
            seq='UA',
            in_chan=64,
            out_chan=32
        )
        
        decoder = ResNetBlock3D(
            seq='CAC',
            in_chan=32,
            out_chan=16
        )
        
        x = random.normal(key, (1, 16, 16, 16, 16))
        
        # Encode
        enc_params = encoder.init(key, x)
        x1 = encoder.apply(enc_params, x)  # (16,16,16) -> (12,12,12)
        
        down_params = downsample.init(random.fold_in(key, 1), x1)
        x2 = downsample.apply(down_params, x1)  # (12,12,12) -> (6,6,6)
        
        # Decode
        up_params = upsample.init(random.fold_in(key, 2), x2)
        x3 = upsample.apply(up_params, x2)  # (6,6,6) -> (12,12,12)
        
        dec_params = decoder.init(random.fold_in(key, 3), x3)
        x4 = decoder.apply(dec_params, x3)  # (12,12,12) -> (8,8,8)
        
        assert x4.shape == (1, 16, 8, 8, 8)
    
    def test_unet_skip_connection_pattern(self):
        """Test U-Net style skip connection pattern"""
        key = random.PRNGKey(42)
        
        # Encoder path
        enc1 = ResNetBlock3D(seq='CA', in_chan=16, out_chan=32)
        down1 = ResampleBlock3D(seq='D', in_chan=32, out_chan=64)
        
        # Decoder path
        up1 = ResampleBlock3D(seq='U', in_chan=64, out_chan=32)
        dec1 = ResNetBlock3D(seq='CA', in_chan=32, out_chan=16)
        
        x = random.normal(key, (1, 16, 10, 10, 10))
        
        # Encode
        enc1_params = enc1.init(key, x)
        e1 = enc1.apply(enc1_params, x)  # (10,10,10) -> (8,8,8)
        
        down1_params = down1.init(random.fold_in(key, 1), e1)
        d1 = down1.apply(down1_params, e1)  # (8,8,8) -> (4,4,4)
        
        # Decode
        up1_params = up1.init(random.fold_in(key, 2), d1)
        u1 = up1.apply(up1_params, d1)  # (4,4,4) -> (8,8,8)
        
        dec1_params = dec1.init(random.fold_in(key, 3), u1)
        output = dec1.apply(dec1_params, u1)  # (8,8,8) -> (6,6,6)
        
        assert output.shape == (1, 16, 6, 6, 6)
    
    def test_multiple_resnet_blocks(self):
        """Test chaining multiple ResNet blocks"""
        key = random.PRNGKey(42)
        
        block1 = ResNetBlock3D(seq='CA', in_chan=16, out_chan=32)
        block2 = ResNetBlock3D(seq='CA', in_chan=32, out_chan=64)
        block3 = ResNetBlock3D(seq='CA', in_chan=64, out_chan=32)
        
        x = random.normal(key, (1, 16, 10, 10, 10))
        
        params1 = block1.init(key, x)
        x1 = block1.apply(params1, x)  # (10,10,10) -> (8,8,8)
        
        params2 = block2.init(random.fold_in(key, 1), x1)
        x2 = block2.apply(params2, x1)  # (8,8,8) -> (6,6,6)
        
        params3 = block3.init(random.fold_in(key, 2), x2)
        x3 = block3.apply(params3, x2)  # (6,6,6) -> (4,4,4)
        
        assert x3.shape == (1, 32, 4, 4, 4)


class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation_resample(self):
        """Test that resample block works with JIT"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3D(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = block.init(key, x)
        
        jit_apply = jax.jit(block.apply)
        output = jit_apply(params, x)
        
        assert output.shape == (1, 32, 8, 16, 16)
        assert jnp.all(jnp.isfinite(output))
    
    def test_jit_compilation_resnet(self):
        """Test that resnet block works with JIT"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        
        jit_apply = jax.jit(block.apply)
        output = jit_apply(params, x)
        
        assert output.shape == (1, 32, 4, 4, 4)
        assert jnp.all(jnp.isfinite(output))
    
    def test_gradient_computation_resample(self):
        """Test gradient computation through resample block"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3D(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = block.init(key, x)
        
        def loss_fn(params):
            output = block.apply(params, x)
            return jnp.mean(output**2)
        
        grad = jax.grad(loss_fn)(params)
        
        # Check gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    
    def test_gradient_computation_resnet(self):
        """Test gradient computation through resnet block"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        
        def loss_fn(params):
            output = block.apply(params, x)
            return jnp.mean(output**2)
        
        grad = jax.grad(loss_fn)(params)
        
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)


class TestNumericalStability:
    """Test numerical stability of blocks"""
    
    def test_small_input_resample(self):
        """Test resample block with small input values"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3D(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8)) * 1e-6
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_small_input_resnet(self):
        """Test resnet block with small input values"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e-6
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_large_input_values(self):
        """Test blocks with large input values"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e3
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        assert jnp.all(jnp.isfinite(output))


class TestEquivalenceWithStyleBlocks:
    """Test relationship between premodulated blocks and style blocks"""
    
    def test_same_output_shape_resample(self):
        """Test that output shapes match style block counterparts"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3D(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Should produce same shape as StyleResampleBlock3D
        expected_shape = (1, 32, 8, 16, 16)
        assert output.shape == expected_shape
    
    def test_same_output_shape_resnet(self):
        """Test that output shapes match style block counterparts"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3D(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        output = block.apply(params, x)
        
        # Should produce same shape as StyleResNetBlock3D
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_no_style_parameter(self):
        """Test that blocks don't require style parameter"""
        key = random.PRNGKey(42)
        
        resample = ResampleBlock3D(seq='UA', in_chan=16, out_chan=32)
        resnet = ResNetBlock3D(seq='CAC', in_chan=16, out_chan=32)
        
        x_resample = random.normal(key, (1, 16, 4, 8, 8))
        x_resnet = random.normal(key, (1, 16, 8, 8, 8))
        
        # Should work without any style parameter
        resample_params = resample.init(key, x_resample)
        resnet_params = resnet.init(key, x_resnet)
        
        resample_output = resample.apply(resample_params, x_resample)
        resnet_output = resnet.apply(resnet_params, x_resnet)
        
        assert resample_output.shape == (1, 32, 8, 16, 16)
        assert resnet_output.shape == (1, 32, 4, 4, 4)
