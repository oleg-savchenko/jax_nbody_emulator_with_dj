"""
Tests for blocks_vel.py module.

High-level blocks composed of layers with manual forward-mode AD for computing
both outputs and their derivatives. These are standard blocks without style
conditioning, used when style parameters have been premodulated into the input.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.blocks_vel import (
    ResampleBlock3DVel, ResNetBlock3DVel
)


class TestResampleBlock3DVel:
    """Test ResampleBlock3DVel"""
    
    def test_returns_tuple(self):
        """Test that forward pass returns (y, dy) tuple"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3DVel(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = block.init(key, x)
        result = block.apply(params, x)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_upsampling_block(self):
        """Test block with upsampling layers"""
        key = random.PRNGKey(42)
        
        # Block: Upsample -> Activation -> Upsample
        block = ResampleBlock3DVel(
            seq='UAU',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        # Two upsamplings: (4,8,8) -> (8,16,16) -> (16,32,32)
        expected_shape = (1, 32, 16, 32, 32)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_downsampling_block(self):
        """Test block with downsampling layers"""
        key = random.PRNGKey(42)
        
        # Block: Downsample -> Activation
        block = ResampleBlock3DVel(
            seq='DA',
            in_chan=32,
            out_chan=64
        )
        
        x = random.normal(key, (1, 32, 8, 16, 16))
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        # One downsampling: (8,16,16) -> (4,8,8)
        expected_shape = (1, 64, 4, 8, 8)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_mixed_resampling(self):
        """Test block with both up and down sampling"""
        key = random.PRNGKey(42)
        
        # Block: Down -> Act -> Up -> Act
        block = ResampleBlock3DVel(
            seq='DAUA',
            in_chan=16,
            out_chan=16
        )
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 16, *spatial_shape))
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        # Down then up: (8,16,16) -> (4,8,8) -> (8,16,16)
        expected_shape = (1, 16, *spatial_shape)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_without_input_tangent(self):
        """Test block without input tangent (dx=None)"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3DVel(
            seq='DA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        
        params = block.init(key, x, None)
        y, dy = block.apply(params, x, None)
        
        expected_shape = (1, 32, 4, 8, 8)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_with_input_tangent(self):
        """Test block with input tangent (dx provided)"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3DVel(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        dx = random.normal(random.fold_in(key, 1), (1, 16, 4, 8, 8))
        
        params = block.init(key, x, dx)
        y, dy = block.apply(params, x, dx)
        
        expected_shape = (1, 32, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_tangent_differs_with_and_without_dx(self):
        """Test that providing dx changes the tangent computation"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3DVel(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        dx = random.normal(random.fold_in(key, 1), (1, 16, 4, 8, 8))
        
        params = block.init(key, x)
        
        # Without dx
        y1, dy1 = block.apply(params, x, None)
        
        # With dx
        y2, dy2 = block.apply(params, x, dx)
        
        # Outputs should be the same
        assert jnp.allclose(y1, y2)
        
        # Tangents should be different
        assert not jnp.allclose(dy1, dy2)
    
    def test_invalid_layer_type(self):
        """Test that invalid layer type raises error during initialization"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3DVel(
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
        
        block_fp16 = ResampleBlock3DVel(
            seq='UA',
            in_chan=16,
            out_chan=32,
            dtype=jnp.float16
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8)).astype(jnp.float16)
        
        params = block_fp16.init(key, x)
        
        first_param = jax.tree_util.tree_leaves(params)[0]
        assert first_param.dtype == jnp.float16
    
    def test_batched_input(self):
        """Test block with batched input"""
        key = random.PRNGKey(42)
        batch_size = 4
        
        block = ResampleBlock3DVel(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (batch_size, 16, 4, 8, 8))
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        expected_shape = (batch_size, 32, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestResNetBlock3DVel:
    """Test ResNetBlock3DVel"""
    
    def test_returns_tuple(self):
        """Test that forward pass returns (y, dy) tuple"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        result = block.apply(params, x)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_basic_resnet_block(self):
        """Test basic ResNet block: Conv -> Act -> Conv"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        # Two 3x3x3 convs crop 2 voxels per side: (8,8,8) -> (4,4,4)
        expected_shape = (1, 32, 4, 4, 4)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_resnet_with_final_activation(self):
        """Test ResNet block with final activation after residual"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CACA',  # Final A applies after residual addition
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_single_conv_resnet(self):
        """Test ResNet block with single convolution"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        # One conv crops 1 voxel per side: (8,8,8) -> (6,6,6)
        expected_shape = (1, 32, 6, 6, 6)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_skip_connection_cropping(self):
        """Test that skip connection is cropped correctly for both y and dy"""
        key = random.PRNGKey(42)
        
        # Three convolutions should crop 3 voxels per side from skip
        block = ResNetBlock3DVel(
            seq='CACAC',
            in_chan=16,
            out_chan=16
        )
        
        x = random.normal(key, (1, 16, 10, 10, 10))
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        # Three convs: (10,10,10) -> (4,4,4)
        expected_shape = (1, 16, 4, 4, 4)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_without_input_tangent(self):
        """Test block without input tangent (dx=None)"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x, None)
        y, dy = block.apply(params, x, None)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_with_input_tangent(self):
        """Test block with input tangent (dx provided)"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        dx = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        
        params = block.init(key, x, dx)
        y, dy = block.apply(params, x, dx)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_tangent_addition_in_residual(self):
        """Test that tangents are added in residual connection"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CA',
            in_chan=16,
            out_chan=16
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        dx = jnp.ones((1, 16, 8, 8, 8))
        
        params = block.init(key, x, dx)
        y, dy = block.apply(params, x, dx)
        
        # Tangent should not be all zeros or all ones (sum of two contributions)
        assert not jnp.allclose(dy, 0.0)
        assert not jnp.allclose(dy, 1.0)
    
    def test_tangent_differs_with_and_without_dx(self):
        """Test that providing dx changes the tangent computation"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        dx = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        
        # Without dx
        y1, dy1 = block.apply(params, x, None)
        
        # With dx
        y2, dy2 = block.apply(params, x, dx)
        
        # Outputs should be the same
        assert jnp.allclose(y1, y2)
        
        # Tangents should be different
        assert not jnp.allclose(dy1, dy2)
    
    def test_final_activation_with_tangent(self):
        """Test final activation is applied to tangents correctly"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CACA',  # Final A after residual
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        dx = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        
        params = block.init(key, x, dx)
        y, dy = block.apply(params, x, dx)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
        
        # Both should be finite
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_invalid_layer_type(self):
        """Test that invalid layer type raises error during initialization"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CXA',  # X is invalid
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        with pytest.raises(ValueError, match='Layer type "X" not supported'):
            params = block.init(key, x)
    
    def test_batched_input(self):
        """Test block with batched input"""
        key = random.PRNGKey(42)
        batch_size = 4
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (batch_size, 16, 8, 8, 8))
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        expected_shape = (batch_size, 32, 4, 4, 4)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestTangentPropagation:
    """Test that tangents propagate correctly through blocks"""
    
    def test_resample_tangent_chain(self):
        """Test tangent propagation through multiple resample operations"""
        key = random.PRNGKey(42)
        
        block1 = ResampleBlock3DVel(
            seq='DA',
            in_chan=16,
            out_chan=32
        )
        
        block2 = ResampleBlock3DVel(
            seq='UA',
            in_chan=32,
            out_chan=16
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        # First block (no initial tangent)
        params1 = block1.init(key, x, None)
        y1, dy1 = block1.apply(params1, x, None)
        
        # Second block (with tangent from first)
        params2 = block2.init(random.fold_in(key, 1), y1, dy1)
        y2, dy2 = block2.apply(params2, y1, dy1)
        
        # Should return to original spatial size
        assert y2.shape == (1, 16, 8, 8, 8)
        assert dy2.shape == (1, 16, 8, 8, 8)
        
        # Both should be finite
        assert jnp.all(jnp.isfinite(y2))
        assert jnp.all(jnp.isfinite(dy2))
    
    def test_resnet_tangent_chain(self):
        """Test tangent propagation through multiple ResNet blocks"""
        key = random.PRNGKey(42)
        
        block1 = ResNetBlock3DVel(
            seq='CA',
            in_chan=16,
            out_chan=32
        )
        
        block2 = ResNetBlock3DVel(
            seq='CA',
            in_chan=32,
            out_chan=64
        )
        
        x = random.normal(key, (1, 16, 10, 10, 10))
        
        # First block
        params1 = block1.init(key, x, None)
        y1, dy1 = block1.apply(params1, x, None)
        
        # Second block
        params2 = block2.init(random.fold_in(key, 1), y1, dy1)
        y2, dy2 = block2.apply(params2, y1, dy1)
        
        # Check shapes: (10,10,10) -> (8,8,8) -> (6,6,6)
        assert y2.shape == (1, 64, 6, 6, 6)
        assert dy2.shape == (1, 64, 6, 6, 6)
    
    def test_mixed_block_tangent_chain(self):
        """Test tangent propagation through mixed block types"""
        key = random.PRNGKey(42)
        
        resnet = ResNetBlock3DVel(
            seq='CA',
            in_chan=16,
            out_chan=32
        )
        
        resample = ResampleBlock3DVel(
            seq='DA',
            in_chan=32,
            out_chan=64
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        # ResNet first (no initial tangent)
        params1 = resnet.init(key, x, None)
        y1, dy1 = resnet.apply(params1, x, None)
        
        # Resample second (with tangent)
        params2 = resample.init(random.fold_in(key, 1), y1, dy1)
        y2, dy2 = resample.apply(params2, y1, dy1)
        
        # (8,8,8) -> (6,6,6) -> (3,3,3)
        assert y2.shape == (1, 64, 3, 3, 3)
        assert dy2.shape == (1, 64, 3, 3, 3)


class TestBlockIntegration:
    """Test integration between different velocity blocks"""
    
    def test_encoder_decoder_pipeline(self):
        """Test typical encoder-decoder architecture with tangents"""
        key = random.PRNGKey(42)
        
        # Encoder: ResNet -> Downsample
        encoder = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        downsample = ResampleBlock3DVel(
            seq='DA',
            in_chan=32,
            out_chan=64
        )
        
        # Decoder: Upsample -> ResNet
        upsample = ResampleBlock3DVel(
            seq='UA',
            in_chan=64,
            out_chan=32
        )
        
        decoder = ResNetBlock3DVel(
            seq='CAC',
            in_chan=32,
            out_chan=16
        )
        
        x = random.normal(key, (1, 16, 16, 16, 16))
        
        # Encode
        enc_params = encoder.init(key, x, None)
        x1, dx1 = encoder.apply(enc_params, x, None)
        
        down_params = downsample.init(random.fold_in(key, 1), x1, dx1)
        x2, dx2 = downsample.apply(down_params, x1, dx1)
        
        # Decode
        up_params = upsample.init(random.fold_in(key, 2), x2, dx2)
        x3, dx3 = upsample.apply(up_params, x2, dx2)
        
        dec_params = decoder.init(random.fold_in(key, 3), x3, dx3)
        x4, dx4 = decoder.apply(dec_params, x3, dx3)
        
        assert x4.shape == (1, 16, 8, 8, 8)
        assert dx4.shape == (1, 16, 8, 8, 8)
        
        # All outputs should be finite
        assert jnp.all(jnp.isfinite(x4))
        assert jnp.all(jnp.isfinite(dx4))
    
    def test_multiple_resnet_blocks(self):
        """Test chaining multiple ResNet blocks with tangents"""
        key = random.PRNGKey(42)
        
        block1 = ResNetBlock3DVel(seq='CA', in_chan=16, out_chan=32)
        block2 = ResNetBlock3DVel(seq='CA', in_chan=32, out_chan=64)
        block3 = ResNetBlock3DVel(seq='CA', in_chan=64, out_chan=32)
        
        x = random.normal(key, (1, 16, 10, 10, 10))
        
        params1 = block1.init(key, x, None)
        y1, dy1 = block1.apply(params1, x, None)
        
        params2 = block2.init(random.fold_in(key, 1), y1, dy1)
        y2, dy2 = block2.apply(params2, y1, dy1)
        
        params3 = block3.init(random.fold_in(key, 2), y2, dy2)
        y3, dy3 = block3.apply(params3, y2, dy2)
        
        assert y3.shape == (1, 32, 4, 4, 4)
        assert dy3.shape == (1, 32, 4, 4, 4)


class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation_resample(self):
        """Test that resample block works with JIT"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3DVel(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = block.init(key, x)
        
        jit_apply = jax.jit(block.apply)
        y, dy = jit_apply(params, x)
        
        assert y.shape == (1, 32, 8, 16, 16)
        assert dy.shape == (1, 32, 8, 16, 16)
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_jit_compilation_resnet(self):
        """Test that resnet block works with JIT"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        
        jit_apply = jax.jit(block.apply)
        y, dy = jit_apply(params, x)
        
        assert y.shape == (1, 32, 4, 4, 4)
        assert dy.shape == (1, 32, 4, 4, 4)
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_gradient_computation_resample(self):
        """Test gradient computation through resample block"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3DVel(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = block.init(key, x)
        
        def loss_fn(params):
            y, dy = block.apply(params, x)
            return jnp.mean(y**2) + jnp.mean(dy**2)
        
        grad = jax.grad(loss_fn)(params)
        
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    
    def test_gradient_computation_resnet(self):
        """Test gradient computation through resnet block"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x)
        
        def loss_fn(params):
            y, dy = block.apply(params, x)
            return jnp.mean(y**2) + jnp.mean(dy**2)
        
        grad = jax.grad(loss_fn)(params)
        
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)


class TestNumericalStability:
    """Test numerical stability of velocity blocks"""
    
    def test_small_input_resample(self):
        """Test resample block with small input values"""
        key = random.PRNGKey(42)
        
        block = ResampleBlock3DVel(
            seq='UA',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8)) * 1e-6
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_small_input_resnet(self):
        """Test resnet block with small input values"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e-6
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_large_input_values(self):
        """Test blocks with large input values"""
        key = random.PRNGKey(42)
        
        block = ResNetBlock3DVel(
            seq='CAC',
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e3
        
        params = block.init(key, x)
        y, dy = block.apply(params, x)
        
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))


class TestEquivalenceWithNonVelBlocks:
    """Test relationship between velocity and non-velocity blocks"""
    
    def test_same_output_shape_resample(self):
        """Test that output shapes match non-velocity counterparts"""
        key = random.PRNGKey(42)
        
        from jax_nbody_emulator.blocks import ResampleBlock3D
        
        vel_block = ResampleBlock3DVel(seq='UA', in_chan=16, out_chan=32)
        std_block = ResampleBlock3D(seq='UA', in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        vel_params = vel_block.init(key, x)
        std_params = std_block.init(key, x)
        
        y_vel, dy_vel = vel_block.apply(vel_params, x)
        y_std = std_block.apply(std_params, x)
        
        # Shapes should match
        assert y_vel.shape == y_std.shape
    
    def test_same_output_shape_resnet(self):
        """Test that output shapes match non-velocity counterparts"""
        key = random.PRNGKey(42)
        
        from jax_nbody_emulator.blocks import ResNetBlock3D
        
        vel_block = ResNetBlock3DVel(seq='CAC', in_chan=16, out_chan=32)
        std_block = ResNetBlock3D(seq='CAC', in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        vel_params = vel_block.init(key, x)
        std_params = std_block.init(key, x)
        
        y_vel, dy_vel = vel_block.apply(vel_params, x)
        y_std = std_block.apply(std_params, x)
        
        # Shapes should match
        assert y_vel.shape == y_std.shape
    
    def test_no_style_parameter(self):
        """Test that blocks don't require style parameter"""
        key = random.PRNGKey(42)
        
        resample = ResampleBlock3DVel(seq='UA', in_chan=16, out_chan=32)
        resnet = ResNetBlock3DVel(seq='CAC', in_chan=16, out_chan=32)
        
        x_resample = random.normal(key, (1, 16, 4, 8, 8))
        x_resnet = random.normal(key, (1, 16, 8, 8, 8))
        
        # Should work without any style parameter
        resample_params = resample.init(key, x_resample)
        resnet_params = resnet.init(key, x_resnet)
        
        y_resample, dy_resample = resample.apply(resample_params, x_resample)
        y_resnet, dy_resnet = resnet.apply(resnet_params, x_resnet)
        
        assert y_resample.shape == (1, 32, 8, 16, 16)
        assert dy_resample.shape == (1, 32, 8, 16, 16)
        assert y_resnet.shape == (1, 32, 4, 4, 4)
        assert dy_resnet.shape == (1, 32, 4, 4, 4)
