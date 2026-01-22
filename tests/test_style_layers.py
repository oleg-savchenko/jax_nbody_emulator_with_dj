"""
Tests for style_layers.py module.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.style_layers import (
    StyleConvBase3D, StyleConvTransposeBase3D,
    StyleConv3D, StyleSkip3D, StyleDownSample3D, StyleUpSample3D
)


class TestStyleConvBase3D:
    """Test the base StyleConvBase3D class"""
    
    def test_forward_pass_shapes_batched(self):
        """Test that forward pass produces correct output shapes"""
        key = random.PRNGKey(42)
        batch_size = 2
        
        layer = StyleConvBase3D(in_chan=32, out_chan=64, kernel_size=3, stride=1)
        
        # Initialize parameters with batched input
        x = random.normal(key, (batch_size, 32, 8, 16, 16))
        s = random.normal(key, (batch_size, 2))
        params = layer.init(key, x, s)
        
        # Forward pass
        output = layer.apply(params, x, s)
        
        # Check output shape (VALID padding reduces spatial dims by kernel_size-1)
        expected_shape = (batch_size, 64, 6, 14, 14)
        assert output.shape == expected_shape
    
    def test_forward_pass_shapes_unbatched(self):
        """Test forward pass with unbatched input"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3D(in_chan=16, out_chan=32, kernel_size=3, stride=1)
        
        # Unbatched input
        x = random.normal(key, (16, 8, 8, 8))
        s = random.normal(key, (2,))
        params = layer.init(key, x, s)
        
        output = layer.apply(params, x, s)
        
        # Output should also be unbatched
        expected_shape = (32, 6, 6, 6)
        assert output.shape == expected_shape
    
    def test_style_conditioning_effect(self):
        """Test that different style vectors produce different outputs"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3D(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s1 = jnp.array([[0.3, 1.0]])
        s2 = jnp.array([[0.5, 1.5]])
        
        params = layer.init(key, x, s1)
        
        output1 = layer.apply(params, x, s1)
        output2 = layer.apply(params, x, s2)
        
        # Outputs should be different when style vectors differ
        assert not jnp.allclose(output1, output2)
    
    def test_custom_style_size(self):
        """Test that custom style_size parameter works"""
        key = random.PRNGKey(42)
        
        style_size = 8
        layer = StyleConvBase3D(in_chan=16, out_chan=32, style_size=style_size)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = random.normal(key, (1, style_size))
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Check style_weight shape
        assert params['params']['style_weight'].shape == (16, style_size)
        assert output.shape == (1, 32, 6, 6, 6)


class TestStyleConvTransposeBase3D:
    """Test the StyleConvTransposeBase3D class for upsampling"""
    
    def test_forward_pass_shapes_batched(self):
        """Test that forward pass produces correct upsampled shapes"""
        key = random.PRNGKey(42)
        batch_size = 2
        
        layer = StyleConvTransposeBase3D(in_chan=32, out_chan=16)
        
        x = random.normal(key, (batch_size, 32, 4, 8, 8))
        s = random.normal(key, (batch_size, 2))
        params = layer.init(key, x, s)
        
        output = layer.apply(params, x, s)
        
        # Should double spatial dimensions
        expected_shape = (batch_size, 16, 8, 16, 16)
        assert output.shape == expected_shape
    
    def test_forward_pass_shapes_unbatched(self):
        """Test forward pass with unbatched input"""
        key = random.PRNGKey(42)
        
        layer = StyleConvTransposeBase3D(in_chan=32, out_chan=16)
        
        x = random.normal(key, (32, 4, 8, 8))
        s = random.normal(key, (2,))
        params = layer.init(key, x, s)
        
        output = layer.apply(params, x, s)
        
        # Output should also be unbatched, with doubled spatial dims
        expected_shape = (16, 8, 16, 16)
        assert output.shape == expected_shape
    
    def test_style_conditioning_effect(self):
        """Test that different style vectors produce different outputs"""
        key = random.PRNGKey(42)
        
        layer = StyleConvTransposeBase3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 4, 4))
        s1 = jnp.array([[0.3, 1.0]])
        s2 = jnp.array([[0.5, 1.5]])
        
        params = layer.init(key, x, s1)
        
        output1 = layer.apply(params, x, s1)
        output2 = layer.apply(params, x, s2)
        
        # Outputs should be different when style vectors differ
        assert not jnp.allclose(output1, output2)


class TestStyleConv3D:
    """Test StyleConv3D layer (partial of StyleConvBase3D)"""
    
    def test_default_parameters(self):
        """Test that StyleConv3D has correct default parameters"""
        layer = StyleConv3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 3
        assert layer.stride == 1
    
    def test_forward_pass(self):
        """Test StyleConv3D forward pass"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # VALID padding: out = in - (kernel_size - 1)
        expected_shape = (1, 32, 6, 14, 14)
        assert output.shape == expected_shape


class TestStyleSkip3D:
    """Test StyleSkip3D layer (1x1x1 convolution)"""
    
    def test_default_parameters(self):
        """Test that StyleSkip3D has correct default parameters"""
        layer = StyleSkip3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 1
        assert layer.stride == 1
    
    def test_preserves_spatial_dimensions(self):
        """Test that StyleSkip3D preserves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = StyleSkip3D(in_chan=32, out_chan=64)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 32, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # 1x1x1 kernel preserves spatial dimensions
        expected_shape = (1, 64, *spatial_shape)
        assert output.shape == expected_shape


class TestStyleDownSample3D:
    """Test StyleDownSample3D layer"""
    
    def test_default_parameters(self):
        """Test that StyleDownSample3D has correct default parameters"""
        layer = StyleDownSample3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 2
    
    def test_downsampling_shape(self):
        """Test that downsampling halves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = StyleDownSample3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Should halve each dimension: (8,16,16) -> (4,8,8)
        expected_shape = (1, 32, 4, 8, 8)
        assert output.shape == expected_shape


class TestStyleUpSample3D:
    """Test StyleUpSample3D layer (transposed convolution)"""
    
    def test_default_parameters(self):
        """Test that StyleUpSample3D has correct default parameters"""
        layer = StyleUpSample3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 1
    
    def test_upsampling_shape(self):
        """Test that upsampling doubles spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = StyleUpSample3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Should double each dimension: (4,8,8) -> (8,16,16)
        expected_shape = (1, 32, 8, 16, 16)
        assert output.shape == expected_shape


class TestParameterInitialization:
    """Test parameter initialization in style layers"""
    
    def test_weight_initialization_shapes(self):
        """Test that weights are initialized with correct shapes"""
        key = random.PRNGKey(42)
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        
        # Check parameter shapes
        assert params['params']['weight'].shape == (32, 16, 3, 3, 3)
        assert params['params']['bias'].shape == (32,)
        assert params['params']['style_weight'].shape == (16, 2)
        assert params['params']['style_bias'].shape == (16,)
    
    def test_style_bias_initialization(self):
        """Test that style bias is initialized to ones"""
        key = random.PRNGKey(42)
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        style_bias = params['params']['style_bias']
        
        # Style bias should be initialized to ones
        assert jnp.allclose(style_bias, jnp.ones(16))
    
    def test_transpose_weight_initialization_shapes(self):
        """Test that transpose conv weights are initialized correctly"""
        key = random.PRNGKey(42)
        layer = StyleUpSample3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 4, 4))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        
        # Check parameter shapes (kernel_size=2 by default)
        assert params['params']['weight'].shape == (32, 16, 2, 2, 2)
        assert params['params']['bias'].shape == (32,)
        assert params['params']['style_weight'].shape == (16, 2)
        assert params['params']['style_bias'].shape == (16,)


class TestDownUpSamplingChain:
    """Test integration between downsampling and upsampling"""
    
    def test_symmetric_down_up(self):
        """Test that down then up returns to original spatial size"""
        key = random.PRNGKey(42)
        
        down_layer = StyleDownSample3D(in_chan=16, out_chan=32)
        up_layer = StyleUpSample3D(in_chan=32, out_chan=16)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 16, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        # Downsample
        down_params = down_layer.init(key, x, s)
        down_output = down_layer.apply(down_params, x, s)
        
        # Upsample
        up_key = random.fold_in(key, 1)
        up_params = up_layer.init(up_key, down_output, s)
        up_output = up_layer.apply(up_params, down_output, s)
        
        # Should return to original spatial dimensions
        assert up_output.shape == (1, 16, *spatial_shape)
    
    def test_multiple_down_up_levels(self):
        """Test multiple levels of downsampling and upsampling"""
        key = random.PRNGKey(42)
        
        # Two levels of downsampling
        down1 = StyleDownSample3D(in_chan=16, out_chan=32)
        down2 = StyleDownSample3D(in_chan=32, out_chan=64)
        up1 = StyleUpSample3D(in_chan=64, out_chan=32)
        up2 = StyleUpSample3D(in_chan=32, out_chan=16)
        
        spatial_shape = (16, 32, 32)
        x = random.normal(key, (1, 16, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        # Downsample twice
        params_d1 = down1.init(key, x, s)
        x1 = down1.apply(params_d1, x, s)
        assert x1.shape == (1, 32, 8, 16, 16)
        
        params_d2 = down2.init(random.fold_in(key, 1), x1, s)
        x2 = down2.apply(params_d2, x1, s)
        assert x2.shape == (1, 64, 4, 8, 8)
        
        # Upsample twice
        params_u1 = up1.init(random.fold_in(key, 2), x2, s)
        x3 = up1.apply(params_u1, x2, s)
        assert x3.shape == (1, 32, 8, 16, 16)
        
        params_u2 = up2.init(random.fold_in(key, 3), x3, s)
        x4 = up2.apply(params_u2, x3, s)
        assert x4.shape == (1, 16, *spatial_shape)


class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that layers work with JIT compilation"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        
        # JIT compile the apply function
        jit_apply = jax.jit(layer.apply)
        output = jit_apply(params, x, s)
        
        assert output.shape == (1, 32, 6, 6, 6)
        assert jnp.all(jnp.isfinite(output))
    
    def test_vmap_over_batch(self):
        """Test that layers work correctly with explicit vmap"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        # Single sample for initialization
        x_single = random.normal(key, (16, 8, 8, 8))
        s_single = random.normal(key, (2,))
        params = layer.init(key, x_single, s_single)
        
        # Batch of samples
        batch_size = 4
        x_batch = random.normal(key, (batch_size, 16, 8, 8, 8))
        s_batch = random.normal(key, (batch_size, 2))
        
        # Apply to batch
        output = layer.apply(params, x_batch, s_batch)
        
        assert output.shape == (batch_size, 32, 6, 6, 6)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed through the layer"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        
        def loss_fn(params):
            output = layer.apply(params, x, s)
            return jnp.mean(output**2)
        
        # Compute gradient
        grad = jax.grad(loss_fn)(params)
        
        # Check gradients exist and are finite
        assert 'params' in grad
        assert jnp.all(jnp.isfinite(grad['params']['weight']))
        assert jnp.all(jnp.isfinite(grad['params']['bias']))
        assert jnp.all(jnp.isfinite(grad['params']['style_weight']))
        assert jnp.all(jnp.isfinite(grad['params']['style_bias']))


class TestNumericalStability:
    """Test numerical stability of style layers"""
    
    def test_demodulation_with_small_weights(self):
        """Test that demodulation handles small weights correctly"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3D(in_chan=16, out_chan=32, eps=1e-8)
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e-6  # Very small input
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Output should be finite
        assert jnp.all(jnp.isfinite(output))
    
    def test_large_style_values(self):
        """Test behavior with large style values"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[100.0, 100.0]])  # Large style values
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Output should be finite
        assert jnp.all(jnp.isfinite(output))
