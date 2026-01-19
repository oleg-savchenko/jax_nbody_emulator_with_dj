"""
Tests for layers_vel.py module.

These are 3D convolutional layers with manual forward-mode AD for computing
both outputs and their derivatives w.r.t. the input parameter (growth factor).
Used when style parameters have been premodulated into the input.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.layers_vel import (
    ConvBase3DVel, ConvTransposeBase3DVel, LeakyReLUVel,
    Conv3DVel, Skip3DVel, DownSample3DVel, UpSample3DVel
)


class TestConvBase3DVel:
    """Test the base ConvBase3DVel class"""
    
    def test_forward_pass_returns_tuple(self):
        """Test that forward pass returns (y, dy) tuple"""
        key = random.PRNGKey(42)
        
        layer = ConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        result = layer.apply(params, x)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_forward_pass_shapes_batched(self):
        """Test that forward pass produces correct output shapes"""
        key = random.PRNGKey(42)
        batch_size = 2
        
        layer = ConvBase3DVel(in_chan=32, out_chan=64, kernel_size=3, stride=1)
        
        x = random.normal(key, (batch_size, 32, 8, 16, 16))
        params = layer.init(key, x)
        
        y, dy = layer.apply(params, x)
        
        # Check output shapes (VALID padding reduces spatial dims by kernel_size-1)
        expected_shape = (batch_size, 64, 6, 14, 14)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_forward_pass_single_batch(self):
        """Test forward pass with batch_size=1"""
        key = random.PRNGKey(42)
        
        layer = ConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3, stride=1)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        params = layer.init(key, x)
        
        y, dy = layer.apply(params, x)
        
        expected_shape = (1, 32, 6, 6, 6)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_without_input_tangent(self):
        """Test layer without input tangent (dx=None)"""
        key = random.PRNGKey(42)
        
        layer = ConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x, None)
        y, dy = layer.apply(params, x, None)
        
        expected_shape = (1, 32, 6, 6, 6)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_with_input_tangent(self):
        """Test layer with input tangent (dx provided)"""
        key = random.PRNGKey(42)
        
        layer = ConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        dx = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        
        params = layer.init(key, x, dx)
        y, dy = layer.apply(params, x, dx)
        
        expected_shape = (1, 32, 6, 6, 6)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_tangent_differs_with_and_without_dx(self):
        """Test that providing dx changes the tangent computation"""
        key = random.PRNGKey(42)
        
        layer = ConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        dx = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        
        # Without dx
        y1, dy1 = layer.apply(params, x, None)
        
        # With dx
        y2, dy2 = layer.apply(params, x, dx)
        
        # Outputs should be the same
        assert jnp.allclose(y1, y2)
        
        # Tangents should be different
        assert not jnp.allclose(dy1, dy2)
    
    def test_different_kernel_sizes(self):
        """Test with different kernel sizes"""
        key = random.PRNGKey(42)
        
        for kernel_size in [1, 2, 3, 4]:
            layer = ConvBase3DVel(in_chan=16, out_chan=32, kernel_size=kernel_size)
            
            x = random.normal(key, (1, 16, 10, 10, 10))
            params = layer.init(key, x)
            
            y, dy = layer.apply(params, x)
            
            # VALID padding: out = in - (kernel_size - 1)
            expected_spatial = 10 - (kernel_size - 1)
            expected_shape = (1, 32, expected_spatial, expected_spatial, expected_spatial)
            assert y.shape == expected_shape
            assert dy.shape == expected_shape
    
    def test_stride_parameter(self):
        """Test that stride parameter works correctly"""
        key = random.PRNGKey(42)
        
        layer = ConvBase3DVel(in_chan=16, out_chan=32, kernel_size=2, stride=2)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        params = layer.init(key, x)
        
        y, dy = layer.apply(params, x)
        
        # With kernel=2, stride=2: out = (in - kernel) // stride + 1
        expected_shape = (1, 32, 4, 8, 8)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_dtype_parameter(self):
        """Test that dtype parameter works correctly"""
        key = random.PRNGKey(42)
        
        layer_fp32 = ConvBase3DVel(in_chan=16, out_chan=32, dtype=jnp.float32)
        layer_fp16 = ConvBase3DVel(in_chan=16, out_chan=32, dtype=jnp.float16)
        
        x_fp32 = random.normal(key, (1, 16, 8, 8, 8))
        
        params_fp32 = layer_fp32.init(key, x_fp32)
        params_fp16 = layer_fp16.init(key, x_fp32.astype(jnp.float16))
        
        # Check parameter dtypes
        assert params_fp32['params']['weight'].dtype == jnp.float32
        assert params_fp16['params']['weight'].dtype == jnp.float16


class TestConvTransposeBase3DVel:
    """Test the ConvTransposeBase3DVel class for upsampling"""
    
    def test_forward_pass_returns_tuple(self):
        """Test that forward pass returns (y, dy) tuple"""
        key = random.PRNGKey(42)
        
        layer = ConvTransposeBase3DVel(in_chan=32, out_chan=16)
        
        x = random.normal(key, (1, 32, 4, 8, 8))
        
        params = layer.init(key, x)
        result = layer.apply(params, x)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_forward_pass_shapes_batched(self):
        """Test that forward pass produces correct upsampled shapes"""
        key = random.PRNGKey(42)
        batch_size = 2
        
        layer = ConvTransposeBase3DVel(in_chan=32, out_chan=16)
        
        x = random.normal(key, (batch_size, 32, 4, 8, 8))
        params = layer.init(key, x)
        
        y, dy = layer.apply(params, x)
        
        # Should double spatial dimensions
        expected_shape = (batch_size, 16, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_forward_pass_single_batch(self):
        """Test forward pass with batch_size=1"""
        key = random.PRNGKey(42)
        
        layer = ConvTransposeBase3DVel(in_chan=32, out_chan=16)
        
        x = random.normal(key, (1, 32, 4, 8, 8))
        params = layer.init(key, x)
        
        y, dy = layer.apply(params, x)
        
        expected_shape = (1, 16, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_with_input_tangent(self):
        """Test layer with input tangent (dx provided)"""
        key = random.PRNGKey(42)
        
        layer = ConvTransposeBase3DVel(in_chan=32, out_chan=16)
        
        x = random.normal(key, (1, 32, 4, 8, 8))
        dx = random.normal(random.fold_in(key, 1), (1, 32, 4, 8, 8))
        
        params = layer.init(key, x, dx)
        y, dy = layer.apply(params, x, dx)
        
        expected_shape = (1, 16, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_upsampling_factor(self):
        """Test that upsampling doubles spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = ConvTransposeBase3DVel(in_chan=16, out_chan=32)
        
        spatial_sizes = [(4, 4, 4), (4, 8, 8), (8, 16, 16)]
        
        for d, h, w in spatial_sizes:
            x = random.normal(key, (1, 16, d, h, w))
            params = layer.init(key, x)
            
            y, dy = layer.apply(params, x)
            
            # Should double each dimension
            expected_shape = (1, 32, 2*d, 2*h, 2*w)
            assert y.shape == expected_shape
            assert dy.shape == expected_shape


class TestLeakyReLUVel:
    """Test LeakyReLUVel activation function with tangent computation"""
    
    def test_forward_pass_returns_tuple(self):
        """Test that forward pass returns (y, dy) tuple"""
        key = random.PRNGKey(42)
        layer = LeakyReLUVel()
        
        x = jnp.array([[-1.0, 0.0, 1.0]])
        dx = jnp.ones_like(x)
        
        params = layer.init(key, x, dx)
        result = layer.apply(params, x, dx)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_default_negative_slope(self):
        """Test that default negative slope is 0.01"""
        layer = LeakyReLUVel()
        assert layer.negative_slope == 0.01
    
    def test_output_values(self):
        """Test forward pass output values"""
        key = random.PRNGKey(42)
        layer = LeakyReLUVel()
        
        x = jnp.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        dx = jnp.ones_like(x)
        
        params = layer.init(key, x, dx)
        y, dy = layer.apply(params, x, dx)
        
        # Test leaky ReLU output
        expected_y = jnp.array([[-0.02, -0.01, 0.0, 1.0, 2.0]])
        assert jnp.allclose(y, expected_y)
    
    def test_tangent_computation(self):
        """Test that tangent is computed correctly"""
        key = random.PRNGKey(42)
        layer = LeakyReLUVel(negative_slope=0.1)
        
        x = jnp.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        dx = jnp.ones_like(x)
        
        params = layer.init(key, x, dx)
        y, dy = layer.apply(params, x, dx)
        
        # Check output
        expected_y = jnp.array([[-0.2, -0.1, 0.0, 1.0, 2.0]])
        assert jnp.allclose(y, expected_y)
        
        # Check tangent (slope for negative, 1.0 for positive)
        # At x=0, condition x > 0 is False, so uses slope
        expected_dy = jnp.array([[0.1, 0.1, 0.1, 1.0, 1.0]])
        assert jnp.allclose(dy, expected_dy)
    
    def test_tangent_scaling(self):
        """Test that tangent is scaled correctly by dx"""
        key = random.PRNGKey(42)
        layer = LeakyReLUVel(negative_slope=0.1)
        
        x = jnp.array([[-1.0, 1.0]])
        dx = jnp.array([[2.0, 3.0]])
        
        params = layer.init(key, x, dx)
        y, dy = layer.apply(params, x, dx)
        
        # For negative x: dy = slope * dx = 0.1 * 2.0 = 0.2
        # For positive x: dy = dx = 3.0
        expected_dy = jnp.array([[0.2, 3.0]])
        assert jnp.allclose(dy, expected_dy)
    
    def test_custom_negative_slope(self):
        """Test with custom negative slope"""
        key = random.PRNGKey(42)
        layer = LeakyReLUVel(negative_slope=0.2)
        
        x = jnp.array([[-1.0, 1.0]])
        dx = jnp.ones_like(x)
        
        params = layer.init(key, x, dx)
        y, dy = layer.apply(params, x, dx)
        
        expected_y = jnp.array([[-0.2, 1.0]])
        expected_dy = jnp.array([[0.2, 1.0]])
        
        assert jnp.allclose(y, expected_y)
        assert jnp.allclose(dy, expected_dy)
    
    def test_dtype_parameter(self):
        """Test that dtype parameter works correctly"""
        key = random.PRNGKey(42)
        
        layer_fp32 = LeakyReLUVel(dtype=jnp.float32)
        layer_fp16 = LeakyReLUVel(dtype=jnp.float16)
        
        x_fp32 = jnp.array([[-1.0, 0.0, 1.0]])
        dx_fp32 = jnp.ones_like(x_fp32)
        
        x_fp16 = x_fp32.astype(jnp.float16)
        dx_fp16 = dx_fp32.astype(jnp.float16)
        
        params_fp32 = layer_fp32.init(key, x_fp32, dx_fp32)
        params_fp16 = layer_fp16.init(key, x_fp16, dx_fp16)
        
        y_fp32, dy_fp32 = layer_fp32.apply(params_fp32, x_fp32, dx_fp32)
        y_fp16, dy_fp16 = layer_fp16.apply(params_fp16, x_fp16, dx_fp16)
        
        assert y_fp32.dtype == jnp.float32
        assert y_fp16.dtype == jnp.float16


class TestConv3DVel:
    """Test Conv3DVel layer (partial of ConvBase3DVel)"""
    
    def test_default_parameters(self):
        """Test that Conv3DVel has correct default parameters"""
        layer = Conv3DVel(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 3
        assert layer.stride == 1
    
    def test_forward_pass(self):
        """Test Conv3DVel forward pass"""
        key = random.PRNGKey(42)
        
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        
        params = layer.init(key, x)
        y, dy = layer.apply(params, x)
        
        # VALID padding: out = in - (kernel_size - 1)
        expected_shape = (1, 32, 6, 14, 14)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestSkip3DVel:
    """Test Skip3DVel layer (1x1x1 convolution)"""
    
    def test_default_parameters(self):
        """Test that Skip3DVel has correct default parameters"""
        layer = Skip3DVel(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 1
        assert layer.stride == 1
    
    def test_preserves_spatial_dimensions(self):
        """Test that Skip3DVel preserves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = Skip3DVel(in_chan=32, out_chan=64)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 32, *spatial_shape))
        
        params = layer.init(key, x)
        y, dy = layer.apply(params, x)
        
        # 1x1x1 kernel preserves spatial dimensions
        expected_shape = (1, 64, *spatial_shape)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestDownSample3DVel:
    """Test DownSample3DVel layer"""
    
    def test_default_parameters(self):
        """Test that DownSample3DVel has correct default parameters"""
        layer = DownSample3DVel(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 2
    
    def test_downsampling_shape(self):
        """Test that downsampling halves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = DownSample3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        
        params = layer.init(key, x)
        y, dy = layer.apply(params, x)
        
        # Should halve each dimension: (8,16,16) -> (4,8,8)
        expected_shape = (1, 32, 4, 8, 8)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestUpSample3DVel:
    """Test UpSample3DVel layer (transposed convolution)"""
    
    def test_default_parameters(self):
        """Test that UpSample3DVel has correct default parameters"""
        layer = UpSample3DVel(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 1
    
    def test_upsampling_shape(self):
        """Test that upsampling doubles spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = UpSample3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = layer.init(key, x)
        y, dy = layer.apply(params, x)
        
        # Should double each dimension: (4,8,8) -> (8,16,16)
        expected_shape = (1, 32, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestTangentPropagation:
    """Test that tangents propagate correctly through layers"""
    
    def test_tangent_propagation_chain(self):
        """Test tangent propagation through multiple layers"""
        key = random.PRNGKey(42)
        
        layer1 = ConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        layer2 = ConvBase3DVel(in_chan=32, out_chan=64, kernel_size=3)
        
        x = random.normal(key, (1, 16, 12, 12, 12))
        
        params1 = layer1.init(key, x, None)
        params2_key = random.fold_in(key, 1)
        
        # First layer (no input tangent)
        y1, dy1 = layer1.apply(params1, x, None)
        
        # Second layer (with tangent from first)
        params2 = layer2.init(params2_key, y1, dy1)
        y2, dy2 = layer2.apply(params2, y1, dy1)
        
        # Check shapes are consistent
        assert y2.shape[1] == 64  # out_chan
        assert dy2.shape == y2.shape
        
        # Both outputs should be finite
        assert jnp.all(jnp.isfinite(y2))
        assert jnp.all(jnp.isfinite(dy2))
    
    def test_conv_activation_chain(self):
        """Test tangent propagation through conv + activation"""
        key = random.PRNGKey(42)
        
        conv = Conv3DVel(in_chan=16, out_chan=32)
        activation = LeakyReLUVel()
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        conv_params = conv.init(key, x)
        y1, dy1 = conv.apply(conv_params, x)
        
        act_params = activation.init(key, y1, dy1)
        y2, dy2 = activation.apply(act_params, y1, dy1)
        
        # Shape should be preserved through activation
        assert y2.shape == y1.shape
        assert dy2.shape == dy1.shape
        
        # Outputs should be finite
        assert jnp.all(jnp.isfinite(y2))
        assert jnp.all(jnp.isfinite(dy2))


class TestParameterInitialization:
    """Test parameter initialization in velocity layers"""
    
    def test_conv_weight_initialization_shapes(self):
        """Test that conv weights are initialized with correct shapes"""
        key = random.PRNGKey(42)
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        
        # Check parameter shapes (includes dweight for tangent)
        assert params['params']['weight'].shape == (32, 16, 3, 3, 3)
        assert params['params']['bias'].shape == (32,)
        assert params['params']['dweight'].shape == (32, 16, 3, 3, 3)
    
    def test_transpose_weight_initialization_shapes(self):
        """Test that transpose conv weights are initialized correctly"""
        key = random.PRNGKey(42)
        layer = UpSample3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 4, 4))
        
        params = layer.init(key, x)
        
        # Check parameter shapes (kernel_size=2 by default)
        assert params['params']['weight'].shape == (32, 16, 2, 2, 2)
        assert params['params']['bias'].shape == (32,)
        assert params['params']['dweight'].shape == (32, 16, 2, 2, 2)
    
    def test_bias_initialization_zeros(self):
        """Test that biases are initialized to zeros"""
        key = random.PRNGKey(42)
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        bias = params['params']['bias']
        
        assert jnp.allclose(bias, jnp.zeros(32))


class TestDownUpSamplingChainVel:
    """Test integration between downsampling and upsampling with tangents"""
    
    def test_symmetric_down_up(self):
        """Test that down then up returns to original spatial size"""
        key = random.PRNGKey(42)
        
        down_layer = DownSample3DVel(in_chan=16, out_chan=32)
        up_layer = UpSample3DVel(in_chan=32, out_chan=16)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 16, *spatial_shape))
        
        # Downsample
        down_params = down_layer.init(key, x)
        down_y, down_dy = down_layer.apply(down_params, x)
        
        # Upsample (passing tangent through)
        up_key = random.fold_in(key, 1)
        up_params = up_layer.init(up_key, down_y, down_dy)
        up_y, up_dy = up_layer.apply(up_params, down_y, down_dy)
        
        # Should return to original spatial dimensions
        assert up_y.shape == (1, 16, *spatial_shape)
        assert up_dy.shape == (1, 16, *spatial_shape)
    
    def test_multiple_down_up_levels(self):
        """Test multiple levels of downsampling and upsampling"""
        key = random.PRNGKey(42)
        
        # Two levels of downsampling
        down1 = DownSample3DVel(in_chan=16, out_chan=32)
        down2 = DownSample3DVel(in_chan=32, out_chan=64)
        up1 = UpSample3DVel(in_chan=64, out_chan=32)
        up2 = UpSample3DVel(in_chan=32, out_chan=16)
        
        spatial_shape = (16, 32, 32)
        x = random.normal(key, (1, 16, *spatial_shape))
        
        # Downsample twice
        params_d1 = down1.init(key, x)
        y1, dy1 = down1.apply(params_d1, x)
        assert y1.shape == (1, 32, 8, 16, 16)
        
        params_d2 = down2.init(random.fold_in(key, 1), y1, dy1)
        y2, dy2 = down2.apply(params_d2, y1, dy1)
        assert y2.shape == (1, 64, 4, 8, 8)
        
        # Upsample twice
        params_u1 = up1.init(random.fold_in(key, 2), y2, dy2)
        y3, dy3 = up1.apply(params_u1, y2, dy2)
        assert y3.shape == (1, 32, 8, 16, 16)
        
        params_u2 = up2.init(random.fold_in(key, 3), y3, dy3)
        y4, dy4 = up2.apply(params_u2, y3, dy3)
        assert y4.shape == (1, 16, *spatial_shape)
        assert dy4.shape == (1, 16, *spatial_shape)


class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that layers work with JIT compilation"""
        key = random.PRNGKey(42)
        
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        
        # JIT compile the apply function
        jit_apply = jax.jit(layer.apply)
        y, dy = jit_apply(params, x)
        
        assert y.shape == (1, 32, 6, 6, 6)
        assert dy.shape == (1, 32, 6, 6, 6)
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_vmap_over_batch(self):
        """Test that layers work correctly with batched input"""
        key = random.PRNGKey(42)
        
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        # Batch of samples
        batch_size = 4
        x_batch = random.normal(key, (batch_size, 16, 8, 8, 8))
        
        params = layer.init(key, x_batch)
        y, dy = layer.apply(params, x_batch)
        
        assert y.shape == (batch_size, 32, 6, 6, 6)
        assert dy.shape == (batch_size, 32, 6, 6, 6)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed through the layer"""
        key = random.PRNGKey(42)
        
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        
        def loss_fn(params):
            y, dy = layer.apply(params, x)
            return jnp.mean(y**2) + jnp.mean(dy**2)
        
        # Compute gradient
        grad = jax.grad(loss_fn)(params)
        
        # Check gradients exist and are finite
        assert 'params' in grad
        assert jnp.all(jnp.isfinite(grad['params']['weight']))
        assert jnp.all(jnp.isfinite(grad['params']['bias']))
        assert jnp.all(jnp.isfinite(grad['params']['dweight']))
    
    def test_gradient_through_activation(self):
        """Test that gradients flow through LeakyReLUVel"""
        key = random.PRNGKey(42)
        
        conv = Conv3DVel(in_chan=16, out_chan=32)
        activation = LeakyReLUVel()
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        conv_params = conv.init(key, x)
        
        def loss_fn(params):
            y, dy = conv.apply(params, x)
            y_act, dy_act = activation.apply({}, y, dy)
            return jnp.mean(y_act**2) + jnp.mean(dy_act**2)
        
        grad = jax.grad(loss_fn)(conv_params)
        
        assert jnp.all(jnp.isfinite(grad['params']['weight']))
        assert jnp.all(jnp.isfinite(grad['params']['bias']))
        assert jnp.all(jnp.isfinite(grad['params']['dweight']))


class TestNumericalStability:
    """Test numerical stability of velocity layers"""
    
    def test_small_input_values(self):
        """Test behavior with very small input values"""
        key = random.PRNGKey(42)
        
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e-6
        
        params = layer.init(key, x)
        y, dy = layer.apply(params, x)
        
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_large_input_values(self):
        """Test behavior with large input values"""
        key = random.PRNGKey(42)
        
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e3
        
        params = layer.init(key, x)
        y, dy = layer.apply(params, x)
        
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_leaky_relu_vel_numerical_stability(self):
        """Test LeakyReLUVel with extreme values"""
        key = random.PRNGKey(42)
        
        layer = LeakyReLUVel()
        
        # Test with large negative and positive values
        x = jnp.array([[-1e6, -1e3, 0.0, 1e3, 1e6]])
        dx = jnp.ones_like(x)
        
        params = layer.init(key, x, dx)
        y, dy = layer.apply(params, x, dx)
        
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))


class TestDweightParameter:
    """Test the dweight parameter specific to velocity layers"""
    
    def test_dweight_affects_tangent(self):
        """Test that dweight parameter affects tangent output"""
        key = random.PRNGKey(42)
        
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        y1, dy1 = layer.apply(params, x)
        
        # Modify dweight
        modified_params = {
            'params': {
                **params['params'],
                'dweight': params['params']['dweight'] * 2.0
            }
        }
        y2, dy2 = layer.apply(modified_params, x)
        
        # Output y should be the same
        assert jnp.allclose(y1, y2)
        
        # Tangent dy should be different (scaled by 2)
        assert not jnp.allclose(dy1, dy2)
    
    def test_dweight_independent_of_weight(self):
        """Test that dweight is independent of weight"""
        key = random.PRNGKey(42)
        
        layer = Conv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        
        # Check that weight and dweight are different
        weight = params['params']['weight']
        dweight = params['params']['dweight']
        
        assert not jnp.allclose(weight, dweight)


class TestEquivalenceWithNonVelLayers:
    """Test relationship between velocity and non-velocity layers"""
    
    def test_same_output_shape(self):
        """Test that output shapes match non-velocity counterparts"""
        key = random.PRNGKey(42)
        
        from jax_nbody_emulator.layers import Conv3D
        
        vel_layer = Conv3DVel(in_chan=16, out_chan=32)
        std_layer = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        vel_params = vel_layer.init(key, x)
        std_params = std_layer.init(key, x)
        
        y_vel, dy_vel = vel_layer.apply(vel_params, x)
        y_std = std_layer.apply(std_params, x)
        
        # Shapes should match
        assert y_vel.shape == y_std.shape
    
    def test_same_param_structure_except_dweight(self):
        """Test that parameter structure matches except for dweight"""
        key = random.PRNGKey(42)
        
        from jax_nbody_emulator.layers import Conv3D
        
        vel_layer = Conv3DVel(in_chan=16, out_chan=32)
        std_layer = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        vel_params = vel_layer.init(key, x)
        std_params = std_layer.init(key, x)
        
        # Both should have weight and bias
        assert 'weight' in vel_params['params']
        assert 'bias' in vel_params['params']
        assert 'weight' in std_params['params']
        assert 'bias' in std_params['params']
        
        # Only vel should have dweight
        assert 'dweight' in vel_params['params']
        assert 'dweight' not in std_params['params']

