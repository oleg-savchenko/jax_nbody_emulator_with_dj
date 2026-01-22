"""
Tests for layers.py module.

These are standard 3D convolutional layers without style conditioning,
used when style parameters have been premodulated into the input.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.layers import (
    ConvBase3D, ConvTransposeBase3D, LeakyReLU,
    Conv3D, Skip3D, DownSample3D, UpSample3D
)


class TestConvBase3D:
    """Test the base ConvBase3D class"""
    
    def test_forward_pass_shapes_batched(self):
        """Test that forward pass produces correct output shapes"""
        key = random.PRNGKey(42)
        batch_size = 2
        
        layer = ConvBase3D(in_chan=32, out_chan=64, kernel_size=3, stride=1)
        
        x = random.normal(key, (batch_size, 32, 8, 16, 16))
        params = layer.init(key, x)
        
        output = layer.apply(params, x)
        
        # Check output shape (VALID padding reduces spatial dims by kernel_size-1)
        expected_shape = (batch_size, 64, 6, 14, 14)
        assert output.shape == expected_shape
    
    def test_forward_pass_single_batch(self):
        """Test forward pass with batch_size=1"""
        key = random.PRNGKey(42)
        
        layer = ConvBase3D(in_chan=16, out_chan=32, kernel_size=3, stride=1)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        params = layer.init(key, x)
        
        output = layer.apply(params, x)
        
        expected_shape = (1, 32, 6, 6, 6)
        assert output.shape == expected_shape
    
    def test_different_kernel_sizes(self):
        """Test with different kernel sizes"""
        key = random.PRNGKey(42)
        
        for kernel_size in [1, 2, 3, 4]:
            layer = ConvBase3D(in_chan=16, out_chan=32, kernel_size=kernel_size)
            
            x = random.normal(key, (1, 16, 10, 10, 10))
            params = layer.init(key, x)
            
            output = layer.apply(params, x)
            
            # VALID padding: out = in - (kernel_size - 1)
            expected_spatial = 10 - (kernel_size - 1)
            expected_shape = (1, 32, expected_spatial, expected_spatial, expected_spatial)
            assert output.shape == expected_shape
    
    def test_stride_parameter(self):
        """Test that stride parameter works correctly"""
        key = random.PRNGKey(42)
        
        layer = ConvBase3D(in_chan=16, out_chan=32, kernel_size=2, stride=2)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        params = layer.init(key, x)
        
        output = layer.apply(params, x)
        
        # With kernel=2, stride=2: out = (in - kernel) // stride + 1 = (8-2)//2 + 1 = 4
        expected_shape = (1, 32, 4, 8, 8)
        assert output.shape == expected_shape
 

class TestConvTransposeBase3D:
    """Test the ConvTransposeBase3D class for upsampling"""
    
    def test_forward_pass_shapes_batched(self):
        """Test that forward pass produces correct upsampled shapes"""
        key = random.PRNGKey(42)
        batch_size = 2
        
        layer = ConvTransposeBase3D(in_chan=32, out_chan=16)
        
        x = random.normal(key, (batch_size, 32, 4, 8, 8))
        params = layer.init(key, x)
        
        output = layer.apply(params, x)
        
        # Should double spatial dimensions
        expected_shape = (batch_size, 16, 8, 16, 16)
        assert output.shape == expected_shape
    
    def test_forward_pass_single_batch(self):
        """Test forward pass with batch_size=1"""
        key = random.PRNGKey(42)
        
        layer = ConvTransposeBase3D(in_chan=32, out_chan=16)
        
        x = random.normal(key, (1, 32, 4, 8, 8))
        params = layer.init(key, x)
        
        output = layer.apply(params, x)
        
        expected_shape = (1, 16, 8, 16, 16)
        assert output.shape == expected_shape
    
    def test_upsampling_factor(self):
        """Test that upsampling doubles spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = ConvTransposeBase3D(in_chan=16, out_chan=32)
        
        spatial_sizes = [(4, 4, 4), (4, 8, 8), (8, 16, 16)]
        
        for d, h, w in spatial_sizes:
            x = random.normal(key, (1, 16, d, h, w))
            params = layer.init(key, x)
            
            output = layer.apply(params, x)
            
            # Should double each dimension
            expected_shape = (1, 32, 2*d, 2*h, 2*w)
            assert output.shape == expected_shape

class TestLeakyReLU:
    """Test LeakyReLU activation function"""
    
    def test_default_negative_slope(self):
        """Test that default negative slope is 0.01"""
        layer = LeakyReLU()
        assert layer.negative_slope == 0.01
    
    def test_forward_pass(self):
        """Test forward pass behavior"""
        key = random.PRNGKey(42)
        layer = LeakyReLU()
        
        x = jnp.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        # Test leaky ReLU behavior: negative values scaled by 0.01
        expected = jnp.array([[-0.02, -0.01, 0.0, 1.0, 2.0]])
        assert jnp.allclose(output, expected)
    
    def test_custom_negative_slope(self):
        """Test with custom negative slope"""
        key = random.PRNGKey(42)
        layer = LeakyReLU(negative_slope=0.1)
        
        x = jnp.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        expected = jnp.array([[-0.2, -0.1, 0.0, 1.0, 2.0]])
        assert jnp.allclose(output, expected)
    
    def test_positive_values_unchanged(self):
        """Test that positive values pass through unchanged"""
        key = random.PRNGKey(42)
        layer = LeakyReLU()
        
        x = random.uniform(key, (10,), minval=0.0, maxval=10.0)
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        assert jnp.allclose(output, x)
    
class TestConv3D:
    """Test Conv3D layer (partial of ConvBase3D)"""
    
    def test_default_parameters(self):
        """Test that Conv3D has correct default parameters"""
        layer = Conv3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 3
        assert layer.stride == 1
    
    def test_forward_pass(self):
        """Test Conv3D forward pass"""
        key = random.PRNGKey(42)
        
        layer = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        # VALID padding: out = in - (kernel_size - 1)
        expected_shape = (1, 32, 6, 14, 14)
        assert output.shape == expected_shape


class TestSkip3D:
    """Test Skip3D layer (1x1x1 convolution)"""
    
    def test_default_parameters(self):
        """Test that Skip3D has correct default parameters"""
        layer = Skip3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 1
        assert layer.stride == 1
    
    def test_preserves_spatial_dimensions(self):
        """Test that Skip3D preserves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = Skip3D(in_chan=32, out_chan=64)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 32, *spatial_shape))
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        # 1x1x1 kernel preserves spatial dimensions
        expected_shape = (1, 64, *spatial_shape)
        assert output.shape == expected_shape
    
    def test_channel_transformation(self):
        """Test that Skip3D correctly transforms channels"""
        key = random.PRNGKey(42)
        
        for in_chan, out_chan in [(16, 32), (32, 16), (64, 64)]:
            layer = Skip3D(in_chan=in_chan, out_chan=out_chan)
            
            x = random.normal(key, (1, in_chan, 8, 8, 8))
            
            params = layer.init(key, x)
            output = layer.apply(params, x)
            
            assert output.shape == (1, out_chan, 8, 8, 8)


class TestDownSample3D:
    """Test DownSample3D layer"""
    
    def test_default_parameters(self):
        """Test that DownSample3D has correct default parameters"""
        layer = DownSample3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 2
    
    def test_downsampling_shape(self):
        """Test that downsampling halves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = DownSample3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        # Should halve each dimension: (8,16,16) -> (4,8,8)
        expected_shape = (1, 32, 4, 8, 8)
        assert output.shape == expected_shape


class TestUpSample3D:
    """Test UpSample3D layer (transposed convolution)"""
    
    def test_default_parameters(self):
        """Test that UpSample3D has correct default parameters"""
        layer = UpSample3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 1
    
    def test_upsampling_shape(self):
        """Test that upsampling doubles spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = UpSample3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        # Should double each dimension: (4,8,8) -> (8,16,16)
        expected_shape = (1, 32, 8, 16, 16)
        assert output.shape == expected_shape


class TestParameterInitialization:
    """Test parameter initialization in layers"""
    
    def test_conv_weight_initialization_shapes(self):
        """Test that conv weights are initialized with correct shapes"""
        key = random.PRNGKey(42)
        layer = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        
        # Check parameter shapes
        assert params['params']['weight'].shape == (32, 16, 3, 3, 3)
        assert params['params']['bias'].shape == (32,)
    
    def test_transpose_weight_initialization_shapes(self):
        """Test that transpose conv weights are initialized correctly"""
        key = random.PRNGKey(42)
        layer = UpSample3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 4, 4))
        
        params = layer.init(key, x)
        
        # Check parameter shapes (kernel_size=2 by default)
        assert params['params']['weight'].shape == (32, 16, 2, 2, 2)
        assert params['params']['bias'].shape == (32,)
    
    def test_bias_initialization_zeros(self):
        """Test that biases are initialized to zeros"""
        key = random.PRNGKey(42)
        layer = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        bias = params['params']['bias']
        
        assert jnp.allclose(bias, jnp.zeros(32))


class TestDownUpSamplingChain:
    """Test integration between downsampling and upsampling"""
    
    def test_symmetric_down_up(self):
        """Test that down then up returns to original spatial size"""
        key = random.PRNGKey(42)
        
        down_layer = DownSample3D(in_chan=16, out_chan=32)
        up_layer = UpSample3D(in_chan=32, out_chan=16)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 16, *spatial_shape))
        
        # Downsample
        down_params = down_layer.init(key, x)
        down_output = down_layer.apply(down_params, x)
        
        # Upsample
        up_key = random.fold_in(key, 1)
        up_params = up_layer.init(up_key, down_output)
        up_output = up_layer.apply(up_params, down_output)
        
        # Should return to original spatial dimensions
        assert up_output.shape == (1, 16, *spatial_shape)
    
    def test_multiple_down_up_levels(self):
        """Test multiple levels of downsampling and upsampling"""
        key = random.PRNGKey(42)
        
        # Two levels of downsampling
        down1 = DownSample3D(in_chan=16, out_chan=32)
        down2 = DownSample3D(in_chan=32, out_chan=64)
        up1 = UpSample3D(in_chan=64, out_chan=32)
        up2 = UpSample3D(in_chan=32, out_chan=16)
        
        spatial_shape = (16, 32, 32)
        x = random.normal(key, (1, 16, *spatial_shape))
        
        # Downsample twice
        params_d1 = down1.init(key, x)
        x1 = down1.apply(params_d1, x)
        assert x1.shape == (1, 32, 8, 16, 16)
        
        params_d2 = down2.init(random.fold_in(key, 1), x1)
        x2 = down2.apply(params_d2, x1)
        assert x2.shape == (1, 64, 4, 8, 8)
        
        # Upsample twice
        params_u1 = up1.init(random.fold_in(key, 2), x2)
        x3 = up1.apply(params_u1, x2)
        assert x3.shape == (1, 32, 8, 16, 16)
        
        params_u2 = up2.init(random.fold_in(key, 3), x3)
        x4 = up2.apply(params_u2, x3)
        assert x4.shape == (1, 16, *spatial_shape)


class TestConvWithActivation:
    """Test convolution followed by activation"""
    
    def test_conv_leaky_relu_chain(self):
        """Test Conv3D followed by LeakyReLU"""
        key = random.PRNGKey(42)
        
        conv = Conv3D(in_chan=16, out_chan=32)
        activation = LeakyReLU()
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        conv_params = conv.init(key, x)
        conv_out = conv.apply(conv_params, x)
        
        act_params = activation.init(key, conv_out)
        act_out = activation.apply(act_params, conv_out)
        
        # Shape should be preserved through activation
        assert act_out.shape == conv_out.shape
        
        # Negative values should be scaled
        negative_mask = conv_out < 0
        if jnp.any(negative_mask):
            assert jnp.all(act_out[negative_mask] == conv_out[negative_mask] * 0.01)


class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that layers work with JIT compilation"""
        key = random.PRNGKey(42)
        
        layer = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        
        # JIT compile the apply function
        jit_apply = jax.jit(layer.apply)
        output = jit_apply(params, x)
        
        assert output.shape == (1, 32, 6, 6, 6)
        assert jnp.all(jnp.isfinite(output))
    
    def test_vmap_over_batch(self):
        """Test that layers work correctly with batched input"""
        key = random.PRNGKey(42)
        
        layer = Conv3D(in_chan=16, out_chan=32)
        
        # Batch of samples
        batch_size = 4
        x_batch = random.normal(key, (batch_size, 16, 8, 8, 8))
        
        params = layer.init(key, x_batch)
        output = layer.apply(params, x_batch)
        
        assert output.shape == (batch_size, 32, 6, 6, 6)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed through the layer"""
        key = random.PRNGKey(42)
        
        layer = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x)
        
        def loss_fn(params):
            output = layer.apply(params, x)
            return jnp.mean(output**2)
        
        # Compute gradient
        grad = jax.grad(loss_fn)(params)
        
        # Check gradients exist and are finite
        assert 'params' in grad
        assert jnp.all(jnp.isfinite(grad['params']['weight']))
        assert jnp.all(jnp.isfinite(grad['params']['bias']))
    
    def test_gradient_through_activation(self):
        """Test that gradients flow through LeakyReLU"""
        key = random.PRNGKey(42)
        
        conv = Conv3D(in_chan=16, out_chan=32)
        activation = LeakyReLU()
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        conv_params = conv.init(key, x)
        
        def loss_fn(params):
            conv_out = conv.apply(params, x)
            act_out = activation.apply({}, conv_out)
            return jnp.mean(act_out**2)
        
        grad = jax.grad(loss_fn)(conv_params)
        
        assert jnp.all(jnp.isfinite(grad['params']['weight']))
        assert jnp.all(jnp.isfinite(grad['params']['bias']))


class TestNumericalStability:
    """Test numerical stability of layers"""
    
    def test_small_input_values(self):
        """Test behavior with very small input values"""
        key = random.PRNGKey(42)
        
        layer = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e-6
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_large_input_values(self):
        """Test behavior with large input values"""
        key = random.PRNGKey(42)
        
        layer = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e3
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_leaky_relu_numerical_stability(self):
        """Test LeakyReLU with extreme values"""
        key = random.PRNGKey(42)
        
        layer = LeakyReLU()
        
        # Test with large negative and positive values
        x = jnp.array([[-1e6, -1e3, 0.0, 1e3, 1e6]])
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        assert jnp.all(jnp.isfinite(output))
        
        # Check correct scaling
        assert output[0, 0] == -1e6 * 0.01
        assert output[0, 4] == 1e6


class TestEquivalenceWithStyleLayers:
    """Test that these layers can be used interchangeably with style layers when style is fixed"""
    
    def test_same_output_shape_as_style_conv(self):
        """Test that output shapes match style layer counterparts"""
        key = random.PRNGKey(42)
        
        # Standard conv
        conv = Conv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        
        params = conv.init(key, x)
        output = conv.apply(params, x)
        
        # Should produce same shape as StyleConv3D with same parameters
        expected_shape = (1, 32, 6, 6, 6)
        assert output.shape == expected_shape
    
    def test_same_output_shape_as_style_transpose(self):
        """Test that upsample output shapes match style layer counterparts"""
        key = random.PRNGKey(42)
        
        # Standard transpose conv
        up = UpSample3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 4, 4))
        
        params = up.init(key, x)
        output = up.apply(params, x)
        
        # Should produce same shape as StyleUpSample3D
        expected_shape = (1, 32, 8, 8, 8)
        assert output.shape == expected_shape
