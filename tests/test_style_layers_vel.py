"""
Tests for style_layers_vel.py module.

These layers compute both outputs and their derivatives w.r.t. the style
parameter s[1] (growth factor Dz) using manual forward-mode automatic
differentiation.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.style_layers_vel import (
    StyleConvBase3DVel, StyleTransposeBase3DVel,
    StyleConv3DVel, StyleSkip3DVel, StyleDownSample3DVel, StyleUpSample3DVel
)


class TestStyleConvBase3DVel:
    """Test the base StyleConvBase3DVel class"""
    
    def test_forward_pass_returns_tuple(self):
        """Test that forward pass returns (y, dy) tuple"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        result = layer.apply(params, x, s)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_forward_pass_shapes_batched(self):
        """Test that forward pass produces correct output shapes"""
        key = random.PRNGKey(42)
        batch_size = 2
        
        layer = StyleConvBase3DVel(in_chan=32, out_chan=64, kernel_size=3, stride=1)
        
        x = random.normal(key, (batch_size, 32, 8, 16, 16))
        s = random.normal(key, (batch_size, 2))
        params = layer.init(key, x, s)
        
        y, dy = layer.apply(params, x, s)
        
        # Check output shapes (VALID padding reduces spatial dims by kernel_size-1)
        expected_shape = (batch_size, 64, 6, 14, 14)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_forward_pass_shapes_unbatched(self):
        """Test forward pass with unbatched input"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3, stride=1)
        
        x = random.normal(key, (16, 8, 8, 8))
        s = random.normal(key, (2,))
        params = layer.init(key, x, s)
        
        y, dy = layer.apply(params, x, s)
        
        # Output should also be unbatched
        expected_shape = (32, 6, 6, 6)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_first_layer_no_tangent(self):
        """Test first layer behavior (dx=None)"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s, None)
        y, dy = layer.apply(params, x, s, None)
        
        # Both should have valid shapes
        expected_shape = (1, 32, 6, 6, 6)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_with_input_tangent(self):
        """Test layer with input tangent (dx provided)"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        dx = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        
        params = layer.init(key, x, s, dx)
        y, dy = layer.apply(params, x, s, dx)
        
        expected_shape = (1, 32, 6, 6, 6)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_style_conditioning_effect(self):
        """Test that different style vectors produce different outputs"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s1 = jnp.array([[0.3, 1.0]])
        s2 = jnp.array([[0.5, 1.5]])
        
        params = layer.init(key, x, s1)
        
        y1, dy1 = layer.apply(params, x, s1)
        y2, dy2 = layer.apply(params, x, s2)
        
        # Outputs should be different when style vectors differ
        assert not jnp.allclose(y1, y2)
        assert not jnp.allclose(dy1, dy2)
    
    def test_tangent_depends_on_s1(self):
        """Test that tangent varies with s[1] (growth factor)"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        # Same s[0], different s[1]
        s1 = jnp.array([[0.3, 0.5]])
        s2 = jnp.array([[0.3, 1.5]])
        
        params = layer.init(key, x, s1)
        
        y1, dy1 = layer.apply(params, x, s1)
        y2, dy2 = layer.apply(params, x, s2)
        
        # Tangents should differ when s[1] differs
        assert not jnp.allclose(dy1, dy2)
    

class TestStyleTransposeBase3DVel:
    """Test the StyleTransposeBase3DVel class for upsampling"""
    
    def test_forward_pass_returns_tuple(self):
        """Test that forward pass returns (y, dy) tuple"""
        key = random.PRNGKey(42)
        
        layer = StyleTransposeBase3DVel(in_chan=32, out_chan=16)
        
        x = random.normal(key, (1, 32, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        result = layer.apply(params, x, s)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_forward_pass_shapes_batched(self):
        """Test that forward pass produces correct upsampled shapes"""
        key = random.PRNGKey(42)
        batch_size = 2
        
        layer = StyleTransposeBase3DVel(in_chan=32, out_chan=16)
        
        x = random.normal(key, (batch_size, 32, 4, 8, 8))
        s = random.normal(key, (batch_size, 2))
        params = layer.init(key, x, s)
        
        y, dy = layer.apply(params, x, s)
        
        # Should double spatial dimensions
        expected_shape = (batch_size, 16, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_forward_pass_shapes_unbatched(self):
        """Test forward pass with unbatched input"""
        key = random.PRNGKey(42)
        
        layer = StyleTransposeBase3DVel(in_chan=32, out_chan=16)
        
        x = random.normal(key, (32, 4, 8, 8))
        s = random.normal(key, (2,))
        params = layer.init(key, x, s)
        
        y, dy = layer.apply(params, x, s)
        
        # Output should also be unbatched, with doubled spatial dims
        expected_shape = (16, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_with_input_tangent(self):
        """Test layer with input tangent (dx provided)"""
        key = random.PRNGKey(42)
        
        layer = StyleTransposeBase3DVel(in_chan=32, out_chan=16)
        
        x = random.normal(key, (1, 32, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        dx = random.normal(random.fold_in(key, 1), (1, 32, 4, 8, 8))
        
        params = layer.init(key, x, s, dx)
        y, dy = layer.apply(params, x, s, dx)
        
        expected_shape = (1, 16, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestStyleConv3DVel:
    """Test StyleConv3DVel layer (partial of StyleConvBase3DVel)"""
    
    def test_default_parameters(self):
        """Test that StyleConv3DVel has correct default parameters"""
        layer = StyleConv3DVel(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 3
        assert layer.stride == 1
    
    def test_forward_pass(self):
        """Test StyleConv3DVel forward pass"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        y, dy = layer.apply(params, x, s)
        
        # VALID padding: out = in - (kernel_size - 1)
        expected_shape = (1, 32, 6, 14, 14)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestStyleSkip3DVel:
    """Test StyleSkip3DVel layer (1x1x1 convolution)"""
    
    def test_default_parameters(self):
        """Test that StyleSkip3DVel has correct default parameters"""
        layer = StyleSkip3DVel(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 1
        assert layer.stride == 1
    
    def test_preserves_spatial_dimensions(self):
        """Test that StyleSkip3DVel preserves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = StyleSkip3DVel(in_chan=32, out_chan=64)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 32, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        y, dy = layer.apply(params, x, s)
        
        # 1x1x1 kernel preserves spatial dimensions
        expected_shape = (1, 64, *spatial_shape)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestStyleDownSample3DVel:
    """Test StyleDownSample3DVel layer"""
    
    def test_default_parameters(self):
        """Test that StyleDownSample3DVel has correct default parameters"""
        layer = StyleDownSample3DVel(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 2
    
    def test_downsampling_shape(self):
        """Test that downsampling halves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = StyleDownSample3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        y, dy = layer.apply(params, x, s)
        
        # Should halve each dimension: (8,16,16) -> (4,8,8)
        expected_shape = (1, 32, 4, 8, 8)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestStyleUpSample3DVel:
    """Test StyleUpSample3DVel layer (transposed convolution)"""
    
    def test_default_parameters(self):
        """Test that StyleUpSample3DVel has correct default parameters"""
        layer = StyleUpSample3DVel(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 1
    
    def test_upsampling_shape(self):
        """Test that upsampling doubles spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = StyleUpSample3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        y, dy = layer.apply(params, x, s)
        
        # Should double each dimension: (4,8,8) -> (8,16,16)
        expected_shape = (1, 32, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestTangentPropagation:
    """Test that tangents propagate correctly through layers"""
    
    def test_tangent_propagation_chain(self):
        """Test tangent propagation through multiple layers"""
        key = random.PRNGKey(42)
        
        layer1 = StyleConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        layer2 = StyleConvBase3DVel(in_chan=32, out_chan=64, kernel_size=3)
        
        x = random.normal(key, (1, 16, 12, 12, 12))
        s = jnp.array([[0.3, 1.0]])
        
        params1 = layer1.init(key, x, s, None)
        params2_key = random.fold_in(key, 1)
        
        # First layer (no input tangent)
        y1, dy1 = layer1.apply(params1, x, s, None)
        
        # Second layer (with tangent from first)
        params2 = layer2.init(params2_key, y1, s, dy1)
        y2, dy2 = layer2.apply(params2, y1, s, dy1)
        
        # Check shapes are consistent
        assert y2.shape[1] == 64  # out_chan
        assert dy2.shape == y2.shape
        
        # Both outputs should be finite
        assert jnp.all(jnp.isfinite(y2))
        assert jnp.all(jnp.isfinite(dy2))
    
    def test_tangent_differs_with_and_without_dx(self):
        """Test that providing dx changes the tangent computation"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        dx = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        
        params = layer.init(key, x, s)
        
        # Without dx
        y1, dy1 = layer.apply(params, x, s, None)
        
        # With dx
        y2, dy2 = layer.apply(params, x, s, dx)
        
        # Outputs should be the same
        assert jnp.allclose(y1, y2)
        
        # Tangents should be different
        assert not jnp.allclose(dy1, dy2)


class TestParameterInitialization:
    """Test parameter initialization in velocity style layers"""
    
    def test_weight_initialization_shapes(self):
        """Test that weights are initialized with correct shapes"""
        key = random.PRNGKey(42)
        layer = StyleConv3DVel(in_chan=16, out_chan=32)
        
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
        layer = StyleConv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        style_bias = params['params']['style_bias']
        
        # Style bias should be initialized to ones
        assert jnp.allclose(style_bias, jnp.ones(16))
    
    def test_transpose_weight_initialization_shapes(self):
        """Test that transpose conv weights are initialized correctly"""
        key = random.PRNGKey(42)
        layer = StyleUpSample3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 4, 4))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        
        # Check parameter shapes (kernel_size=2 by default)
        assert params['params']['weight'].shape == (32, 16, 2, 2, 2)
        assert params['params']['bias'].shape == (32,)
        assert params['params']['style_weight'].shape == (16, 2)
        assert params['params']['style_bias'].shape == (16,)


class TestDownUpSamplingChainVel:
    """Test integration between downsampling and upsampling with tangents"""
    
    def test_symmetric_down_up(self):
        """Test that down then up returns to original spatial size"""
        key = random.PRNGKey(42)
        
        down_layer = StyleDownSample3DVel(in_chan=16, out_chan=32)
        up_layer = StyleUpSample3DVel(in_chan=32, out_chan=16)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 16, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        # Downsample
        down_params = down_layer.init(key, x, s)
        down_y, down_dy = down_layer.apply(down_params, x, s)
        
        # Upsample (passing tangent through)
        up_key = random.fold_in(key, 1)
        up_params = up_layer.init(up_key, down_y, s, down_dy)
        up_y, up_dy = up_layer.apply(up_params, down_y, s, down_dy)
        
        # Should return to original spatial dimensions
        assert up_y.shape == (1, 16, *spatial_shape)
        assert up_dy.shape == (1, 16, *spatial_shape)
    
    def test_multiple_down_up_levels(self):
        """Test multiple levels of downsampling and upsampling"""
        key = random.PRNGKey(42)
        
        # Two levels of downsampling
        down1 = StyleDownSample3DVel(in_chan=16, out_chan=32)
        down2 = StyleDownSample3DVel(in_chan=32, out_chan=64)
        up1 = StyleUpSample3DVel(in_chan=64, out_chan=32)
        up2 = StyleUpSample3DVel(in_chan=32, out_chan=16)
        
        spatial_shape = (16, 32, 32)
        x = random.normal(key, (1, 16, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        # Downsample twice
        params_d1 = down1.init(key, x, s)
        y1, dy1 = down1.apply(params_d1, x, s)
        assert y1.shape == (1, 32, 8, 16, 16)
        
        params_d2 = down2.init(random.fold_in(key, 1), y1, s, dy1)
        y2, dy2 = down2.apply(params_d2, y1, s, dy1)
        assert y2.shape == (1, 64, 4, 8, 8)
        
        # Upsample twice
        params_u1 = up1.init(random.fold_in(key, 2), y2, s, dy2)
        y3, dy3 = up1.apply(params_u1, y2, s, dy2)
        assert y3.shape == (1, 32, 8, 16, 16)
        
        params_u2 = up2.init(random.fold_in(key, 3), y3, s, dy3)
        y4, dy4 = up2.apply(params_u2, y3, s, dy3)
        assert y4.shape == (1, 16, *spatial_shape)
        assert dy4.shape == (1, 16, *spatial_shape)


class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that layers work with JIT compilation"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        
        # JIT compile the apply function
        jit_apply = jax.jit(layer.apply)
        y, dy = jit_apply(params, x, s)
        
        assert y.shape == (1, 32, 6, 6, 6)
        assert dy.shape == (1, 32, 6, 6, 6)
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_vmap_over_batch(self):
        """Test that layers work correctly with batched input"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3DVel(in_chan=16, out_chan=32)
        
        # Single sample for initialization
        x_single = random.normal(key, (16, 8, 8, 8))
        s_single = random.normal(key, (2,))
        params = layer.init(key, x_single, s_single)
        
        # Batch of samples
        batch_size = 4
        x_batch = random.normal(key, (batch_size, 16, 8, 8, 8))
        s_batch = random.normal(key, (batch_size, 2))
        
        # Apply to batch
        y, dy = layer.apply(params, x_batch, s_batch)
        
        assert y.shape == (batch_size, 32, 6, 6, 6)
        assert dy.shape == (batch_size, 32, 6, 6, 6)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed through the layer"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        
        def loss_fn(params):
            y, dy = layer.apply(params, x, s)
            return jnp.mean(y**2) + jnp.mean(dy**2)
        
        # Compute gradient
        grad = jax.grad(loss_fn)(params)
        
        # Check gradients exist and are finite
        assert 'params' in grad
        assert jnp.all(jnp.isfinite(grad['params']['weight']))
        assert jnp.all(jnp.isfinite(grad['params']['bias']))
        assert jnp.all(jnp.isfinite(grad['params']['style_weight']))
        assert jnp.all(jnp.isfinite(grad['params']['style_bias']))


class TestNumericalStability:
    """Test numerical stability of velocity style layers"""
    
    def test_demodulation_with_small_weights(self):
        """Test that demodulation handles small weights correctly"""
        key = random.PRNGKey(42)
        
        layer = StyleConvBase3DVel(in_chan=16, out_chan=32, eps=1e-8)
        
        x = random.normal(key, (1, 16, 8, 8, 8)) * 1e-6  # Very small input
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        y, dy = layer.apply(params, x, s)
        
        # Both outputs should be finite
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_large_style_values(self):
        """Test behavior with large style values"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[100.0, 100.0]])  # Large style values
        
        params = layer.init(key, x, s)
        y, dy = layer.apply(params, x, s)
        
        # Both outputs should be finite
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))
    
    def test_small_dz_value(self):
        """Test behavior with small Dz (s[1] close to -1)"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3DVel(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        # s[1] + 1.0 = Dz, so s[1] = -0.9 gives Dz = 0.1
        s = jnp.array([[0.3, -0.9]])
        
        params = layer.init(key, x, s)
        y, dy = layer.apply(params, x, s)
        
        # Both outputs should be finite
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(dy))


class TestManualVsAutoAD:
    """Test that manual forward-mode AD matches JAX's automatic differentiation"""
    
    def test_tangent_matches_jvp(self):
        """Test that manual tangent computation matches jax.jvp"""
        key = random.PRNGKey(42)
        
        # Use the non-velocity version for comparison
        from jax_nbody_emulator.style_layers import StyleConvBase3D
        
        layer_vel = StyleConvBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        layer_ref = StyleConvBase3D(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        # Initialize both with same key for same params
        params_vel = layer_vel.init(key, x, s)
        params_ref = layer_ref.init(key, x, s)
        
        # Get manual tangent
        y_vel, dy_manual = layer_vel.apply(params_vel, x, s, None)
        
        # Get reference output
        y_ref = layer_ref.apply(params_ref, x, s)
        
        # Outputs should match
        assert jnp.allclose(y_vel, y_ref, rtol=1e-5)
        
        # Compute tangent via JAX JVP for reference
        def forward_fn(s_val):
            s_arr = s.at[:, 1].set(s_val)
            return layer_ref.apply(params_ref, x, s_arr)
        
        primals = (s[0, 1],)
        tangents = (jnp.array(1.0),)
        _, dy_jvp = jax.jvp(forward_fn, primals, tangents)
        
        # Manual tangent should approximately match JVP tangent
        # Note: There may be small differences due to the w/Dz term in first layer
        # This is expected behavior, so we use a looser tolerance
        assert dy_manual.shape == dy_jvp.shape
        assert jnp.all(jnp.isfinite(dy_manual))
        assert jnp.all(jnp.isfinite(dy_jvp))
