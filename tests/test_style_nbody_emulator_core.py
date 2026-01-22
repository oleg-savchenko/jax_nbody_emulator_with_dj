"""
Tests for style_nbody_emulator_core.py module.

Main N-body emulator model implementation with style conditioning.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore


class TestStyleNBodyEmulatorCoreInitialization:
    """Test StyleNBodyEmulatorCore initialization"""
    
    def test_default_initialization(self):
        """Test that model initializes with default parameters"""
        model = StyleNBodyEmulatorCore()
        
        assert model.style_size == 2
        assert model.in_chan == 3
        assert model.out_chan == 3
        assert model.mid_chan == 64
        assert model.eps == 1e-8
    
    def test_custom_channels(self):
        """Test model with custom channel configuration"""
        model = StyleNBodyEmulatorCore(
            in_chan=1,
            out_chan=1,
            mid_chan=32
        )
        
        assert model.in_chan == 1
        assert model.out_chan == 1
        assert model.mid_chan == 32
    
    def test_custom_style_size(self):
        """Test model with custom style size"""
        model = StyleNBodyEmulatorCore(style_size=4)
        
        assert model.style_size == 4


class TestStyleNBodyEmulatorCoreForwardPass:
    """Test StyleNBodyEmulatorCore forward pass"""
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        # Input spatial size must be at least 128 to have valid output after cropping
        # The model crops 48 voxels per side: output = input - 96
        batch_size = 1
        spatial_size = 128  # Output will be 128 - 96 = 32
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Output is cropped by 48 voxels per side: 128 - 96 = 32
        expected_shape = (batch_size, 3, 32, 32, 32)
        assert output.shape == expected_shape
    
    def test_larger_input(self):
        """Test with larger input spatial size"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        batch_size = 1
        spatial_size = 160  # Output will be 160 - 96 = 64
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        expected_shape = (batch_size, 3, 64, 64, 64)
        assert output.shape == expected_shape
    
    def test_batch_processing(self):
        """Test that model handles batched inputs correctly"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        batch_size = 2
        spatial_size = 128
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.2, 0.3])
        Dz = jnp.array([0.9, 1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Check batch dimension is preserved
        assert output.shape[0] == batch_size
        assert output.shape == (batch_size, 3, 32, 32, 32)
    
    def test_output_finite(self):
        """Test that output values are finite"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        assert jnp.all(jnp.isfinite(output))


class TestStyleNBodyEmulatorCoreCosmology:
    """Test cosmological parameter handling"""
    
    def test_cosmology_affects_output(self):
        """Test that cosmology parameters affect output"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        params = model.init(key, x, jnp.array([0.3]), jnp.array([1.0]))
        
        # Test with different cosmological parameters
        output1 = model.apply(params, x, jnp.array([0.3]), jnp.array([1.0]))
        output2 = model.apply(params, x, jnp.array([0.5]), jnp.array([1.2]))
        
        # Outputs should differ when cosmology changes
        assert not jnp.allclose(output1, output2)
    
    def test_om_scaling(self):
        """Test that Om parameter affects output"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, jnp.array([0.3]), Dz)
        
        output1 = model.apply(params, x, jnp.array([0.2]), Dz)
        output2 = model.apply(params, x, jnp.array([0.4]), Dz)
        
        assert not jnp.allclose(output1, output2)
    
    def test_dz_scaling(self):
        """Test that Dz parameter affects output (both style and input scaling)"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        
        params = model.init(key, x, Om, jnp.array([1.0]))
        
        output1 = model.apply(params, x, Om, jnp.array([0.8]))
        output2 = model.apply(params, x, Om, jnp.array([1.2]))
        
        assert not jnp.allclose(output1, output2)
    
    def test_style_vector_construction(self):
        """Test that style vector is constructed correctly from Om and Dz"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        # Style vector: s0 = (Om - 0.3) * 5, s1 = Dz - 1
        # For Om=0.3, Dz=1.0: s = [0.0, 0.0]
        # For Om=0.5, Dz=2.0: s = [1.0, 1.0]
        
        params = model.init(key, x, jnp.array([0.3]), jnp.array([1.0]))
        
        # Both should work without error
        output1 = model.apply(params, x, jnp.array([0.3]), jnp.array([1.0]))
        output2 = model.apply(params, x, jnp.array([0.5]), jnp.array([2.0]))
        
        assert output1.shape == output2.shape
        assert not jnp.allclose(output1, output2)


class TestStyleNBodyEmulatorCoreResidualConnection:
    """Test residual connection behavior"""
    
    def test_residual_connection_working(self):
        """Test that residual connection is working"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = jnp.ones((1, 3, spatial_size, spatial_size, spatial_size)) * 0.5
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Output should not be constant (network is working)
        assert output.std() > 0
        # Output should not be all zeros
        assert not jnp.allclose(output, 0.0)
    
    def test_input_affects_output(self):
        """Test that changing input changes output"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x1 = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        x2 = random.normal(random.fold_in(key, 1), (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x1, Om, Dz)
        
        output1 = model.apply(params, x1, Om, Dz)
        output2 = model.apply(params, x2, Om, Dz)
        
        assert not jnp.allclose(output1, output2)


class TestStyleNBodyEmulatorCoreDtype:
    """Test dtype handling"""
    
    def test_dtype_fp32(self):
        """Test model with FP32 precision (default)"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        assert output.dtype == jnp.float32
    
    def test_dtype_fp16(self):
        """Test model with FP16 precision"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        assert output.dtype == jnp.float16
        assert output.shape == (1, 3, 32, 32, 32)
    
    def test_dtype_bfloat16(self):
        """Test model with BF16 precision"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.bfloat16)
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        assert output.dtype == jnp.bfloat16


class TestStyleNBodyEmulatorCoreJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that model can be JIT compiled"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        
        # JIT compile the apply function
        jitted_apply = jax.jit(model.apply)
        
        output = jitted_apply(params, x, Om, Dz)
        
        assert output.shape == (1, 3, 32, 32, 32)
        assert jnp.all(jnp.isfinite(output))
    
    def test_gradient_computation(self):
        """Test that gradients can be computed"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        
        def loss_fn(params):
            output = model.apply(params, x, Om, Dz)
            return jnp.mean(output**2)
        
        grad = jax.grad(loss_fn)(params)
        
        # Check gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)


class TestStyleNBodyEmulatorCoreParameterStructure:
    """Test parameter structure"""
    
    def test_parameter_structure(self):
        """Test that parameters have expected structure"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        
        # Should have params key
        assert 'params' in params
        
        # Check for expected blocks
        expected_blocks = [
            'conv_l00', 'conv_l01', 'down_l0',
            'conv_l1', 'down_l1',
            'conv_l2', 'down_l2',
            'conv_c',
            'up_r2', 'conv_r2',
            'up_r1', 'conv_r1',
            'up_r0', 'conv_r00', 'conv_r01'
        ]
        
        for block in expected_blocks:
            assert block in params['params'], f"Missing block: {block}"
    
    def test_parameter_count(self):
        """Test that parameter count is reasonable"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        
        # Count total parameters
        total_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
        
        # Should have millions of parameters for a U-Net
        assert total_params > 1_000_000
    
    def test_all_parameters_finite(self):
        """Test that all initialized parameters are finite"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        
        for leaf in jax.tree_util.tree_leaves(params):
            assert jnp.all(jnp.isfinite(leaf))


class TestStyleNBodyEmulatorCoreEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_zero_input(self):
        """Test model behavior with zero input"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = jnp.zeros((1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Should produce output (not crash)
        assert output.shape == (1, 3, 32, 32, 32)
        # Output should not be NaN
        assert not jnp.any(jnp.isnan(output))
    
    def test_extreme_cosmology_low(self):
        """Test with low cosmology values"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        # Very low matter density and growth
        Om = jnp.array([0.1])
        Dz = jnp.array([0.5])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_extreme_cosmology_high(self):
        """Test with high cosmology values"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        # High matter density and growth
        Om = jnp.array([0.5])
        Dz = jnp.array([2.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_numerical_stability_fp16(self):
        """Test numerical stability with FP16"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Should not overflow to inf or underflow to zero everywhere
        assert jnp.all(jnp.isfinite(output))
        assert output.std() > 0
    
    def test_small_input_values(self):
        """Test with very small input values"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)) * 1e-6
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_large_input_values(self):
        """Test with large input values"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)) * 1e3
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        assert jnp.all(jnp.isfinite(output))


class TestStyleNBodyEmulatorCoreArchitecture:
    """Test architectural properties"""
    
    def test_encoder_decoder_symmetry(self):
        """Test that encoder and decoder have symmetric structure"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        
        # Count encoder and decoder blocks
        encoder_blocks = ['conv_l00', 'conv_l01', 'down_l0', 'conv_l1', 'down_l1', 'conv_l2', 'down_l2']
        decoder_blocks = ['up_r2', 'conv_r2', 'up_r1', 'conv_r1', 'up_r0', 'conv_r00', 'conv_r01']
        
        for block in encoder_blocks:
            assert block in params['params']
        
        for block in decoder_blocks:
            assert block in params['params']
    
    def test_bottleneck_exists(self):
        """Test that bottleneck layer exists"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        
        assert 'conv_c' in params['params']
    
    def test_custom_mid_channels(self):
        """Test model with different mid_chan values"""
        key = random.PRNGKey(42)
        
        for mid_chan in [32, 64, 128]:
            model = StyleNBodyEmulatorCore(mid_chan=mid_chan)
            
            spatial_size = 128
            x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
            Om = jnp.array([0.3])
            Dz = jnp.array([1.0])
            
            params = model.init(key, x, Om, Dz)
            output = model.apply(params, x, Om, Dz)
            
            # Output shape should be independent of mid_chan
            assert output.shape == (1, 3, 32, 32, 32)
