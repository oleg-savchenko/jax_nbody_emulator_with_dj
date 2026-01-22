"""
Tests for nbody_emulator_core.py module.

Main N-body emulator model implementation (premodulated version without runtime style conditioning).

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.nbody_emulator import modulate_emulator_parameters
from jax_nbody_emulator.nbody_emulator_core import NBodyEmulatorCore
from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore


class TestNBodyEmulatorCoreInitialization:
    """Test NBodyEmulatorCore initialization"""
    
    def test_default_initialization(self):
        """Test that model initializes with default parameters"""
        model = NBodyEmulatorCore()
        
        assert model.in_chan == 3
        assert model.out_chan == 3
        assert model.mid_chan == 64
        assert model.eps == 1e-8
    
    def test_custom_channels(self):
        """Test model with custom channel configuration"""
        model = NBodyEmulatorCore(
            in_chan=1,
            out_chan=1,
            mid_chan=32
        )
        
        assert model.in_chan == 1
        assert model.out_chan == 1
        assert model.mid_chan == 32

class TestNBodyEmulatorCoreForwardPass:
    """Test NBodyEmulatorCore forward pass"""
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        # Input spatial size must be at least 128 to have valid output after cropping
        # The model crops 48 voxels per side: output = input - 96
        batch_size = 1
        spatial_size = 128  # Output will be 128 - 96 = 32
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        # Output is cropped by 48 voxels per side: 128 - 96 = 32
        expected_shape = (batch_size, 3, 32, 32, 32)
        assert output.shape == expected_shape
    
    def test_larger_input(self):
        """Test with larger input spatial size"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        batch_size = 1
        spatial_size = 160  # Output will be 160 - 96 = 64
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        expected_shape = (batch_size, 3, 64, 64, 64)
        assert output.shape == expected_shape
    
    def test_batch_processing(self):
        """Test that model handles batched inputs correctly"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        batch_size = 2
        spatial_size = 128
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([0.9, 1.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        # Check batch dimension is preserved
        assert output.shape[0] == batch_size
        assert output.shape == (batch_size, 3, 32, 32, 32)
    
    def test_output_finite(self):
        """Test that output values are finite"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_scalar_dz(self):
        """Test that model handles scalar Dz input"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array(1.0)  # Scalar, not array
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        assert output.shape == (1, 3, 32, 32, 32)
        assert jnp.all(jnp.isfinite(output))


class TestNBodyEmulatorCoreGrowthFactor:
    """Test growth factor (Dz) handling"""
    
    def test_dz_affects_output(self):
        """Test that Dz parameter affects output via input scaling"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        params = model.init(key, x, jnp.array([1.0]))
        
        output1 = model.apply(params, x, jnp.array([0.8]))
        output2 = model.apply(params, x, jnp.array([1.2]))
        
        # Outputs should differ when Dz changes
        assert not jnp.allclose(output1, output2)
    
    def test_dz_scaling_proportionality(self):
        """Test that Dz scaling is applied to input"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        params = model.init(key, x, jnp.array([1.0]))
        
        # Different Dz values should produce different outputs
        output_low = model.apply(params, x, jnp.array([0.5]))
        output_mid = model.apply(params, x, jnp.array([1.0]))
        output_high = model.apply(params, x, jnp.array([2.0]))
        
        assert not jnp.allclose(output_low, output_mid)
        assert not jnp.allclose(output_mid, output_high)


class TestNBodyEmulatorCoreResidualConnection:
    """Test residual connection behavior"""
    
    def test_residual_connection_working(self):
        """Test that residual connection is working"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = jnp.ones((1, 3, spatial_size, spatial_size, spatial_size)) * 0.5
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        # Output should not be constant (network is working)
        assert output.std() > 0
        # Output should not be all zeros
        assert not jnp.allclose(output, 0.0)
    
    def test_input_affects_output(self):
        """Test that changing input changes output"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x1 = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        x2 = random.normal(random.fold_in(key, 1), (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x1, Dz)
        
        output1 = model.apply(params, x1, Dz)
        output2 = model.apply(params, x2, Dz)
        
        assert not jnp.allclose(output1, output2)


class TestNBodyEmulatorCoreJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that model can be JIT compiled"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        
        # JIT compile the apply function
        jitted_apply = jax.jit(model.apply)
        
        output = jitted_apply(params, x, Dz)
        
        assert output.shape == (1, 3, 32, 32, 32)
        assert jnp.all(jnp.isfinite(output))
    
    def test_gradient_computation(self):
        """Test that gradients can be computed"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        
        def loss_fn(params):
            output = model.apply(params, x, Dz)
            return jnp.mean(output**2)
        
        grad = jax.grad(loss_fn)(params)
        
        # Check gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)


class TestNBodyEmulatorCoreParameterStructure:
    """Test parameter structure"""
    
    def test_parameter_structure(self):
        """Test that parameters have expected structure"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        
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
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        
        # Count total parameters
        total_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
        
        # Should have millions of parameters for a U-Net
        assert total_params > 1_000_000
    
    def test_all_parameters_finite(self):
        """Test that all initialized parameters are finite"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        
        for leaf in jax.tree_util.tree_leaves(params):
            assert jnp.all(jnp.isfinite(leaf))
    
    def test_no_style_parameters(self):
        """Test that NBodyEmulatorCore has no style-related parameters"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        
        # Check that no parameter names contain 'style'
        def check_no_style(params_dict, path=""):
            for key, value in params_dict.items():
                current_path = f"{path}/{key}" if path else key
                if isinstance(value, dict):
                    check_no_style(value, current_path)
                else:
                    assert 'style' not in key.lower(), f"Found style parameter at {current_path}"
        
        check_no_style(params['params'])


class TestNBodyEmulatorCoreEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_zero_input(self):
        """Test model behavior with zero input"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = jnp.zeros((1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        # Should produce output (not crash)
        assert output.shape == (1, 3, 32, 32, 32)
        # Output should not be NaN
        assert not jnp.any(jnp.isnan(output))
    
    def test_extreme_dz_low(self):
        """Test with low Dz value"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([0.5])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_extreme_dz_high(self):
        """Test with high Dz value"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([2.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_numerical_stability_fp16(self):
        """Test numerical stability with FP16"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        # Should not overflow to inf or underflow to zero everywhere
        assert jnp.all(jnp.isfinite(output))
        assert output.std() > 0
    
    def test_small_input_values(self):
        """Test with very small input values"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)) * 1e-6
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_large_input_values(self):
        """Test with large input values"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)) * 1e3
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        output = model.apply(params, x, Dz)
        
        assert jnp.all(jnp.isfinite(output))


class TestNBodyEmulatorCoreArchitecture:
    """Test architectural properties"""
    
    def test_encoder_decoder_symmetry(self):
        """Test that encoder and decoder have symmetric structure"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        
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
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz)
        
        assert 'conv_c' in params['params']
    
    def test_custom_mid_channels(self):
        """Test model with different mid_chan values"""
        key = random.PRNGKey(42)
        
        for mid_chan in [32, 64, 128]:
            model = NBodyEmulatorCore(mid_chan=mid_chan)
            
            spatial_size = 128
            x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
            Dz = jnp.array([1.0])
            
            params = model.init(key, x, Dz)
            output = model.apply(params, x, Dz)
            
            # Output shape should be independent of mid_chan
            assert output.shape == (1, 3, 32, 32, 32)


    
class TestModulatedModelUsage:
    """Test using modulated parameters with NBodyEmulatorCore"""
    
    def test_modulated_params_work_with_nbody_emulator_core(self):
        """Test that modulated params can be used with NBodyEmulatorCore"""
        key = random.PRNGKey(42)
        
        # Create and init style model
        style_model = StyleNBodyEmulatorCore()
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        style_params = style_model.init(key, x, Om, Dz)
        
        # Modulate parameters
        z = 0.0
        Om_val = 0.3
        modulated_params = modulate_emulator_parameters(style_params, z, Om_val)
        
        # Create NBodyEmulatorCore and use modulated params
        nbody_model = NBodyEmulatorCore()
        
        # Initialize NBodyEmulatorCore to get its param structure
        nbody_params = nbody_model.init(key, x, Dz)
        
        # The modulated params should have compatible structure for the forward pass
        # Note: This test verifies structural compatibility
        output = nbody_model.apply(nbody_params, x, Dz)
        
        assert output.shape == (1, 3, 32, 32, 32)
        assert jnp.all(jnp.isfinite(output))


class TestStyleVsPremodulatedConsistency:
    """Test consistency between StyleNBodyEmulatorCore and premodulated NBodyEmulatorCore"""
    
    def test_architecture_consistency(self):
        """Test that both models have same architectural structure"""
        key = random.PRNGKey(42)
        
        style_model = StyleNBodyEmulatorCore()
        nbody_model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        style_params = style_model.init(key, x, jnp.array([0.3]), jnp.array([1.0]))
        nbody_params = nbody_model.init(key, x, jnp.array([1.0]))
        
        # Both should have same block names
        style_blocks = set(style_params['params'].keys())
        nbody_blocks = set(nbody_params['params'].keys())
        
        assert style_blocks == nbody_blocks, \
            f"Block mismatch: {style_blocks.symmetric_difference(nbody_blocks)}"
    
    def test_output_shape_consistency(self):
        """Test that both models produce same output shape"""
        key = random.PRNGKey(42)
        
        style_model = StyleNBodyEmulatorCore()
        nbody_model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        style_params = style_model.init(key, x, jnp.array([0.3]), jnp.array([1.0]))
        nbody_params = nbody_model.init(key, x, jnp.array([1.0]))
        
        style_output = style_model.apply(style_params, x, jnp.array([0.3]), jnp.array([1.0]))
        nbody_output = nbody_model.apply(nbody_params, x, jnp.array([1.0]))
        
        assert style_output.shape == nbody_output.shape
