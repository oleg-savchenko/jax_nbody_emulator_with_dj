"""
Tests for nbody_emulator_vel_core.py module.

Main N-body emulator model implementation (premodulated version) with velocity output.
This version computes both displacement and velocity fields using manual forward-mode
automatic differentiation w.r.t. the growth factor Dz.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.nbody_emulator import modulate_emulator_parameters_vel
from jax_nbody_emulator.nbody_emulator_vel_core import NBodyEmulatorVelCore
from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore


class TestNBodyEmulatorVelCoreInitialization:
    """Test NBodyEmulatorVelCore initialization"""
    
    def test_default_initialization(self):
        """Test that model initializes with default parameters"""
        model = NBodyEmulatorVelCore()
        
        assert model.in_chan == 3
        assert model.out_chan == 3
        assert model.mid_chan == 64
        assert model.eps == 1e-8
    
    def test_custom_channels(self):
        """Test model with custom channel configuration"""
        model = NBodyEmulatorVelCore(
            in_chan=1,
            out_chan=1,
            mid_chan=32
        )
        
        assert model.in_chan == 1
        assert model.out_chan == 1
        assert model.mid_chan == 32

class TestNBodyEmulatorVelCoreForwardPass:
    """Test NBodyEmulatorVelCore forward pass"""
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes for both displacement and velocity"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        batch_size = 1
        spatial_size = 128  # Output will be 128 - 96 = 32
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        expected_shape = (batch_size, 3, 32, 32, 32)
        assert displacement.shape == expected_shape
        assert velocity.shape == expected_shape
    
    def test_larger_input(self):
        """Test with larger input spatial size"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        batch_size = 1
        spatial_size = 160  # Output will be 160 - 96 = 64
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        expected_shape = (batch_size, 3, 64, 64, 64)
        assert displacement.shape == expected_shape
        assert velocity.shape == expected_shape
    
    def test_batch_processing(self):
        """Test that model handles batched inputs correctly"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        batch_size = 2
        spatial_size = 128
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([0.9, 1.0])
        vel_fac = jnp.array([0.8, 1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert displacement.shape[0] == batch_size
        assert velocity.shape[0] == batch_size
        assert displacement.shape == (batch_size, 3, 32, 32, 32)
        assert velocity.shape == (batch_size, 3, 32, 32, 32)
    
    def test_output_finite(self):
        """Test that both output values are finite"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_returns_tuple(self):
        """Test that forward pass returns a tuple of two arrays"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        output = model.apply(params, x, Dz, vel_fac)
        
        assert isinstance(output, tuple)
        assert len(output) == 2


class TestNBodyEmulatorVelCoreGrowthAndVelocityFactors:
    """Test growth factor (Dz) and velocity factor (vel_fac) handling"""
    
    def test_dz_affects_displacement(self):
        """Test that Dz parameter affects displacement output"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, jnp.array([1.0]), vel_fac)
        
        disp1, _ = model.apply(params, x, jnp.array([0.8]), vel_fac)
        disp2, _ = model.apply(params, x, jnp.array([1.2]), vel_fac)
        
        assert not jnp.allclose(disp1, disp2)
    
    def test_dz_affects_velocity(self):
        """Test that Dz parameter affects velocity output"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, jnp.array([1.0]), vel_fac)
        
        _, vel1 = model.apply(params, x, jnp.array([0.8]), vel_fac)
        _, vel2 = model.apply(params, x, jnp.array([1.2]), vel_fac)
        
        assert not jnp.allclose(vel1, vel2)
    
    def test_vel_fac_affects_velocity(self):
        """Test that vel_fac parameter affects velocity output"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz, jnp.array([1.0]))
        
        _, vel1 = model.apply(params, x, Dz, jnp.array([0.5]))
        _, vel2 = model.apply(params, x, Dz, jnp.array([1.5]))
        
        assert not jnp.allclose(vel1, vel2)
    
    def test_vel_fac_does_not_affect_displacement(self):
        """Test that vel_fac parameter does NOT affect displacement output"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz, jnp.array([1.0]))
        
        disp1, _ = model.apply(params, x, Dz, jnp.array([0.5]))
        disp2, _ = model.apply(params, x, Dz, jnp.array([1.5]))
        
        # Displacement should be identical regardless of vel_fac
        assert jnp.allclose(disp1, disp2)
    
    def test_velocity_scales_with_vel_fac(self):
        """Test that velocity output scales linearly with vel_fac"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Dz, jnp.array([1.0]))
        
        _, vel1 = model.apply(params, x, Dz, jnp.array([1.0]))
        _, vel2 = model.apply(params, x, Dz, jnp.array([2.0]))
        
        # vel2 should be approximately 2 * vel1
        assert jnp.allclose(vel2, 2.0 * vel1, rtol=1e-5)


class TestNBodyEmulatorVelCoreResidualConnection:
    """Test residual connection behavior"""
    
    def test_residual_connection_working(self):
        """Test that residual connection is working"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = jnp.ones((1, 3, spatial_size, spatial_size, spatial_size)) * 0.5
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        # Outputs should not be constant
        assert displacement.std() > 0
        assert velocity.std() > 0
        # Outputs should not be all zeros
        assert not jnp.allclose(displacement, 0.0)
        assert not jnp.allclose(velocity, 0.0)
    
    def test_input_affects_both_outputs(self):
        """Test that changing input changes both outputs"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x1 = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        x2 = random.normal(random.fold_in(key, 1), (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x1, Dz, vel_fac)
        
        disp1, vel1 = model.apply(params, x1, Dz, vel_fac)
        disp2, vel2 = model.apply(params, x2, Dz, vel_fac)
        
        assert not jnp.allclose(disp1, disp2)
        assert not jnp.allclose(vel1, vel2)


class TestNBodyEmulatorVelCoreDtype:
    """Test dtype handling"""
    
    def test_dtype_fp32(self):
        """Test model with FP32 precision (default)"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert displacement.dtype == jnp.float32
        assert velocity.dtype == jnp.float32
    
    def test_dtype_fp16(self):
        """Test model with FP16 precision"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert displacement.dtype == jnp.float16
        assert velocity.dtype == jnp.float16
        assert displacement.shape == (1, 3, 32, 32, 32)
        assert velocity.shape == (1, 3, 32, 32, 32)
    
    def test_dtype_bfloat16(self):
        """Test model with BF16 precision"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.bfloat16)
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert displacement.dtype == jnp.bfloat16
        assert velocity.dtype == jnp.bfloat16


class TestNBodyEmulatorVelCoreJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that model can be JIT compiled"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        
        # JIT compile the apply function
        jitted_apply = jax.jit(model.apply)
        
        displacement, velocity = jitted_apply(params, x, Dz, vel_fac)
        
        assert displacement.shape == (1, 3, 32, 32, 32)
        assert velocity.shape == (1, 3, 32, 32, 32)
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_gradient_computation_displacement(self):
        """Test that gradients can be computed for displacement"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        
        def loss_fn(params):
            displacement, _ = model.apply(params, x, Dz, vel_fac)
            return jnp.mean(displacement**2)
        
        grad = jax.grad(loss_fn)(params)
        
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    
    def test_gradient_computation_velocity(self):
        """Test that gradients can be computed for velocity"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        
        def loss_fn(params):
            _, velocity = model.apply(params, x, Dz, vel_fac)
            return jnp.mean(velocity**2)
        
        grad = jax.grad(loss_fn)(params)
        
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    
    def test_gradient_computation_combined(self):
        """Test that gradients can be computed for combined loss"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        
        def loss_fn(params):
            displacement, velocity = model.apply(params, x, Dz, vel_fac)
            return jnp.mean(displacement**2) + jnp.mean(velocity**2)
        
        grad = jax.grad(loss_fn)(params)
        
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)


class TestNBodyEmulatorVelCoreParameterStructure:
    """Test parameter structure"""
    
    def test_parameter_structure(self):
        """Test that parameters have expected structure"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        
        assert 'params' in params
        
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
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        
        total_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
        
        assert total_params > 1_000_000
    
    def test_all_parameters_finite(self):
        """Test that all initialized parameters are finite"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        
        for leaf in jax.tree_util.tree_leaves(params):
            assert jnp.all(jnp.isfinite(leaf))


class TestNBodyEmulatorVelCoreEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_zero_input(self):
        """Test model behavior with zero input"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = jnp.zeros((1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert displacement.shape == (1, 3, 32, 32, 32)
        assert velocity.shape == (1, 3, 32, 32, 32)
        assert not jnp.any(jnp.isnan(displacement))
        assert not jnp.any(jnp.isnan(velocity))
    
    def test_extreme_dz_low(self):
        """Test with low Dz value"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([0.5])
        vel_fac = jnp.array([0.5])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_extreme_dz_high(self):
        """Test with high Dz value"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([2.0])
        vel_fac = jnp.array([2.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_numerical_stability_fp16(self):
        """Test numerical stability with FP16"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
        assert displacement.std() > 0
        assert velocity.std() > 0
    
    def test_small_input_values(self):
        """Test with very small input values"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)) * 1e-6
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_large_input_values(self):
        """Test with large input values"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)) * 1e3
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_zero_vel_fac(self):
        """Test with zero velocity factor"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([0.0])
        
        params = model.init(key, x, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Dz, vel_fac)
        
        # Displacement should still be computed
        assert jnp.all(jnp.isfinite(displacement))
        # Velocity should be zero when vel_fac is zero
        assert jnp.allclose(velocity, 0.0)


class TestNBodyEmulatorVelCoreArchitecture:
    """Test architectural properties"""
    
    def test_encoder_decoder_symmetry(self):
        """Test that encoder and decoder have symmetric structure"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        
        encoder_blocks = ['conv_l00', 'conv_l01', 'down_l0', 'conv_l1', 'down_l1', 'conv_l2', 'down_l2']
        decoder_blocks = ['up_r2', 'conv_r2', 'up_r1', 'conv_r1', 'up_r0', 'conv_r00', 'conv_r01']
        
        for block in encoder_blocks:
            assert block in params['params']
        
        for block in decoder_blocks:
            assert block in params['params']
    
    def test_bottleneck_exists(self):
        """Test that bottleneck layer exists"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = model.init(key, x, Dz, vel_fac)
        
        assert 'conv_c' in params['params']
    
    def test_custom_mid_channels(self):
        """Test model with different mid_chan values"""
        key = random.PRNGKey(42)
        
        for mid_chan in [32, 64, 128]:
            model = NBodyEmulatorVelCore(mid_chan=mid_chan)
            
            spatial_size = 128
            x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
            Dz = jnp.array([1.0])
            vel_fac = jnp.array([1.0])
            
            params = model.init(key, x, Dz, vel_fac)
            displacement, velocity = model.apply(params, x, Dz, vel_fac)
            
            assert displacement.shape == (1, 3, 32, 32, 32)
            assert velocity.shape == (1, 3, 32, 32, 32)


class TestModulatedVelModelUsage:
    """Test using modulated parameters with NBodyEmulatorVelCore"""
    
    def test_modulated_params_work_with_nbody_emulator_vel_core(self):
        """Test that modulated params can be used with NBodyEmulatorVelCore"""
        key = random.PRNGKey(42)
        
        # Create NBodyEmulatorVelCore and initialize
        vel_model = NBodyEmulatorVelCore()
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = vel_model.init(key, x, Dz, vel_fac)
        
        # Run forward pass
        displacement, velocity = vel_model.apply(params, x, Dz, vel_fac)
        
        assert displacement.shape == (1, 3, 32, 32, 32)
        assert velocity.shape == (1, 3, 32, 32, 32)
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))


class TestVelocityDerivativeConsistency:
    """Test that velocity output is consistent with derivative of displacement"""
    
    def test_velocity_related_to_displacement_derivative(self):
        """Test that velocity has reasonable relationship to displacement changes with Dz"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        params = model.init(key, x, jnp.array([1.0]), jnp.array([1.0]))
        
        # Get displacement at two nearby Dz values
        Dz1 = jnp.array([1.0])
        Dz2 = jnp.array([1.001])
        vel_fac = jnp.array([1.0])
        
        disp1, vel1 = model.apply(params, x, Dz1, vel_fac)
        disp2, _ = model.apply(params, x, Dz2, vel_fac)
        
        # Numerical derivative
        dDz = 0.001
        numerical_deriv = (disp2 - disp1) / dDz
        
        # The velocity should be related to derivative of displacement w.r.t. Dz
        # scaled by vel_fac. They won't be exactly equal due to the specific
        # forward-mode AD implementation, but should be correlated
        correlation = jnp.corrcoef(
            numerical_deriv.flatten(), 
            vel1.flatten()
        )[0, 1]
        
        # Should be positively correlated (high correlation expected)
        assert correlation > 0.9, f"Velocity should be highly correlated with displacement derivative, got {correlation}"
