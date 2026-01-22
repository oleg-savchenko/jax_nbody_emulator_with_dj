"""
Tests for style_nbody_emulator_vel_core.py module.

Main N-body emulator model implementation with style conditioning and
velocity computation using manual forward-mode automatic differentiation.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.style_nbody_emulator_vel_core import StyleNBodyEmulatorVelCore


class TestStyleNBodyEmulatorVelCoreInitialization:
    """Test StyleNBodyEmulatorVelCore initialization"""
    
    def test_default_initialization(self):
        """Test that model initializes with default parameters"""
        model = StyleNBodyEmulatorVelCore()
        
        assert model.style_size == 2
        assert model.in_chan == 3
        assert model.out_chan == 3
        assert model.mid_chan == 64
        assert model.eps == 1e-8
    
    def test_custom_channels(self):
        """Test model with custom channel configuration"""
        model = StyleNBodyEmulatorVelCore(
            in_chan=1,
            out_chan=1,
            mid_chan=32
        )
        
        assert model.in_chan == 1
        assert model.out_chan == 1
        assert model.mid_chan == 32
    
    def test_custom_style_size(self):
        """Test model with custom style size"""
        model = StyleNBodyEmulatorVelCore(style_size=4)
        
        assert model.style_size == 4


class TestStyleNBodyEmulatorVelCoreForwardPass:
    """Test StyleNBodyEmulatorVelCore forward pass"""
    
    def test_returns_tuple(self):
        """Test that forward pass returns (displacement, velocity) tuple"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        result = model.apply(params, x, Om, Dz, vel_fac)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        batch_size = 1
        spatial_size = 128  # Output will be 128 - 96 = 32
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # Both outputs should have same shape
        expected_shape = (batch_size, 3, 32, 32, 32)
        assert displacement.shape == expected_shape
        assert velocity.shape == expected_shape
    
    def test_larger_input(self):
        """Test with larger input spatial size"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        batch_size = 1
        spatial_size = 160  # Output will be 160 - 96 = 64
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        expected_shape = (batch_size, 3, 64, 64, 64)
        assert displacement.shape == expected_shape
        assert velocity.shape == expected_shape
    
    def test_batch_processing(self):
        """Test that model handles batched inputs correctly"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        batch_size = 2
        spatial_size = 128
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.2, 0.3])
        Dz = jnp.array([0.9, 1.0])
        vel_fac = jnp.array([90.0, 100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert displacement.shape[0] == batch_size
        assert velocity.shape[0] == batch_size
        assert displacement.shape == (batch_size, 3, 32, 32, 32)
        assert velocity.shape == (batch_size, 3, 32, 32, 32)
    
    def test_outputs_finite(self):
        """Test that output values are finite"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))


class TestStyleNBodyEmulatorVelCoreVelocity:
    """Test velocity-specific behavior"""
    
    def test_velocity_scaling_with_vel_fac(self):
        """Test that velocity scales with vel_fac parameter"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz, jnp.array([100.0]))
        
        # Test with different velocity factors
        _, vel1 = model.apply(params, x, Om, Dz, jnp.array([100.0]))
        _, vel2 = model.apply(params, x, Om, Dz, jnp.array([200.0]))
        
        # Velocity should scale linearly with vel_fac
        # vel2 should be approximately 2x vel1
        ratio = jnp.mean(jnp.abs(vel2)) / jnp.mean(jnp.abs(vel1))
        assert jnp.isclose(ratio, 2.0, rtol=0.01)
    
    def test_displacement_independent_of_vel_fac(self):
        """Test that displacement is independent of vel_fac"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz, jnp.array([100.0]))
        
        # Test with different velocity factors
        dis1, _ = model.apply(params, x, Om, Dz, jnp.array([100.0]))
        dis2, _ = model.apply(params, x, Om, Dz, jnp.array([200.0]))
        
        # Displacement should be the same regardless of vel_fac
        assert jnp.allclose(dis1, dis2)
    
    def test_velocity_not_zero(self):
        """Test that velocity output is non-zero"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        _, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # Velocity should have variation
        assert velocity.std() > 0
        # Should not be all zeros
        assert not jnp.allclose(velocity, 0.0)
    
    def test_velocity_differs_from_displacement(self):
        """Test that velocity and displacement are different"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # They should not be identical
        assert not jnp.allclose(displacement, velocity)


class TestStyleNBodyEmulatorVelCoreCosmology:
    """Test cosmological parameter handling"""
    
    def test_cosmology_affects_displacement(self):
        """Test that cosmology parameters affect displacement"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, jnp.array([0.3]), jnp.array([1.0]), vel_fac)
        
        dis1, _ = model.apply(params, x, jnp.array([0.3]), jnp.array([1.0]), vel_fac)
        dis2, _ = model.apply(params, x, jnp.array([0.5]), jnp.array([1.2]), vel_fac)
        
        assert not jnp.allclose(dis1, dis2)
    
    def test_cosmology_affects_velocity(self):
        """Test that cosmology parameters affect velocity"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, jnp.array([0.3]), jnp.array([1.0]), vel_fac)
        
        _, vel1 = model.apply(params, x, jnp.array([0.3]), jnp.array([1.0]), vel_fac)
        _, vel2 = model.apply(params, x, jnp.array([0.5]), jnp.array([1.2]), vel_fac)
        
        assert not jnp.allclose(vel1, vel2)
    
    def test_dz_scaling_affects_both_outputs(self):
        """Test that Dz affects both displacement and velocity"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, jnp.array([1.0]), vel_fac)
        
        dis1, vel1 = model.apply(params, x, Om, jnp.array([0.8]), vel_fac)
        dis2, vel2 = model.apply(params, x, Om, jnp.array([1.2]), vel_fac)
        
        assert not jnp.allclose(dis1, dis2)
        assert not jnp.allclose(vel1, vel2)


class TestStyleNBodyEmulatorVelCoreDtype:
    """Test dtype handling"""
    
    def test_dtype_fp32(self):
        """Test model with FP32 precision (default)"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert displacement.dtype == jnp.float32
        assert velocity.dtype == jnp.float32
    
    def test_dtype_fp16(self):
        """Test model with FP16 precision"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert displacement.dtype == jnp.float16
        assert velocity.dtype == jnp.float16
    
    def test_dtype_bfloat16(self):
        """Test model with BF16 precision"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.bfloat16)
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert displacement.dtype == jnp.bfloat16
        assert velocity.dtype == jnp.bfloat16


class TestStyleNBodyEmulatorVelCoreJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that model can be JIT compiled"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        
        # JIT compile the apply function
        jitted_apply = jax.jit(model.apply)
        
        displacement, velocity = jitted_apply(params, x, Om, Dz, vel_fac)
        
        assert displacement.shape == (1, 3, 32, 32, 32)
        assert velocity.shape == (1, 3, 32, 32, 32)
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_gradient_computation(self):
        """Test that gradients can be computed"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        
        def loss_fn(params):
            displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
            return jnp.mean(displacement**2) + jnp.mean(velocity**2)
        
        grad = jax.grad(loss_fn)(params)
        
        # Check gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(grad)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)


class TestStyleNBodyEmulatorVelCoreParameterStructure:
    """Test parameter structure"""
    
    def test_parameter_structure(self):
        """Test that parameters have expected structure"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        
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
    
    def test_parameter_count_matches_non_vel(self):
        """Test that parameter count matches non-velocity version"""
        key = random.PRNGKey(42)
        
        from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore
        
        model_vel = StyleNBodyEmulatorVelCore()
        model_std = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params_vel = model_vel.init(key, x, Om, Dz, vel_fac)
        params_std = model_std.init(key, x, Om, Dz)
        
        def count_params(params):
            return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
        
        n_params_vel = count_params(params_vel)
        n_params_std = count_params(params_std)
        
        # Should have exactly the same number of parameters
        assert n_params_vel == n_params_std
    
    def test_all_parameters_finite(self):
        """Test that all initialized parameters are finite"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        
        for leaf in jax.tree_util.tree_leaves(params):
            assert jnp.all(jnp.isfinite(leaf))


class TestStyleNBodyEmulatorVelCoreEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_zero_input(self):
        """Test model behavior with zero input"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = jnp.zeros((1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # Should produce output (not crash)
        assert displacement.shape == (1, 3, 32, 32, 32)
        assert velocity.shape == (1, 3, 32, 32, 32)
        # Outputs should not be NaN
        assert not jnp.any(jnp.isnan(displacement))
        assert not jnp.any(jnp.isnan(velocity))
    
    def test_extreme_cosmology_low(self):
        """Test with low cosmology values"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        Om = jnp.array([0.1])
        Dz = jnp.array([0.5])
        vel_fac = jnp.array([50.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_extreme_cosmology_high(self):
        """Test with high cosmology values"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        Om = jnp.array([0.5])
        Dz = jnp.array([2.0])
        vel_fac = jnp.array([200.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_numerical_stability_fp16(self):
        """Test numerical stability with FP16"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
        assert displacement.std() > 0
        assert velocity.std() > 0
    
    def test_small_input_values(self):
        """Test with very small input values"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)) * 1e-6
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_large_input_values(self):
        """Test with large input values"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)) * 1e3
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
    
    def test_small_dz_value(self):
        """Test behavior with small Dz value"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([0.1])  # Small but non-zero
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # Should handle small Dz without division issues
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))


class TestStyleNBodyEmulatorVelCoreModelComparison:
    """Compare behavior with non-velocity model"""
    
    def test_displacement_consistency(self):
        """Test that displacement output is similar to non-velocity model"""
        key = random.PRNGKey(42)
        
        from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore
        
        model_dis = StyleNBodyEmulatorCore()
        model_vel = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        # Initialize with same key
        params_dis = model_dis.init(key, x, Om, Dz)
        params_vel = model_vel.init(key, x, Om, Dz, vel_fac)
        
        # Get outputs
        dis_output = model_dis.apply(params_dis, x, Om, Dz)
        dis_from_vel, _ = model_vel.apply(params_vel, x, Om, Dz, vel_fac)
        
        # Shapes should match
        assert dis_output.shape == dis_from_vel.shape
        
        # Both should be non-trivial
        assert dis_output.std() > 0
        assert dis_from_vel.std() > 0
    
    def test_same_architecture(self):
        """Test that both models have same architecture"""
        key = random.PRNGKey(42)
        
        from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore
        
        model_dis = StyleNBodyEmulatorCore()
        model_vel = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params_dis = model_dis.init(key, x, Om, Dz)
        params_vel = model_vel.init(key, x, Om, Dz, jnp.array([100.0]))
        
        # Should have same parameter structure
        keys_dis = set(params_dis['params'].keys())
        keys_vel = set(params_vel['params'].keys())
        
        assert keys_dis == keys_vel


class TestStyleNBodyEmulatorVelCoreResidualConnection:
    """Test residual connection and tangent propagation"""
    
    def test_residual_connection_working(self):
        """Test that residual connection is working"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = jnp.ones((1, 3, spatial_size, spatial_size, spatial_size)) * 0.5
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # Both outputs should have variation
        assert displacement.std() > 0
        assert velocity.std() > 0
    
    def test_tangent_propagation_through_network(self):
        """Test that tangents propagate correctly through the network"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # Velocity is computed from tangents, should be different from displacement
        assert not jnp.allclose(displacement, velocity)
        
        # Both should be finite
        assert jnp.all(jnp.isfinite(displacement))
        assert jnp.all(jnp.isfinite(velocity))
