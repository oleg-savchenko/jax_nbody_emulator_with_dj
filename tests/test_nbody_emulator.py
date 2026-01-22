"""
Tests for nbody_emulator.py module.

Factory functions for creating N-body emulator models and processors.
Provides a unified interface for constructing emulator bundles with
the appropriate model variant, parameters, and subbox processor.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.nbody_emulator import (
    NBodyEmulator,
    create_emulator,
    load_default_parameters,
    modulate_emulator_parameters,
    modulate_emulator_parameters_vel,
    _modulate_weights,
    _modulate_weights_vel,
)
from jax_nbody_emulator.subbox import SubboxConfig


# =============================================================================
# NBodyEmulator Bundle Tests
# =============================================================================

class TestNBodyEmulatorBundle:
    """Test NBodyEmulator dataclass and its methods"""
    
    @pytest.fixture
    def style_emulator_no_vel(self):
        """Create a style emulator without velocity (no params loaded)"""
        return create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            processor_config=None,
        )
    
    @pytest.fixture
    def style_emulator_vel(self):
        """Create a style emulator with velocity (no params loaded)"""
        return create_emulator(
            premodulate=False,
            compute_vel=True,
            load_params=False,
            processor_config=None,
        )
    
    def test_bundle_attributes(self, style_emulator_no_vel):
        """Test that bundle has correct attributes"""
        emulator = style_emulator_no_vel
        
        assert hasattr(emulator, 'model')
        assert hasattr(emulator, 'params')
        assert hasattr(emulator, 'processor')
        assert hasattr(emulator, 'premodulate')
        assert hasattr(emulator, 'compute_vel')
        assert hasattr(emulator, 'dtype')
    
    def test_bundle_default_values(self, style_emulator_no_vel):
        """Test bundle default attribute values"""
        emulator = style_emulator_no_vel
        
        assert emulator.params is None
        assert emulator.processor is None
        assert emulator.premodulate is False
        assert emulator.compute_vel is False
        assert emulator.dtype == jnp.float32
    
    def test_bundle_compute_vel_true(self, style_emulator_vel):
        """Test bundle with compute_vel=True"""
        assert style_emulator_vel.compute_vel is True
    
    def test_apply_raises_without_params(self, style_emulator_no_vel):
        """Test that apply() raises ValueError without loaded params"""
        emulator = style_emulator_no_vel
        x = jnp.ones((1, 3, 128, 128, 128))
        
        with pytest.raises(ValueError, match="No parameters loaded"):
            emulator.apply(x, z=0.0, Om=0.3)
    
    def test_process_box_raises_without_processor(self, style_emulator_no_vel):
        """Test that process_box() raises ValueError without processor"""
        emulator = style_emulator_no_vel
        input_box = np.ones((3, 64, 64, 64), dtype=np.float32)
        
        with pytest.raises(ValueError, match="No processor created"):
            emulator.process_box(input_box, z=0.0, Om=0.3)
    
    def test_call_is_alias_for_apply(self, style_emulator_no_vel):
        """Test that __call__ is an alias for apply()"""
        emulator = style_emulator_no_vel
        x = jnp.ones((1, 3, 128, 128, 128))
        
        # Both should raise the same error
        with pytest.raises(ValueError, match="No parameters loaded"):
            emulator(x, z=0.0, Om=0.3)


class TestNBodyEmulatorApply:
    """Test NBodyEmulator.apply() method with initialized params"""
    
    @pytest.fixture
    def initialized_style_emulator(self):
        """Create emulator with randomly initialized params"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            processor_config=None,
        )
        
        # Initialize params manually
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = emulator.model.init(key, x, Om, Dz)
        
        return NBodyEmulator(
            model=emulator.model,
            params=params,
            processor=None,
            premodulate=False,
            compute_vel=False,
            dtype=jnp.float32,
        )
    
    @pytest.fixture
    def initialized_style_vel_emulator(self):
        """Create velocity emulator with randomly initialized params"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=True,
            load_params=False,
            processor_config=None,
        )
        
        # Initialize params manually
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        params = emulator.model.init(key, x, Om, Dz, vel_fac)
        
        return NBodyEmulator(
            model=emulator.model,
            params=params,
            processor=None,
            premodulate=False,
            compute_vel=True,
            dtype=jnp.float32,
        )
    
    def test_apply_output_shape(self, initialized_style_emulator):
        """Test that apply produces correct output shape"""
        emulator = initialized_style_emulator
        x = jnp.ones((1, 3, 128, 128, 128))
        
        output = emulator.apply(x, z=0.0, Om=0.3)
        
        # Output should have same spatial dimensions (or reduced by network)
        assert output.ndim == 5
        assert output.shape[0] == 1  # batch
        assert output.shape[1] == 3  # channels
    
    def test_apply_vel_returns_tuple(self, initialized_style_vel_emulator):
        """Test that velocity emulator returns (displacement, velocity) tuple"""
        emulator = initialized_style_vel_emulator
        x = jnp.ones((1, 3, 128, 128, 128))
        
        output = emulator.apply(x, z=0.0, Om=0.3)
        
        assert isinstance(output, tuple)
        assert len(output) == 2
        disp, vel = output
        assert disp.shape == vel.shape
    
    def test_apply_scalar_z_om(self, initialized_style_emulator):
        """Test apply with scalar z and Om"""
        emulator = initialized_style_emulator
        x = jnp.ones((1, 3, 128, 128, 128))
        
        # Should work with scalars
        output = emulator.apply(x, z=0.5, Om=0.3)
        assert jnp.all(jnp.isfinite(output))
    
    def test_apply_array_z_om(self, initialized_style_emulator):
        """Test apply with array z and Om"""
        emulator = initialized_style_emulator
        x = jnp.ones((1, 3, 128, 128, 128))
        
        # Should work with arrays
        output = emulator.apply(x, z=jnp.array([0.5]), Om=jnp.array([0.3]))
        assert jnp.all(jnp.isfinite(output))
    
    def test_apply_different_redshifts(self, initialized_style_emulator):
        """Test that different redshifts produce different outputs"""
        emulator = initialized_style_emulator
        x = jnp.ones((1, 3, 128, 128, 128))
        
        output_z0 = emulator.apply(x, z=0.0, Om=0.3)
        output_z1 = emulator.apply(x, z=1.0, Om=0.3)
        
        assert not jnp.allclose(output_z0, output_z1)
    
    def test_apply_different_cosmology(self, initialized_style_emulator):
        """Test that different Om produces different outputs"""
        emulator = initialized_style_emulator
        x = jnp.ones((1, 3, 128, 128, 128))
        
        output_om02 = emulator.apply(x, z=0.5, Om=0.2)
        output_om04 = emulator.apply(x, z=0.5, Om=0.4)
        
        assert not jnp.allclose(output_om02, output_om04)


class TestNBodyEmulatorPremodulated:
    """Test NBodyEmulator with premodulated parameters"""
    
    @pytest.fixture
    def premodulated_emulator(self):
        """Create premodulated emulator with randomly initialized params"""
        # First create a style model to get compatible params
        from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore
        from jax_nbody_emulator.nbody_emulator_core import NBodyEmulatorCore
        
        style_model = StyleNBodyEmulatorCore()
        core_model = NBodyEmulatorCore()
        
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        style_params = style_model.init(key, x, Om, Dz)
        
        # Modulate params for fixed cosmology
        modulated_params = modulate_emulator_parameters(style_params, z=0.0, Om=0.3)
        
        return NBodyEmulator(
            model=core_model,
            params=modulated_params,
            processor=None,
            premodulate=True,
            compute_vel=False,
            dtype=jnp.float32,
        )
    
    @pytest.fixture
    def premodulated_vel_emulator(self):
        """Create premodulated velocity emulator"""
        from jax_nbody_emulator.style_nbody_emulator_vel_core import StyleNBodyEmulatorVelCore
        from jax_nbody_emulator.nbody_emulator_vel_core import NBodyEmulatorVelCore
        
        style_model = StyleNBodyEmulatorVelCore()
        core_model = NBodyEmulatorVelCore()
        
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        style_params = style_model.init(key, x, Om, Dz, vel_fac)
        
        # Modulate params for fixed cosmology
        modulated_params = modulate_emulator_parameters_vel(style_params, z=0.0, Om=0.3)
        
        return NBodyEmulator(
            model=core_model,
            params=modulated_params,
            processor=None,
            premodulate=True,
            compute_vel=True,
            dtype=jnp.float32,
        )
    
    def test_premodulated_apply(self, premodulated_emulator):
        """Test that premodulated emulator apply works"""
        emulator = premodulated_emulator
        x = jnp.ones((1, 3, 128, 128, 128))
        
        output = emulator.apply(x, z=0.0, Om=0.3)
        
        assert output.ndim == 5
        assert jnp.all(jnp.isfinite(output))
    
    def test_premodulated_vel_apply(self, premodulated_vel_emulator):
        """Test that premodulated velocity emulator returns tuple"""
        emulator = premodulated_vel_emulator
        x = jnp.ones((1, 3, 128, 128, 128))
        
        output = emulator.apply(x, z=0.0, Om=0.3)
        
        assert isinstance(output, tuple)
        assert len(output) == 2
        disp, vel = output
        assert jnp.all(jnp.isfinite(disp))
        assert jnp.all(jnp.isfinite(vel))


# =============================================================================
# create_emulator() Factory Function Tests
# =============================================================================

class TestCreateEmulatorModelSelection:
    """Test that create_emulator selects correct model variants"""
    
    def test_style_model_no_vel(self):
        """Test premodulate=False, compute_vel=False creates StyleNBodyEmulatorCore"""
        from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore
        
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
        )
        
        assert isinstance(emulator.model, StyleNBodyEmulatorCore)
        assert emulator.premodulate is False
        assert emulator.compute_vel is False
    
    def test_style_model_vel(self):
        """Test premodulate=False, compute_vel=True creates StyleNBodyEmulatorVelCore"""
        from jax_nbody_emulator.style_nbody_emulator_vel_core import StyleNBodyEmulatorVelCore
        
        emulator = create_emulator(
            premodulate=False,
            compute_vel=True,
            load_params=False,
        )
        
        assert isinstance(emulator.model, StyleNBodyEmulatorVelCore)
        assert emulator.premodulate is False
        assert emulator.compute_vel is True
    
    def test_core_model_no_vel(self):
        """Test premodulate=True, compute_vel=False creates NBodyEmulatorCore"""
        from jax_nbody_emulator.nbody_emulator_core import NBodyEmulatorCore
        
        emulator = create_emulator(
            premodulate=True,
            compute_vel=False,
            load_params=False,
        )
        
        assert isinstance(emulator.model, NBodyEmulatorCore)
        assert emulator.premodulate is True
        assert emulator.compute_vel is False
    
    def test_core_model_vel(self):
        """Test premodulate=True, compute_vel=True creates NBodyEmulatorVelCore"""
        from jax_nbody_emulator.nbody_emulator_vel_core import NBodyEmulatorVelCore
        
        emulator = create_emulator(
            premodulate=True,
            compute_vel=True,
            load_params=False,
        )
        
        assert isinstance(emulator.model, NBodyEmulatorVelCore)
        assert emulator.premodulate is True
        assert emulator.compute_vel is True


class TestCreateEmulatorValidation:
    """Test create_emulator input validation"""
    
    def test_premodulate_requires_z_and_om(self):
        """Test that premodulate=True with load_params=True requires z and Om"""
        with pytest.raises(ValueError, match="premodulate_z and premodulate_Om"):
            create_emulator(
                premodulate=True,
                compute_vel=False,
                load_params=True,
                premodulate_z=None,
                premodulate_Om=None,
            )
    
    def test_premodulate_requires_both_z_and_om(self):
        """Test that both z and Om are required when premodulating"""
        with pytest.raises(ValueError):
            create_emulator(
                premodulate=True,
                compute_vel=False,
                load_params=True,
                premodulate_z=0.0,
                premodulate_Om=None,  # Missing Om
            )
        
        with pytest.raises(ValueError):
            create_emulator(
                premodulate=True,
                compute_vel=False,
                load_params=True,
                premodulate_z=None,  # Missing z
                premodulate_Om=0.3,
            )


class TestCreateEmulatorParams:
    """Test create_emulator parameter loading and modulation"""
    
    def test_load_params_false_returns_none(self):
        """Test that load_params=False results in params=None"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
        )
        
        assert emulator.params is None
    
    def test_no_processor_without_config(self):
        """Test that no processor is created without config"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            processor_config=None,
        )
        
        assert emulator.processor is None


class TestCreateEmulatorWithProcessor:
    """Test create_emulator with SubboxProcessor creation"""
    
    @pytest.fixture
    def subbox_config(self):
        """Create SubboxConfig for testing"""
        return SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            in_chan=3,
            padding=((48, 48), (48, 48), (48, 48)),
            dtype=jnp.float32,
        )
    
    def test_processor_created_with_config(self, subbox_config):
        """Test that processor is created when config is provided"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            processor_config=subbox_config,
        )
        
        assert emulator.processor is not None
    
    def test_processor_dtype_overrides_factory_dtype(self, subbox_config):
        """Test that processor_config dtype overrides dtype argument"""
        subbox_config.dtype = jnp.float16
        
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            processor_config=subbox_config,
            dtype=jnp.float32,  # This should be overridden
        )
        
        assert emulator.dtype == jnp.float16


class TestCreateEmulatorDtype:
    """Test create_emulator dtype handling"""
    
    def test_default_dtype_is_float32(self):
        """Test that default dtype is float32"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
        )
        
        assert emulator.dtype == jnp.float32
    
    def test_explicit_dtype_float16(self):
        """Test setting dtype to float16"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            dtype=jnp.float16,
        )
        
        assert emulator.dtype == jnp.float16
    
    def test_explicit_dtype_bfloat16(self):
        """Test setting dtype to bfloat16"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            dtype=jnp.bfloat16,
        )
        
        assert emulator.dtype == jnp.bfloat16


class TestCreateEmulatorModelKwargs:
    """Test that model_kwargs are passed through to model constructor"""
    
    def test_custom_channels(self):
        """Test passing custom channel configuration"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            in_chan=1,
            out_chan=1,
        )
        
        # Model should have the custom configuration
        assert emulator.model is not None


# =============================================================================
# Parameter Loading and Modulation Tests
# =============================================================================

class TestLoadDefaultParameters:
    """Test load_default_parameters function"""
    
    @pytest.fixture
    def params_file_exists(self):
        """Check if the default parameters file exists"""
        from pathlib import Path
        params_path = Path(__file__).parent.parent / "src" / "jax_nbody_emulator" / "model_parameters" / "nbody_emulator_params.npz"
        return params_path.exists()
    
    def test_load_default_parameters_structure(self, params_file_exists):
        """Test that loaded parameters have correct structure"""
        if not params_file_exists:
            pytest.skip("Default parameters file not found")
        
        params = load_default_parameters()
        
        assert 'params' in params
        assert isinstance(params['params'], dict)
    
    def test_load_default_parameters_contains_blocks(self, params_file_exists):
        """Test that loaded parameters contain expected blocks"""
        if not params_file_exists:
            pytest.skip("Default parameters file not found")
        
        params = load_default_parameters()
        
        # Should contain conv blocks
        param_keys = list(params['params'].keys())
        assert len(param_keys) > 0
        
        # Should have some conv layers
        has_conv = any('conv' in k.lower() for k in param_keys)
        assert has_conv


class TestModulateWeights:
    """Test _modulate_weights helper function"""
    
    def test_modulate_weights_output_shape(self):
        """Test that modulated weights have correct shape"""
        # Create dummy weights matching style layer structure
        c_in, c_out, k = 3, 8, 3
        style_weight = jnp.ones((c_in, 2))  # 2 style dims (Om, Dz)
        style_bias = jnp.zeros((c_in,))
        weight = jnp.ones((c_out, c_in, k, k, k))
        s = jnp.array([[0.0, 0.0]])  # style vector
        
        w_mod = _modulate_weights(style_weight, style_bias, weight, s)
        
        # Should add batch dimension
        assert w_mod.shape == (1, c_out, c_in, k, k, k)
    
    def test_modulate_weights_normalization(self):
        """Test that modulated weights are normalized"""
        c_in, c_out, k = 3, 8, 3
        style_weight = jnp.ones((c_in, 2))
        style_bias = jnp.zeros((c_in,))
        weight = jnp.ones((c_out, c_in, k, k, k)) * 10.0  # Large weights
        s = jnp.array([[0.0, 0.0]])
        
        w_mod = _modulate_weights(style_weight, style_bias, weight, s)
        
        # Weights should be normalized (reasonable magnitude)
        norm = jnp.sqrt(jnp.sum(w_mod[0]**2, axis=(1, 2, 3, 4)))
        assert jnp.all(norm < 100)  # Should be normalized
    
    def test_modulate_weights_1d_style(self):
        """Test modulate_weights with 1D style vector"""
        c_in, c_out, k = 3, 8, 3
        style_weight = jnp.ones((c_in, 2))
        style_bias = jnp.zeros((c_in,))
        weight = jnp.ones((c_out, c_in, k, k, k))
        s = jnp.array([0.0, 0.0])  # 1D style (no batch)
        
        w_mod = _modulate_weights(style_weight, style_bias, weight, s)
        
        assert w_mod.shape == (1, c_out, c_in, k, k, k)


class TestModulateWeightsVel:
    """Test _modulate_weights_vel helper function for velocity computation"""
    
    def test_modulate_weights_vel_returns_tuple(self):
        """Test that velocity modulation returns (weights, dweights) tuple"""
        c_in, c_out, k = 3, 8, 3
        style_weight = jnp.ones((c_in, 2))
        style_bias = jnp.zeros((c_in,))
        weight = jnp.ones((c_out, c_in, k, k, k))
        s = jnp.array([[0.0, 0.0]])
        
        result = _modulate_weights_vel(style_weight, style_bias, weight, s)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_modulate_weights_vel_output_shapes(self):
        """Test that velocity weight and dweight have same shape"""
        c_in, c_out, k = 3, 8, 3
        style_weight = jnp.ones((c_in, 2))
        style_bias = jnp.zeros((c_in,))
        weight = jnp.ones((c_out, c_in, k, k, k))
        s = jnp.array([[0.0, 0.0]])
        
        w_mod, dw_mod = _modulate_weights_vel(style_weight, style_bias, weight, s)
        
        assert w_mod.shape == dw_mod.shape
        assert w_mod.shape == (1, c_out, c_in, k, k, k)
    
    def test_modulate_weights_vel_first_layer_handling(self):
        """Test dx=None handling for first layer"""
        c_in, c_out, k = 3, 8, 3
        style_weight = jnp.ones((c_in, 2))
        style_bias = jnp.zeros((c_in,))
        weight = jnp.ones((c_out, c_in, k, k, k))
        s = jnp.array([[0.0, 1.0]])  # Dz = s[1] + 1 = 2
        
        # dx=None means first layer (different derivative calculation)
        w_mod_first, dw_mod_first = _modulate_weights_vel(
            style_weight, style_bias, weight, s, dx=None
        )
        
        # dx=1 means subsequent layers
        w_mod_later, dw_mod_later = _modulate_weights_vel(
            style_weight, style_bias, weight, s, dx=1
        )
        
        # Weights should be same, but derivatives differ
        assert jnp.allclose(w_mod_first, w_mod_later)
        assert not jnp.allclose(dw_mod_first, dw_mod_later)


class TestModulateEmulatorParameters:
    """Test modulate_emulator_parameters function"""
    
    @pytest.fixture
    def style_params(self):
        """Create dummy style model parameters"""
        from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore
        
        model = StyleNBodyEmulatorCore()
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        return model.init(key, x, Om, Dz)
    
    def test_modulate_returns_dict(self, style_params):
        """Test that modulation returns dict with params key"""
        modulated = modulate_emulator_parameters(style_params, z=0.0, Om=0.3)
        
        assert isinstance(modulated, dict)
        assert 'params' in modulated
    
    def test_modulate_preserves_structure(self, style_params):
        """Test that modulated params have same block structure"""
        modulated = modulate_emulator_parameters(style_params, z=0.0, Om=0.3)
        
        original_blocks = set(style_params['params'].keys())
        modulated_blocks = set(modulated['params'].keys())
        
        assert original_blocks == modulated_blocks
    
    def test_modulate_removes_style_keys(self, style_params):
        """Test that modulated params have style_weight/bias removed"""
        modulated = modulate_emulator_parameters(style_params, z=0.0, Om=0.3)
        
        for block_name, block_params in modulated['params'].items():
            for layer_name, layer_params in block_params.items():
                assert 'style_weight' not in layer_params
                assert 'style_bias' not in layer_params
    
    def test_modulate_different_cosmology_different_weights(self, style_params):
        """Test that different cosmology produces different modulated weights"""
        mod_om02 = modulate_emulator_parameters(style_params, z=0.5, Om=0.2)
        mod_om04 = modulate_emulator_parameters(style_params, z=0.5, Om=0.4)
        
        # Get first weight from first block
        first_block = list(mod_om02['params'].keys())[0]
        first_layer = list(mod_om02['params'][first_block].keys())[0]
        
        w1 = mod_om02['params'][first_block][first_layer]['weight']
        w2 = mod_om04['params'][first_block][first_layer]['weight']
        
        assert not jnp.allclose(w1, w2)


class TestModulateEmulatorParametersVel:
    """Test modulate_emulator_parameters_vel function"""
    
    @pytest.fixture
    def style_vel_params(self):
        """Create dummy style velocity model parameters"""
        from jax_nbody_emulator.style_nbody_emulator_vel_core import StyleNBodyEmulatorVelCore
        
        model = StyleNBodyEmulatorVelCore()
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        
        return model.init(key, x, Om, Dz, vel_fac)
    
    def test_modulate_vel_returns_dict(self, style_vel_params):
        """Test that velocity modulation returns dict"""
        modulated = modulate_emulator_parameters_vel(style_vel_params, z=0.0, Om=0.3)
        
        assert isinstance(modulated, dict)
        assert 'params' in modulated
    
    def test_modulate_vel_has_dweight(self, style_vel_params):
        """Test that velocity modulation adds dweight keys"""
        modulated = modulate_emulator_parameters_vel(style_vel_params, z=0.0, Om=0.3)
        
        # Check that at least one layer has dweight
        has_dweight = False
        for block_name, block_params in modulated['params'].items():
            for layer_name, layer_params in block_params.items():
                if 'dweight' in layer_params:
                    has_dweight = True
                    break
        
        assert has_dweight
    
    def test_modulate_vel_dweight_shape_matches_weight(self, style_vel_params):
        """Test that dweight shape matches weight shape"""
        modulated = modulate_emulator_parameters_vel(style_vel_params, z=0.0, Om=0.3)
        
        for block_name, block_params in modulated['params'].items():
            for layer_name, layer_params in block_params.items():
                if 'dweight' in layer_params:
                    assert layer_params['dweight'].shape == layer_params['weight'].shape


# =============================================================================
# Integration Tests
# =============================================================================

class TestEmulatorIntegration:
    """Integration tests for full emulator workflow"""
    
    @pytest.fixture
    def subbox_config(self):
        """Create SubboxConfig for integration testing"""
        return SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            in_chan=3,
            padding=((48, 48), (48, 48), (48, 48)),
            dtype=jnp.float32,
        )
    
    def test_style_emulator_full_workflow(self, subbox_config):
        """Test complete workflow: create -> init params -> process box"""
        # Create emulator (without loading params since we don't have the file)
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            processor_config=subbox_config,
        )
        
        # Manually initialize params
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        params = emulator.model.init(key, x, Om, Dz)
        
        # Update processor with params
        emulator.processor.params = params
        
        # Process a box
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = emulator.processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.shape == (3, 64, 64, 64)
        assert np.all(np.isfinite(output))
    
    def test_style_vel_emulator_full_workflow(self, subbox_config):
        """Test complete workflow with velocity"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=True,
            load_params=False,
            processor_config=subbox_config,
        )
        
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        params = emulator.model.init(key, x, Om, Dz, vel_fac)
        
        emulator.processor.params = params
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        result = emulator.processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert isinstance(result, tuple)
        disp, vel = result
        assert disp.shape == (3, 64, 64, 64)
        assert vel.shape == (3, 64, 64, 64)
    
    def test_deterministic_output(self, subbox_config):
        """Test that same input produces same output"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
            processor_config=subbox_config,
        )
        
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        params = emulator.model.init(key, x, Om, Dz)
        emulator.processor.params = params
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output1 = emulator.processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        output2 = emulator.processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        np.testing.assert_array_equal(output1, output2)


class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation_apply(self):
        """Test that apply can be JIT compiled"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
        )
        
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        params = emulator.model.init(key, x, Om, Dz)
        
        # Create bundle with params
        emulator_with_params = NBodyEmulator(
            model=emulator.model,
            params=params,
            processor=None,
            premodulate=False,
            compute_vel=False,
        )
        
        # JIT compile the apply function
        jit_apply = jax.jit(emulator_with_params.apply)
        
        # Should work without error
        output = jit_apply(x, z=0.0, Om=0.3)
        assert jnp.all(jnp.isfinite(output))
    
    def test_vmap_over_batch(self):
        """Test that model handles batched inputs correctly"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
        )
        
        # Use smaller spatial size (8 + 2*48 = 104) to avoid OOM with batch=4
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 104, 104, 104))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        params = emulator.model.init(key, x, Om, Dz)
        
        # Test with batch size 4 - need matching Om and Dz batch sizes
        batch_x = jnp.ones((4, 3, 104, 104, 104))
        batch_Om = jnp.array([0.3, 0.3, 0.3, 0.3])
        batch_Dz = jnp.array([1.0, 1.0, 1.0, 1.0])
        
        output = emulator.model.apply(params, batch_x, batch_Om, batch_Dz)
        
        assert output.shape[0] == 4
        assert jnp.all(jnp.isfinite(output))


class TestEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_high_redshift(self):
        """Test emulator at high redshift"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
        )
        
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        Om = jnp.array([0.3])
        Dz = jnp.array([0.1])  # High z -> low Dz
        params = emulator.model.init(key, x, Om, Dz)
        
        output = emulator.model.apply(params, x, Om, Dz)
        assert jnp.all(jnp.isfinite(output))
    
    def test_extreme_omega_m(self):
        """Test with extreme omega_m values"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
        )
        
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128))
        
        for Om_val in [0.1, 0.5, 0.9]:
            Om = jnp.array([Om_val])
            Dz = jnp.array([1.0])
            params = emulator.model.init(key, x, Om, Dz)
            
            output = emulator.model.apply(params, x, Om, Dz)
            assert jnp.all(jnp.isfinite(output)), f"Failed for Om={Om_val}"
    
    def test_small_input(self):
        """Test with very small input values"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
        )
        
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128)) * 1e-8
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        params = emulator.model.init(key, x, Om, Dz)
        
        output = emulator.model.apply(params, x, Om, Dz)
        assert jnp.all(jnp.isfinite(output))
    
    def test_large_input(self):
        """Test with large input values"""
        emulator = create_emulator(
            premodulate=False,
            compute_vel=False,
            load_params=False,
        )
        
        key = random.PRNGKey(42)
        x = jnp.ones((1, 3, 128, 128, 128)) * 100.0
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        params = emulator.model.init(key, x, Om, Dz)
        
        output = emulator.model.apply(params, x, Om, Dz)
        assert jnp.all(jnp.isfinite(output))


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
