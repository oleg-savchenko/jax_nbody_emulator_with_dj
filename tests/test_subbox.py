"""
Tests for subbox.py module.

Subbox processing utilities for handling large volumes by splitting them into
smaller overlapping subboxes for GPU processing, then reassembling results.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.subbox import (
    SubboxConfig,
    SubboxProcessor,
    SubboxProcessorVel,
    StyleSubboxProcessor,
    StyleSubboxProcessorVel,
)
from jax_nbody_emulator.nbody_emulator import NBodyEmulator
from jax_nbody_emulator.nbody_emulator_vel import NBodyEmulatorVel
from jax_nbody_emulator.style_nbody_emulator import StyleNBodyEmulator
from jax_nbody_emulator.style_nbody_emulator_vel import StyleNBodyEmulatorVel


class TestSubboxConfig:
    """Test SubboxConfig initialization and index computation"""
    
    def test_default_initialization(self):
        """Test that SubboxConfig initializes correctly"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        assert config.in_chan == 3
        assert config.size == (256, 256, 256)
        assert config.ndiv == (2, 2, 2)
        assert config.NDIM == 3
        assert config.n_subboxes == 8  # 2 * 2 * 2
        assert config.crop_size == (128, 128, 128)  # 256 // 2
    
    def test_asymmetric_divisions(self):
        """Test with asymmetric division factors"""
        config = SubboxConfig(
            in_chan=3,
            size=(512, 256, 128),
            ndiv=(4, 2, 1),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        assert config.n_subboxes == 8  # 4 * 2 * 1
        assert config.crop_size == (128, 128, 128)
    
    def test_single_subbox(self):
        """Test with single subbox (no division)"""
        config = SubboxConfig(
            in_chan=3,
            size=(128, 128, 128),
            ndiv=(1, 1, 1),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        assert config.n_subboxes == 1
        assert config.crop_size == (128, 128, 128)
    
    def test_precomputed_indices(self):
        """Test that indices are precomputed during initialization"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        # Should have precomputed indices for all subboxes
        assert len(config.all_crop_inds) == config.n_subboxes
        assert len(config.all_add_inds) == config.n_subboxes
    
    def test_anchor_computation(self):
        """Test anchor point computation for different subbox indices"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        # First subbox should have anchor at origin
        anchor0 = config._get_anchor(0)
        assert anchor0 == (0, 0, 0)
        
        # Last subbox in z should have anchor offset in z
        anchor1 = config._get_anchor(1)
        assert anchor1 == (0, 0, 128)
        
        # Last subbox should have anchor at (128, 128, 128)
        anchor7 = config._get_anchor(7)
        assert anchor7 == (128, 128, 128)
    
    def test_crop_indices_shape(self):
        """Test that crop indices have correct structure"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        crop_inds = config.all_crop_inds[0]
        
        # Should have 4 elements: channel slice + 3 spatial index arrays
        assert len(crop_inds) == 4
        assert crop_inds[0] == slice(None)  # Channel dimension
    
    def test_add_indices_no_padding(self):
        """Test that add indices have no padding"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        add_inds = config.all_add_inds[0]
        
        # Add indices should cover crop_size without padding
        # Each spatial index array should have length crop_size
        for i in range(1, 4):
            assert add_inds[i].size == config.crop_size[i-1]
    
    def test_periodic_boundary_conditions(self):
        """Test that crop indices handle periodic boundaries"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        # First subbox needs to wrap around for negative indices
        crop_inds = config.all_crop_inds[0]
        
        # The first spatial dimension indices should wrap
        # anchor=0, padding=48, so we need indices from -48 to 128+48
        # With periodic BC, -48 should wrap to 256-48=208
        z_inds = crop_inds[1].flatten()
        
        # Should contain wrapped indices (values >= 208)
        assert np.any(z_inds >= 208)
        # Should also contain normal indices
        assert np.any(z_inds < 128)


class TestSubboxConfigEdgeCases:
    """Test SubboxConfig edge cases"""
    
    def test_asymmetric_padding(self):
        """Test with asymmetric padding"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((32, 64), (48, 48), (24, 72))
        )
        
        assert config.n_subboxes == 8
        # Should still work with asymmetric padding
        assert len(config.all_crop_inds) == 8
    
    def test_large_division(self):
        """Test with many divisions"""
        config = SubboxConfig(
            in_chan=3,
            size=(512, 512, 512),
            ndiv=(4, 4, 4),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        assert config.n_subboxes == 64
        assert config.crop_size == (128, 128, 128)
    
    def test_single_channel(self):
        """Test with single input channel"""
        config = SubboxConfig(
            in_chan=1,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        assert config.in_chan == 1


class TestSubboxProcessor:
    """Test SubboxProcessor for displacement-only models"""
    
    @pytest.fixture
    def setup_processor(self):
        """Set up model and processor for testing"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        # Initialize with dummy input
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            in_chan=3,
            size=(64, 64, 64),  # Small for testing
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
            dtype=jnp.float32
        )
        
        return processor, config
    
    def test_processor_initialization(self, setup_processor):
        """Test that processor initializes correctly"""
        processor, config = setup_processor
        
        assert processor.config == config
        assert processor.dtype == jnp.float32
        assert processor.batch_size == 1
    
    def test_process_box_output_shape(self, setup_processor):
        """Test that process_box returns correct output shape"""
        processor, config = setup_processor
        
        # Create input box
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert output.shape == (3, 64, 64, 64)
        assert output.dtype == np.float32
    
    def test_process_box_finite_output(self, setup_processor):
        """Test that process_box returns finite values"""
        processor, config = setup_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert np.all(np.isfinite(output))
    
    def test_process_box_different_cosmology(self, setup_processor):
        """Test that different cosmology produces different output via Dz scaling.
        
        Note: For the premodulated NBodyEmulator, Om only affects the output through
        the growth factor D(z, Om). At z=0, D(z=0, Om) = 1 for all Om, so we need
        to test at non-zero redshift where D(z, Om) varies with Om.
        """
        processor, config = setup_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        # At z > 0, D(z, Om) depends on Om, so outputs should differ
        output1 = processor.process_box(input_box, z=0.5, Om=0.2, show_progress=False)
        output2 = processor.process_box(input_box, z=0.5, Om=0.4, show_progress=False)
        
        assert not np.allclose(output1, output2)
    
    def test_process_box_different_redshift(self, setup_processor):
        """Test that different redshift produces different output"""
        processor, config = setup_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output1 = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        output2 = processor.process_box(input_box, z=1.0, Om=0.3, show_progress=False)
        
        assert not np.allclose(output1, output2)


class TestSubboxProcessorVel:
    """Test SubboxProcessorVel for displacement+velocity models"""
    
    @pytest.fixture
    def setup_processor_vel(self):
        """Set up velocity model and processor for testing"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVel()
        
        # Initialize with dummy input
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        params = model.init(key, x, Dz, vel_fac)
        
        config = SubboxConfig(
            in_chan=3,
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        processor = SubboxProcessorVel(
            model=model,
            params=params,
            config=config,
            dtype=jnp.float32
        )
        
        return processor, config
    
    def test_processor_vel_initialization(self, setup_processor_vel):
        """Test that velocity processor initializes correctly"""
        processor, config = setup_processor_vel
        
        assert processor.config == config
        assert processor.dtype == jnp.float32
    
    def test_process_box_returns_tuple(self, setup_processor_vel):
        """Test that process_box returns tuple of displacement and velocity"""
        processor, config = setup_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        result = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_process_box_output_shapes(self, setup_processor_vel):
        """Test that both outputs have correct shapes"""
        processor, config = setup_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        displacement, velocity = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert displacement.shape == (3, 64, 64, 64)
        assert velocity.shape == (3, 64, 64, 64)
        assert displacement.dtype == np.float32
        assert velocity.dtype == np.float32
    
    def test_process_box_finite_outputs(self, setup_processor_vel):
        """Test that both outputs are finite"""
        processor, config = setup_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        displacement, velocity = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert np.all(np.isfinite(displacement))
        assert np.all(np.isfinite(velocity))
    
    def test_displacement_and_velocity_different(self, setup_processor_vel):
        """Test that displacement and velocity are different"""
        processor, config = setup_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        displacement, velocity = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        # They should not be identical
        assert not np.allclose(displacement, velocity)


class TestStyleSubboxProcessor:
    """Test StyleSubboxProcessor for style-conditioned displacement models"""
    
    @pytest.fixture
    def setup_style_processor(self):
        """Set up style model and processor for testing"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulator()
        
        # Initialize with dummy input
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        params = model.init(key, x, Om, Dz)
        
        config = SubboxConfig(
            in_chan=3,
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        processor = StyleSubboxProcessor(
            model=model,
            params=params,
            config=config,
            dtype=jnp.float32
        )
        
        return processor, config
    
    def test_style_processor_initialization(self, setup_style_processor):
        """Test that style processor initializes correctly"""
        processor, config = setup_style_processor
        
        assert processor.config == config
        assert processor.dtype == jnp.float32
    
    def test_process_box_output_shape(self, setup_style_processor):
        """Test that process_box returns correct output shape"""
        processor, config = setup_style_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert output.shape == (3, 64, 64, 64)
        assert output.dtype == np.float32
    
    def test_process_box_finite_output(self, setup_style_processor):
        """Test that process_box returns finite values"""
        processor, config = setup_style_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert np.all(np.isfinite(output))
    
    def test_cosmology_affects_output(self, setup_style_processor):
        """Test that cosmology parameters affect output"""
        processor, config = setup_style_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output1 = processor.process_box(input_box, z=0.0, Om=0.2, show_progress=False)
        output2 = processor.process_box(input_box, z=0.0, Om=0.4, show_progress=False)
        
        assert not np.allclose(output1, output2)


class TestStyleSubboxProcessorVel:
    """Test StyleSubboxProcessorVel for style-conditioned displacement+velocity models"""
    
    @pytest.fixture
    def setup_style_processor_vel(self):
        """Set up style velocity model and processor for testing"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVel()
        
        # Initialize with dummy input
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        params = model.init(key, x, Om, Dz, vel_fac)
        
        config = SubboxConfig(
            in_chan=3,
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        processor = StyleSubboxProcessorVel(
            model=model,
            params=params,
            config=config,
            dtype=jnp.float32
        )
        
        return processor, config
    
    def test_style_processor_vel_initialization(self, setup_style_processor_vel):
        """Test that style velocity processor initializes correctly"""
        processor, config = setup_style_processor_vel
        
        assert processor.config == config
        assert processor.dtype == jnp.float32
    
    def test_process_box_returns_tuple(self, setup_style_processor_vel):
        """Test that process_box returns tuple"""
        processor, config = setup_style_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        result = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_process_box_output_shapes(self, setup_style_processor_vel):
        """Test that both outputs have correct shapes"""
        processor, config = setup_style_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        displacement, velocity = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert displacement.shape == (3, 64, 64, 64)
        assert velocity.shape == (3, 64, 64, 64)
    
    def test_process_box_finite_outputs(self, setup_style_processor_vel):
        """Test that both outputs are finite"""
        processor, config = setup_style_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        displacement, velocity = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert np.all(np.isfinite(displacement))
        assert np.all(np.isfinite(velocity))
    
    def test_cosmology_affects_both_outputs(self, setup_style_processor_vel):
        """Test that cosmology affects both displacement and velocity"""
        processor, config = setup_style_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        disp1, vel1 = processor.process_box(input_box, z=0.0, Om=0.2, show_progress=False)
        disp2, vel2 = processor.process_box(input_box, z=0.0, Om=0.4, show_progress=False)
        
        assert not np.allclose(disp1, disp2)
        assert not np.allclose(vel1, vel2)


class TestSubboxProcessorDtypes:
    """Test dtype handling in subbox processors"""
    
    def test_fp16_processing(self):
        """Test processing with FP16 dtype"""
        key = random.PRNGKey(42)
        model = NBodyEmulator(dtype=jnp.float16)
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x.astype(jnp.float16), Dz.astype(jnp.float16))
        
        config = SubboxConfig(
            in_chan=3,
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
            dtype=jnp.float16
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        # Output should be converted back to float32
        assert output.dtype == np.float32
        assert np.all(np.isfinite(output))


class TestSubboxProcessorIntegration:
    """Integration tests for subbox processors"""
    
    def test_full_coverage(self):
        """Test that all subboxes cover the full volume"""
        config = SubboxConfig(
            in_chan=3,
            size=(128, 128, 128),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        # Track which voxels are covered
        coverage = np.zeros((128, 128, 128), dtype=int)
        
        for idx in range(config.n_subboxes):
            add_inds = config.all_add_inds[idx]
            # Create a meshgrid from the indices
            z_inds = add_inds[1].flatten()
            y_inds = add_inds[2].flatten()
            x_inds = add_inds[3].flatten()
            
            # Mark covered voxels
            for z in z_inds:
                for y in y_inds:
                    for x in x_inds:
                        coverage[z, y, x] += 1
        
        # Every voxel should be covered exactly once
        assert np.all(coverage == 1)
    
    def test_deterministic_output(self):
        """Test that processing is deterministic"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            in_chan=3,
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
            dtype=jnp.float32
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output1 = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        output2 = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert np.allclose(output1, output2)
    
    def test_input_preserved(self):
        """Test that input array is not modified"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            in_chan=3,
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
            dtype=jnp.float32
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        input_box_copy = input_box.copy()
        
        _ = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert np.array_equal(input_box, input_box_copy)


class TestSubboxConfigIndexConsistency:
    """Test index consistency across different configurations"""
    
    def test_crop_indices_within_bounds_with_periodic(self):
        """Test that crop indices are valid (may wrap with periodic BC)"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        for idx in range(config.n_subboxes):
            crop_inds = config.all_crop_inds[idx]
            
            for dim in range(1, 4):
                inds = crop_inds[dim].flatten()
                # All indices should be valid after periodic wrapping
                assert np.all(inds >= 0)
                assert np.all(inds < config.size[dim-1])
    
    def test_add_indices_within_bounds(self):
        """Test that add indices are within bounds"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        for idx in range(config.n_subboxes):
            add_inds = config.all_add_inds[idx]
            
            for dim in range(1, 4):
                inds = add_inds[dim].flatten()
                assert np.all(inds >= 0)
                assert np.all(inds < config.size[dim-1])
    
    def test_add_indices_contiguous(self):
        """Test that add indices form contiguous blocks"""
        config = SubboxConfig(
            in_chan=3,
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        for idx in range(config.n_subboxes):
            add_inds = config.all_add_inds[idx]
            
            for dim in range(1, 4):
                inds = add_inds[dim].flatten()
                # Should be contiguous (consecutive integers)
                assert np.all(np.diff(inds) == 1)


class TestSubboxProcessorProgressBar:
    """Test progress bar functionality"""
    
    def test_progress_bar_disabled(self):
        """Test that processing works with progress bar disabled"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            in_chan=3,
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
            dtype=jnp.float32
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        # Should not raise any errors with progress disabled
        output = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            show_progress=False
        )
        
        assert output.shape == (3, 64, 64, 64)
    
    def test_custom_description(self):
        """Test that custom description works"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            in_chan=3,
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            padding=((48, 48), (48, 48), (48, 48))
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
            dtype=jnp.float32
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        # Should not raise any errors with custom description
        output = processor.process_box(
            input_box,
            z=0.0,
            Om=0.3,
            desc="Custom description",
            show_progress=False
        )
        
        assert output.shape == (3, 64, 64, 64)
