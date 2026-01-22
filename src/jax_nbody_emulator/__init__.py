"""
N-Body Emulator: A JAX-based neural network emulator for cosmological simulations.

Quick start:
    from jax_nbody_emulator import create_emulator, SubboxConfig
    
    config = SubboxConfig(size=(512, 512, 512), ndiv=(4, 4, 4))
    emulator = create_emulator(processor_config=config)
    displacement, velocity = emulator.process_box(input_box, z=0.0, Om=0.3)

For fixed cosmology (faster repeated inference):
    emulator = create_emulator(
        premodulate=True,
        premodulate_z=0.0,
        premodulate_Om=0.3,
        processor_config=config,
    )
    displacement, velocity = emulator.process_box(input_box, z=0.0, Om=0.3)

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

# =============================================================================
# Tier 1: Primary API (most users only need these)
# =============================================================================

from .nbody_emulator import (
    NBodyEmulator,
    create_emulator,
    load_default_parameters,
    modulate_emulator_parameters,
    modulate_emulator_parameters_vel,
)
from .subbox import SubboxConfig, SubboxProcessor
from .cosmology import growth_factor, hubble_rate, growth_rate, dlogH_dloga, vel_norm, acc_norm

# =============================================================================
# Tier 2: Core models (for users who want direct model access)
# =============================================================================

from .style_nbody_emulator_core import StyleNBodyEmulatorCore
from .style_nbody_emulator_vel_core import StyleNBodyEmulatorVelCore
from .nbody_emulator_core import NBodyEmulatorCore
from .nbody_emulator_vel_core import NBodyEmulatorVelCore

# =============================================================================
# Tier 3: Building blocks (for custom architectures)
# These are importable but not in __all__ to reduce clutter
# Usage: from jax_nbody_emulator.layers import Conv3D
# =============================================================================

# Layers and blocks are accessible via submodule imports:
#   from jax_nbody_emulator.layers import Conv3D, Skip3D, ...
#   from jax_nbody_emulator.blocks import ResNetBlock3D, ...
#   from jax_nbody_emulator.style_layers import StyleConv3D, ...
#   from jax_nbody_emulator.style_blocks import StyleResNetBlock3D, ...

# =============================================================================
# Package metadata
# =============================================================================

__version__ = "0.1.0"
__author__ = "Drew Jamieson"
__email__ = "drew.s.jamieson@gmail.com"

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Primary API - Factory and processing
    "create_emulator",
    "NBodyEmulator",
    "SubboxConfig",
    "SubboxProcessor",
    "load_default_parameters",
    "modulate_emulator_parameters",
    "modulate_emulator_parameters_vel",
    # Cosmology functions
    "growth_factor",
    "hubble_rate",
    "growth_rate",
    "dlogH_dloga",
    "vel_norm",
    "acc_norm",
    # Core models (Style - flexible cosmology)
    "StyleNBodyEmulatorCore",
    "StyleNBodyEmulatorVelCore",
    # Core models (Premodulated - fixed cosmology)
    "NBodyEmulatorCore",
    "NBodyEmulatorVelCore",
]
