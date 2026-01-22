"""
N-Body Emulator: A JAX-based neural network emulator for cosmological simulations.

This package provides two inference modes:

**Preprocessed Mode (Fast):**
- Preprocess parameters once per cosmology: `modulate_emulator_parameters()`
- Run inference many times: `NBodyEmulator`
- Use: `SubboxProcessor` or `SubboxProcessorVel`
- Best for: Processing many boxes with same (Om, z)

**Style Mode (Flexible):**
- Pass (Om, Dz) as inputs to each forward pass
- Models: `StyleNBodyEmulator` or `StyleNBodyEmulatorVel`
- Use: `StyleSubboxProcessor` or `StyleSubboxProcessorVel`
- Best for: Varying cosmology per box

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

from importlib import resources
import numpy as np
import jax.numpy as jnp
import jax

# Cosmology
from .cosmology import D, H, f, dlogH_dloga, vel_norm, acc_norm

# Preprocessed models (no style)
from .layers import ConvBase3D, Conv3D, Skip3D, DownSample3D, UpSample3D, LeakyReLU
from .blocks import ResampleBlock3D, ResNetBlock3D
from .nbody_emulator_core import NBodyEmulatorCore

# Preprocessed models with velocity
from .layers_vel import ConvBase3DVel, Conv3DVel, Skip3DVel, DownSample3DVel, UpSample3DVel, LeakyReLUVel
from .blocks_vel import ResampleBlock3DVel, ResNetBlock3DVel
from .nbody_emulator_vel_core import NBodyEmulatorVelCore

# Style models (Om, Dz inputs)
from .style_layers import StyleConvBase3D, StyleConv3D, StyleSkip3D, StyleDownSample3D, StyleUpSample3D
from .style_blocks import StyleResampleBlock3D, StyleResNetBlock3D
from .style_nbody_emulator_core import StyleNBodyEmulatorCore

# Style models with velocity
from .style_layers_vel import StyleConvBase3DVel, StyleConv3DVel, StyleSkip3DVel, StyleDownSample3DVel, StyleUpSample3DVel
from .style_blocks_vel import StyleResampleBlock3DVel, StyleResNetBlock3DVel
from .style_nbody_emulator_vel_core import StyleNBodyEmulatorVelCore

# Subbox processing
from .subbox import SubboxConfig, SubboxProcessor

# Factory for creating emulator objects with subbox processors, and loading/premodulating model parameters
from .nbody_emulator import NBodyEmulator, load_default_parameters, modulate_emulator_parameters, modulate_emulator_parameters_vel, create_emulator

__version__ = "0.1.0"
__author__ = "Drew Jamieson"
__email__ = "drew.s.jamieson@gmail.com"

__all__ = [
    # Cosmology functions
    "D",
    "H", 
    "f",
    "dlogH_dloga",
    "vel_norm",
    "acc_norm",
    # Layers
    "ConvBase3D",
    "Conv3D",
    "Skip3D", 
    "DownSample3D",
    "UpSample3D",
    "LeakyReLU",
    # Blocks
    "ResampleBlock3D",
    "ResNetBlock3D",
    # Emulator
    "NBodyEmulatorCore",
    # Layers with velocity
    "ConvBase3DVel",
    "Conv3DVel",
    "Skip3DVel", 
    "DownSample3DVel",
    "UpSample3DVel",
    "LeakyReLUVel",
    # Blocks with velocity
    "ResampleBlock3DVel",
    "ResNetBlock3DVel",
    # Emulator with velocity
    "NBodyEmulatorVelCore",
    # Style layers
    "StyleConvBase3D",
    "StyleConv3D",
    "StyleSkip3D", 
    "StyleDownSample3D",
    "StyleUpSample3D",
    # Style blocks
    "StyleResampleBlock3D",
    "StyleResNetBlock3D",
    # Style emulator
    "StyleNBodyEmulatorCore",
    # Style layers with velocity
    "StyleConvBase3DVel",
    "StyleConv3DVel",
    "StyleSkip3DVel", 
    "StyleDownSample3DVel",
    "StyleUpSample3DVel",
    # Style blocks with velocity
    "StyleResampleBlock3DVel",
    "StyleResNetBlock3DVel",
    # Emulator with velocity
    "StyleNBodyEmulatorVelCore",
    # Subbox processors
    "SubboxConfig",
    "SubboxProcessor",
    # Factory
    "NBodyEmulator",
    "load_default_parameters",
    "modulate_emulator_parameters",
    "modulate_emulator_parameters_vel",
    "create_emulator"
]
