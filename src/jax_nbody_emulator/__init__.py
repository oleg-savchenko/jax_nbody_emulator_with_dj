"""
N-Body Emulator: A JAX-based neural network emulator for cosmological simulations.

This package provides two inference modes:

**Preprocessed Mode (Fast):**
- Preprocess parameters once per cosmology: `modulate_emulator_parameters()`
- Run inference many times: `NBodyEmulator` or `NBodyEmulatorVel`
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

def load_default_parameters(dtype=jnp.float32):
    """
    Load the default pretrained N-body emulator parameters.
    
    Parameters
    ----------
    dtype : jnp.dtype, optional
        Desired dtype for all parameter arrays (default: jnp.float32).
        Common options: jnp.float32, jnp.float16, jnp.bfloat16
    
    Returns
    -------
    dict
        Dictionary containing the parameter pytree and any metadata.
        Access parameters with data['params'].
    
    Examples
    --------
    >>> # Load in FP32 (default)
    >>> data = load_default_parameters()
    >>> params = data['params']
    
    >>> # Load in FP16 for faster inference
    >>> data = load_default_parameters(dtype=jnp.float16)
    >>> params_fp16 = data['params']
    """
    with resources.files(__package__).joinpath(
        "model_parameters/nbody_emulator_params.npz"
    ).open("rb") as f:
        # Load as NumPy arrays first
        raw = np.load(f, allow_pickle=True)
        data = {k: raw[k] for k in raw.files}
    
    # Extract and convert params
    params = data.get("params")
    if params is not None:
        params = params.item()  # Unwrap the pytree from numpy object array
        
        # Convert to JAX arrays with desired dtype
        def convert_array(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)):
                # Handle both NumPy and JAX arrays
                # Convert to target dtype, preserving JAX array status
                if isinstance(x, np.ndarray):
                    return jnp.asarray(x, dtype=dtype)
                else:
                    return x.astype(dtype)
            return x
        
        # Use tree_map with is_leaf to ensure we traverse everything
        params = jax.tree_util.tree_map(
            convert_array, 
            params,
            is_leaf=lambda x: isinstance(x, (np.ndarray, jnp.ndarray))
        )
        data["params"] = params
    
    return data

# Cosmology
from .cosmology import D, H, f, dlogH_dloga, vel_norm, acc_norm

# Preprocessing functions
from .nbody_emulator import modulate_emulator_parameters
from .nbody_emulator_vel import modulate_emulator_parameters_vel

# Preprocessed models (no style)
from .layers import ConvBase3D, Conv3D, Skip3D, DownSample3D, UpSample3D, LeakyReLU
from .blocks import ResampleBlock3D, ResNetBlock3D
from .nbody_emulator import NBodyEmulator

# Preprocessed models with velocity
from .layers_vel import ConvBase3DVel, Conv3DVel, Skip3DVel, DownSample3DVel, UpSample3DVel, LeakyReLUVel
from .blocks_vel import ResampleBlock3DVel, ResNetBlock3DVel
from .nbody_emulator_vel import NBodyEmulatorVel

# Style models (Om, Dz inputs)
from .style_layers import StyleConvBase3D, StyleConv3D, StyleSkip3D, StyleDownSample3D, StyleUpSample3D
from .style_blocks import StyleResampleBlock3D, StyleResNetBlock3D
from .style_nbody_emulator import StyleNBodyEmulator

# Style models with velocity
from .style_layers_vel import StyleConvBase3DVel, StyleConv3DVel, StyleSkip3DVel, StyleDownSample3DVel, StyleUpSample3DVel
from .style_blocks_vel import StyleResampleBlock3DVel, StyleResNetBlock3DVel
from .style_nbody_emulator_vel import StyleNBodyEmulatorVel

# Subbox processing
from .subbox import (
    SubboxConfig,
    SubboxProcessor,
    SubboxProcessorVel,
    StyleSubboxProcessor,
    StyleSubboxProcessorVel
)

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
    "modulate_emulator_parameters",
    "NBodyEmulator",
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
    "modulate_emulator_parameters_vel",
    "NBodyEmulatorVel",
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
    "StyleNBodyEmulator",
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
    "StyleNBodyEmulatorVel",
    # Subbox processors
    "SubboxConfig",
    "SubboxProcessor",
    "StyleSubboxProcessor",
    "SubboxProcessorVel",
    "StyleSubboxProcessorVel"
]
