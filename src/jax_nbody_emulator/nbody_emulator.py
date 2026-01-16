"""
Main N-body emulator model implementation.

This module contains the primary NBodyEmulator class that implements
a 3D U-Net-like architecture for cosmological
N-body simulations.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .cosmology import D
from .blocks import ResNetBlock3D, ResampleBlock3D

def _modulate_weights(style_weight, style_bias, weight, s, eps = 1.e-8) :

        if s.ndim == 1:
            s = s[None]

        s = jnp.array(s, dtype=style_weight.dtype)
        
        s_mod = jnp.dot(s, style_weight.T) + style_bias
        # s_mod: (B, C_in) -> (B, 1, C_in, 1, 1, 1)
        s_mod = s_mod[:, None, :, None, None, None]
        
        # w: (C_out, C_in, K, K, K) -> (B, C_out, C_in, K, K, K)
        w = weight[None] * s_mod
        
        # Demodulation (normalize over spatial + input channels)
        norm = jnp.sqrt(jnp.sum(w**2, axis=(2,3,4,5), keepdims=True) + jnp.array(eps, dtype=w.dtype))
        
        w_normalized = w / norm
        
        return w_normalized

def modulate_emulator_parameters(params, z, Om, eps = 1.e-8, dtype = jnp.float32):
    """
    Preprocess all network parameters for fixed (z, Om).
    
    Returns new params dict with modulated weights.
    """
    
    Dz = D(z, Om)
    
    # Compute style vector
    s0 = (Om - 0.3) * 5.
    s1 = Dz - 1.
    s = jnp.stack([s0, s1], axis=-1)
    
    # Process each block
    processed_params = {'params':{}}
    for block_name, block_params in params['params'].items():
        processed_params['params'][block_name] = {}
        for layer_name, layer_params in block_params.items():
            if 'style_weight' in layer_params:
                # This layer has style modulation - preprocess it
                w_norm = _modulate_weights(
                    layer_params['style_weight'],
                    layer_params['style_bias'],
                    layer_params['weight'],
                    s,
                    eps=eps
                )
                processed_params['params'][block_name][layer_name] = {
                    'weight': w_norm[0].astype(dtype),
                    'bias': layer_params['bias'].astype(dtype)
                }
            else:
                # Pass through unmodified
                print(f'skipping {block_name} {layer_name}')
                processed_params['params'][block_name][layer_name] = layer_params
    
    return processed_params   

class NBodyEmulator(nn.Module):
    """
    3D U-Net-like neural network for N-body simulations.
    
    Processes 3D volumetric data through encoder-bottleneck-decoder architecture
    with skip connections.
    
    Attributes:
        in_chan: Number of input channels
        out_chan: Number of output channels
        mid_chan: Number of channels in intermediate layers
        eps: Small constant for numerical stability
    """
    in_chan: int = 3
    out_chan: int = 3
    mid_chan: int = 64
    eps: float = 1e-8
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        """Initialize all network layers."""
        mid_chan_1 = self.mid_chan
        mid_chan_2 = 2 * mid_chan_1  # For concatenation with skip connections
        
        # Encoder Path (Downsampling)
        self.conv_l00 = ResNetBlock3D(
            'CACA', self.in_chan, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_l01 = ResNetBlock3D(
            'CACA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.down_l0 = ResampleBlock3D(
            'DA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.conv_l1 = ResNetBlock3D(
            'CACA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.down_l1 = ResampleBlock3D(
            'DA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.conv_l2 = ResNetBlock3D(
            'CACA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.down_l2 = ResampleBlock3D(
            'DA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        # Bottleneck
        self.conv_c = ResNetBlock3D(
            'CACA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        # Decoder Path (Upsampling)
        self.up_r2 = ResampleBlock3D(
            'UA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r2 = ResNetBlock3D(
            'CACA', mid_chan_2, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.up_r1 = ResampleBlock3D(
            'UA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r1 = ResNetBlock3D(
            'CACA', mid_chan_2, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.up_r0 = ResampleBlock3D(
            'UA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r00 = ResNetBlock3D(
            'CACA', mid_chan_2, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r01 = ResNetBlock3D(
            'CAC', mid_chan_1, self.out_chan, eps=self.eps, dtype=self.dtype
        )
    
    def __call__(self, x, Dz):
        """
        Forward pass of the NBodyEmulator.
        
        Args:
            x: Input 3D data tensor of shape (B, C_in, D, H, W)
            Dz: Linear growth factors at output redshifts of shape (B,)
            
        Returns:
            displacement:
                - displacement: Predicted displacement field (B, C_out, D', H', W')
                - ocity: Predicted ocity field (B, C_out, D', H', W')
        """
        
        
        if Dz.ndim == 0 :
            Dz = Dz[None]

        Dz = Dz.astype(self.dtype)[:, None, None, None, None]
        
        # Apply growth factor scaling to input
        # Factor of 6 is from original model input normalization
        x = x * (Dz / 6.)
        dx = None  # Will be computed by first layer
        
        # Store cropped input for final residual connection
        # Assumes input spatial dimensions are large enough (e.g., 256^3)
        x0 = x[:, :, 48:-48, 48:-48, 48:-48]
        
        # ===== Encoder Path =====
        x = self.conv_l00(x)
        y0 = self.conv_l01(x)
        
        x = self.down_l0(y0)
        # Crop skip connection to match decoder dimensions
        y0 = y0[:, :, 40:-40, 40:-40, 40:-40]
        
        y1 = self.conv_l1(x)
        x = self.down_l1(y1)
        # Crop skip connection
        y1 = y1[:, :, 16:-16, 16:-16, 16:-16]
        
        y2 = self.conv_l2(x)
        x = self.down_l2(y2)
        # Crop skip connection
        y2 = y2[:, :, 4:-4, 4:-4, 4:-4]
        
        # ===== Bottleneck =====
        x = self.conv_c(x)

        # ===== Decoder Path =====
        x = self.up_r2(x)
        # Concatenate with skip connection
        x = jnp.concatenate([y2, x], axis=1)
        x = self.conv_r2(x)

        x = self.up_r1(x)
        # Concatenate with skip connection
        x = jnp.concatenate([y1, x], axis=1)
        x = self.conv_r1(x)

        x = self.up_r0(x)
        # Concatenate with skip connection
        x = jnp.concatenate([y0, x], axis=1)
        x = self.conv_r00(x)
        x = self.conv_r01(x)

        # ===== Final Residual Connection =====
        # Displacement field
        displacement = (x + x0) * 6.
        
        return displacement
