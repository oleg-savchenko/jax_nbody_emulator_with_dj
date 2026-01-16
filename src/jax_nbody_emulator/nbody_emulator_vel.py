"""
Main N-body emulator model implementation.

This module contains the primary NBodyEmulatorVel class that implements
a 3D U-Net-like architecture for cosmological
N-body simulations.

This is the "Vel" (velocity) version that computes both outputs and their derivatives
w.r.t. the input parameter (growth factor) Dz using manual forward-mode automatic
differentiation. Used for predicting both displacement and velocity fields simultaneously.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .cosmology import D
from .blocks_vel import ResNetBlock3DVel, ResampleBlock3DVel

def _modulate_weights_vel(style_weight, style_bias, weight, s, dx=None, eps = 1.e-8) :

        if s.ndim == 1:
            s = s[None]

        s = jnp.array(s, dtype=style_weight.dtype)
        
        s_mod = jnp.dot(s, style_weight.T) + style_bias
        # s_mod: (B, C_in) -> (B, 1, C_in, 1, 1, 1)
        s_mod = s_mod[:, None, :, None, None, None]
        ds = jnp.zeros_like(s).at[:, 1].set(1.0)
        ds_mod = jnp.dot(ds, style_weight.T)
        ds_mod = ds_mod[:, None, :, None, None, None]
        
        # w: (C_out, C_in, K, K, K) -> (B, C_out, C_in, K, K, K)
        w = weight[None] * s_mod
        dw_style = weight[None] * ds_mod
        
        # Demodulation (normalize over spatial + input channels)
        norm = jnp.sqrt(jnp.sum(w**2, axis=(2,3,4,5), keepdims=True) + eps)
        dnorm = -jnp.sum(w * dw_style, axis=(2,3,4,5), keepdims=True) / (norm**3)
        
        w_normalized = w / norm
        dw_normalized = (dw_style / norm) + (w * dnorm)

        if dx is None:
            # First layer handling
            Dz = (s[:, 1] + 1.0)[:, None, None, None, None, None]
            dw_normalized = dw_normalized + w_normalized / Dz
        else:
            dw_normalized = dw_normalized
        
        return w_normalized, dw_normalized
    
def modulate_emulator_parameters_vel(params, z, Om, eps = 1.e-8, dtype=jnp.float32):
    """
    Preprocess all network parameters for fixed (z, Om).
    
    Returns new params dict with modulated weights.
    """
    # Compute style vector
    #s0 = (Om - 0.3) * 5.
    #s1 = D(jnp.array([z]), jnp.array([Om]))[0] - 1.
    #s = jnp.array([[s0, s1]])

    Dz = D(z, Om)
    
    # Compute style vector
    s0 = (Om - 0.3) * 5.
    s1 = Dz - 1.
    s = jnp.stack([s0, s1], axis=-1)
    
    # Process each block
    processed_params = {'params':{}}
    # Process each block
    for block_name, block_params in params['params'].items():
        processed_params['params'][block_name] = {}
        for layer_name, layer_params in block_params.items():
            if 'style_weight' in layer_params:
                # Only the first block's conv_0 and skip layers have input that is linear in Dz
                if block_name == 'conv_l00' and (layer_name == 'conv_0' or layer_name == 'skip') :
                    dx = None
                else :
                    dx = 1
                # This layer has style modulation - preprocess it
                w_norm, dw_norm = _modulate_weights_vel(
                    layer_params['style_weight'],
                    layer_params['style_bias'],
                    layer_params['weight'],
                    s,
                    dx=dx,
                    eps=eps
                )
                processed_params['params'][block_name][layer_name] = {
                    'weight': w_norm[0].astype(dtype),
                    'dweight': dw_norm[0].astype(dtype),
                    'bias': layer_params['bias'].astype(dtype)
                }
            else:
                # Pass through unmodified
                print(f'skipping {block_name} {layer_name}')
                processed_params['params'][block_name][layer_name] = layer_params

    return processed_params

class NBodyEmulatorVel(nn.Module):
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
        self.conv_l00 = ResNetBlock3DVel(
            'CACA', self.in_chan, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_l01 = ResNetBlock3DVel(
            'CACA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.down_l0 = ResampleBlock3DVel(
            'DA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.conv_l1 = ResNetBlock3DVel(
            'CACA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.down_l1 = ResampleBlock3DVel(
            'DA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.conv_l2 = ResNetBlock3DVel(
            'CACA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.down_l2 = ResampleBlock3DVel(
            'DA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        # Bottleneck
        self.conv_c = ResNetBlock3DVel(
            'CACA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        # Decoder Path (Upsampling)
        self.up_r2 = ResampleBlock3DVel(
            'UA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r2 = ResNetBlock3DVel(
            'CACA', mid_chan_2, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.up_r1 = ResampleBlock3DVel(
            'UA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r1 = ResNetBlock3DVel(
            'CACA', mid_chan_2, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.up_r0 = ResampleBlock3DVel(
            'UA', mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r00 = ResNetBlock3DVel(
            'CACA', mid_chan_2, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r01 = ResNetBlock3DVel(
            'CAC', mid_chan_1, self.out_chan, eps=self.eps, dtype=self.dtype
        )
    
    def __call__(self, x, Dz, vel_fac):
        """
        Forward pass of the NBodyEmulator.
        
        Args:
            x: Input 3D data tensor of shape (B, C_in, D, H, W)
            Dz: Linear growth factors at output redshifts of shape (B,)
            vel_fac: f(z) D(z) H(z) / (1 + z), Linear velocity factors at output redshifts of shape (B,)
            
        Returns:
            Tuple of (displacement, velocity):
                - displacement: Predicted displacement field (B, C_out, D', H', W')
                - velocity: Predicted velocity field (B, C_out, D', H', W')
        """
        
        Dz = Dz.astype(self.dtype)[:, None, None, None, None]
        vel_fac = vel_fac.astype(self.dtype)[:, None, None, None, None]
        
        # Apply growth factor scaling to input
        # Factor of 6 is from original model input normalization
        x = x * (Dz / 6.)
        dx = None  # Will be computed by first layer
        
        # Store cropped input for final residual connection
        # Assumes input spatial dimensions are large enough (e.g., 256^3)
        x0 = x[:, :, 48:-48, 48:-48, 48:-48]
        
        # ===== Encoder Path =====
        x, dx = self.conv_l00(x, dx)
        y0, dy0 = self.conv_l01(x, dx)
        
        x, dx = self.down_l0(y0, dy0)
        # Crop skip connection to match decoder dimensions
        y0 = y0[:, :, 40:-40, 40:-40, 40:-40]
        dy0 = dy0[:, :, 40:-40, 40:-40, 40:-40]
        
        y1, dy1 = self.conv_l1(x, dx)
        x, dx = self.down_l1(y1, dy1)
        # Crop skip connection
        y1 = y1[:, :, 16:-16, 16:-16, 16:-16]
        dy1 = dy1[:, :, 16:-16, 16:-16, 16:-16]
        
        y2, dy2 = self.conv_l2(x, dx)
        x, dx = self.down_l2(y2, dy2)
        # Crop skip connection
        y2 = y2[:, :, 4:-4, 4:-4, 4:-4]
        dy2 = dy2[:, :, 4:-4, 4:-4, 4:-4]
        
        # ===== Bottleneck =====
        x, dx = self.conv_c(x, dx)

        # ===== Decoder Path =====
        x, dx = self.up_r2(x, dx)
        # Concatenate with skip connection
        x = jnp.concatenate([y2, x], axis=1)
        dx = jnp.concatenate([dy2, dx], axis=1)
        x, dx = self.conv_r2(x, dx)

        x, dx = self.up_r1(x, dx)
        # Concatenate with skip connection
        x = jnp.concatenate([y1, x], axis=1)
        dx = jnp.concatenate([dy1, dx], axis=1)
        x, dx = self.conv_r1(x, dx)

        x, dx = self.up_r0(x, dx)
        # Concatenate with skip connection
        x = jnp.concatenate([y0, x], axis=1)
        dx = jnp.concatenate([dy0, dx], axis=1)
        x, dx = self.conv_r00(x, dx)
        x, dx = self.conv_r01(x, dx)

        # ===== Final Residual Connection =====
        # Displacement field
        displacement = (x + x0) * 6.
        
        # Velocity field (includes derivative w.r.t. Dz)
        velocity = (dx + x0 / Dz) * vel_fac * 6.
        
        return displacement, velocity
