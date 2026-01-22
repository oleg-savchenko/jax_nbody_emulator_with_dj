"""
3D convolutional layers with manual forward-mode AD.

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
from jax import vmap
from functools import partial

class ConvBase3DVel(nn.Module):
    """
    Base class for standard 3D convolutions.
    
    Handles weight modulation, demodulation, and standard convolution.
    Simpler, JAX-idiomatic implementation without explicit shape tracking.
    """
    in_chan: int
    out_chan: int
    kernel_size: int = 3
    stride: int = 1
    eps: float = 1e-8
    
    @nn.compact
    def __call__(self, x, dx=None):
        """
        Args:
            x: Input (B, C_in, D, H, W)
            dx: Optional tangent (same shape as x)
            
        Returns:
            (y, dy): Output and tangent
        """
        
        # Convolution weights
        kernel_shape = (self.kernel_size,) * 3
        weight = self.param('weight',
                           nn.initializers.lecun_normal(),
                           (self.out_chan, self.in_chan, *kernel_shape))
        bias = self.param('bias',
                         nn.initializers.zeros,
                         (self.out_chan,))
        dweight = self.param('dweight',
                           nn.initializers.lecun_normal(),
                           (self.out_chan, self.in_chan, *kernel_shape))
    
        weight = weight.astype(x.dtype)
        dweight = dweight.astype(x.dtype)
        bias = bias.astype(x.dtype)
            
        # Convolution using vmap
        def single_conv(x_i, w_i):
            out = jax.lax.conv_general_dilated(
                lhs=x_i[None],
                rhs=w_i,
                window_strides=(self.stride,) * 3,
                padding='VALID',
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
            )
            return out[0]
        
        def single_conv_b(x_i, w_i, b_i):
            out = jax.lax.conv_general_dilated(
                lhs=x_i[None],
                rhs=w_i,
                window_strides=(self.stride,) * 3,
                padding='VALID',
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
            )
            return out[0] + b_i[:,None,None,None]
        
        
        if x.shape[0] == 1 :
            y = single_conv_b(x[0], weight, bias)[None]
            dy = single_conv(x[0], dweight)[None]
        else :
            # Vectorize over batch
            y = vmap(single_conv_b, in_axes=(0, None, None))(x, weight, bias)
            dy = vmap(single_conv, in_axes=(0, None))(x, dweight)

        if dx is not None:
            if x.shape[0] == 1 :
                dy = dy + single_conv(dx[0], weight)[None]
            else :
                dy = dy + vmap(single_conv, in_axes=(0, None))(dx, weight)
        
        return y, dy


class ConvTransposeBase3DVel(nn.Module):
    """
    Base class for upsampling 3D convolutions.
    
    Uses lhs_dilation for efficient upsampling. Avoids checkerboard artifacts
    and is more numerically stable than true transposed convolution.
    """
    in_chan: int
    out_chan: int
    kernel_size: int = 2
    stride: int = 1
    eps: float = 1e-8
    
    @nn.compact
    def __call__(self, x, dx=None):
        """
        Args:
            x: Input (B, C_in, D, H, W)
            dx: Optional tangent (same shape as x)
            
        Returns:
            (y, dy): Upsampled output and tangent
        """
        
        # Weights
        kernel_shape = (self.kernel_size,) * 3
        weight = self.param('weight',
                           nn.initializers.lecun_normal(),
                           (self.out_chan, self.in_chan, *kernel_shape))
        bias = self.param('bias',
                         nn.initializers.zeros,
                         (self.out_chan,))
        dweight = self.param('dweight',
                           nn.initializers.lecun_normal(),
                           (self.out_chan, self.in_chan, *kernel_shape))
        
        weight = weight.astype(x.dtype)
        dweight = dweight.astype(x.dtype)
        bias = bias.astype(x.dtype)

        # Upsampling convolution (using lhs_dilation)
        def single_upsample_conv(x_i, w_i):
            out = jax.lax.conv_general_dilated(
                lhs=x_i[None],
                rhs=w_i,
                window_strides=(self.stride,) * 3,
                padding=((1, 1),) * 3,  # Computed here
                lhs_dilation=(2, 2, 2),  # Computed here
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
            )
            return out[0]
        
        def single_upsample_conv_b(x_i, w_i, b_i):
            out = jax.lax.conv_general_dilated(
                lhs=x_i[None],
                rhs=w_i,
                window_strides=(self.stride,) * 3,
                padding=((1, 1),) * 3,  # Computed here
                lhs_dilation=(2, 2, 2),  # Computed here
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
            )
            return out[0] + b_i[:,None,None,None]
        
        if x.shape[0] == 1 :
            y = single_upsample_conv_b(x[0], weight, bias)[None]
            dy = single_upsample_conv(x[0], dweight)[None]
        else :
            # Vectorize over batch
            y = vmap(single_upsample_conv_b, in_axes=(0, None, None))(x, weight, bias)
            dy = vmap(single_upsample_conv, in_axes=(0, None))(x, dweight)
        
        if dx is not None:
            if x.shape[0] == 1 :
                dy = dy + single_upsample_conv(dx[0], weight)[None]
            else :
                dy = dy + vmap(single_upsample_conv, in_axes=(0, None))(dx, weight)
        
        return y, dy
    
class LeakyReLUVel(nn.Module):
    """Leaky ReLU with tangent computation."""
    negative_slope: float = 0.01
    
    def __call__(self, x, dx):
        slope = jnp.array(self.negative_slope, x.dtype)
        y = jax.nn.leaky_relu(x, negative_slope=slope)
        dy = jnp.where(x > 0, dx, slope * dx)
        return y, dy

# Specialized layers using partial
Conv3DVel = partial(ConvBase3DVel, kernel_size=3, stride=1)
Skip3DVel = partial(ConvBase3DVel, kernel_size=1, stride=1)
DownSample3DVel = partial(ConvBase3DVel, kernel_size=2, stride=2)
UpSample3DVel = ConvTransposeBase3DVel
