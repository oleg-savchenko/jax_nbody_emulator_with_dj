"""
3D convolutional layers.

This module contains the basic building blocks for 3D convolutions,
including the base class and specific layer implementations.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import vmap
from functools import partial

class ConvBase3D(nn.Module):
    """
    Base class for standard 3D convolutions.
    """
    in_chan: int
    out_chan: int
    kernel_size: int = 3
    stride: int = 1
    eps: float = 1e-8
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input (B, C_in, D, H, W)
            
        Returns:
            y: Output and tangent
        """
        
        # Convolution weights
        kernel_shape = (self.kernel_size,) * 3
        weight = self.param('weight',
                           nn.initializers.lecun_normal(),
                           (self.out_chan, self.in_chan, *kernel_shape),
                           self.dtype)
        bias = self.param('bias',
                         nn.initializers.zeros,
                         (self.out_chan,),
                         self.dtype)
                
        # Convolution using vmap        
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
        else :
            # Vectorize over batch
            y = vmap(single_conv_b, in_axes=(0, None, None))(x, weight, bias)
        
        return y


class ConvTransposeBase3D(nn.Module):
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
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input (B, C_in, D, H, W)
            
        Returns:
            y: Upsampled output and tangent
        """
        
        # Weights
        kernel_shape = (self.kernel_size,) * 3
        weight = self.param('weight',
                           nn.initializers.lecun_normal(),
                           (self.out_chan, self.in_chan, *kernel_shape),
                           self.dtype)
        bias = self.param('bias',
                         nn.initializers.zeros,
                         (self.out_chan,),
                         self.dtype)
        
        # Upsampling convolution (using lhs_dilation)        
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
        else :
            # Vectorize over batch
            y = vmap(single_upsample_conv_b, in_axes=(0, None, None))(x, weight, bias)
        
        return y
    
class LeakyReLU(nn.Module):
    """Leaky ReLU."""
    negative_slope: float = 0.01
    dtype: jnp.dtype = jnp.float32
    
    def __call__(self, x):
        slope = jnp.array(self.negative_slope, dtype=self.dtype)
        return jax.nn.leaky_relu(x, negative_slope=slope)

# Specialized layers using partial
Conv3D = partial(ConvBase3D, kernel_size=3, stride=1)
Skip3D = partial(ConvBase3D, kernel_size=1, stride=1)
DownSample3D = partial(ConvBase3D, kernel_size=2, stride=2)
UpSample3D = ConvTransposeBase3D  # No partial needed, defaults are correct
