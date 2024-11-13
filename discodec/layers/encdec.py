import math
from jax import random
from flax import linen as nn
from typing import Sequence, Callable

from discodec.layers.convs import Snake1d, WNConv, WNConvTranspose, ConvNeXtBlock
from discodec.layers.attention import LocalMHA


class ResidualUnit(nn.Module):
    dim: int = 16
    dilation: int = 1
    pad: bool = True
    groups: int = 1
    kernel: int = 7

    @nn.compact
    def __call__(self, x):

        pad = ((self.kernel - 1) * self.dilation) // 2

        y = Snake1d(name='snake_0')(x)
        y = WNConv(features=self.dim, kernel_size=(self.kernel,), padding=0 if not self.pad else (pad,), kernel_dilation=(self.dilation,),
                   feature_group_count=self.groups, name='conv_1')(y)
        y = Snake1d(name='snake_2')(y)
        y = WNConv(features=self.dim, kernel_size=(1,), name='conv_3', padding=0)(y)

        pad = (x.shape[-2] - y.shape[-2]) // 2
        if pad > 0:
            x = x[..., pad:-pad, :]

        return x + y



class EncoderBlock(nn.Module):
    dim: int = 16
    stride: int = 1
    pad: bool = True
    groups: int = 1

    @nn.compact
    def __call__(self, x):
        x = ResidualUnit(self.dim // 2, dilation=1, pad=self.pad, groups=self.groups, name='res_unit_0')(x)
        x = ResidualUnit(self.dim // 2, dilation=3, pad=self.pad, groups=self.groups, name='res_unit_1')(x)
        x = ResidualUnit(self.dim // 2, dilation=9, pad=self.pad, groups=self.groups, name='res_unit_2')(x)

        x = Snake1d(name='snake_3')(x)

        x = WNConv(features=self.dim,
                    kernel_size=(2*self.stride,),
                    strides=(self.stride,),
                    padding=0 if not self.pad else (math.ceil(self.stride / 2),),
                    name='conv_4')(x)

        return x


class Encoder(nn.Module):
    d_model: int = 64
    strides: Sequence[int] = (3, 3, 7, 7)
    d_latent: int = 64
    pad: bool = True
    depthwise: bool = False

    @nn.compact
    def __call__(self, x):
        d_model = self.d_model
        x = WNConv(features=d_model, kernel_size=(7,), padding=0 if not self.pad else (3,), name='conv_0')(x)
        for i, stride in enumerate(self.strides):
            d_model *= 2
            groups = d_model // 2 if self.depthwise else 1
            x = EncoderBlock(d_model, stride, pad=self.pad, groups=groups, name=f'block_{i+1}',)(x)

            x = ConvNeXtBlock(d_model*2, layer_scale_init_value=1/len(self.strides))(x)
        
        x = LocalMHA()(x)

        groups = d_model if self.depthwise else 1
        x = WNConv(features=d_model, kernel_size=(7,), padding=0 if not self.pad else (3,),
                   feature_group_count=groups, name=f'conv_{i+3}')(x)

        return x


class NoiseBlock(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        rng = self.make_rng('noise')
        b, t, c = x.shape
        noise = random.normal(rng, (b, t, 1))
        h = WNConv(features=self.dim, kernel_size=(1,), use_bias=False, name='conv_0')(x)
        n = noise * h
        x = x + n
        return x


class DecoderBlock(nn.Module):
    output_dim: int = 8
    stride: int = 1
    noise: bool = False
    groups: int = 1

    @nn.compact
    def __call__(self, x, train=False):
        x = Snake1d(name='snake_0')(x)

        torch_padding = math.ceil(self.stride / 2)
        padding = torch_padding + self.stride - 1

        x = WNConvTranspose(features=self.output_dim,
                             kernel_size=(2 * self.stride,),
                             strides=(self.stride,),
                             padding=(padding,),
                             transpose_kernel=True,
                             name='conv_t_1')(x)

        if self.noise and train:
            x = NoiseBlock(self.output_dim, name='noise')(x)
        
        x = ResidualUnit(self.output_dim, dilation=1, groups=self.groups, name='res_unit_2')(x)
        x = ResidualUnit(self.output_dim, dilation=3, groups=self.groups, name='res_unit_3')(x)
        x = ResidualUnit(self.output_dim, dilation=9, groups=self.groups, name='res_unit_4')(x)
        return x


class Decoder(nn.Module):
    channels: int
    rates: list
    d_out: int = 1
    output_act: Callable = nn.tanh
    noise: bool = False
    depthwise: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        in_channels = x.shape[-1]

        if self.depthwise:
            x = WNConv(features=in_channels, kernel_size=(7,), padding=(3,),
                       feature_group_count=in_channels, name='conv_0')(x)
            x = WNConv(features=self.channels, kernel_size=(1,))(x)
        else:
            x = WNConv(features=self.channels, kernel_size=(7,), padding=(3,), name='conv_0')(x)

        x = LocalMHA()(x)
        
        for i, stride in enumerate(self.rates):
            x = ConvNeXtBlock(x.shape[-1]*2, layer_scale_init_value=1/len(self.rates))(x)
            
            output_dim = self.channels // 2 ** (i + 1)
            groups = output_dim if self.depthwise else 1
            x = DecoderBlock(output_dim, stride, noise=self.noise, groups=groups,
                             name=f'block_{1+i}')(x, train=train)

        x = Snake1d(name=f'snake_{i+2}')(x)
        x = WNConv(features=self.d_out, kernel_size=(7,), padding=(3,), name=f'conv_{i+3}')(x)

        x = self.output_act(x)

        return x
