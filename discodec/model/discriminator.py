from functools import partial
from jax import numpy as jnp
from flax import linen as nn
from einops import rearrange
from typing import Sequence

from discodec.layers.convs import WNConv
from discodec.utils.utils import stft, as_real


act = partial(nn.leaky_relu, negative_slope=0.1)


class MPD(nn.Module):
    period: int = 256

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = jnp.pad(x, ((0, 0), (0, 0), (0, self.period - t % self.period)), mode='reflect')
        return x

    @nn.compact
    def __call__(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b l p c", p=self.period)

        wn_conv2ds = [
            WNConv(features=32, kernel_size=(5, 1), strides=(3, 1),
                                padding=((2,2), (0,0))),
            WNConv(features=128, kernel_size=(5, 1), strides=(3, 1),
                                padding=((2,2), (0,0))),
            WNConv(features=512, kernel_size=(5, 1), strides=(3, 1),
                                padding=((2,2), (0,0))),
            WNConv(features=1024, kernel_size=(5, 1), strides=(3, 1),
                                padding=((2,2), (0,0))),
            WNConv(features=1024, kernel_size=(5, 1), strides=1,
                                padding=((2,2), (0,0)))
        ]
        
        for conv in wn_conv2ds:
            x = act(conv(x))
            fmap.append(x)
        
        x = WNConv(features=1, kernel_size=(3, 1), padding=((1,1), (0,0)))(x)
        fmap.append(x)

        return fmap
        
class MSD(nn.Module):
    rate: int = 1
    sample_rate: int = 44100

    @nn.compact
    def __call__(self, x):
        raise NotImplementedError()

BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))

class MRD(nn.Module):
    window_length: int
    hop_factor: float = 0.25
    sample_rate: int = 44100
    bands_ratios: list = BANDS

    def setup(self):
        self.hop_length=int(self.window_length * self.hop_factor)
        n_fft = self.window_length // 2 + 1
        self.bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in self.bands_ratios]

        ch = 32
        convs = \
            [WNConv(features=ch, kernel_size=(3, 9), strides=(1, 1), padding=((1,1), (4,4))), act] + \
            [WNConv(features=ch, kernel_size=(3, 9), strides=(1, 2), padding=((1,1), (4,4))), act]*3 + \
            [WNConv(features=ch, kernel_size=(3, 3), strides=(1, 1), padding=((1,1), (1,1))), act]
        
        self.convs = nn.Sequential(convs)

        self.band_convs = [convs for _ in range(len(self.bands))]
        self.conv_post = WNConv(features=1, kernel_size=(3, 3), strides=(1, 1), padding=((1,1), (1,1)))
    
    def spectrogram(self, audio):
        _, _, spec = stft(audio, self.window_length, self.hop_length, True, self.sample_rate, axis=-1)
        spec = as_real(spec)

        spec = rearrange(spec, "b 1 f t c -> (b 1) t f c")

        spec_bands = [spec[..., start:end, :] for start, end in self.bands]

        return spec_bands

    def __call__(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)
        
        x = jnp.concatenate(x, axis=-2)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap

class Discriminator(nn.Module):
    rates: Sequence[int] = ()
    periods: Sequence[int] = (2, 3, 5, 7, 11)
    fft_sizes: Sequence[int] = (2048, 1024, 512)
    sample_rate: int = 44100
    bands: Sequence[tuple] = BANDS

    def setup(self):
        discs = []
        discs += [MPD(p) for p in self.periods]
        discs += [MSD(r, sample_rate=self.sample_rate) for r in self.rates]
        discs += [MRD(f, sample_rate=self.sample_rate, bands_ratios=self.bands) for f in self.fft_sizes]
        self.discriminators = discs
    
    def preprocess(self, y):
        y = y - y.mean(axis=-1, keepdims=True)
        y = 0.8 * y / (jnp.abs(y).max(axis=-1, keepdims=True) + 1e-9)

        return y

    def __call__(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps