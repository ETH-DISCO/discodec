import math

from jax import numpy as jnp
from flax import linen as nn

from typing import Sequence, Union, Callable

from discodec.layers.encdec import Encoder, Decoder
from discodec.layers.rvq import ResidualVectorQuantize

class Model(nn.Module):
    encoder_dim: int = 64
    encoder_rates: Sequence[int] = (3, 3, 7, 7)
    init_latent_dim: int = None
    decoder_dim: int = 1536
    decoder_rates: Sequence[int] = (7, 7, 3, 3)
    vq_strides: Sequence[int] = (8, 4, 2, 1)
    codebook_size: int = 1024
    codebook_dim: Union[int, list] = 8
    quantizer_dropout: float = 0.0
    sample_rate: int = 44100
    decoder_output_act: Callable = nn.tanh
    noise: bool = True
    depthwise: bool = True
    attn_window_size: int = None

    def setup(self):
        if self.init_latent_dim is None:
            self.latent_dim = self.encoder_dim * (2 ** len(self.encoder_rates))
        else:
            self.latent_dim = self.init_latent_dim
        
        self.n_codebooks = len(self.vq_strides)
        self.hop_length = math.prod(self.encoder_rates)

        self.encoder = Encoder(self.encoder_dim,
                                self.encoder_rates,
                                self.latent_dim,
                                depthwise=self.depthwise)
        
        self.quantizer = ResidualVectorQuantize(
            input_dim=self.latent_dim,
            vq_strides=self.vq_strides,
            codebook_size=self.codebook_size,
            init_codebook_dim=self.codebook_dim,
            quantizer_dropout=self.quantizer_dropout,
        )
        
        self.decoder = Decoder(
            self.decoder_dim,
            self.decoder_rates,
            output_act=self.decoder_output_act,
            noise=self.noise,
            depthwise=self.depthwise,
        )
        
    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate, "Sample rate mismatch"

        length = audio_data.shape[-2]

        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = jnp.pad(audio_data, ((0, 0), (0, right_pad), (0, 0)))

        return audio_data

    def encode(self, audio_data, n_quantizers: int = None, training: bool = False):
        z = self.encoder(audio_data)
        
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers, training=training
        )

        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z, training: bool = False):
        return self.decoder(z, train=training)
    
    def __call__(self, audio_data, sample_rate: int = None, n_quantizers: int = None, training: bool = False):
        length = audio_data.shape[-2]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(audio_data, n_quantizers, training)
        x = self.decode(z, training)

        return {
            "audio": x[..., :length, :],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }