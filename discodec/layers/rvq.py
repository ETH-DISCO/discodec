import jax
from jax import numpy as jnp
from flax import linen as nn
from einops import rearrange
from typing import Sequence, Union

from discodec.layers.convs import WNConv


def normalize(x, axis=-1, ord=2, eps=1e-12):
    x_norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    return x / jnp.maximum(x_norm, eps)


def squared_err(x, y):
    return (x - y) ** 2


def detach(x):
    return jax.lax.stop_gradient(x)


class VectorQuantize(nn.Module):
    input_dim: int
    codebook_size: int
    codebook_dim: int
    stride: int = 1
    sync_nu: float = 0.0
    num_groups: int = 1

    def setup(self):
        self.codebook = nn.Embed(
            num_embeddings=self.codebook_size, features=self.codebook_dim, name="embed"
        )
        self.in_proj = WNConv(
            features=self.codebook_dim, kernel_size=(1,)
        )
        self.out_proj = WNConv(
            features=self.input_dim, kernel_size=(1,)
        )

        self.code_bias = self.param(
            "code_bias",
            lambda key, shape: jnp.zeros(shape),
            (self.num_groups, self.codebook_dim),
        )
        self.code_scale = self.param(
            "code_scale",
            lambda key, shape: jnp.ones(shape),
            (self.num_groups, self.codebook_dim),
        )

        self.codebook_usage = self.variable(
            "codebook_stats",
            "usage",
            lambda s: jnp.zeros(s, jnp.float32),
            (self.codebook_size,),
        )

    def __call__(self, x, training: bool = False):
        if self.stride > 1:
            x = nn.avg_pool(
                x, window_shape=(self.stride,), strides=(self.stride,)
            )

        z_e = self.in_proj(x)

        z_q, indices = self.encode_latents(z_e)

        commitment_loss = squared_err(z_e, detach(z_q)).mean((1, 2))
        codebook_loss = squared_err(z_q, detach(z_e)).mean((1, 2))

        z_q = (
            z_e + detach(z_q - z_e) + (self.sync_nu * z_q) + detach(-self.sync_nu * z_q)
        )

        z_q = self.out_proj(z_q)

        if self.stride > 1:
            z_q = jnp.repeat(z_q, self.stride, axis=-2)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def decode_code(self, x):
        embed = self.codebook(x)

        unique_indices, counts = jnp.unique(
            x, return_counts=True, size=self.codebook_size, fill_value=-1
        )
        usage = jnp.zeros(self.codebook_size, jnp.float32)
        usage = usage.at[unique_indices].add(counts)
        self.codebook_usage.value = usage

        return embed

    def out_proj_decoded(self, x):
        return self.out_proj(x)

    def encode_latents(self, latents):
        encodings = rearrange(latents, "b t d -> (b t) d")
        codebook = self.codebook.embedding

        encodings = normalize(encodings, axis=1)
        codebook = normalize(codebook, axis=1)

        codebook = rearrange(codebook, "(n g) d -> n g d", g=self.num_groups)

        codebook = codebook * self.code_scale + self.code_bias

        codebook = rearrange(codebook, "n g d -> (n g) d")

        dist = (
            (encodings**2).sum(axis=1, keepdims=True)
            - 2 * encodings @ codebook.T
            + (codebook**2).sum(axis=1, keepdims=True).T
        )

        indices = rearrange((-dist).argmax(axis=1), "(b t) -> b t", b=latents.shape[0])

        z_q = self.decode_code(indices)

        return z_q, indices



prefix = "vq"


class ResidualVectorQuantize(nn.Module):
    input_dim: int = 512
    vq_strides: Sequence[int] = (1, 1, 1, 1)
    codebook_size: int = 1024
    init_codebook_dim: Union[int, list] = 8
    quantizer_dropout: float = 0.0
    precision: str = "float32"

    def setup(self):
        self.n_codebooks = len(self.vq_strides)

        if isinstance(self.init_codebook_dim, int):
            self.codebook_dim = [self.init_codebook_dim] * self.n_codebooks
        else:
            self.codebook_dim = self.init_codebook_dim

        self.vqs = [
            VectorQuantize(
                input_dim=self.input_dim,
                codebook_size=self.codebook_size,
                codebook_dim=self.codebook_dim[i],
                stride=self.vq_strides[i],
                sync_nu=1.0,
                num_groups=8,
                name=f"{prefix}_{i}",
            )
            for i in range(self.n_codebooks)
        ]

    def __call__(self, z, n_quantizers: int = None, training: bool = False):
        z_q = 0
        residual = z
        commitment_loss, codebook_loss = 0, 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        if training and self.quantizer_dropout > 0:
            rng = self.make_rng("quant_dropout")
            n_quantizers = (
                jnp.ones((z.shape[0],), dtype=jnp.int32) * self.n_codebooks + 1
            )
            dropout = jax.random.randint(rng, (z.shape[0],), 1, self.n_codebooks + 1)
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers = n_quantizers.at[:n_dropout].set(dropout[:n_dropout])

        for i, quantizer in enumerate(self.vqs):
            if not training and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices, z_e_i = quantizer(
                residual, training=training
            )

            mask = jnp.full((z.shape[0],), fill_value=i, dtype=jnp.int32) < n_quantizers

            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices)
            latents.append(z_e_i)

        return z_q, codebook_indices, latents, commitment_loss, codebook_loss
