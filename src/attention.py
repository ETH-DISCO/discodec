from jax import numpy as jnp
from flax import linen as nn
from einops import rearrange
import functools


class LocalMHA(nn.Module):
    head_dim: int = 64
    use_rotary_pos_emb: bool = True

    @nn.compact
    def __call__(self, x, windows=1):
        dim = x.shape[-1]
        residual = x
        n_heads = dim // self.head_dim

        x = nn.LayerNorm()(x)
        qkv_proj = functools.partial(
            nn.DenseGeneral,
            features=(n_heads, self.head_dim),
            axis=-1,
            use_bias=False,
        )
        q, k, v = (
            qkv_proj(name='q_proj')(x),
            qkv_proj(name='k_proj')(x),
            qkv_proj(name='v_proj')(x),
        )
        
        q, k, v = map(lambda t: rearrange(t, "b (w n) h d -> b h w n d", w=windows), (q, k, v))

        if self.use_rotary_pos_emb:
            pos_emb, scale = SinusoidalEmbeddings(self.head_dim, scale_base=16)(k)
            q, k = apply_rotary_pos_emb(q, k, pos_emb, scale)

        q, k, v = map(lambda t: rearrange(t, "b h w n d -> b w n h d"), (q, k, v))

        attn = nn.dot_product_attention(q, k, v)
        attn = rearrange(attn, "b w n h d -> b (w n) (h d)")
        out = nn.Dense(dim, use_bias=False)(attn) + residual

        return out


class SinusoidalEmbeddings(nn.Module):
    dim: int
    scale_base: int = None
    use_xpos: bool = False
    dtype = jnp.float32

    def setup(self):
        assert not (self.use_xpos and self.scale_base is None), "scale base must be defined if using xpos"
        self.inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2).astype(self.dtype) / self.dim))
        self.scale = (jnp.arange(0, self.dim, 2) + 0.4 * self.dim) / (1.4 * self.dim)
    
    def __call__(self, x):
        length = x.shape[-2]
        t = jnp.arange(length).astype(self.dtype)
        freqs = jnp.einsum("i , j -> i j", t, self.inv_freq)
        freqs = jnp.concatenate([freqs, freqs], axis=-1)
        if not self.use_xpos:
            return freqs, jnp.ones(1)

        power = (t - (length // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = jnp.concatenate([scale, scale], axis=-1)
        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "b ... (r d) -> b ... r d", r=2)
    x1 = x[..., 0, :]
    x2 = x[..., 1, :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, freqs, scale=1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    inv_scale = scale**-1
    if scale.ndim == 2:
        scale = scale[-q_len:, :]
    q = (q * jnp.cos(q_freqs) * scale) + (rotate_half(q) * jnp.sin(q_freqs) * scale)
    k = (k * jnp.cos(freqs) * inv_scale) + (rotate_half(k) * jnp.sin(freqs) * inv_scale)
    return q, k