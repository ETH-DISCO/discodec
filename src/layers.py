import dataclasses
import jax
from jax import numpy as jnp
from flax import linen as nn

from einops import rearrange

from flax.typing import (
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Axes,
)
from typing import Iterable, Optional, Callable
from flax.linen import dtypes, transforms
from flax.linen.normalization import _canonicalize_axes, _l2_normalize


default_kernel_init = nn.initializers.truncated_normal(stddev=0.02)


def WNConv(*args, **kwargs):
    kernel_init = kwargs.pop('kernel_init', default_kernel_init)
    layer = nn.Conv(*args, kernel_init=kernel_init, **kwargs)
    return WeightNorm(layer)


def WNConvTranspose(*args, **kwargs):
    kernel_init = kwargs.pop('kernel_init', default_kernel_init)
    layer = nn.ConvTranspose(*args, kernel_init=kernel_init, **kwargs)
    return WeightNorm(layer)


def WNDense(*args, **kwargs):
    kernel_init = kwargs.pop('kernel_init', default_kernel_init)
    layer = nn.Dense(*args, kernel_init=kernel_init, **kwargs)
    return WeightNorm(layer)


def snake(x, alpha):
    shape = x.shape
    x = rearrange(x, 'n ... c -> n (...) c')

    x = x + 1.0 / (alpha + 1e-9) * jnp.sin(alpha * x)**2
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', lambda key, shape: jnp.ones(shape), (1, 1, x.shape[-1]))
        return snake(x, alpha)


class WeightNorm(nn.Module):
    layer_instance: nn.Module
    epsilon: float = 1e-12
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_scale: bool = True
    feature_axes: Optional[Axes] = -1
    variable_filter: Optional[Iterable] = dataclasses.field(
      default_factory=lambda: {'kernel'}
    )

    @nn.compact
    def __call__(self, *args, **kwargs):
      """Compute the l2-norm of the weights in ``self.layer_instance``
      and normalize the weights using this value before computing the
      ``__call__`` output.

      Args:
        *args: positional arguments to be passed into the call method of the
          underlying layer instance in ``self.layer_instance``.
        **kwargs: keyword arguments to be passed into the call method of the
          underlying layer instance in ``self.layer_instance``.

      Returns:
        Output of the layer using l2-normalized weights.
      """

      def layer_forward(layer_instance):
        return layer_instance(*args, **kwargs)

      return transforms.map_variables(
        layer_forward,
        trans_in_fn=lambda vs: jax.tree_util.tree_map_with_path(
          self._l2_normalize,
          vs,
        ),
        init=self.is_initializing(),
      )(self.layer_instance)


    def _l2_normalize(self, path, vs):
      """Compute the l2-norm and normalize the variables ``vs`` using this
      value. This is intended to be a helper function used in this Module's
      ``__call__`` method in conjunction with ``nn.transforms.map_variables``
      and ``jax.tree_util.tree_map_with_path``.

      Args:
        path: dict key path, used for naming the ``scale`` variable
        vs: variables to be l2-normalized
      """
      value = jnp.asarray(vs)
      str_path = (
        self.layer_instance.name
        + '/'
        + '/'.join((dict_key.key for dict_key in path[1:]))
      )
      if self.variable_filter:
        for variable_name in self.variable_filter:
          if variable_name in str_path:
            break
        else:
          return value

      if self.feature_axes is None:
        feature_axes = ()
        reduction_axes = tuple(i for i in range(value.ndim))
      else:
        feature_axes = _canonicalize_axes(value.ndim, self.feature_axes)
        reduction_axes = tuple(
          i for i in range(value.ndim) if i not in feature_axes
        )

      feature_shape = [1] * value.ndim
      reduced_feature_shape = []
      for ax in feature_axes:
        feature_shape[ax] = value.shape[ax]
        reduced_feature_shape.append(value.shape[ax])

      value_bar = _l2_normalize(value, axis=reduction_axes, eps=self.epsilon)

      args = [vs]
      if self.use_scale:
        # initialize scale with l2 norm of the weights
        norm_val = jnp.sqrt((value * value).sum(axis=reduction_axes, keepdims=True) + self.epsilon)

        scale = self.param(
          str_path + '/scale',
          lambda key: norm_val,
        ).reshape(feature_shape)
        value_bar *= scale
        args.append(scale)

      dtype = dtypes.canonicalize_dtype(*args, dtype=self.dtype)
      return jnp.asarray(value_bar, dtype)


class ConvNeXtBlock(nn.Module):
    intermediate_dim: int
    layer_scale_init_value: float = None
    adanorm_num_embeddings: int = None
    act: Callable = nn.gelu
    
    @nn.compact
    def __call__(self, x, cond_embedding_id = None):
        dim = x.shape[-1]
        
        residual = x
        x = nn.Conv(features=dim, kernel_size=(7,), padding=(3,), feature_group_count=dim)(x)
        
        if self.adanorm_num_embeddings is not None:
            assert cond_embedding_id is not None
            x = AdaLayerNorm(self.adanorm_num_embeddings, dim)(x, cond_embedding_id)
        else:
            x = nn.LayerNorm()(x)
        
        x = nn.Dense(features=self.intermediate_dim)(x)
        x = self.act(x)
        x = nn.Dense(features=dim)(x)
        
        if self.layer_scale_init_value > 0:
            gamma = self.param('gamma', lambda key, shape: self.layer_scale_init_value * jnp.ones(shape), (dim,))
            x = x * gamma
        
        x = x + residual
        
        return x


class AdaLayerNorm(nn.Module):
    num_embeddings: int
    embedding_dim: int
    eps: float = 1e-6
    
    def setup(self):
        self.scale = nn.Embed(self.num_embeddings, self.embedding_dim, embedding_init=nn.initializers.ones)
        self.shift = nn.Embed(self.num_embeddings, self.embedding_dim, embedding_init=nn.initializers.zeros)
    
    def __call__(self, x, cond_embedding_id):
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.LayerNorm(epsilon=self.eps, use_scale=False, use_bias=False)(x)
        x = x * scale + shift

        return x
