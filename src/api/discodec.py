import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.training import train_state
import flax.linen as nn
import optax
from einops import rearrange

from src.model.generator import Model
from src.model.discriminator import Discriminator
from src.utils.utils import load_model

class DisCodec:
    def __init__(self, data_shape, variant):
        path = load_model(variant)
        self.ckpt_mngr = ocp.CheckpointManager(path,
                                               item_names=('gen_state', 'disc_state', 'config'))

        self.restore(data_shape)
    
    def _init_fn(self, model, tx, batch, key, **kwargs):
        model_vars = model.init(key, batch, **kwargs)
        state = train_state.TrainState.create(apply_fn=model.apply, params=model_vars, tx=tx)
        return state

    def _create_gen_state(self, config, batch, key):
        generator = Model(**config['dac'], decoder_output_act=nn.tanh)

        gen_scheduler = optax.exponential_decay(config['adamw']['lr'], 1, config['exponential_lr']['gamma'])
        gen_opt = optax.chain(
            optax.clip_by_global_norm(1e3),
            optax.adamw(b1=config['adamw']['betas'][0], b2=config['adamw']['betas'][1], learning_rate=gen_scheduler)
        )

        gen_state = self._init_fn(generator, gen_opt, batch, key, training=True)
        return gen_state
    
    def _create_disc_state(self, config, batch, key):
        discriminator = Discriminator(**config['discriminator'])

        disc_scheduler = optax.exponential_decay(config['adamw']['lr'], 1, config['exponential_lr']['gamma'])
        disc_opt = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adamw(b1=config['adamw']['betas'][0], b2=config['adamw']['betas'][1], learning_rate=disc_scheduler)
        )

        disc_state = self._init_fn(discriminator, disc_opt, rearrange(batch, 'b t c -> b c t'), key)
        return disc_state

    def _restore_optimizer_state(self, opt_state, restored):
        """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
        np_restored = jax.tree.map(lambda x: np.array(x), restored)
        return jax.tree.unflatten(
            jax.tree.structure(opt_state), jax.tree.leaves(np_restored)
        )

    def restore(self, data_shape):
        restored = self.ckpt_mngr.restore(
            self.ckpt_mngr.latest_step(),
            args=ocp.args.Composite(
                gen_state=ocp.args.StandardRestore(),
                disc_state=ocp.args.StandardRestore(),
                config=ocp.args.JsonRestore(),
            ),
        )

        config = restored['config']

        key = jax.random.PRNGKey(0)
        batch = jnp.ones(data_shape)

        gen_state = self._create_gen_state(config, batch, key)
        disc_state = self._create_disc_state(config, batch, key)

        rest_gen_state = restored['gen_state']
        rest_disc_state = restored['disc_state']

        rest_gen_params = jax.tree.map(lambda x: np.array(x), rest_gen_state['params'])
        rest_disc_params = jax.tree.map(lambda x: np.array(x), rest_disc_state['params'])
        rest_gen_opt = self._restore_optimizer_state(gen_state.opt_state, rest_gen_state['opt_state'])
        rest_disc_opt = self._restore_optimizer_state(disc_state.opt_state, rest_disc_state['opt_state'])

        self.gen_state = gen_state.replace(params=rest_gen_params, step=np.array(rest_gen_state['step']), opt_state=rest_gen_opt)
        self.disc_state = disc_state.replace(params=rest_disc_params, step=np.array(rest_disc_state['step']), opt_state=rest_disc_opt)

    def encode(self, data, n_quantizers=None):
        data = self.gen_state.apply_fn(self.gen_state.params, data, method='preprocess', sample_rate=None)
        (z, codes, latents, _, _), _ = self.gen_state.apply_fn(self.gen_state.params, data, method='encode',
                                                          n_quantizers=n_quantizers, training=False, mutable=['codebook_stats'])
        return z, codes, latents
    
    def decode(self, z):
        return self.gen_state.apply_fn(self.gen_state.params, z, method='decode', training=False)