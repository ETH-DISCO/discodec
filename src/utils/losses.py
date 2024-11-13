import jax
from jax import numpy as jnp
from librosa.filters import mel as librosa_mel_fn

from src.utils.utils import stft, audiotools_mel_spec


def l1_loss(x, y):
    return jnp.mean(jnp.abs(x - y))


class SpecLoss():
    def __init__(self, window_lengths=[2048, 512], loss_fn=l1_loss, clamp_eps=1e-5, mag_weight=1.0, log_weight=1.0, pow=2.0, match_stride=False, sample_rate=44100,
                 n_mels=[150, 80], mel_fmins=[0.0, 0.0], mel_fmaxs=[None, None]):
        self.window_lengths = window_lengths
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.mag_weight = mag_weight
        self.log_weight = log_weight
        self.pow = pow
        self.match_stride = match_stride
        self.sample_rate = sample_rate

        self.mel_banks = [librosa_mel_fn(
            sr=sample_rate,
            n_fft=w,
            n_mels=n_mel,
            fmin=mel_fmin,
            fmax=mel_fmax,
        ) for w, n_mel, mel_fmin, mel_fmax in zip(window_lengths, n_mels, mel_fmins, mel_fmaxs)]


    def multiscale_stft_loss(self, x, y):
        loss = 0.0
        for w in self.window_lengths:
            hop_length = w // 4
            _, _, x_spec = stft(x, w, hop_length, self.match_stride, self.sample_rate, axis=-1)
            _, _, y_spec = stft(y, w, hop_length, self.match_stride, self.sample_rate, axis=-1)
            log_loss = self.loss_fn(
                jnp.log10(jnp.maximum(jnp.abs(x_spec), self.clamp_eps) ** self.pow),
                jnp.log10(jnp.maximum(jnp.abs(y_spec), self.clamp_eps) ** self.pow)
            )
            mag_loss = self.loss_fn(
                jnp.abs(x_spec),
                jnp.abs(y_spec)
            )
            loss += self.log_weight * log_loss + self.mag_weight * mag_loss
        
        return loss

    def multiscale_mel_loss(self, x, y):
        loss = 0.0
        for w, mel_bank in zip(self.window_lengths, self.mel_banks):
            hop_length = w // 4
            _, _, x_spec = stft(x, w, hop_length, self.match_stride, self.sample_rate, axis=-1)
            _, _, y_spec = stft(y, w, hop_length, self.match_stride, self.sample_rate, axis=-1)
            
            x_mel = audiotools_mel_spec(x_spec, mel_bank)
            y_mel = audiotools_mel_spec(y_spec, mel_bank)

            log_loss = self.loss_fn(
                jnp.log10(jnp.maximum(x_mel, self.clamp_eps) ** self.pow),
                jnp.log10(jnp.maximum(y_mel, self.clamp_eps) ** self.pow)
            )

            mag_loss = self.loss_fn(
                x_mel,
                y_mel
            )

            loss += self.log_weight * log_loss + self.mag_weight * mag_loss
        
        return loss


def disc_adv_loss(d_fake, d_real):
    loss_d = 0
    accuracies_real = []
    accuracies_fake = []

    for i in range(len(d_fake)):
        loss_d += jnp.mean(d_fake[i][-1] ** 2)
        loss_d += jnp.mean((1 - d_real[i][-1]) ** 2)

        accuracies_real.append(jnp.mean(d_real[i][-1] > 0.5))
        accuracies_fake.append(jnp.mean(d_fake[i][-1] < 0.5))
    
    accuracies_real = jnp.array(accuracies_real).mean()
    accuracies_fake = jnp.array(accuracies_fake).mean()
    
    return loss_d, accuracies_real, accuracies_fake

def gen_adv_loss(d_fake):
    loss_g = 0
    for x_fake in d_fake:
        loss_g += jnp.mean((1 - x_fake[-1]) ** 2)
    
    return loss_g

def gen_feat_loss(d_fake, d_real):
    loss_feature = 0
    for i in range(len(d_fake)):
        for j in range(len(d_fake[i]) - 1):
            loss_feature += l1_loss(d_fake[i][j], jax.lax.stop_gradient(d_real[i][j]))
    
    return loss_feature