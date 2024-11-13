import jax
from jax import numpy as jnp
import math
from enum import Enum
import os
import zipfile
import requests


class Variant(Enum):
    VQ_QUANTDROPOUT = 0

CONFIG_MAP = {
    Variant.VQ_QUANTDROPOUT: ('https://github.com/ETH-DISCO/discodec/releases/download/v0.0.1/vq_quantdropout.zip', 'vq_quantdropout'),
}
MODEL_DIR = '~/.cache/discodec'


def load_model(variant):
    """Checks if folder with variant name exists in cache and downloads the model if it doesn't. Returns the path to the model directory."""
    model_dir = os.path.expanduser(MODEL_DIR)
    model_name = CONFIG_MAP[variant][1]
    model_dir = os.path.join(model_dir, model_name)

    if not os.path.exists(model_dir):
        print(f'{model_name} model not found in cache. Downloading...')
        download_model(variant)
        print(f'{model_name} model downloaded to {model_dir}')

    return model_dir

def download_model(variant):
    """Downloads the model weights as a zip file and extracts it to the cache directory."""
    url, model_name = CONFIG_MAP[variant]
    model_dir = os.path.expanduser(MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f'{model_name}.zip')

    if not os.path.exists(model_file):
        print(f'Downloading {model_name} model...')
        r = requests.get(url, allow_redirects=True)
        with open(model_file, 'wb') as f:
            f.write(r.content)

    with zipfile.ZipFile(model_file, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

    return model_dir

def compute_stft_padding(length, window_length, hop_length, match_stride):
    if match_stride:
        right_pad = math.ceil(length / hop_length) * hop_length - length
        pad = (window_length - hop_length) // 2
    else:
        right_pad, pad = 0, 0
    
    return right_pad, pad

def stft(audio, window_length, hop_length, match_stride, sample_rate, axis=-1):
    length = audio.shape[axis]
    right_pad, pad = compute_stft_padding(length, window_length, hop_length, match_stride)
    audio = jnp.pad(audio, ((0, 0), (0, 0), (pad, pad + right_pad)), mode='reflect')

    window = jnp.hanning(window_length)

    freqs, times, spec = jax.scipy.signal.stft(
        audio,
        fs=sample_rate,
        window=window,
        nperseg=window_length,
        noverlap=window_length - hop_length,
    )

    scale = jnp.sqrt(1.0 / window.sum() ** 2)
    spec = spec / scale

    if match_stride:
        spec = spec[..., 2:-2]
    
    return freqs, times, spec

def as_real(x):
    """Converts complex array to real array with last dimension being real & imaginary parts."""

    if not jnp.issubdtype(x.dtype, jnp.complexfloating):
        return x

    xr = jnp.zeros(x.shape+(2,), dtype=x.real.dtype)
    xr = xr.at[...,0].set(x.real)
    xr = xr.at[...,1].set(x.imag)
    return xr

def audiotools_mel_spec(spec, mel_basis, **kwargs):
    mag = jnp.abs(spec)
    mel_spec = jnp.einsum('bcft,nf->bcnt', mag, mel_basis)

    return mel_spec