# DisCodec: Neural Audio Codec for Latent Music Representations

[📜Paper](https://openreview.net/forum?id=vzrk9ACbIb)

## Usage

### Installation

```bash
pip install git+https://github.com/ETH-DISCO/discodec
```

### Programmatic usage

```python
from discodec import DisCodec, Variant

# Create a DisCodec model with the weights specified by the variant.
# The first parameter is the input shape of the model. DisCodec expects
# a tensor in the shape of (batch_size, num_samples, num_channels).
# This will download weights to ~/.cache/discodec (if not already downloaded).
model = DisCodec((8, 16758, 1), Variant.VQ_QUANTDROPOUT)

# Encode an audio tensor. 'z' is the latent as reconstructed by RVQ, 'codes'
# is a list of codebook indices, and 'latents' is a list of corresponding continuous
# latents.
z, codes, latents = model.encode(audio)

# Decode the latent tensor.
recon = model.decode(z)
```

### TODOs

- [x] Add quantized variant.
- [x] Add usage example.
- [ ] Add VAE variant.