# SNAC 🍿

Multi-**S**cale **N**eural **A**udio **C**odec (SNAC) compresses audio into discrete codes at a low bitrate.

| 🎸 Music samples                                                                                         | 🗣️ Speech samples                                                                                       |
|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| <video src='https://github.com/hubertsiuzdak/snac/assets/35269911/e8adac68-d3f1-4fc1-8cf9-f48d9bcd95ed'> | <video src='https://github.com/hubertsiuzdak/snac/assets/35269911/65ac2547-c711-49d4-8a5d-64d52e6d6ba1'> |

🎧 More audio samples available at https://hubertsiuzdak.github.io/snac/

## Overview

SNAC encodes audio into hierarchical tokens similarly to SoundStream, EnCodec, and DAC (see the image
on the left). However, SNAC introduces a simple change where coarse tokens are sampled less frequently,
covering a broader time span (see the image on the right).

This can not only save on bitrate, but more importantly this might be very useful for language modeling approaches to
audio generation. E.g. with coarse tokens of ~10 Hz and a context window of 2048 you can effectively model a
consistent structure of an audio track for ~3 minutes.

![snac.png](img%2Fsnac.png)

## Pretrained models

Currently, all models support only single audio channel (mono).

| Model                                                                       | Bitrate   | Sample Rate | Params | Recommended use case     | 
|-----------------------------------------------------------------------------|-----------|-------------|--------|--------------------------|
| [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) | 0.98 kbps | 24 kHz      | 19.8 M | 🗣️ Speech               | 
| [hubertsiuzdak/snac_32khz](https://huggingface.co/hubertsiuzdak/snac_32khz) | 1.9 kbps  | 32 kHz      | 54.5 M | 🎸 Music / Sound Effects | 
| [hubertsiuzdak/snac_44khz](https://huggingface.co/hubertsiuzdak/snac_44khz) | 2.6 kbps  | 44 kHz      | 54.5 M | 🎸 Music / Sound Effects |

## Usage

Install it using:

```bash
pip install snac
```

To encode (and decode) audio with SNAC in Python, use the following code:

```python
import torch
from snac import SNAC

model = SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().cuda()
audio = torch.randn(1, 1, 32000).cuda()  # placeholder for actual audio with shape (B, 1, T)

with torch.inference_mode():
    codes = model.encode(audio)
    audio_hat = model.decode(codes)
```

You can also encode and reconstruct in a single call:

```python
with torch.inference_mode():
    audio_hat, codes = model(audio)
```

⚠️ Note that `codes` is a list of token sequences of variable lengths, each corresponding to a different temporal
resolution.

```
>>> [code.shape[1] for code in codes]
[12, 24, 48, 96]
```

## Acknowledgements

Module definitions are adapted from the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
