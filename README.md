# [WIP] SNAC 🍿

Multi-**S**cale **N**eural **A**udio **C**odec (SNAC) compressess 44.1 kHz audio into discrete codes at a low bitrate.

## Overview

SNAC encodes audio into hierarchical tokens similarly to SoundStream, EnCodec, and DAC (see the image
on the left). However, SNAC introduces a simple change where coarse tokens are sampled less frequently,
covering a broader time span (see the image on the right).

This can not only save on bitrate, but more importantly this might be very useful for language modeling approaches to
audio generation. E.g. with coarse tokens of ~10 Hz and a context window of 2048 you can effectively model a
consistent structure of an audio track for ~3 minutes.

![snac.png](img%2Fsnac.png)

## Usage

Install it using:

```bash
pip install snac
```

A pretrained model that compresses audio into discrete codes at a 2.2 kbps bitrate is available
at [Hugging Face](https://huggingface.co/hubertsiuzdak/snac). It uses 4 RVQ levels with token rates of 12.5, 25, 50, and
100 Hz.

To encode (and reconstruct) audio with SNAC in Python, use the following code:

```python
import torch
from snac import SNAC

model = SNAC.from_pretrained("hubertsiuzdak/snac").eval().cuda()
audio = torch.randn(1, 1, 44100).cuda()  # B, 1, T

with torch.inference_mode():
    audio_hat, _, codes, _, _ = model(audio)
```

⚠️ Note that `codes` is a list of token sequences of variable lengths, each corresponding to a different temporal
resolution.

```
>>> [code.shape[1] for code in codes]
[13, 26, 52, 104]
```

## Acknowledgements

Module definitions are adapted from the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
