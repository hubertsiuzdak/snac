# SNAC 🍿

Multi-**S**cale **N**eural **A**udio **C**odec (SNAC) compresses audio into discrete codes at a low bitrate.

https://github.com/hubertsiuzdak/snac/assets/35269911/e8adac68-d3f1-4fc1-8cf9-f48d9bcd95ed

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

| Model                                                                       | Bitrate  | Sample Rate | 
|-----------------------------------------------------------------------------|----------|-------------|
| [hubertsiuzdak/snac_32khz](https://huggingface.co/hubertsiuzdak/snac_32khz) | 1.9 kbps | 32 kHz      | 
| [hubertsiuzdak/snac_44khz](https://huggingface.co/hubertsiuzdak/snac_44khz) | 2.6 kbps | 44 kHz      |

These models were trained mostly on music. 

## Usage

Install it using:

```bash
pip install snac
```

To encode (and reconstruct) audio with SNAC in Python, use the following code:

```python
import torch
from snac import SNAC

model = SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().cuda()
audio = torch.randn(1, 1, 32000).cuda()  # B, 1, T

with torch.inference_mode():
    audio_hat, _, codes, _, _ = model(audio)
```

⚠️ Note that `codes` is a list of token sequences of variable lengths, each corresponding to a different temporal
resolution.

```
>>> [code.shape[1] for code in codes]
[12, 24, 48, 96]
```

## Acknowledgements

Module definitions are adapted from the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
