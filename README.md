# [WIP] SNAC 🍿

Multi-**S**cale **N**eural **A**udio **C**odec (SNAC) compressess 44.1 kHz audio into discrete codes at a low bitrate.

It encodes audio into hierarchical tokens similarly to SoundStream, EnCodec, and DAC (see the image
on the left). However, SNAC introduces a simple change where coarse tokens are sampled less frequently,
covering a broader time span (see the image on the right).

This can not only save on bitrate, but more importantly this might be very useful for language modeling approaches to
audio generation. E.g. with coarse tokens of ~10 Hz and a context window of 2048 you can effectively model a
consistent structure of an audio track for ~3 minutes.

![snac.png](img%2Fsnac.png)
