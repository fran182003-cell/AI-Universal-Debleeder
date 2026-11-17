# Modifications to Demucs

This folder is a modified copy of [Demucs](https://github.com/facebookresearch/demucs) v4.1.0a2 by Meta Platforms, Inc. (MIT License, see `LICENSE`).

It is adapted so that HDemucs outputs **one signal per source**: source *i* is the separated version of input channel *i* (the beamformer output pointing at direction *i*).

- `hdemucs.py`
  - The first encoder layer takes `audio_channels` inputs instead of `audio_channels * len(sources)`.
  - The output keeps only channel *i* for source *i*, so the model returns `[batch, sources, time]` instead of `[batch, sources, channels, time]`.
  - The time-branch denormalisation is adapted to that shape.
  - The channel-selection index is created on the tensor's device (originally hard-coded to `cuda`, which failed on CPU).
- `apply.py`: output buffers and segment handling adapted to the `[batch, sources, time]` shape.
