"""
utils.py - Utility functions for audio processing.
"""

import wave
import io
import struct

def combine_channels(wf: wave.Wave_read) -> wave.Wave_read:
    """
    Combine all channels of a wave file into a single mono channel.
    Args:
        wf: wave.Wave_read object
    Returns:
        wave.Wave_read: New mono wave.Wave_read object (in-memory)
    """
    nchannels = wf.getnchannels()
    if nchannels == 1:
        return wf

    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    nframes = wf.getnframes()
    wf.rewind()
    frames = wf.readframes(nframes)

    # Unpack all samples
    fmt = '<' + 'h' * nchannels * nframes if sampwidth == 2 else None
    if fmt is None:
        raise NotImplementedError("Only 16-bit PCM supported.")
    samples = struct.unpack(fmt, frames)

    # Combine channels: average samples for each frame
    mono_samples = []
    for i in range(nframes):
        frame = samples[i * nchannels:(i + 1) * nchannels]
        avg = int(sum(frame) / nchannels)
        mono_samples.append(avg)

    # Pack mono samples
    mono_frames = struct.pack('<' + 'h' * nframes, *mono_samples)

    # Write to in-memory buffer
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as mono_wf:
        mono_wf.setnchannels(1)
        mono_wf.setsampwidth(sampwidth)
        mono_wf.setframerate(framerate)
        mono_wf.writeframes(mono_frames)
    buffer.seek(0)
    return wave.open(buffer, 'rb')
