"""Shared 16-bit PCM decoding and utterance-preserving preprocessing."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sample_rate: int
    original_frames: int
    start: int
    end: int


def preprocess(wf, settings):
    if wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Only uncompressed 16-bit PCM WAV is supported.")
    count, channels, rate = wf.getnframes(), wf.getnchannels(), wf.getframerate()
    if count < 4:
        raise ValueError("Recording is empty or too short (need at least 4 frames).")
    wf.rewind()
    raw = wf.readframes(count)
    if len(raw) != count * channels * 2:
        raise ValueError("Truncated WAV data.")
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).mean(axis=1) / 32768.0
    samples -= samples.mean()
    peak = float(np.max(np.abs(samples)))
    if peak < 1 / 32768:
        raise ValueError("Recording is silent or contains only DC.")
    active = np.flatnonzero(np.abs(samples) >= peak * 10 ** (settings["trim_db"] / 20))
    padding = round(rate * settings["trim_padding_ms"] / 1000)
    start = max(0, int(active[0]) - padding)
    end = min(count, int(active[-1]) + padding + 1)
    if end - start < 4:
        raise ValueError("Trimmed recording is too short.")
    samples = samples[start:end] * (settings["normalize_peak"] / peak)
    return Audio(samples, rate, count, start, end)
