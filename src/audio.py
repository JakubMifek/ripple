"""Shared 16-bit PCM decoding and utterance-preserving preprocessing."""

from dataclasses import dataclass
import hashlib
import math
import struct
import numpy as np


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sample_rate: int
    original_frames: int
    start: int
    end: int
    start_angle: float


def recording_angle(raw, rate, channels):
    """Stable clockwise angle from original PCM and its format, not filename or trim."""
    digest = hashlib.sha256(b"ripple-orientation-v1\0" + struct.pack("<IH", rate, channels) + raw).digest()
    return int.from_bytes(digest[:4], "big") * (360.0 / 2**32)


def activity_bounds(samples, rate, settings):
    """Sustained RMS anchors with quieter boundary extension; never cut internal gaps."""
    window = min(len(samples), max(1, round(rate * 0.020)))
    hop = max(1, round(rate * 0.005))
    starts = np.arange(0, len(samples) - window + 1, hop)
    if starts[-1] != len(samples) - window:
        starts = np.append(starts, len(samples) - window)
    squares = np.r_[0.0, np.cumsum(samples * samples)]
    rms = np.sqrt(np.maximum(0, (squares[starts + window] - squares[starts]) / window))
    peak = float(rms.max())
    strong = rms >= peak * 10 ** (settings["trim_trigger_db"] / 20)
    transitions = np.diff(np.r_[False, strong, False].astype(int))
    beginnings, endings = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    minimum = min(len(starts), max(1, math.ceil(rate * 0.030 / hop)))
    sustained = endings - beginnings >= minimum
    if not sustained.any():
        raise ValueError("No sustained activity found for RMS trimming; use a clearer recording "
                         "or --config presets/legacy-trim.toml.")
    first, last = int(beginnings[sustained][0]), int(endings[sustained][-1] - 1)
    weak = rms >= peak * 10 ** (settings["trim_db"] / 20)
    max_gap = max(1, math.ceil(rate * 0.040 / hop))

    def extend(anchor, direction):
        boundary, gap = anchor, 0
        index = anchor + direction
        while 0 <= index < len(starts):
            if weak[index]:
                boundary, gap = index, 0
            else:
                gap += 1
                if gap >= max_gap:
                    break
            index += direction
        return boundary

    return int(starts[extend(first, -1)]), int(starts[extend(last, 1)] + window)


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
    if settings["trim_method"] == "rms":
        start, end = activity_bounds(samples, rate, settings)
    elif settings["trim_method"] == "peak":
        active = np.flatnonzero(np.abs(samples) >= peak * 10 ** (settings["trim_db"] / 20))
        start, end = int(active[0]), int(active[-1]) + 1
    else:
        raise ValueError("trim_method must be rms or peak.")
    padding = round(rate * settings["trim_padding_ms"] / 1000)
    start = max(0, start - padding)
    end = min(count, end + padding)
    if end - start < 4:
        raise ValueError("Trimmed recording is too short.")
    samples = samples[start:end] * (settings["normalize_peak"] / peak)
    return Audio(samples, rate, count, start, end, recording_angle(raw, rate, channels))
