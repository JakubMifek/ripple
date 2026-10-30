"""Strict, audio-bound known-generating-control pitch input; no estimation."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PROVENANCE = ("known-generating-controls", "direct-source-measurement")


@dataclass(frozen=True)
class Pitch:
    intervals: tuple

    def values(self, indices):
        indices = np.asarray(indices)
        ends = np.array([end for _, end, _ in self.intervals])
        if np.any(indices < 0) or np.any(indices >= ends[-1]):
            raise ValueError("Pitch lookup outside original recording.")
        values = np.array([hz if hz is not None else 0.0 for _, _, hz in self.intervals])
        return values[np.searchsorted(ends, indices, side="right")]


def wav_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _keys(obj, expected, label):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise ValueError(f"{label} must contain exactly: {', '.join(expected)}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate pitch JSON key: {key}")
        result[key] = value
    return result


def load_pitch(path, wav_path, sample_rate, frames):
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream, object_pairs_hook=_unique_object)
    _keys(data, ("version", "source", "wav_sha256", "sample_rate", "intervals"), "Pitch sidecar")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Pitch sidecar version must be 1.")
    _keys(data["source"], ("category", "description"), "Pitch source")
    source = data["source"]
    if source["category"] not in PROVENANCE:
        raise ValueError("Pitch provenance must be known generating controls or direct source measurement; estimated pitch is excluded.")
    if not isinstance(source["description"], str) or not source["description"].strip():
        raise ValueError("Pitch source description is required.")
    if data["wav_sha256"] != wav_hash(wav_path):
        raise ValueError("Pitch WAV SHA-256 mismatch.")
    if type(data["sample_rate"]) is not int or data["sample_rate"] != sample_rate:
        raise ValueError("Pitch sample rate mismatch.")
    if not isinstance(data["intervals"], list) or not data["intervals"]:
        raise ValueError("Pitch intervals must completely cover the original recording.")
    result, previous = [], 0
    for interval in data["intervals"]:
        _keys(interval, ("start", "end", "hz"), "Pitch interval")
        start, end, hz = (interval[k] for k in ("start", "end", "hz"))
        if type(start) is not int or type(end) is not int or start != previous or end <= start or end > frames:
            raise ValueError("Pitch intervals must be contiguous, ordered, nonoverlapping sample-index intervals.")
        if hz is not None and (type(hz) not in (int, float) or not 0 < hz < sample_rate / 2 or not math.isfinite(hz)):
            raise ValueError("Known pitch Hz must be finite, positive and below Nyquist; use null for unvoiced.")
        result.append((start, end, hz))
        previous = end
    if previous != frames:
        raise ValueError("Pitch intervals must cover every original WAV frame.")
    return Pitch(tuple(result))
