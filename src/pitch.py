"""Known-source pitch sidecars and explicitly labeled automatic Praat estimates."""

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
    source: str = "known-generating-controls"
    estimator: str | None = None

    @property
    def metadata(self):
        result = {"pitch-source": self.source}
        if self.estimator is not None:
            result["pitch-estimator"] = self.estimator
        return result

    def values(self, indices):
        indices = np.asarray(indices)
        ends = np.array([end for _, end, _ in self.intervals])
        if np.any(indices < 0) or np.any(indices >= ends[-1]):
            raise ValueError("Pitch lookup outside original recording.")
        values = np.array([hz if hz is not None else 0.0 for _, _, hz in self.intervals])
        return values[np.searchsorted(ends, indices, side="right")]


def estimate_pitch(audio, settings):
    rate = audio.sample_rate
    if settings["ceiling_hz"] >= rate / 2:
        raise ValueError("pitch_estimation.ceiling_hz must be below the WAV Nyquist frequency.")
    step = settings["hop_ms"] / 1000
    if step * rate < 1:
        raise ValueError("Pitch estimation hop must span at least one audio frame.")
    minimum_duration = 6 / settings["floor_hz"]
    if len(audio.samples) / rate < minimum_duration:
        raise ValueError(f"Recording is too short for pitch estimation: need at least "
                         f"{minimum_duration:.3f} seconds after trimming at the configured pitch floor.")
    try:
        import parselmouth
    except (ImportError, OSError) as error:
        raise ValueError("Automatic pitch estimation requires praat-parselmouth; "
                         "install dependencies with python3 -m pip install -r requirements.txt.") from error
    try:
        track = parselmouth.Sound(audio.samples, sampling_frequency=rate).to_pitch_ac(
            time_step=step, pitch_floor=settings["floor_hz"], pitch_ceiling=settings["ceiling_hz"],
            voicing_threshold=settings["voicing_threshold"], very_accurate=True)
    except parselmouth.PraatError as error:
        raise ValueError(f"Praat pitch estimation failed: {error}") from error
    hz = np.asarray(track.selected_array["frequency"])
    centers = np.asarray(track.xs())
    if (hz.ndim != 1 or centers.shape != hz.shape or not len(hz)
            or not np.isfinite(hz).all() or not np.isfinite(centers).all()
            or np.any(hz < 0) or np.any(hz >= rate / 2)):
        raise ValueError("Praat returned an invalid pitch track.")
    if not np.any(hz > 0):
        raise ValueError("No voiced pitch detected by Praat. Record a clear voiced word, adjust "
                         "pitch_estimation settings, or explicitly select an energy-only method.")
    boundaries = np.r_[centers[0] - step / 2,
                       (centers[:-1] + centers[1:]) / 2, centers[-1] + step / 2]
    # Praat's first audio sample is centered at 0.5/rate; retain original WAV indices.
    frames = np.ceil(boundaries * rate - 0.5).astype(int) + audio.start
    frames = np.clip(frames, audio.start, audio.end)
    if np.any(np.diff(frames) <= 0):
        raise ValueError("Praat pitch frames do not map to increasing audio intervals.")
    intervals = [(0, int(frames[0]), None)] if frames[0] > 0 else []
    intervals.extend((int(start), int(end), float(value) if value > 0 else None)
                     for start, end, value in zip(frames[:-1], frames[1:], hz))
    if frames[-1] < audio.original_frames:
        intervals.append((int(frames[-1]), audio.original_frames, None))
    return Pitch(tuple(intervals), source="estimated",
                 estimator=f"Praat {parselmouth.PRAAT_VERSION} autocorrelation "
                           f"(Parselmouth {parselmouth.VERSION})")


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
        raise ValueError("Version-1 pitch sidecars require known generating controls or direct source "
                         "measurement, not estimates. Omit --pitch for automatic Praat estimation.")
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
    return Pitch(tuple(result), source=source["category"])
