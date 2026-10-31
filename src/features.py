"""Interpretable normalized-time RMS, relative bands and aligned-pitch features."""

import numpy as np


def frame_features(audio, settings, pitch=None):
    samples, rate = audio.samples, audio.sample_rate
    window = max(2, round(rate * settings["window_ms"] / 1000))
    hop = max(1, round(rate * settings["hop_ms"] / 1000))
    # Centers include both utterance boundaries. Padding is zero, not cyclic audio.
    centers = np.arange(0, len(samples), hop)
    if centers[-1] != len(samples) - 1:
        centers = np.append(centers, len(samples) - 1)
    padded = np.pad(samples, (window // 2, window))
    energy, timbre = [], []
    frequencies = np.fft.rfftfreq(window, 1 / rate)
    masks = [(frequencies >= a) & (frequencies < b)
             for a, b in zip(settings["bands_hz"], settings["bands_hz"][1:])]
    if settings["bands_hz"][-1] < rate / 2:
        raise ValueError("bands_hz must cover the WAV Nyquist frequency.")
    masks[-1] |= frequencies == settings["bands_hz"][-1]
    weights = np.asarray(settings["band_weights"])
    for center in centers:
        frame = padded[center:center + window]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        energy.append(rms ** settings["compression"])
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(window))) ** 2
        bands = np.array([spectrum[mask].sum() for mask in masks])
        total = bands.sum()
        relative = bands / total if total > 1e-20 else np.zeros(len(bands))
        timbre.append(float(relative @ weights) * rms ** settings["compression"])
    time = centers / (len(samples) - 1)
    result = {"time": time, "energy": np.array(energy), "timbre": np.array(timbre)}
    if pitch is not None:
        # Sample-index lookup retains exact original offsets; never interpolate Hz across unvoiced intervals.
        original = np.arange(audio.start, audio.end)
        hz = pitch.values(original)
        voiced = hz > 0
        log_pitch = np.zeros(len(hz))
        log_pitch[voiced] = np.log2(hz[voiced] / settings["pitch_reference_hz"])
        mean = float(log_pitch[voiced].mean()) if voiced.any() else 0.0
        absolute = np.zeros(len(hz))
        inflection = np.zeros(len(hz))
        absolute[voiced] = np.tanh(mean / settings["pitch_octaves"])
        inflection[voiced] = np.tanh((log_pitch[voiced] - mean) / settings["pitch_octaves"])
        sample_time = np.arange(len(hz)) / len(hz)
        modulation = (settings["pitch_absolute_weight"] * absolute * np.cos(2 * np.pi * sample_time)
                      + settings["pitch_inflection_weight"] * inflection)
        result.update(pitch_time=sample_time, pitch=modulation, hz=hz,
                      absolute=absolute, inflection=inflection)
    return result


def feature_profile(features, algorithm, settings, count=2048):
    time = np.arange(count) / count
    energy = np.interp(time, features["time"], features["energy"])
    profile = settings["energy_weight"] * (energy - energy.mean())
    if algorithm == "energy-timbre":
        secondary = np.interp(time, features["time"], features["timbre"])
        profile += settings["timbre_weight"] * secondary
    elif algorithm == "energy-pitch":
        # Interpolate modulation, not acoustic Hz; unvoiced source values remain zero.
        secondary = np.interp(time, features["pitch_time"], features["pitch"])
        profile += settings["pitch_weight"] * secondary
    return profile
