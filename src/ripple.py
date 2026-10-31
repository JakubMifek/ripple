"""Audio-to-art orchestration; ripple() returns (diagnostic SVG, circular SVG)."""

import math
import wave
import numpy as np
from audio import preprocess
from config import PITCH_ALGORITHMS, TIMBRE_ALGORITHMS, load_config, validate_config
from features import frame_features, feature_profile
from geometry import periodic_coefficients, circular_svg
from pitch import estimate_pitch
from utils import create_waveform_svg, fit_polynomial_curve, fit_fourier_curve, smooth_waveform


def _scale_smoothness(smoothness):
    return 0.0 if smoothness <= 0 else (1.0 if smoothness >= 1 else 10 ** math.log2(smoothness))


def _extract_extremes(smoothed):
    values = np.asarray(smoothed)
    min_points, max_points = [], []
    for i in range(1, len(values) - 1):
        left, value, right = values[i-1:i+2]
        if (value >= left and value > right) or (value > left and value >= right):
            max_points.append((i, float(value)))
        if (value <= left and value < right) or (value < left and value <= right):
            min_points.append((i, float(value)))
    for points in (min_points, max_points):
        if points:
            points.insert(0, (0, float(values[0])))
            points.append((len(values) - 1, float(values[-1])))
    return min_points, max_points


def _extract_midpoints(min_points, max_points):
    """Retain the legacy recursive signed-extrema midpoint, without integer loss."""
    minima, maxima = dict(min_points), dict(max_points)
    result = []
    for x in sorted(minima.keys() | maxima.keys()):
        if x in minima and x in maxima:
            value = (minima[x] + maxima[x]) / 2
        else:
            current = minima[x] if x in minima else maxima[x]
            value = (result[-1][1] + current) / 2 if result else current
        result.append((x, value))
    return result


def create_svg_path(points, close_path=False):
    """Float Catmull-Rom-to-Bezier conversion, including the periodic seam."""
    if not points:
        return ""
    values = np.array(points, dtype=float)
    if len(values) < 3:
        return "M " + " L ".join(f"{x:.10f},{y:.10f}" for x, y in values) + (" Z" if close_path else "")
    parts = [f"M {values[0, 0]:.10f},{values[0, 1]:.10f}"]
    for i in range(len(values) if close_path else len(values) - 1):
        j = (i + 1) % len(values)
        previous = values[(i - 1) % len(values)] if close_path or i else values[i]
        following = values[(j + 1) % len(values)] if close_path or j < len(values) - 1 else values[j]
        a = values[i] + (values[j] - previous) / 6
        b = values[j] - (following - values[i]) / 6
        parts.append(f"C {a[0]:.10f},{a[1]:.10f} {b[0]:.10f},{b[1]:.10f} {values[j,0]:.10f},{values[j,1]:.10f}")
    return " ".join(parts) + (" Z" if close_path else "")


def to_svg_points(indices, vals, svg_width, svg_height, smoothed):
    peak = max((abs(s) for s in smoothed), default=1) or 1
    return [(i * svg_width / max(1, len(smoothed)-1),
             svg_height / 2 - v / peak * (svg_height / 2 - 2)) for i, v in zip(indices, vals)]


def baseline_profile(audio, render):
    smoothness = _scale_smoothness(render["smoothness"])
    smoothed = smooth_waveform(audio.samples, smoothness)
    minima, maxima = _extract_extremes(smoothed)
    envelopes = []
    for points in (minima, maxima):
        values = smooth_waveform([y for _, y in points], smoothness / 2)
        envelopes.append([(i / (len(smoothed)-1), v) for (i, _), v in zip(points, values)])
    midpoints = _extract_midpoints(*envelopes)
    if len(midpoints) < 4:
        raise ValueError("Not enough midpoints for baseline fitting (need at least 4); use less smoothing or an energy method.")
    if render["fitting"] == "poly":
        fitted = fit_polynomial_curve(midpoints, render["detail"])
    else:
        fitted = fit_fourier_curve(midpoints, render["detail"])
    profile = np.array([y for _, y in fitted])
    return profile, smoothed, envelopes


def diagnostic_svg(audio, render, profile, waveform, envelopes, metadata=None):
    width, height = render["max_width"], render["height"]
    paths = []

    def add(time, values, color):
        pairs = list(zip(time, values))
        # Diagnostics are bounded in size without changing extraction or fitted geometry.
        if len(pairs) > 2048:
            pairs = [pairs[i] for i in np.linspace(0, len(pairs)-1, 2048, dtype=int)]
        points = [(x * width, height * (0.5 - 0.42 * y)) for x, y in pairs]
        paths.append({"d": create_svg_path(points), "stroke": color, "stroke_width": 1})

    add(np.linspace(0, 1, len(waveform)), waveform, "#888")
    for points, color in zip(envelopes, ("red", "#023e8a", "#2a9d8f")):
        if points:
            time, values = zip(*points)
            add(time, values, color)
    add(np.arange(len(profile)) / len(profile), profile, render["color"])
    details = {"trim-start": audio.start, "trim-end": audio.end, "algorithm": render["algorithm"]}
    details.update(metadata or {})
    return create_waveform_svg(width, height, paths, details)


def ripple(wf: wave.Wave_read, smoothness=None, poly_degree=None, max_width=None,
           height=None, fitting=None, color=None, circle=None, *,
           algorithm=None, settings=None, pitch_data=None, baseline_gain=None) -> tuple[str, str]:
    """Shared preprocessing and periodic geometry, independent of export dimensions."""
    if settings is None:
        settings = load_config(overrides={
            "smoothness": smoothness, "detail": poly_degree, "max_width": max_width,
            "height": height, "fitting": fitting, "color": color, "circle": circle,
            "algorithm": algorithm, "baseline_gain": baseline_gain})
    else:
        validate_config(settings)
    render = settings["render"]
    if render["algorithm"] not in PITCH_ALGORITHMS and pitch_data is not None:
        raise ValueError("Pitch input is only supported by energy-pitch or energy-timbre-pitch.")
    audio = preprocess(wf, settings["preprocess"])
    if render["algorithm"] in PITCH_ALGORITHMS and pitch_data is None:
        pitch_data = estimate_pitch(audio, settings["pitch_estimation"])
    metadata = pitch_data.metadata if pitch_data is not None else None
    if render["algorithm"] == "baseline":
        profile, waveform, envelopes = baseline_profile(audio, render)
    else:
        features = frame_features(audio, settings["features"], pitch_data)
        profile = feature_profile(features, render["algorithm"], settings["features"])
        waveform, envelopes = audio.samples, [list(zip(features["time"], features["energy"]))]
        if pitch_data is None or render["algorithm"] in TIMBRE_ALGORITHMS:
            envelopes.append(list(zip(features["time"], features["timbre"])))
        if pitch_data is not None:
            envelopes.append(list(zip(features["pitch_time"], features["pitch"])))
    gain = render["baseline_gain"] if render["algorithm"] == "baseline" else 1
    coefficients = periodic_coefficients(profile * gain, render["detail"], render["roundness"])
    diagnostic_metadata = {**(metadata or {}), "trim-method": settings["preprocess"]["trim_method"]}
    return (diagnostic_svg(audio, render, profile, waveform, envelopes, diagnostic_metadata),
            circular_svg(coefficients, render, metadata))
