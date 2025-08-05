"""
ripple.py - Contains the ripple method for processing a wave audio object and returning an SVG object.
"""

import wave
import struct
import statistics
import math
import numpy as np
from utils import combine_channels


def ripple(
    wf: wave.Wave_read,
    smoothness: float = 0.0,
    poly_degree: int = 10,
    max_width: int = 2048,
    height: int = 256,
    fitting: str = 'fourier',
) -> str:
    """
    Process the wave audio object and return an SVG string.
    Args:
        wf: wave.Wave_read object
    Returns:
        str: SVG content as a string
    """

    wf = combine_channels(wf)

    channels = wf.getnchannels() # Should be 1 after combining
    if channels != 1:
        raise ValueError("Expected mono audio after combining channels, but got multiple channels.")
    
    sampwidth = wf.getsampwidth()
    # framerate = wf.getframerate()
    nframes = wf.getnframes()

    # print(f"Channels: {channels}")
    # print(f"Sample width: {sampwidth}")
    # print(f"Frame rate: {framerate}")
    # print(f"Number of frames: {nframes}")
    
    # Extract amplitude time series (first 1024 samples for simplicity)
    wf.rewind()
    num_samples = nframes # min(1024, nframes)
    print(f"Reading {num_samples} samples from {nframes} total frames with {sampwidth}-bit depth.")

    frames = wf.readframes(num_samples)

    # Only 16-bit PCM supported
    if sampwidth != 2:
        raise NotImplementedError("Only 16-bit PCM supported for waveform SVG.")
    samples = list(struct.unpack('<' + 'h' * num_samples, frames))

    # Logarithmic scaling for smoothness
    if smoothness <= 0:
        scaled_smoothness = 0.0
    elif smoothness >= 1:
        scaled_smoothness = 1.0
    else:
        # Map 0 < s < 1 to a log scale: s' = 10**(-2 * (1-s))
        # s=0.5 -> 0.1, s=0.25 -> 0.01, s=0.75 -> 0.385, etc.
        scaled_smoothness = 1/math.pow(10, -1 * math.log2(smoothness))

    print(f"Smoothness: {smoothness} (scaled: {scaled_smoothness})")

    # Smooth the waveform samples
    smoothed = smooth_waveform(samples, scaled_smoothness)

    svg_width = min(max_width, nframes)
    svg_height = height

    # Second round: extract local maxima and minima
    max_points = []
    min_points = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1] or smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            max_points.append((i, smoothed[i]))
        if smoothed[i] <= smoothed[i - 1] and smoothed[i] < smoothed[i + 1] or smoothed[i] < smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
            min_points.append((i, smoothed[i]))
    # Always include endpoints
    if max_points and max_points[0][0] != 0:
        max_points = [(0, smoothed[0])] + max_points
    if max_points and max_points[-1][0] != len(smoothed) - 1:
        max_points.append((len(smoothed) - 1, smoothed[-1]))
    if min_points and min_points[0][0] != 0:
        min_points = [(0, smoothed[0])] + min_points
    if min_points and min_points[-1][0] != len(smoothed) - 1:
        min_points.append((len(smoothed) - 1, smoothed[-1]))

    # Smooth the max and min points (optional: use a fixed window for envelope)
    max_indices, max_vals = zip(*max_points) if max_points else ([], [])
    min_indices, min_vals = zip(*min_points) if min_points else ([], [])
    max_vals_smooth = smooth_waveform(list(max_vals), scaled_smoothness / 2) if max_vals else []
    min_vals_smooth = smooth_waveform(list(min_vals), scaled_smoothness / 2) if min_vals else []

    # Map to SVG coordinates
    def to_svg_points(indices, vals):
        return [(
            int(i * svg_width / len(smoothed)),
            int(svg_height // 2 - (v / (max(abs(s) for s in smoothed) or 1)) * (svg_height // 2 - 2))
        ) for i, v in zip(indices, vals)]

    max_svg_points = to_svg_points(max_indices, max_vals_smooth) if max_vals else []
    min_svg_points = to_svg_points(min_indices, min_vals_smooth) if min_vals else []

    # Collect SVG path strings
    svg_paths = []
    # Main waveform
    waveform_path = create_svg_path([
        (int(i * svg_width / len(smoothed)),
         int(svg_height // 2 - (v / (max(abs(s) for s in smoothed) or 1)) * (svg_height // 2 - 2)))
        for i, v in enumerate(smoothed)
    ])
    svg_paths.append({'d': waveform_path, 'stroke': 'gray', 'stroke_width': 1})

    # Envelope paths
    if max_svg_points and min_svg_points:
        max_path = create_svg_path(max_svg_points)
        min_path = create_svg_path(min_svg_points)
        svg_paths.append({'d': max_path, 'stroke': 'red', 'stroke_width': 1})
        svg_paths.append({'d': min_path, 'stroke': '#023e8a', 'stroke_width': 1})

        # Use the average of the min and max y for each x present in both
        xs = sorted(set([x for x, _ in max_svg_points] + [x for x, _ in min_svg_points]))
        min_dict = dict(min_svg_points)
        max_dict = dict(max_svg_points)
        mid_points = []
        for x in xs:
            y_vals = []
            if x in min_dict:
                y_vals.append(min_dict[x])
            if x in max_dict:
                y_vals.append(max_dict[x])
            if y_vals and len(y_vals) > 1:
                y = sum(y_vals) // len(y_vals)
                mid_points.append((x, y))

        if len(mid_points) >= 4:
            # Use fitting method from parameter
            if fitting == 'poly':
                poly_points = fit_polynomial_curve(mid_points, poly_degree)
                poly_path = create_svg_path(poly_points)
                svg_paths.append({'d': poly_path, 'stroke': 'green', 'stroke_width': 1})
            elif fitting == 'fourier':
                fourier_points = fit_fourier_curve(mid_points, n_terms=poly_degree)
                fourier_path = create_svg_path(fourier_points)
                svg_paths.append({'d': fourier_path, 'stroke': 'orange', 'stroke_width': 1})
            else:
                raise ValueError(f"Unknown fitting method: {fitting}")

    # Compose SVG at the end
    return create_waveform_svg(svg_width, svg_height, svg_paths)

def create_svg_path(points):
    if len(points) < 3:
        return 'M ' + ' L '.join(f"{x},{y}" for x, y in points)
    def control_points(p0, p1, p2, t=0.2):
        d01 = ((p1[0] - p0[0]), (p1[1] - p0[1]))
        d12 = ((p2[0] - p1[0]), (p2[1] - p1[1]))
        c1 = (p1[0] - d01[0] * t, p1[1] - d01[1] * t)
        c2 = (p1[0] + d12[0] * t, p1[1] + d12[1] * t)
        return c1, c2
    path_data = f"M {points[0][0]},{points[0][1]}"
    for i in range(1, len(points) - 1):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]
        c1, c2 = control_points(p0, p1, p2)
        path_data += f" C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {p2[0]},{p2[1]}"
    return path_data

def create_waveform_svg(svg_width, svg_height, svg_paths):
    svg_content = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">'
    for path in svg_paths:
        svg_content += f'<path d="{path["d"]}" fill="none" stroke="{path["stroke"]}" stroke-width="{path["stroke_width"]}"/>'
    svg_content += '</svg>'
    return svg_content

def smooth_waveform(samples: list, smoothness: float) -> list:
    """
    Smooth the waveform samples based on the specified smoothness.
    Args:
        samples: List of audio samples
        smoothness: Float between 0 and 1 indicating smoothness level
    Returns:
        List of smoothed samples
    """
    
    n = len(samples)
    if smoothness <= 0:
        return samples
    elif smoothness >= 1:
        avg = int(sum(samples) / n)
        return [avg] * n
    else:
        smoothed = []
        window = int(1 + smoothness * (n - 1))
        if window % 2 == 0:
            window += 1  # median filter window should be odd

        print(f"Window size for smoothing: {window}")
        half = window // 2

        for i in range(n):
            # Cyclic window: wrap indices around the array
            indices = [(i + j) % n for j in range(-half, half + 1)]
            window_samples = [samples[idx] for idx in indices]
            if len(window_samples) < 3:
                smoothed.append(int(sum(window_samples) / len(window_samples)))
            else:
                smoothed.append(int(statistics.median(window_samples)))
        return smoothed
def fit_polynomial_curve(mid_points, poly_degree):
    """
    Fit a polynomial curve through the mid_points and return the fitted points.
    """
    margin = min(len(mid_points), 128)
    x_vals = np.array(list(reversed([-x-mid_points[1][0] for x, y in mid_points[:margin]])) + [x for x, y in mid_points] + [mid_points[-1][0] + x for x, y in mid_points[:margin]])
    y_vals = np.array([y for x, y in mid_points[len(mid_points) - margin:]] + [y for x, y in mid_points] + [y for x, y in mid_points[:margin]])
    degree = min(poly_degree, len(mid_points) - 1)
    coeffs = np.polyfit(x_vals, y_vals, degree)
    poly = np.poly1d(coeffs)
    poly_points = [(x, int(poly(x))) for x in x_vals[margin:-margin]]
    return poly_points

def fit_fourier_curve(mid_points, n_terms=10):
    """
    Fit a truncated Fourier series (sum of sines and cosines) to the mid_points.
    Returns a list of (x, y) points for the fitted curve.
    """
    # Sort by x to ensure correct order
    mid_points = sorted(mid_points, key=lambda p: p[0])
    x_vals = np.array([x for x, y in mid_points])
    y_vals = np.array([y for x, y in mid_points])
    N = len(x_vals)
    # Normalize x to [0, 2pi] for periodicity
    x_norm = 2 * np.pi * (x_vals - x_vals[0]) / (x_vals[-1] - x_vals[0] if x_vals[-1] != x_vals[0] else 1)
    # Build design matrix for least squares fit
    A = [np.ones(N)]
    for n in range(1, n_terms + 1):
        A.append(np.sin(n * x_norm))
        A.append(np.cos(n * x_norm))
    A = np.vstack(A).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y_vals, rcond=None)
    # Generate fitted curve (dense sampling for smoothness)
    x_dense = np.linspace(x_vals[0], x_vals[-1], 4 * N)
    x_dense_norm = 2 * np.pi * (x_dense - x_vals[0]) / (x_vals[-1] - x_vals[0] if x_vals[-1] != x_vals[0] else 1)
    y_dense = coeffs[0] * np.ones_like(x_dense)
    for n in range(1, n_terms + 1):
        y_dense += coeffs[2 * n - 1] * np.sin(n * x_dense_norm)
        y_dense += coeffs[2 * n] * np.cos(n * x_dense_norm)
    fourier_points = [(int(x), int(y)) for x, y in zip(x_dense, y_dense)]
    return fourier_points
