"""
ripple.py - Contains the ripple method for processing a wave audio object and returning an SVG object.
"""

import wave
import struct
import statistics
import math
import numpy as np
from utils import combine_channels


def ripple(wf: wave.Wave_read, smoothness: float = 0.0, poly_degree: int = 10) -> str:
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

    svg_width = min(2048, nframes)
    svg_height = 512

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

    # Create SVG for both curves (envelope)
    svg_content = create_waveform_svg(smoothed, svg_width, svg_height)
    if max_svg_points and min_svg_points:
        max_path = create_svg_path(max_svg_points)
        min_path = create_svg_path(min_svg_points)
        svg_content = svg_content.replace('</svg>',
            f'<path d="{max_path}" fill="none" stroke="red" stroke-width="1"/>'
            f'<path d="{min_path}" fill="none" stroke="#023e8a" stroke-width="1"/>'
        )

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
            # Fit a higher-degree polynomial for a smooth center curve that stays within the envelope
            margin = min(len(mid_points), 128)
            print("Mid point length: ", len(mid_points))
            x_vals = np.array(list(reversed([-x-mid_points[1][0] for x, y in mid_points[:margin]])) + [x for x, y in mid_points] + [mid_points[-1][0] + x for x, y in mid_points[:margin]])
            y_vals = np.array([y for x, y in mid_points[len(mid_points) - margin:]] + [y for x, y in mid_points] + [y for x, y in mid_points[:margin]])
            print("X values length: ", len(x_vals))
            print(x_vals)
            degree = min(poly_degree, len(mid_points) - 1)
            coeffs = np.polyfit(x_vals, y_vals, degree)
            poly = np.poly1d(coeffs)
            poly_points = [(x, int(poly(x))) for x in x_vals[margin:-margin]]
            poly_path = create_svg_path(poly_points)
            svg_content += f'<path d="{poly_path}" fill="none" stroke="green" stroke-width="1"/>'
        svg_content += '</svg>'
    return svg_content

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

def create_waveform_svg(samples, svg_width, svg_height):
    mid_y = svg_height // 2
    num_samples = len(samples)
    max_ampl = max(abs(s) for s in samples) or 1
    points = []
    for i, s in enumerate(samples):
        x = int(i * svg_width / num_samples)
        y = int(mid_y - (s / max_ampl) * (svg_height // 2 - 2))
        points.append((x, y))

    def control_points(p0, p1, p2, t=0.2):
        d01 = ((p1[0] - p0[0]), (p1[1] - p0[1]))
        d12 = ((p2[0] - p1[0]), (p2[1] - p1[1]))
        c1 = (p1[0] - d01[0] * t, p1[1] - d01[1] * t)
        c2 = (p1[0] + d12[0] * t, p1[1] + d12[1] * t)
        return c1, c2

    if len(points) < 3:
        path_data = 'M ' + ' L '.join(f"{x},{y}" for x, y in points)
    else:
        path_data = f"M {points[0][0]},{points[0][1]}"
        for i in range(1, len(points) - 1):
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]
            c1, c2 = control_points(p0, p1, p2)
            path_data += f" C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {p2[0]},{p2[1]}"

    svg_content = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg"><path d="{path_data}" fill="none" stroke="black" stroke-width="1"/></svg>'
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
    
    smoothed = []

    n = len(samples)
    if smoothness <= 0:
        smoothed = samples
    elif smoothness >= 1:
        avg = int(sum(samples) / n)
        smoothed = [avg] * n
    else:
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
