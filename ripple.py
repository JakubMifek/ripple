# ripple.py - Contains the ripple method for processing a wave audio object and returning SVG visualizations.

import wave
import struct
import math
import numpy as np
from utils import combine_channels, create_waveform_svg, fit_polynomial_curve, fit_fourier_curve, smooth_waveform


def _get_wave_samples(wf: wave.Wave_read) -> tuple[wave.Wave_read, list[int], int, int]:
    """
    Convert stereo to mono if needed, check for mono, and extract all samples from a wave file.
    Returns the wave object, sample list, sample width, and number of frames.
    """
    wf = combine_channels(wf)
    channels = wf.getnchannels() # Should be 1 after combining
    if channels != 1:
        raise ValueError("Expected mono audio after combining channels, but got multiple channels.")

    sampwidth = wf.getsampwidth()
    nframes = wf.getnframes()
    wf.rewind()
    num_samples = nframes

    print(f"Reading {num_samples} samples from {nframes} total frames with {sampwidth}-bit depth.")
    frames = wf.readframes(num_samples)

    if sampwidth != 2:
        raise NotImplementedError("Only 16-bit PCM supported for waveform SVG.")

    samples = list(struct.unpack('<' + 'h' * num_samples, frames))

    return wf, samples, sampwidth, nframes

def _scale_smoothness(smoothness: float) -> float:
    """
    Logarithmically scale the smoothness parameter for perceptual control.
    """
    if smoothness <= 0:
        scaled_smoothness = 0.0
    elif smoothness >= 1:
        scaled_smoothness = 1.0
    else:
        # Map 0 < s < 1 to a log scale: s' = 10**(-2 * (1-s))
        # s=0.5 -> 0.1, s=0.25 -> 0.01, s=0.75 -> 0.385, etc.
        scaled_smoothness = 1/math.pow(10, -1 * math.log2(smoothness))
    print(f"Smoothness: {smoothness} (scaled: {scaled_smoothness})")
    return scaled_smoothness

def _extract_extremes(smoothed: list[float]) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Extract local minima and maxima from the smoothed waveform.
    Returns (min_points, max_points) as lists of (index, value).
    """
    max_points = []
    min_points = []

    for i in range(1, len(smoothed) - 1):
        # Local maxima
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1] or smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            max_points.append((i, smoothed[i]))
        # Local minima
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

    return min_points, max_points

def to_svg_points(
    indices: list[int],
    vals: list[float],
    svg_width: int,
    svg_height: int,
    smoothed: list[float]
) -> list[tuple[int, int]]:
    """
    Map waveform or envelope points to SVG coordinates for visualization.
    """
    return [
        (
            int(i * svg_width / len(smoothed)),
            int(svg_height // 2 - (v / (max(abs(s) for s in smoothed) or 1)) * (svg_height // 2 - 2))
        )
        for i, v in zip(indices, vals)
    ]


def _extract_midpoints(
    min_svg_points: list[tuple[int, int]],
    max_svg_points: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """
    Compute midpoints between min and max SVG points for envelope fitting.
    Returns a list of (x, y) tuples for the midpoints.
    """
    mid_points = []
    xs = sorted(set([x for x, _ in max_svg_points] + [x for x, _ in min_svg_points]))
    min_dict = dict(min_svg_points)
    max_dict = dict(max_svg_points)

    for x in xs:
        if x in min_dict and x in max_dict:
            mid_points.append((x, (min_dict[x] + max_dict[x]) // 2))
        elif x in min_dict:
            mid_points.append((x, (mid_points[-1][1] + min_dict[x]) // 2))
        elif x in max_dict:
            mid_points.append((x, (mid_points[-1][1] + max_dict[x]) // 2))

    print(f"Found {len(mid_points)} mid points for fitting.")
    return mid_points


def _path_to_circle(
    fit_points: list[tuple[int, int]],
    svg_width: int,
    svg_height: int
) -> tuple[list[tuple[int, int]], str, str]:
    """
    Map a fitted curve to a circular SVG path, with the circle centered at the top-left.
    Returns the circle points, SVG path, and base circle SVG string.
    """
    print(f"Mapping fitted curve to circle with {len(fit_points)} points.")
    r_min = 0.75
    r_max = 1.00
    center_to_corner = True
    circle_points = curve_to_circle(fit_points, svg_width, svg_height, r_min=r_min, r_max=r_max, center_to_corner=center_to_corner)
    circle_path = create_svg_path(circle_points, close_path=True)

    # Draw the base circle (average radius), centered at top-left
    base_r = min(svg_width, svg_height) / 2
    avg_r = ((r_min + r_max) / 2) * base_r
    if center_to_corner:
        cx = base_r
        cy = base_r
    else:
        cx = svg_width / 2
        cy = svg_height / 2
    base_circle_svg = f'M {cx + avg_r},{cy} ' \
        f'A {avg_r},{avg_r} 0 1,0 {cx - avg_r},{cy} ' \
        f'A {avg_r},{avg_r} 0 1,0 {cx + avg_r},{cy}'
    return circle_points, circle_path, base_circle_svg


def create_svg_path(
    points: list[tuple[int, int]],
    close_path: bool = False
) -> str:
    """
    Create an SVG path string from a list of points, using Bezier curves for smoothness.
    If close_path is True, the path is closed.
    """
    if len(points) < 3:
        path = 'M ' + ' L '.join(f"{x},{y}" for x, y in points)
        if close_path:
            path += ' Z'
        return path

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
    if close_path:
        path_data += ' Z'
    return path_data


def curve_to_circle(
    points: list[tuple[int, int]],
    width: int,
    height: int,
    r_min: float = 0.5,
    r_max: float = 0.75,
    center_to_corner: bool = False
) -> list[tuple[int, int]]:
    """
    Map a list of (x, y) points to a circle for circular SVG visualization.
    The x axis is mapped to angle, y to radius modulation.
    r_min and r_max are relative to min(width, height)/2.
    If center_to_corner is True, the circle is centered at (base_r, base_r) (top-left), else at (width/2, height/2).
    Returns list of (cx, cy) points.
    """
    if not points:
        return []

    N = len(points)

    # Normalize x to [0, 2pi]
    xs = np.array([x for x, y in points])
    ys = np.array([y for x, y in points])
    x_min, x_max = xs.min(), xs.max()
    x_norm = 2 * np.pi * (xs - x_min) / (x_max - x_min if x_max != x_min else 1)

    # Normalize y to [0, 1]
    y_min, y_max = ys.min(), ys.max()
    y_norm = (ys - y_min) / (y_max - y_min if y_max != y_min else 1)

    # Map to radius
    base_r = min(width, height) / 2
    r0 = r_min * base_r
    r1 = r_max * base_r
    radii = r0 + (r1 - r0) * y_norm

    if center_to_corner:
        cx = base_r
        cy = base_r
    else:
        cx = width / 2
        cy = height / 2

    circle_points = [
        (
            int(cx + r * np.cos(theta)),
            int(cy + r * np.sin(theta))
        )
        for theta, r in zip(x_norm, radii)
    ]
    return circle_points

def ripple(
    wf: wave.Wave_read,
    smoothness: float = 0.0,
    poly_degree: int = 10,
    max_width: int = 2048,
    height: int = 256,
    fitting: str = 'fourier',
) -> tuple[str, str]:
    """
    Process the wave audio object and return two SVG strings: a linear (processing) SVG and a circular SVG.
    Args:
        wf: wave.Wave_read object
        smoothness: Smoothing parameter (0-1)
        poly_degree: Degree/terms for polynomial or Fourier fitting
        max_width: Max SVG width
        height: SVG height
        fitting: 'poly' or 'fourier' for curve fitting
    Returns:
        Tuple[str, str]: (processing SVG, circular SVG)
    """

    # --- Extract amplitude time series ---
    wf, samples, _, nframes = _get_wave_samples(wf)

    # --- Scale smoothness ---
    scaled_smoothness = _scale_smoothness(smoothness)

    # --- Smooth the waveform samples ---
    print(f"Smoothing waveform with smoothness: {scaled_smoothness}")
    smoothed = smooth_waveform(samples, scaled_smoothness)

    # --- Extract local minima and maxima ---
    min_points, max_points = _extract_extremes(smoothed)

    # --- Smooth the max and min points (envelope) ---
    print(f"Smoothing max/min points with smoothness: {scaled_smoothness / 2}")
    max_indices, max_vals = zip(*max_points) if max_points else (list[int](), list[float]())
    min_indices, min_vals = zip(*min_points) if min_points else (list[int](), list[float]())
    max_vals_smooth = smooth_waveform(list(max_vals), scaled_smoothness / 2) if max_vals else []
    min_vals_smooth = smooth_waveform(list(min_vals), scaled_smoothness / 2) if min_vals else []

    # --- Map to SVG coordinates ---
    svg_width = min(max_width, nframes)
    svg_height = height
    max_svg_points = to_svg_points(list[int](max_indices), max_vals_smooth, svg_width, svg_height, smoothed) if max_vals else []
    min_svg_points = to_svg_points(list[int](min_indices), min_vals_smooth, svg_width, svg_height, smoothed) if min_vals else []

    # --- Main waveform path (processing only) ---
    waveform_path = create_svg_path([
        (int(i * svg_width / len(smoothed)),
         int(svg_height // 2 - (v / (max(abs(s) for s in smoothed) or 1)) * (svg_height // 2 - 2)))
        for i, v in enumerate(smoothed)
    ])

    # --- Collect SVG path strings for processing (non-circular) and circular SVGs ---
    processing_paths = [{'d': waveform_path, 'stroke': 'gray', 'stroke_width': 1}]
    circular_paths = []

    # --- Envelope paths (processing only) ---
    max_path = create_svg_path(max_svg_points)
    min_path = create_svg_path(min_svg_points)
    processing_paths.append({'d': max_path, 'stroke': 'red', 'stroke_width': 1})
    processing_paths.append({'d': min_path, 'stroke': '#023e8a', 'stroke_width': 1})

    # --- Use the average of the min and max y for each x present in both ---
    mid_points = _extract_midpoints(min_svg_points, max_svg_points)

    if len(mid_points) < 4:
        raise ValueError("Not enough mid points for fitting. Need at least 4 points.")

    # --- Use fitting method from parameter ---
    if fitting == 'poly':
        fit_points = fit_polynomial_curve(mid_points, poly_degree)
        color = 'green'
    elif fitting == 'fourier':
        fit_points = fit_fourier_curve(mid_points, n_terms=poly_degree)
        color = 'orange'
    else:
        raise ValueError(f"Unknown fitting method: {fitting}")

    # --- Draw the fitted curve as a path before circular mapping (processing only) ---
    fit_path = create_svg_path(fit_points, close_path=False)
    processing_paths.append({'d': fit_path, 'stroke': color, 'stroke_width': 1})

    # --- Map fitted curve to circle (circular SVG only), with center at top-left ---
    _, circle_path, base_circle_svg = _path_to_circle(fit_points, svg_width, svg_height)
    circular_paths.append({'d': base_circle_svg, 'stroke': '#888', 'stroke_width': 1})
    circular_paths.append({'d': circle_path, 'stroke': color, 'stroke_width': 1})

    # --- Mark beginning of the sound curve on the circle (circular SVG only) ---
    # if circle_points:
    #     start_x, start_y = circle_points[0]
    #     circular_paths.append({'d': f'M {start_x},{start_y} m -4,0 a 4,4 0 1,0 8,0 a 4,4 0 1,0 -8,0', 'stroke': 'none', 'stroke_width': 0, 'fill': 'lime'})

    # --- Compose SVGs at the end ---
    print(f"Creating SVGs with width {svg_width} and height {svg_height}.")
    processing_svg = create_waveform_svg(svg_width, svg_height, processing_paths)
    shorter_side = min(svg_width, svg_height)
    circular_svg = create_waveform_svg(shorter_side, shorter_side, circular_paths)
    return processing_svg, circular_svg
