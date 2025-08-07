"""
utils.py - Utility functions for audio processing.
"""

import wave
import io
import struct
import numpy as np
import statistics

# --- Smoothing ---
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
    
# --- SVG Generation ---
def create_waveform_svg(svg_width, svg_height, svg_paths):
    """
    Create an SVG string from a list of path dictionaries.
    """
    svg_content = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">'
    for path in svg_paths:
        # If 'fill' is present, use it, otherwise fill="none"
        fill = path.get('fill', 'none')
        svg_content += f'<path d="{path["d"]}" fill="{fill}" stroke="{path["stroke"]}" stroke-width="{path["stroke_width"]}"/>'
    svg_content += '</svg>'
    return svg_content

# --- Curve Fitting ---
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
