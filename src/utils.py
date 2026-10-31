"""Shared smoothing, fitting and SVG serialization."""

import heapq
import io
import wave
import xml.etree.ElementTree as ET
import numpy as np

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def smooth_waveform(samples, smoothness):
    """Exact cyclic median, with float precision and O(n log n) heap operations."""
    n = len(samples)
    if not n or smoothness <= 0:
        return list(samples)
    if smoothness >= 1:
        return [float(np.mean(samples))] * n
    window = int(1 + smoothness * (n - 1)) | 1
    half = window // 2
    values = [float(samples[i % n]) for i in range(-half, n + half)]
    lower, upper, side, expired = [], [], {}, set()
    counts = [0, 0]

    def prune(heap):
        while heap and heap[0][1] in expired:
            _, idx = heapq.heappop(heap)
            expired.remove(idx)
            del side[idx]

    def balance():
        prune(lower)
        prune(upper)
        while counts[0] > counts[1] + 1:
            value, idx = heapq.heappop(lower)
            heapq.heappush(upper, (-value, idx))
            side[idx] = 1
            counts[0] -= 1
            counts[1] += 1
            prune(lower)
        while counts[0] < counts[1]:
            value, idx = heapq.heappop(upper)
            heapq.heappush(lower, (-value, idx))
            side[idx] = 0
            counts[1] -= 1
            counts[0] += 1
            prune(upper)

    result = []
    for idx, value in enumerate(values):
        prune(lower)
        which = 0 if not lower or value <= -lower[0][0] else 1
        heapq.heappush(lower if which == 0 else upper, (-value if which == 0 else value, idx))
        side[idx] = which
        counts[which] += 1
        if idx >= window:
            old = idx - window
            counts[side[old]] -= 1
            expired.add(old)
        balance()
        if idx >= window - 1:
            result.append(-lower[0][0])
    return result


def create_waveform_svg(svg_width, svg_height, svg_paths, metadata=None):
    root = ET.Element(f"{{{SVG_NS}}}svg", width=str(svg_width), height=str(svg_height),
                      viewBox=f"0 0 {svg_width} {svg_height}")
    if metadata:
        for key, value in metadata.items():
            root.set(f"data-{key}", str(value))
    for path in svg_paths:
        element = ET.SubElement(root, f"{{{SVG_NS}}}path", {
            "d": path["d"], "fill": path.get("fill", "none"),
            "stroke": path["stroke"], "stroke-width": str(path["stroke_width"]),
            "stroke-linejoin": "round", "stroke-linecap": "round"})
        for attribute in ("data-role", "stroke-dasharray"):
            if attribute in path:
                element.set(attribute, str(path[attribute]))
    return ET.tostring(root, encoding="unicode")


def reference_first(elements):
    """Place distinct marked references below artwork, preserving artwork order."""
    references, artwork = {}, []
    for element in elements:
        if element.get("data-role") == "reference-circle":
            references.setdefault(tuple(sorted(element.attrib.items())), element)
        else:
            artwork.append(element)
    return list(references.values()) + artwork


def fit_polynomial_curve(mid_points, poly_degree):
    xs, ys = np.asarray(mid_points, dtype=float).T
    if len(xs) <= poly_degree:
        raise ValueError("Not enough midpoints for the requested polynomial degree.")
    # Polynomial.fit scales the time domain before solving, avoiding pixel-scale conditioning.
    polynomial = np.polynomial.Polynomial.fit(xs, ys, poly_degree)
    dense = np.linspace(xs[0], xs[-1], 512, endpoint=False)
    return list(zip(dense, polynomial(dense)))


def fit_fourier_curve(mid_points, n_terms=10):
    points = np.asarray(sorted(mid_points), dtype=float)
    xs, ys = points.T
    if len(xs) < 4 or xs[-1] <= xs[0]:
        raise ValueError("Not enough distinct midpoints for Fourier fitting (need at least 4).")
    terms = min(n_terms, (len(xs) - 1) // 2)
    angles = 2 * np.pi * (xs - xs[0]) / (xs[-1] - xs[0])
    matrix = [np.ones(len(xs))]
    for k in range(1, terms + 1):
        matrix.extend((np.cos(k * angles), np.sin(k * angles)))
    coefficients = np.linalg.lstsq(np.array(matrix).T, ys, rcond=None)[0]
    time = np.linspace(xs[0], xs[-1], 512, endpoint=False)
    theta = 2 * np.pi * (time - xs[0]) / (xs[-1] - xs[0])
    profile = np.full(len(time), coefficients[0])
    for k in range(1, terms + 1):
        profile += coefficients[2*k-1] * np.cos(k*theta) + coefficients[2*k] * np.sin(k*theta)
    return list(zip(time, profile))


def combine_channels(wf):
    """Legacy helper; the main pipeline averages floating samples without requantization."""
    if wf.getsampwidth() != 2:
        raise ValueError("Only 16-bit PCM supported.")
    if wf.getnchannels() == 1:
        return wf
    wf.rewind()
    samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    mono = np.rint(samples.reshape(-1, wf.getnchannels()).mean(axis=1)).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setparams((1, 2, wf.getframerate(), 0, "NONE", "not compressed"))
        target.writeframes(mono.tobytes())
    buffer.seek(0)
    return wave.open(buffer, "rb")
