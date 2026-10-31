"""Bounded periodic radial curves with analytic, matching cubic join tangents."""

import numpy as np
from utils import create_waveform_svg


def periodic_coefficients(profile, detail, roundness):
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("A finite profile with at least four samples is required.")
    coefficients = np.fft.rfft(values) / len(values)
    coefficients = coefficients[:min(detail + 1, len(coefficients))]
    coefficients[0] = 0  # Nominal diameter is shared; absolute pitch is a broad harmonic, not DC.
    k = np.arange(len(coefficients))
    coefficients *= np.exp(-roundness * (k / 3) ** 2)
    bound = 2 * np.abs(coefficients[1:]).sum()
    if bound > 1:
        count = max(1024, 32 * (len(coefficients) - 1))
        theta = np.arange(count) * (2 * np.pi / count)
        values, _ = evaluate_profile(coefficients, theta)
        # Between samples, the profile can change by at most slope_bound * pi/count.
        slope_bound = 2 * np.sum(k * np.abs(coefficients))
        bound = min(bound, np.max(np.abs(values)) + slope_bound * np.pi / count)
        if bound > 1:
            coefficients /= bound
    return coefficients


def evaluate_profile(coefficients, theta):
    theta = np.asarray(theta)
    values, derivative = np.zeros_like(theta, dtype=float), np.zeros_like(theta, dtype=float)
    for k, coefficient in enumerate(coefficients[1:], 1):
        harmonic = coefficient * np.exp(1j * k * theta)
        values += 2 * harmonic.real
        derivative += 2 * (1j * k * harmonic).real
    return values, derivative


def radial_path(coefficients, render):
    count = render["points"]
    theta = np.arange(count) * (2 * np.pi / count)
    profile, derivative = evaluate_profile(coefficients, theta)
    side = min(render["max_width"], render["height"])
    nominal = side * render["nominal_radius"]
    radius = nominal * (1 + render["deformation"] * profile)
    dr = nominal * render["deformation"] * derivative
    points = side / 2 * (1 + 1j) + radius * np.exp(1j * theta)
    tangents = (dr + 1j * radius) * np.exp(1j * theta)
    step = 2 * np.pi / count

    def point(z):
        return f"{z.real:.10f},{z.imag:.10f}"

    parts = [f"M {point(points[0])}"]
    for i in range(count):
        j = (i + 1) % count
        parts.append(f"C {point(points[i] + tangents[i] * step / 3)} "
                     f"{point(points[j] - tangents[j] * step / 3)} {point(points[j])}")
    return " ".join(parts) + " Z"


def circular_svg(coefficients, render, metadata=None):
    side = min(render["max_width"], render["height"])
    paths = []
    if render["circle"]:
        paths.append({"d": radial_path(np.zeros(1, dtype=complex), render),
                      "stroke": "#c8c8c8", "stroke_width": 1, "stroke-dasharray": "1 5",
                      "data-role": "reference-circle"})
    paths.append({"d": radial_path(coefficients, render), "stroke": render["color"],
                  "stroke_width": render["stroke_width"]})
    details = {
        "algorithm": render["algorithm"], "nominal-diameter": 2 * render["nominal_radius"] * side,
        "center": f"{side / 2},{side / 2}"}
    details.update(metadata or {})
    return create_waveform_svg(side, side, paths, details)
