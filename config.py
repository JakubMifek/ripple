"""Versioned project defaults, strict override merging and CLI resolution."""

import argparse
import copy
import math
from pathlib import Path
import tomllib

ALGORITHMS = ("baseline", "energy", "energy-timbre", "energy-pitch")
DEFAULT_CONFIG = Path(__file__).with_name("ripple.toml")
SCHEMA = {
    "render": {"algorithm", "max_width", "height", "color", "stroke_width", "circle",
               "detail", "deformation", "nominal_radius", "roundness", "points", "fitting", "smoothness", "baseline_gain"},
    "preprocess": {"trim_db", "trim_padding_ms", "normalize_peak"},
    "features": {"window_ms", "hop_ms", "compression", "energy_weight", "timbre_weight",
                 "pitch_weight", "bands_hz", "band_weights", "pitch_reference_hz",
                 "pitch_octaves", "pitch_absolute_weight", "pitch_inflection_weight"},
}


def validate_keys(settings):
    if set(settings) != {"version", *SCHEMA}:
        raise ValueError("Configuration must contain version and only render, preprocess, features tables.")
    if type(settings["version"]) is not int or settings["version"] != 1:
        raise ValueError("Configuration version must be 1.")
    for section, keys in SCHEMA.items():
        if not isinstance(settings[section], dict) or set(settings[section]) != keys:
            raise ValueError(f"Unknown or missing configuration settings in {section}; expected {', '.join(sorted(keys))}.")


def load_config(path=None, overrides=None):
    with DEFAULT_CONFIG.open("rb") as stream:
        settings = tomllib.load(stream)
    validate_keys(settings)
    if path:
        with Path(path).open("rb") as stream:
            selected = tomllib.load(stream)
        if type(selected.get("version")) is not int or selected["version"] != 1:
            raise ValueError("Override configuration must declare version = 1.")
        for section, values in selected.items():
            if section not in settings:
                raise ValueError(f"Unknown configuration setting: {section}")
            if section == "version":
                continue
            if not isinstance(values, dict):
                raise ValueError(f"Configuration section {section} must be a table.")
            for key, value in values.items():
                if key not in settings[section]:
                    raise ValueError(f"Unknown configuration setting: {section}.{key}")
                settings[section][key] = value
    settings = copy.deepcopy(settings)
    for key, value in (overrides or {}).items():
        if value is not None:
            if key not in settings["render"]:
                raise ValueError(f"Unknown render override: {key}")
            settings["render"][key] = value
    validate_config(settings)
    return settings


def validate_config(settings):
    validate_keys(settings)
    r, p, f = (settings[k] for k in ("render", "preprocess", "features"))

    def number(table, key, low, high, integer=False):
        value = table[key]
        if (type(value) not in ((int,) if integer else (int, float))
                or not low <= value <= high or not math.isfinite(value)):
            raise ValueError(f"{key} must be {'an integer' if integer else 'a number'} in [{low}, {high}].")

    if r["algorithm"] not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}.")
    if r["fitting"] not in ("poly", "fourier"):
        raise ValueError("fitting must be poly or fourier.")
    if r["algorithm"] != "baseline" and r["fitting"] != "fourier":
        raise ValueError("Polynomial fitting is supported only by baseline.")
    if type(r["circle"]) is not bool:
        raise ValueError("circle must be boolean.")
    if not isinstance(r["color"], str) or not r["color"].strip():
        raise ValueError("color must be a nonempty SVG color.")
    for key in ("max_width", "height"):
        number(r, key, 16, 100000, True)
    number(r, "detail", 1, 128, True)
    number(r, "points", 64, 8192, True)
    if r["points"] < 8 * r["detail"]:
        raise ValueError("points must be at least 8 * detail.")
    number(r, "stroke_width", 0.01, 1000)
    number(r, "smoothness", 0, 1)
    number(r, "baseline_gain", 0, 64)
    number(r, "roundness", 0, 1)
    number(r, "deformation", 0, 0.35)
    number(r, "nominal_radius", 0.1, 0.45)
    if r["nominal_radius"] * (1 + r["deformation"]) + r["stroke_width"] / (2 * min(r["max_width"], r["height"])) >= 0.5:
        raise ValueError("Nominal radius, deformation and stroke must fit inside the canvas.")
    number(p, "trim_db", -100, -10)
    number(p, "trim_padding_ms", 0, 1000)
    number(p, "normalize_peak", 0.01, 1)
    number(f, "window_ms", 1, 1000)
    number(f, "hop_ms", 0.1, f["window_ms"])
    number(f, "compression", 0.05, 1)
    for key in ("energy_weight", "timbre_weight", "pitch_weight",
                "pitch_absolute_weight", "pitch_inflection_weight"):
        number(f, key, 0, 4)
    number(f, "pitch_reference_hz", 1, 20000)
    number(f, "pitch_octaves", 0.1, 8)
    edges, weights = f["bands_hz"], f["band_weights"]
    if not isinstance(edges, list) or not 2 <= len(edges) <= 33:
        raise ValueError("bands_hz must contain 2 to 33 increasing edges.")
    if any(type(x) not in (int, float) or not 0 <= x <= 1e9 or not math.isfinite(x) for x in edges):
        raise ValueError("bands_hz must contain finite nonnegative numbers.")
    if edges[0] != 0 or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("bands_hz must start at zero and be strictly increasing.")
    if (not isinstance(weights, list) or len(weights) != len(edges) - 1
            or any(type(x) not in (int, float) or abs(x) > 1 or not math.isfinite(x) for x in weights)):
        raise ValueError("band_weights must have one finite [-1, 1] weight per band.")
    return settings


def add_render_arguments(parser):
    parser.add_argument("--config", type=Path, help="Version-1 TOML overrides (project defaults are always loaded first)")
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--mode", choices=["Ripple"], default="Ripple", help="Legacy mode name")
    parser.add_argument("--smoothness", type=float, help="Baseline cyclic signed-waveform smoothing, 0 to 1")
    parser.add_argument("--baseline-gain", type=float, help="Fixed baseline profile contrast, 0 to 64; 1 preserves unscaled extraction")
    parser.add_argument("--poly-degree", "--detail", dest="detail", type=int, help="Fourier terms / baseline polynomial degree")
    parser.add_argument("--max-width", type=int, help="Diagnostic width; circular side is min(width, height)")
    parser.add_argument("--height", type=int)
    parser.add_argument("--fitting", choices=["fourier", "poly"], help="poly is baseline-only")
    parser.add_argument("--color")
    parser.add_argument("--stroke-width", type=float)
    parser.add_argument("--deformation", type=float, help="Fractional radial deformation, 0 to 0.35")
    parser.add_argument("--roundness", type=float, help="High-harmonic attenuation, 0 to 1")
    parser.add_argument("--circle", action=argparse.BooleanOptionalAction, default=None, help="Add a light-gray nominal reference circle")


def config_from_args(args):
    return load_config(args.config, {key: getattr(args, key, None) for key in (
        "algorithm", "smoothness", "detail", "max_width", "height", "fitting",
        "color", "stroke_width", "deformation", "roundness", "circle", "baseline_gain")})
