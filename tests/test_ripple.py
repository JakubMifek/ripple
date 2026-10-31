"""Deterministic analytic fixtures; no natural-speech pitch labels or estimates."""

import copy
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave
import xml.etree.ElementTree as ET

import numpy as np
from svgpathtools import parse_path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from audio import preprocess
from compare import comparison_preview, overlay
from config import ALGORITHMS, DEFAULT_CONFIG, config_from_args, load_config, validate_config
from features import frame_features, feature_profile
from generate_fixtures import generate
from geometry import circular_svg, evaluate_profile, periodic_coefficients
from main import main as main_command, parse_args, render_file
from merge import export_png
from pitch import Pitch, estimate_pitch, load_pitch, wav_hash
from ripple import create_svg_path
from utils import SVG_NS, smooth_waveform

NS = {"s": SVG_NS}


def signal(rate=8000, duration=0.6, hz=220, gain=0.6):
    t = np.arange(round(rate * duration)) / rate
    envelope = 0.6 + 0.35 * np.cos(2 * np.pi * t / duration)
    return gain * envelope * np.sin(2 * np.pi * hz * t)


def write_wav(path, values, rate=8000, channels=1, width=2):
    values = np.asarray(values)
    if channels > 1 and values.ndim == 1:
        values = np.repeat(values[:, None], channels, axis=1)
    if width == 2:
        raw = np.rint(values * 32767).astype("<i2").tobytes()
    else:
        raw = np.full(values.size, 128, dtype=np.uint8).tobytes()
    with wave.open(str(path), "wb") as target:
        target.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        target.writeframes(raw)


def paths(document):
    return [parse_path(p.attrib["d"]) for p in ET.fromstring(document).findall(".//s:path", NS)]


def sample_path(document, count=1024):
    path = paths(document)[-1]
    return np.array([path.point(i / count) for i in range(count)])


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ripple-test-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.wav = self.work / "word.wav"
        write_wav(self.wav, signal())
        self.settings = load_config(overrides={"smoothness": 0, "detail": 8, "max_width": 512, "height": 256,
                                              "stroke_width": 1, "deformation": 0.18, "circle": False})

    def audio(self, path=None):
        with wave.open(str(path or self.wav), "rb") as wf:
            return preprocess(wf, self.settings["preprocess"])

    def assert_reference_circle(self, element):
        self.assertEqual(element.get("data-role"), "reference-circle")
        self.assertEqual(element.get("stroke"), "#e0e0e0")
        self.assertEqual(element.get("stroke-width"), "1")
        self.assertEqual(element.get("stroke-dasharray"), "1 5")
        self.assertEqual(element.get("stroke-linecap"), "round")
        self.assertEqual(element.get("fill"), "none")

    def run_cli(self, script, *args, ok=True):
        result = subprocess.run([sys.executable, str(SRC / script), *map(str, args)],
                                cwd=self.work, text=True, capture_output=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result


class ConfigurationTests(FixtureCase):
    def test_defaults_precedence_and_no_circle_override(self):
        self.assertEqual(DEFAULT_CONFIG, REPO / "ripple.toml")
        defaults = load_config()
        self.assertEqual(defaults["render"]["algorithm"], "baseline")
        self.assertEqual(defaults["render"]["max_width"], 16384)
        self.assertEqual(defaults["render"]["height"], 1024)
        self.assertEqual(defaults["render"]["color"], "black")
        self.assertEqual(defaults["render"]["stroke_width"], 3)
        self.assertTrue(defaults["render"]["circle"])
        self.assertEqual(defaults["render"]["deformation"], 0.24)
        self.assertEqual(defaults["render"]["roundness"], 0.15)
        self.assertEqual(defaults["render"]["baseline_gain"], 19.2)
        self.assertEqual(defaults["features"]["energy_weight"], 3.96)
        self.assertEqual(defaults["features"]["timbre_weight"], 2.64)
        self.assertEqual(defaults["pitch_estimation"], {
            "floor_hz": 60.0, "ceiling_hz": 600.0, "hop_ms": 10.0, "voicing_threshold": 0.45})
        self.assertEqual(defaults, load_config(REPO / "presets/expressive.toml"))
        config = self.work / "custom.toml"
        config.write_text('version = 1\n[render]\ncolor = "blue"\ndetail = 5\ncircle = true\nbaseline_gain = 4\n')
        _, args = parse_args([str(self.wav), "x.svg", "--config", str(config), "--poly-degree", "7",
                              "--no-circle", "--baseline-gain", "8"])
        settings = config_from_args(args)
        self.assertEqual(settings["render"]["color"], "blue")
        self.assertEqual(settings["render"]["detail"], 7)
        self.assertFalse(settings["render"]["circle"])
        self.assertEqual(settings["render"]["baseline_gain"], 8)
        self.assertEqual(settings["preprocess"], defaults["preprocess"])

    def test_unknown_invalid_and_method_restrictions(self):
        cases = [
            "version = 2", 'version = 1\n[unknown]\nx = 1',
            "version = 1\n[render]\nunknown = 1", "version = 1\n[render]\nheight = true",
            "version = 1\n[features]\ncompression = nan",
            "version = 1\n[features]\nbands_hz = [0, 100, 50]",
            "version = 1\n[features]\nband_weights = [0]",
            'version = 1\n[render]\nalgorithm = "energy"\nfitting = "poly"',
            "version = 1\n[render]\npoints = 64\ndetail = 128",
            "version = 1\n[render]\nnominal_radius = 0.45\ndeformation = 0.35",
            "version = 1\n[render]\nsmoothness = -0.1",
            "version = 1\n[render]\nbaseline_gain = -1",
            "version = 1\n[render]\nbaseline_gain = 65",
            "version = 1\n[render]\nbaseline_gain = nan",
            "version = 1\n[pitch_estimation]\nfloor_hz = 0",
            "version = 1\n[pitch_estimation]\nfloor_hz = 600",
            "version = 1\n[pitch_estimation]\nceiling_hz = 50",
            "version = 1\n[pitch_estimation]\nhop_ms = 0",
            "version = 1\n[pitch_estimation]\nvoicing_threshold = 1.1",
            "version = 1\n[pitch_estimation]\nvoicing_threshold = nan",
        ]
        config = self.work / "bad.toml"
        for text in cases:
            with self.subTest(text=text):
                config.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(config)
        result = self.run_cli("main.py", self.wav, "no.svg", "--algorithm", "energy", "--fitting", "poly", ok=False)
        self.assertIn("only", result.stderr)
        self.assertFalse((self.work / "no.svg").exists())

    def test_style_is_xml_escaped(self):
        self.settings["render"]["color"] = 'red" onload="bad & <'
        root = ET.fromstring(render_file(self.wav, self.settings)[1])
        self.assertEqual(root.find("s:path", NS).get("stroke"), self.settings["render"]["color"])
        self.assertNotIn("onload", root.find("s:path", NS).attrib)

    def test_project_defaults_schema_is_strict(self):
        for key in ("typo", "height"):
            settings = copy.deepcopy(self.settings)
            if key == "typo":
                settings["render"][key] = 1
            else:
                del settings["render"][key]
            with self.assertRaises(ValueError):
                validate_config(settings)


class PreprocessingTests(FixtureCase):
    def test_mono_stereo_dc_gain(self):
        mono = self.audio()
        stereo = self.work / "stereo.wav"
        write_wav(stereo, signal(), channels=2)
        np.testing.assert_array_equal(mono.samples, self.audio(stereo).samples)
        shifted = self.work / "shifted.wav"
        write_wav(shifted, signal(gain=0.3) + 0.1)
        np.testing.assert_allclose(mono.samples, self.audio(shifted).samples, atol=0.00015)
        self.assertAlmostEqual(float(np.mean(mono.samples)), 0, places=10)
        self.assertAlmostEqual(float(np.max(np.abs(mono.samples))), 0.9, places=10)
        self.assertEqual(mono.sample_rate, 8000)
        # Channel averaging occurs before normalization, without dropping frames.
        pair = np.column_stack((signal(gain=0.6), signal(gain=0.2)))
        write_wav(stereo, pair, channels=2)
        np.testing.assert_allclose(mono.samples, self.audio(stereo).samples, atol=0.00015)
        self.assertEqual(self.audio(stereo).original_frames, len(pair))

    def test_trim_offsets_internal_pause(self):
        rate = 8000
        tone = signal(duration=0.1, hz=200)
        values = np.concatenate((np.zeros(800), tone, np.zeros(400), tone, np.zeros(800)))
        write_wav(self.wav, values)
        audio = self.audio()
        self.assertGreater(audio.start, 650)
        self.assertLess(audio.start, 810)
        self.assertEqual(audio.end, len(values) - audio.start + 1)
        internal = audio.samples[1600-audio.start:2000-audio.start]
        np.testing.assert_allclose(internal, 0, atol=1e-10)
        self.assertEqual(audio.original_frames, len(values))

    def test_silence_dc_empty_short_and_unsupported(self):
        for values, width in [(np.zeros(400), 2), (np.full(400, 0.2), 2),
                              (np.array([]), 2), (np.ones(3), 2), (signal(), 1)]:
            with self.subTest(size=len(values), width=width):
                write_wav(self.wav, values, width=width)
                with self.assertRaises(ValueError):
                    self.audio()

    def test_truncated_wav(self):
        data = self.wav.read_bytes()
        self.wav.write_bytes(data[:-10])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            self.audio()

    def test_exact_cyclic_float_median(self):
        rng = np.random.default_rng(42)
        for n in (1, 2, 3, 20, 101):
            samples = rng.integers(-4, 5, n) / 10
            for smoothness in (0, 0.02, 0.2, 0.8, 0.999, 1):
                if smoothness == 0:
                    expected = list(samples)
                elif smoothness == 1:
                    expected = [float(samples.mean())] * n
                else:
                    window = int(1 + smoothness * (n - 1)) | 1
                    expected = [statistics.median(samples[(i+j) % n] for j in range(-window//2+1, window//2+1))
                                for i in range(n)]
                np.testing.assert_allclose(smooth_waveform(samples, smoothness), expected, atol=1e-15)


class FeatureTests(FixtureCase):
    def test_rms_analytic_and_time(self):
        # Whole periods in a rectangular frame: RMS = amplitude / sqrt(2).
        self.settings["preprocess"]["normalize_peak"] = 0.8
        write_wav(self.wav, signal(duration=0.5, hz=200, gain=0.5))
        audio = self.audio()
        values = frame_features(audio, self.settings["features"])
        self.assertEqual(values["time"][0], 0)
        self.assertEqual(values["time"][-1], 1)
        constant = np.sin(2*np.pi*200*np.arange(4000)/8000) * 0.8
        audio = type(audio)(constant, 8000, 4000, 0, 4000)
        rms = frame_features(audio, self.settings["features"])["energy"]
        expected = (0.8 / math.sqrt(2)) ** self.settings["features"]["compression"]
        np.testing.assert_allclose(rms[1:-1], expected, atol=1e-12)
        self.assertLess(rms[0], expected)

    def test_energy_polarity_gain_and_padding_resistance(self):
        # Each recording has enough surrounding silence to retain the configured trim padding.
        write_wav(self.wav, np.pad(signal(), (400, 400)))
        for method in ALGORITHMS[1:3]:
            self.settings["render"]["algorithm"] = method
            reference = sample_path(render_file(self.wav, self.settings)[1])
            variants = [np.pad(-signal(), (400, 400)), np.pad(signal(gain=0.25), (400, 400)),
                        np.pad(signal(), (800, 800))]
            for i, variant in enumerate(variants):
                path = self.work / f"variant-{i}.wav"
                write_wav(path, variant)
                actual = sample_path(render_file(path, self.settings)[1])
                np.testing.assert_allclose(actual, reference, atol=0.003)

    def test_energy_envelope_and_relative_timbre(self):
        audio = self.audio()
        energy = frame_features(audio, self.settings["features"])["energy"]
        self.assertGreater(float(energy.max() - energy.min()), 0.2)
        low = frame_features(audio, self.settings["features"])
        other = self.work / "high.wav"
        write_wav(other, signal(hz=2200))
        high = frame_features(self.audio(other), self.settings["features"])
        self.assertGreater(float(high["timbre"].mean() - low["timbre"].mean()), 0.5)
        for method, key in (("energy-timbre", "timbre_weight"), ("energy-pitch", "pitch_weight")):
            settings = copy.deepcopy(self.settings["features"])
            settings[key] = 0
            pitch = Pitch(((0, audio.original_frames, 220),))
            values = frame_features(audio, settings, pitch if method == "energy-pitch" else None)
            np.testing.assert_array_equal(feature_profile(values, method, settings),
                                          feature_profile(values, "energy", settings))

    def test_absolute_pitch_inflection_and_unvoiced(self):
        audio = self.audio()
        features = self.settings["features"]
        low = frame_features(audio, features, Pitch(((0, audio.original_frames, 110),)))
        high = frame_features(audio, features, Pitch(((0, audio.original_frames, 440),)))
        self.assertLess(low["absolute"].mean(), 0)
        self.assertGreater(high["absolute"].mean(), 0)
        np.testing.assert_allclose(low["inflection"], 0, atol=1e-14)
        profile_low = feature_profile(low, "energy-pitch", features)
        profile_high = feature_profile(high, "energy-pitch", features)
        a = periodic_coefficients(profile_low, 8, 0.65)
        b = periodic_coefficients(profile_high, 8, 0.65)
        self.assertGreater(abs(a[1]-b[1]), 0.05)  # Absolute pitch survives removal of DC radius.
        count = audio.original_frames
        pitch = Pitch(((0, count//3, 165), (count//3, 2*count//3, None), (2*count//3, count, 330)))
        varying = frame_features(audio, features, pitch)
        self.assertLess(varying["inflection"][0], 0)
        self.assertGreater(varying["inflection"][-1], 0)
        mask = (np.arange(audio.start, audio.end) >= count//3) & (np.arange(audio.start, audio.end) < 2*count//3)
        np.testing.assert_array_equal(varying["hz"][mask], 0)
        np.testing.assert_array_equal(varying["pitch"][mask], 0)

    def test_pitch_geometry_gain_polarity_and_size_invariance(self):
        known = self.work / "known"
        generate(known)
        wav, sidecar = known / "changing.wav", known / "changing.pitch.json"
        self.settings["render"]["algorithm"] = "energy-pitch"
        reference = sample_path(render_file(wav, self.settings, sidecar)[1])
        with wave.open(str(wav), "rb") as source:
            original = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2") / 32767
        # Polarity/gain transformations preserve the actual known generating frequency controls.
        for multiplier in (-1, 0.5):
            transformed = self.work / f"gain-{multiplier}.wav"
            write_wav(transformed, original * multiplier)
            data = json.loads(sidecar.read_text())
            data["wav_sha256"] = wav_hash(transformed)
            data["source"]["description"] += f" Output polarity/gain multiplied by {multiplier}."
            supplied = self.work / f"gain-{multiplier}.json"
            supplied.write_text(json.dumps(data))
            actual = sample_path(render_file(transformed, self.settings, supplied)[1])
            np.testing.assert_allclose(actual, reference, atol=0.01)
        larger = copy.deepcopy(self.settings)
        larger["render"].update(max_width=1024, height=512)
        scaled = sample_path(render_file(wav, larger, sidecar)[1])
        np.testing.assert_allclose(scaled / 2, reference, atol=1e-9)


class PitchTests(FixtureCase):
    def setUp(self):
        super().setUp()
        self.fixtures = self.work / "known"
        generate(self.fixtures)
        self.wav = self.fixtures / "interrupted.wav"
        self.sidecar = self.fixtures / "interrupted.pitch.json"
        self.data = json.loads(self.sidecar.read_text())
        with wave.open(str(self.wav), "rb") as wf:
            self.rate, self.count = wf.getframerate(), wf.getnframes()

    def load(self, data=None):
        if data is not None:
            self.sidecar.write_text(json.dumps(data))
        return load_pitch(self.sidecar, self.wav, self.rate, self.count)

    def test_valid_contract_and_alignment(self):
        pitch = self.load()
        self.assertEqual(pitch.metadata, {"pitch-source": "known-generating-controls"})
        np.testing.assert_array_equal(pitch.values([0, 1999, 2000, 3199, 3200]), [220, 220, 0, 0, 330])
        self.settings["preprocess"]["trim_padding_ms"] = 0
        audio = self.audio()
        self.assertGreater(audio.start, 0)
        features = frame_features(audio, self.settings["features"], pitch)
        expected = pitch.values(np.arange(audio.start, audio.end))
        np.testing.assert_array_equal(features["hz"], expected)

    def test_reject_malformed_estimated_and_mismatches(self):
        mutations = [
            lambda d: d.update(version=2),
            lambda d: d.update(sample_rate=44100),
            lambda d: d.update(wav_sha256="0" * 64),
            lambda d: d.update(extra=1),
            lambda d: d["source"].update(category="estimated"),
            lambda d: d["source"].update(category="manually-corrected-estimate"),
            lambda d: d["source"].update(description=""),
            lambda d: d["intervals"][0].update(start=1),
            lambda d: d["intervals"][1].update(start=1999),
            lambda d: d["intervals"][1].update(start=2001),
            lambda d: d["intervals"][-1].update(end=self.count-1),
            lambda d: d["intervals"][0].update(hz=0),
            lambda d: d["intervals"][0].update(hz=float("nan")),
            lambda d: d["intervals"][0].update(hz=5000),
            lambda d: d["intervals"][0].update(hz=True),
            lambda d: d["intervals"][0].update(start=False),
            lambda d: d.update(intervals=[]),
            lambda d: d["intervals"][0].pop("hz"),
        ]
        for mutation in mutations:
            data = copy.deepcopy(self.data)
            mutation(data)
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    self.load(data)
        self.sidecar.write_text("{bad")
        with self.assertRaises(ValueError):
            self.load()
        self.sidecar.write_text('{"version": 1, "version": 1}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()
        with self.assertRaises(ValueError):
            Pitch(((0, 100, 220),)).values([100])

    def test_generator_is_deterministic(self):
        second = self.work / "second"
        generate(second)
        for path in self.fixtures.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())


class PitchEstimationTests(FixtureCase):
    def estimate(self, path=None):
        return estimate_pitch(self.audio(path), self.settings["pitch_estimation"])

    def test_constant_pitch_accuracy_across_rates_and_registers(self):
        for rate in (8000, 16000, 44100, 48000):
            for expected in (110, 220, 440):
                with self.subTest(rate=rate, hz=expected):
                    write_wav(self.wav, signal(rate=rate, hz=expected), rate=rate)
                    pitch = self.estimate()
                    self.assertEqual(pitch.source, "estimated")
                    self.assertIn("Praat", pitch.estimator)
                    self.assertIn("Parselmouth", pitch.estimator)
                    self.assertEqual(pitch.intervals[0][0], 0)
                    self.assertEqual(pitch.intervals[-1][1], round(0.6 * rate))
                    for current, following in zip(pitch.intervals, pitch.intervals[1:]):
                        self.assertEqual(current[1], following[0])
                    interior = pitch.values(np.arange(round(0.1 * rate), round(0.5 * rate)))
                    np.testing.assert_allclose(interior, expected, atol=0.5, rtol=0)

    def test_inflection_and_trimmed_original_frame_alignment(self):
        known = self.work / "known"
        generate(known)
        changing = self.estimate(known / "changing.wav")
        for start, expected in ((0, 165), (0.25, 220), (0.5, 330)):
            indices = np.arange(round((start + 0.08) * 8000), round((start + 0.17) * 8000))
            np.testing.assert_allclose(changing.values(indices), expected, atol=1, rtol=0)
        with wave.open(str(known / "interrupted.wav"), "rb") as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2") / 32767
        write_wav(self.wav, np.pad(samples, (1600, 2400)))
        self.settings["preprocess"]["trim_padding_ms"] = 0
        audio = self.audio()
        self.assertGreater(audio.start, 1600)
        pitch = self.estimate()
        self.assertEqual(pitch.intervals[-1][1], audio.original_frames)
        np.testing.assert_array_equal(pitch.values(np.arange(1600)), 0)
        np.testing.assert_array_equal(pitch.values(np.arange(6800, audio.original_frames)), 0)
        np.testing.assert_allclose(pitch.values(np.arange(2400, 2800)), 220, atol=1, rtol=0)
        np.testing.assert_allclose(pitch.values(np.arange(5600, 6000)), 330, atol=1, rtol=0)
        features = frame_features(audio, self.settings["features"], pitch)
        original = np.arange(audio.start, audio.end)
        pause = (original >= 4080) & (original < 4320)
        np.testing.assert_array_equal(features["hz"][pause], 0)
        np.testing.assert_array_equal(features["pitch"][pause], 0)
        np.testing.assert_array_equal(features["hz"], pitch.values(original))

    def test_polarity_gain_stereo_and_padding(self):
        interior = np.arange(800, 4000)
        reference = self.estimate().values(interior)
        for multiplier, channels, padding in ((-1, 1, 0), (0.5, 2, 0), (1, 1, 1600)):
            write_wav(self.wav, np.pad(signal() * multiplier, (padding, padding)), channels=channels)
            actual = self.estimate().values(interior + padding)
            np.testing.assert_allclose(actual, reference, atol=0.1, rtol=0)

    def test_geometry_metadata_ablation_and_size_independence(self):
        write_wav(self.wav, signal(hz=110))
        self.settings["render"]["algorithm"] = "energy-pitch"
        result = render_file(self.wav, self.settings)
        self.assertEqual(result, render_file(self.wav, self.settings))
        for document in result:
            root = ET.fromstring(document)
            self.assertEqual(root.get("data-pitch-source"), "estimated")
            self.assertIn("Praat", root.get("data-pitch-estimator"))
        curve = paths(result[1])[0]
        self.assertTrue(curve.isclosed())
        self.assertTrue(curve.iscontinuous())
        for current, following in zip(curve, list(curve)[1:] + [curve[0]]):
            self.assertAlmostEqual(abs(current.derivative(1) - following.derivative(0)), 0, delta=1e-8)
        points = sample_path(result[1])
        self.assertTrue(np.isfinite(points).all())
        radii = np.abs(points - (128 + 128j))
        self.assertGreater(radii.min(), 102.4 * 0.82 - 1e-6)
        self.assertLess(radii.max(), 102.4 * 1.18 + 1e-6)
        bigger = copy.deepcopy(self.settings)
        bigger["render"].update(max_width=1024, height=512)
        np.testing.assert_allclose(sample_path(render_file(self.wav, bigger)[1]) / 2, points, atol=1e-9)
        self.settings["features"]["pitch_weight"] = 0
        without_pitch = sample_path(render_file(self.wav, self.settings)[1])
        self.assertGreater(float(np.max(np.abs(points - without_pitch))), 0.1)
        self.settings["render"]["algorithm"] = "energy"
        np.testing.assert_array_equal(without_pitch, sample_path(render_file(self.wav, self.settings)[1]))

    def test_short_unvoiced_and_invalid_rate_fail_explicitly(self):
        write_wav(self.wav, signal(duration=0.02))
        with self.assertRaisesRegex(ValueError, "too short for pitch estimation"):
            self.estimate()
        write_wav(self.wav, np.random.default_rng(42).normal(0, 0.1, 6400))
        with self.assertRaisesRegex(ValueError, "No voiced pitch"):
            self.estimate()
        self.settings["render"]["algorithm"] = "energy-pitch"
        self.settings["features"]["pitch_weight"] = 0
        with self.assertRaisesRegex(ValueError, "No voiced pitch"):
            render_file(self.wav, self.settings)
        self.settings["pitch_estimation"]["ceiling_hz"] = 5000
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            self.estimate()

    def test_missing_failed_and_invalid_estimator(self):
        import parselmouth
        with mock.patch.dict(sys.modules, {"parselmouth": None}):
            with self.assertRaisesRegex(ValueError, "requires praat-parselmouth"):
                self.estimate()
            for method in ALGORITHMS[:3]:
                self.settings["render"]["algorithm"] = method
                render_file(self.wav, self.settings)
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as error:
                    main_command([str(self.wav), str(self.work / "missing.svg"), "--algorithm", "energy-pitch"])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("requires praat-parselmouth", stderr.getvalue())
            self.assertFalse((self.work / "missing.svg").exists())
        with mock.patch("parselmouth.Sound", side_effect=parselmouth.PraatError("test failure")):
            with self.assertRaisesRegex(ValueError, "Praat pitch estimation failed: test failure"):
                self.estimate()
        for frequencies in ([220, float("nan")], [220, -1], [220, 5000], [0, 0]):
            track = mock.Mock()
            track.selected_array = {"frequency": np.array(frequencies)}
            track.xs.return_value = np.array([0.1, 0.11])
            with mock.patch("parselmouth.Sound") as sound:
                sound.return_value.to_pitch_ac.return_value = track
                with self.assertRaisesRegex(ValueError, "invalid pitch track|No voiced pitch"):
                    self.estimate()

    def test_known_sidecars_bypass_estimation_and_fail_without_fallback(self):
        known = self.work / "known"
        generate(known)
        wav, sidecar = known / "changing.wav", known / "changing.pitch.json"
        self.settings["render"]["algorithm"] = "energy-pitch"
        with mock.patch("ripple.estimate_pitch", side_effect=AssertionError("Must not estimate supplied pitch")):
            with mock.patch.dict(sys.modules, {"parselmouth": None}):
                diagnostic, circular = render_file(wav, self.settings, sidecar)
            for document in (diagnostic, circular):
                root = ET.fromstring(document)
                self.assertEqual(root.get("data-pitch-source"), "known-generating-controls")
                self.assertIsNone(root.get("data-pitch-estimator"))
            sidecar.write_text("{}")
            with self.assertRaises(ValueError):
                render_file(wav, self.settings, sidecar)

    def test_overlay_retains_each_words_pitch_provenance(self):
        self.settings["render"]["algorithm"] = "energy-pitch"
        estimated = render_file(self.wav, self.settings)[1]
        known = circular_svg(np.zeros(1, dtype=complex), self.settings["render"],
                             {"pitch-source": "known-generating-controls"})
        combined = overlay([known, estimated], ["known", "estimated"], "energy-pitch")
        root = ET.fromstring(combined)
        self.assertEqual(root.get("data-pitch-source"), "mixed")
        self.assertIsNone(root.get("data-pitch-estimator"))
        groups = root.findall("s:g", NS)
        self.assertEqual([g.get("data-pitch-source") for g in groups], ["known-generating-controls", "estimated"])
        self.assertIn("Praat", groups[1].get("data-pitch-estimator"))
        preview = ET.fromstring(comparison_preview({"energy-pitch": combined}, 256))
        self.assertEqual(preview.find(".//s:text", NS).text, "energy-pitch (mixed sources)")


class GeometryTests(FixtureCase):
    def test_all_methods_deterministic_shared_diameter_and_finite_closed_paths(self):
        known = self.work / "known"
        generate(known)
        wav = known / "changing.wav"
        for method in ALGORITHMS:
            self.settings["render"]["algorithm"] = method
            pitch = known / "changing.pitch.json" if method == "energy-pitch" else None
            first = render_file(wav, self.settings, pitch)
            self.assertEqual(first, render_file(wav, self.settings, pitch))
            diagnostic, circular = map(ET.fromstring, first)
            self.assertEqual(diagnostic.tag, f"{{{SVG_NS}}}svg")
            self.assertEqual(circular.get("width"), "256")
            self.assertEqual(circular.get("height"), "256")
            self.assertEqual(float(circular.get("data-nominal-diameter")), 204.8)
            self.assertEqual(circular.get("data-center"), "128.0,128.0")
            self.assertEqual(len(circular.findall("s:path", NS)), 1)
            path = paths(first[1])[0]
            self.assertTrue(path.isclosed())
            self.assertTrue(path.iscontinuous())
            for segment, following in zip(path, list(path)[1:] + [path[0]]):
                self.assertAlmostEqual(abs(segment.derivative(1)-following.derivative(0)), 0, delta=1e-8)
            points = sample_path(first[1], 4096)
            radius = np.abs(points - (128 + 128j))
            self.assertTrue(np.isfinite(points).all())
            self.assertGreater(radius.min(), 102.4 * 0.82 - 1e-6)
            self.assertLess(radius.max(), 102.4 * 1.18 + 1e-6)
            # A strictly increasing polar angle on a closed star-shaped curve cannot self-intersect.
            angles = np.unwrap(np.angle(np.r_[points, points[0]] - (128+128j)))
            self.assertTrue(np.all(np.diff(angles) > 0), method)

    def test_size_invariance_reference_and_zero_deformation(self):
        for method in ALGORITHMS[:3]:
            self.settings["render"]["algorithm"] = method
            before = sample_path(render_file(self.wav, self.settings)[1]) / 256
            bigger = copy.deepcopy(self.settings)
            bigger["render"].update(max_width=1024, height=512)
            after = sample_path(render_file(self.wav, bigger)[1]) / 512
            np.testing.assert_allclose(before, after, atol=1e-11)
        self.settings["render"].update(deformation=0, circle=True)
        circular = render_file(self.wav, self.settings)[1]
        self.assertEqual(len(paths(circular)), 2)
        points = sample_path(circular)
        np.testing.assert_allclose(np.abs(points - (128+128j)), 102.4, atol=1e-7)

    def test_bounded_profiles_do_not_amplify_residuals_and_roundness(self):
        theta = np.arange(2048) * 2*np.pi/2048
        tiny = periodic_coefficients(1e-9 * np.cos(theta), 8, 0)
        values, _ = evaluate_profile(tiny, theta)
        self.assertLess(np.max(np.abs(values)), 2e-9)
        big = periodic_coefficients(100*np.cos(theta) + 40*np.sin(8*theta), 8, 0)
        values, _ = evaluate_profile(big, theta)
        self.assertLessEqual(np.max(np.abs(values)), 1+1e-12)
        raw = periodic_coefficients(np.cos(theta)+np.cos(8*theta), 8, 0)
        rounded = periodic_coefficients(np.cos(theta)+np.cos(8*theta), 8, 1)
        self.assertLess(abs(rounded[8]/rounded[1]), abs(raw[8]/raw[1]))

    def test_light_gray_reference_preserves_curve_and_nominal_radius(self):
        render = self.settings["render"]
        render["stroke_width"] = 3
        theta = np.arange(2048) * 2 * np.pi / 2048
        coefficients = periodic_coefficients(np.sin(theta), render["detail"], render["roundness"])
        original = ET.fromstring(circular_svg(coefficients, render)).find("s:path", NS)
        render["circle"] = True
        root = ET.fromstring(circular_svg(coefficients, render))
        reference, artwork = root.findall("s:path", NS)
        self.assert_reference_circle(reference)
        self.assertEqual(artwork.attrib, original.attrib)
        path = parse_path(reference.get("d"))
        self.assertTrue(path.isclosed())
        self.assertTrue(path.iscontinuous())
        center = complex(*map(float, root.get("data-center").split(",")))
        radius = float(root.get("data-nominal-diameter")) / 2
        samples = [segment.point(t) for segment in path for t in (0, 0.25, 0.5, 0.75, 1)]
        np.testing.assert_allclose(np.abs(np.array(samples) - center), radius, atol=1e-7, rtol=0)

    def test_limiter_preserves_in_bounds_profiles_and_bounds_between_samples(self):
        theta = np.arange(4096) * 2 * np.pi / 4096
        profile = 0.6 * np.cos(theta) + 0.5 * np.sin(2 * theta)
        self.assertLess(np.max(np.abs(profile)), 1)
        coefficients = periodic_coefficients(profile, 8, 0)
        values, _ = evaluate_profile(coefficients, theta)
        np.testing.assert_allclose(values, profile, atol=1e-12, rtol=0)
        for detail in (8, 32, 128):
            with self.subTest(detail=detail):
                profile = 5 * np.cos(detail * theta + 0.13) + 2 * np.sin((detail - 1) * theta + 0.7)
                coefficients = periodic_coefficients(profile, detail, 0)
                # Offset from the limiter's angular grid to check the continuous bound.
                between = (np.arange(16384) + 0.5) * 2 * np.pi / 16384
                values, _ = evaluate_profile(coefficients, between)
                self.assertLessEqual(np.max(np.abs(values)), 1 + 1e-12)

    def test_baseline_gain_is_fixed_and_does_not_inflate_tiny_residuals(self):
        theta = np.arange(2048) * 2 * np.pi / 2048
        render = self.settings["render"]
        audio = self.audio()
        center, nominal = 128 + 128j, 102.4
        with mock.patch("ripple.baseline_profile", return_value=(0.01 * np.cos(theta), audio.samples, [])):
            deviations = {}
            for gain in (0, 1, 16):
                render["baseline_gain"] = gain
                circular = render_file(self.wav, self.settings)[1]
                # Segment anchors share angles; arc-length positions change with gain.
                anchors = np.array([segment.start for segment in paths(circular)[-1]])
                deviations[gain] = np.abs(anchors - center) - nominal
            np.testing.assert_allclose(deviations[0], 0, atol=1e-7, rtol=0)
            np.testing.assert_allclose(deviations[16], 16 * deviations[1], atol=1e-6, rtol=0)
        with mock.patch("ripple.baseline_profile", return_value=(1e-9 * np.cos(theta), audio.samples, [])):
            circular = render_file(self.wav, self.settings)[1]
            self.assertLess(np.max(np.abs(np.abs(sample_path(circular) - center) - nominal)), 1e-6)
        for method in ALGORITHMS[1:3]:
            render["algorithm"] = method
            render["baseline_gain"] = 0
            first = render_file(self.wav, self.settings)[1]
            render["baseline_gain"] = 64
            self.assertEqual(first, render_file(self.wav, self.settings)[1])

    def test_legacy_poly_and_insufficient_extrema(self):
        self.settings["render"].update(fitting="poly", detail=4)
        circular = render_file(self.wav, self.settings)[1]
        path = paths(circular)[0]
        self.assertTrue(path.isclosed())
        self.settings["render"]["smoothness"] = 1
        with self.assertRaisesRegex(ValueError, "midpoints"):
            render_file(self.wav, self.settings)

    def test_generic_bezier_joins(self):
        for closed in (False, True):
            path = parse_path(create_svg_path([(0, 0), (1, 2), (3, 3), (4, 0)], closed))
            pairs = list(zip(path, list(path)[1:]))
            if closed:
                pairs.append((path[-1], path[0]))
            for a, b in pairs:
                self.assertAlmostEqual(abs(a.derivative(1) - b.derivative(0)), 0, delta=1e-12)


class InterfaceTests(FixtureCase):
    def test_main_protects_root_defaults_for_both_outputs(self):
        protected = self.work / "ripple.toml"
        original = DEFAULT_CONFIG.read_bytes()
        protected.write_bytes(original)
        artwork = self.work / "artwork.svg"
        for args in ([str(self.wav), str(protected)],
                     [str(self.wav), str(artwork), "--diagnostic", str(protected)]):
            with self.subTest(args=args), mock.patch("main.DEFAULT_CONFIG", protected):
                with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    with self.assertRaises(SystemExit) as error:
                        main_command(args)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("must not overwrite", stderr.getvalue())
                self.assertEqual(protected.read_bytes(), original)
                self.assertFalse(artwork.exists())

    def test_four_cli_paths_diagnostics_and_errors(self):
        known = self.work / "known"
        generate(known)
        wav, pitch = known / "changing.wav", known / "changing.pitch.json"
        for method in ALGORITHMS:
            extra = ["--pitch", pitch] if method == "energy-pitch" else []
            self.run_cli("main.py", wav, f"{method}.svg", "--algorithm", method,
                         "--max-width", 512, "--height", 256, "--stroke-width", 1, "--smoothness", 0, *extra)
            self.assertTrue((self.work / f"{method}.diagnostic.svg").is_file())
            self.assertEqual(ET.parse(self.work / f"{method}.svg").getroot().get("data-algorithm"), method)
        self.assertFalse((self.work / "tmp.svg").exists())
        result = self.run_cli("main.py", wav, "estimated.svg", "--algorithm", "energy-pitch")
        self.assertIn("estimated automatically with Praat", result.stdout)
        self.assertEqual(ET.parse(self.work / "estimated.svg").getroot().get("data-pitch-source"), "estimated")
        self.assertFalse((self.work / "x.svg").exists())
        self.run_cli("main.py", wav, "x.svg", "--pitch", pitch, ok=False)
        self.run_cli("main.py", wav, wav, ok=False)
        self.run_cli("main.py", wav, "x.svg", "--diagnostic", "x.svg", ok=False)
        self.run_cli("main.py", wav, "x.svg", "--diagnostic", "custom.svg")
        self.assertTrue((self.work / "custom.svg").exists())
        self.assertFalse((self.work / "x.diagnostic.svg").exists())
        root = ET.parse(self.work / "x.svg").getroot()
        self.assertEqual(root.get("width"), "1024")
        self.assertEqual(root.get("height"), "1024")
        reference, artwork = root.findall("s:path", NS)
        self.assert_reference_circle(reference)
        self.assertEqual(artwork.get("stroke"), "black")
        self.assertEqual(artwork.get("stroke-width"), "3.0")

    def test_parallel_cli_diagnostic_isolation(self):
        processes = [subprocess.Popen([sys.executable, str(SRC / "main.py"), str(self.wav), f"word-{i}.svg",
                                      "--smoothness", "0", "--max-width", "512", "--height", "256", "--stroke-width", "1"],
                                     cwd=self.work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for i in range(2)]
        for process in processes:
            _, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr)
        for i in range(2):
            self.assertTrue((self.work / f"word-{i}.diagnostic.svg").exists())
        self.assertFalse((self.work / "tmp.svg").exists())

    def test_comparison_multiword_overlays_and_labels(self):
        known = self.work / "known"
        generate(known)
        inputs = [known / f"{name}.wav" for name in ("constant", "interrupted")]
        pitches = [known / f"{name}.pitch.json" for name in ("constant", "interrupted")]
        output = self.work / "comparison"
        self.run_cli("compare.py", *inputs, "--output-dir", output, "--pitch", *pitches,
                     "--smoothness", 0, "--max-width", 512, "--height", 256, "--stroke-width", 1, "--no-circle")
        preview = ET.parse(output / "comparison.svg").getroot()
        self.assertEqual([text.text for text in preview.findall(".//s:text", NS)],
                         [*ALGORITHMS[:3], "energy-pitch (known)"])
        self.assertEqual(float(preview.get("width")), 1024)
        for method in ALGORITHMS:
            overlay = ET.parse(output / method / "overlay.svg").getroot()
            self.assertEqual(overlay.get("data-center"), "128.0,128.0")
            self.assertEqual(float(overlay.get("data-nominal-diameter")), 204.8)
            self.assertEqual(len(overlay.findall("s:g", NS)), 2)
            self.assertTrue(all("transform" not in g.attrib for g in overlay.findall("s:g", NS)))
            self.assertEqual(len(overlay.findall(".//s:path", NS)), 2)
            self.assertEqual(len(list((output / method / "words").glob("*.svg"))), 2)
        self.run_cli("compare.py", *inputs, "--output-dir", "missing", "--pitch", pitches[0], ok=False)
        self.assertFalse((self.work / "missing").exists())
        bad = self.work / "bad.json"
        bad.write_text("{}")
        self.run_cli("compare.py", *inputs, "--output-dir", "bad", "--pitch", pitches[0], bad, ok=False)
        self.assertFalse((self.work / "bad").exists())
        self.run_cli("compare.py", *inputs, "--output-dir", output, "--pitch", *pitches, ok=False)

    def test_collision_safe_three_method_real_comparison(self):
        # Same stem and even duplicate paths must still receive distinct output names.
        for folder in ("a", "b"):
            (self.work / folder).mkdir()
            write_wav(self.work / folder / "word.wav", signal(duration=0.4 if folder == "a" else 0.6))
        self.run_cli("compare.py", self.work / "a/word.wav", self.work / "b/word.wav", self.work / "a/word.wav",
                     "--output-dir", "three", "--algorithms", "baseline", "energy", "energy-timbre",
                     "--smoothness", 0, "--max-width", 512, "--height", 256, "--stroke-width", 1)
        for method in ALGORITHMS[:3]:
            self.assertEqual(len(list((self.work / "three" / method / "words").glob("*.svg"))), 3)
        real = REPO / "samples/HelloWorld.wav"
        self.run_cli("compare.py", real, "--output-dir", "speech", "--algorithms", *ALGORITHMS[:3],
                     "--smoothness", 0, "--poly-degree", 4, "--max-width", 512, "--height", 256, "--stroke-width", 1)
        self.assertFalse((self.work / "speech/energy-pitch").exists())

    def test_comparison_reference_is_once_below_all_words_and_preview(self):
        known = self.work / "known"
        generate(known)
        labels = ["constant", "changing", "interrupted", "constant"]
        inputs = [known / f"{label}.wav" for label in labels]
        pitches = [known / f"{label}.pitch.json" for label in labels]
        output = self.work / "comparison"
        self.run_cli("compare.py", *inputs, "--output-dir", output, "--pitch", *pitches,
                     "--smoothness", 0, "--max-width", 512, "--height", 256,
                     "--stroke-width", 3, "--deformation", 0.18)
        preview = ET.parse(output / "comparison.svg").getroot()
        self.assertEqual([text.text for text in preview.findall(".//s:text", NS)],
                         [*ALGORITHMS[:3], "energy-pitch (known)"])
        self.assertEqual(len(preview.findall("s:g", NS)), len(ALGORITHMS))
        for method, panel in zip(ALGORITHMS, preview.findall("s:g", NS)):
            with self.subTest(method=method):
                overlay = ET.parse(output / method / "overlay.svg").getroot()
                self.assertEqual(len(overlay), 5)
                self.assert_reference_circle(overlay[0])
                self.assertEqual(len(overlay.findall(".//s:path", NS)), 5)
                groups = overlay.findall("s:g", NS)
                self.assertEqual([g.get("data-word") for g in groups], labels)
                self.assertEqual([g.find("s:title", NS).text for g in groups], labels)
                words = sorted((output / method / "words").glob("*.svg"))
                self.assertEqual(len(words), len(labels))
                for group, word in zip(groups, words):
                    reference, artwork = ET.parse(word).getroot().findall("s:path", NS)
                    self.assertEqual(reference.attrib, overlay[0].attrib)
                    self.assertEqual(len(group.findall("s:path", NS)), 1)
                    self.assertEqual(group.find("s:path", NS).attrib, artwork.attrib)
                    self.assertEqual(artwork.get("stroke"), "black")
                    self.assertEqual(artwork.get("stroke-width"), "3.0")
                    self.assertIsNone(artwork.get("stroke-dasharray"))
                preview_art = panel.find("s:g", NS)
                self.assertEqual([ET.tostring(child) for child in preview_art],
                                 [ET.tostring(child) for child in overlay])

    def test_wav_only_four_method_comparison_and_pitch_failures(self):
        output = self.work / "automatic"
        result = self.run_cli("compare.py", self.wav, REPO / "samples/HelloWorld.wav",
                             "--output-dir", output, "--smoothness", 0,
                             "--max-width", 512, "--height", 256, "--stroke-width", 1)
        self.assertIn("estimated automatically with Praat", result.stdout)
        preview = ET.parse(output / "comparison.svg").getroot()
        self.assertEqual([text.text for text in preview.findall(".//s:text", NS)],
                         [*ALGORITHMS[:3], "energy-pitch (estimated)"])
        art = preview.findall("s:g", NS)[-1].find("s:g", NS)
        self.assertEqual(art.get("data-pitch-source"), "estimated")
        self.assert_reference_circle(art[0])
        for method in ALGORITHMS:
            self.assertEqual(len(list((output / method / "words").glob("*.svg"))), 2)
        for word in (output / "energy-pitch/words").glob("*.svg"):
            self.assertEqual(ET.parse(word).getroot().get("data-pitch-source"), "estimated")
        noise = self.work / "noise.wav"
        write_wav(noise, np.random.default_rng(42).normal(0, 0.1, 6400))
        result = self.run_cli("main.py", noise, "failure.svg", "--algorithm", "energy-pitch", ok=False)
        self.assertIn("No voiced pitch", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.work / "failure.svg").exists())
        self.assertFalse((self.work / "failure.diagnostic.svg").exists())
        self.run_cli("compare.py", self.wav, noise, "--algorithms", "energy", "energy-pitch",
                     "--output-dir", "failure", ok=False)
        self.assertFalse((self.work / "failure").exists())

    def test_merge_reference_is_once_below_artwork_and_keeps_its_style(self):
        inputs = self.work / "merge-inputs"
        inputs.mkdir()
        self.settings["render"].update(circle=True, color="#e0e0e0")
        circular = render_file(self.wav, self.settings)[1]
        for name in ("a.svg", "b.svg"):
            (inputs / name).write_text(circular)
        self.run_cli("merge.py", inputs, "--width", 512, "--height", 512,
                     "--output", "merged.svg", "--thickness", 3, "--no-png")
        elements = ET.parse(self.work / "merged.svg").getroot().findall("s:path", NS)
        self.assertEqual(len(elements), 3)
        self.assert_reference_circle(elements[0])
        for element in elements[1:]:
            self.assertIsNone(element.get("data-role"))
            self.assertIsNone(element.get("stroke-dasharray"))
            self.assertEqual(element.get("stroke"), "#e0e0e0")
            self.assertEqual(element.get("stroke-width"), "3")
        np.testing.assert_allclose(sample_path((self.work / "merged.svg").read_text()),
                                   sample_path(circular) * 2, atol=1e-8)

    def test_merge_compatibility_and_optional_png(self):
        inputs = self.work / "merge-inputs"
        inputs.mkdir()
        circular = render_file(self.wav, self.settings)[1]
        (inputs / "a.svg").write_text(circular)
        (inputs / "b.svg").write_text(circular)
        examples = self.work / "output"
        examples.mkdir()
        for name in ("merged.svg", "merged.png"):
            (examples / name).write_bytes(b"existing example")
        result = self.run_cli("merge.py", inputs, "--no-png", ok=False)
        self.assertIn("--output", result.stderr)
        for name in ("merged.svg", "merged.png"):
            self.assertEqual((examples / name).read_bytes(), b"existing example")
        self.run_cli("merge.py", inputs, "--width", 512, "--height", 512,
                     "--output", "merged.svg", "--thickness", 3, "--no-png")
        root = ET.parse(self.work / "merged.svg").getroot()
        self.assertEqual(len(root.findall("s:path", NS)), 2)
        original = sample_path(circular) * 2
        actual = sample_path((self.work / "merged.svg").read_text())
        np.testing.assert_allclose(actual, original, atol=1e-8)
        self.assertTrue(all(p.get("stroke-width") == "3" for p in root.findall("s:path", NS)))
        self.run_cli("merge.py", inputs, "--output", inputs / "merged.svg", "--no-png", ok=False)
        self.run_cli("merge.py", inputs, "--output", "merged.svg", "--png", "merged.svg", ok=False)
        # Exercise optional and required missing-native-library behavior on every host.
        with mock.patch.dict(sys.modules, {"cairosvg": None}):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertFalse(export_png("x.svg", "x.png", 256, 256))
                self.assertIn("skipped", stderr.getvalue())
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                export_png("x.svg", "x.png", 256, 256, required=True)
        renderer = mock.Mock()
        renderer.svg2png.side_effect = OSError("native render failed")
        with mock.patch.dict(sys.modules, {"cairosvg": renderer}):
            with self.assertRaisesRegex(RuntimeError, "rendering failed"):
                export_png("x.svg", "x.png", 256, 256, required=True)
        try:
            import cairosvg  # noqa: F401
        except (ImportError, OSError):
            result = self.run_cli("merge.py", inputs, "--output", "with-png.svg", "--png", "explicit.png", ok=False)
            self.assertIn("unavailable", result.stderr)
            self.assertFalse((self.work / "explicit.png").exists())
        else:
            self.run_cli("merge.py", inputs, "--width", 256, "--height", 256,
                         "--output", "with-png.svg", "--png", "explicit.png")
            import struct
            data = (self.work / "explicit.png").read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", data[16:24]), (256, 256))


if __name__ == "__main__":
    unittest.main()
