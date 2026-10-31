import argparse
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
from svgpathtools import parse_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pendant import composition, main, render
from utils import SVG_NS


CIRCLE = ("M 85 50 C 85 69.329966 69.329966 85 50 85 "
          "C 30.670034 85 15 69.329966 15 50 "
          "C 15 30.670034 30.670034 15 50 15 "
          "C 69.329966 15 85 30.670034 85 50 Z")


class PendantTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.svg = self.work / "overlay.svg"
        self.root = ET.Element(f"{{{SVG_NS}}}svg", {
            "viewBox": "0 0 100 100", "data-center": "50,50",
            "data-nominal-diameter": "70", "data-orientation": "recording"})
        ET.SubElement(self.root, f"{{{SVG_NS}}}path", {"d": CIRCLE, "data-role": "reference-circle"})
        for name in ("First", "Second"):
            group = ET.SubElement(self.root, f"{{{SVG_NS}}}g", {"data-word": name})
            ET.SubElement(group, f"{{{SVG_NS}}}path", {"d": CIRCLE, "stroke": "black"})
        self.write()
        self.args = argparse.Namespace(input_svg=self.svg, output_dir=self.work / "result",
                                       diameter_mm=32, wire_mm=0.75, resolution=512, samples=32,
                                       threads=2, blender="blender", save_scene=False)

    def write(self):
        ET.ElementTree(self.root).write(self.svg, encoding="utf-8")

    def fake_renderer(self, command, **kwargs):
        self.assertIn("--disable-autoexec", command)
        self.assertIn("--python-exit-code", command)
        job_path = Path(command[-1])
        job = json.loads(job_path.read_text())
        self.assertEqual([w["name"] for w in job["words"]], ["First", "Second"])
        self.assertEqual(job["threads"], 2)
        (job_path.parent / "pendant.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 512, 512))
        (job_path.parent / "render.json").write_text(json.dumps({"body_components": 1, "word_count": 2}))
        if job["save_scene"]:
            (job_path.parent / "pendant.blend").write_bytes(b"scene fixture")

    def test_exact_cubics_shared_scale_and_guide_exclusion(self):
        result = composition(self.svg, 32, 0.75)
        self.assertEqual([word["name"] for word in result["words"]], ["First", "Second"])
        self.assertEqual(result["source_metadata"]["data-orientation"], "recording")
        self.assertEqual(len(result["source_sha256"]), 64)
        path = parse_path(CIRCLE)
        for word in result["words"]:
            for actual, segment in zip(word["cubics"], path):
                expected = [[(z.real - 50) * 0.032 / 70, (50 - z.imag) * 0.032 / 70, 0.000375]
                            for z in (segment.start, segment.control1, segment.control2, segment.end)]
                np.testing.assert_allclose(actual, expected, atol=1e-17, rtol=0)
            self.assertEqual(word["cubics"][0][0], word["cubics"][-1][-1])
        self.assertAlmostEqual(result["anchor_y"], 0.016, places=12)
        doubled = composition(self.svg, 64, 1.5)
        np.testing.assert_allclose(np.array(doubled["words"][0]["cubics"]),
                                   2 * np.array(result["words"][0]["cubics"]), atol=1e-17)

    def test_svg_canvas_scaling_does_not_refit_words(self):
        expected = composition(self.svg, 32, 0.75)
        self.root.set("viewBox", "0 0 200 200")
        self.root.set("data-center", "100,100")
        self.root.set("data-nominal-diameter", "140")
        for path in self.root.iter(f"{{{SVG_NS}}}path"):
            path.set("d", parse_path(CIRCLE).scaled(2, origin=0j).d())
        self.write()
        actual = composition(self.svg, 32, 0.75)
        np.testing.assert_allclose(actual["words"][0]["cubics"], expected["words"][0]["cubics"],
                                   atol=1e-17, rtol=0)
        self.assertAlmostEqual(actual["anchor_y"], expected["anchor_y"], places=12)

    def test_rejects_unsupported_geometry_and_hidden_transforms(self):
        original = copy.deepcopy(self.root)
        changes = [
            lambda r: r.attrib.pop("data-nominal-diameter"),
            lambda r: r.set("data-center", "nan,50"),
            lambda r: r.set("viewBox", "0 0 -1 100"),
            lambda r: r[1].set("transform", "translate(1,0)"),
            lambda r: r[1][0].set("style", "stroke:red"),
            lambda r: r[1][0].set("d", "M 0 0 L 1 1 Z"),
            lambda r: r[1][0].set("d", "M 0 0 C 1 2"),
            lambda r: r[1][0].set("d", CIRCLE[:-1].replace("85 50", "84 50", 1)),
            lambda r: r[1][0].set("d", CIRCLE.replace("69.329966", "1e999")),
            lambda r: ET.SubElement(r, f"{{{SVG_NS}}}rect"),
            lambda r: ET.SubElement(r, f"{{{SVG_NS}}}svg"),
            lambda r: r.remove(r[1]) or r.remove(r[1]),
        ]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                self.root = copy.deepcopy(original)
                change(self.root)
                self.write()
                with self.assertRaises(ValueError):
                    composition(self.svg, 32, 0.75)

    def test_parameters_and_output_collisions_fail_before_rendering(self):
        for key, value in (("diameter_mm", float("nan")), ("diameter_mm", 0),
                           ("wire_mm", float("inf")), ("wire_mm", 0.01),
                           ("resolution", 0), ("samples", 0), ("threads", 0)):
            args = copy.copy(self.args)
            setattr(args, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                render(args)
        self.args.output_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "new directory"):
            render(self.args)
        self.args.output_dir = self.work / "missing" / "nested"
        with self.assertRaisesRegex(ValueError, "new directory"):
            render(self.args)

    def test_renderer_missing_or_failed_leaves_no_partial_output(self):
        with mock.patch("pendant.shutil.which", return_value=None), self.assertRaisesRegex(ValueError, "Blender"):
            render(self.args)
        with mock.patch("pendant.shutil.which", return_value="/blender"):
            def fail(command, **kwargs):
                kwargs["stdout"].write("Cycles failed to start\n")
                kwargs["stdout"].flush()
                raise subprocess.CalledProcessError(1, command)
            with mock.patch("pendant.subprocess.run", side_effect=fail), self.assertRaisesRegex(RuntimeError, "Cycles failed"):
                render(self.args)
        self.assertEqual(list(self.work.iterdir()), [self.svg])

    def test_missing_bad_or_disconnected_outputs_are_not_success(self):
        def invalid(command, **kwargs):
            self.fake_renderer(command, **kwargs)
            path = Path(command[-1]).parent
            if failure == "missing":
                (path / "pendant.png").unlink()
            elif failure == "dimensions":
                (path / "pendant.png").write_bytes(b"not a PNG")
            else:
                (path / "render.json").write_text('{"body_components": 2, "word_count": 2}')
        with mock.patch("pendant.shutil.which", return_value="/blender"):
            for failure in ("missing", "dimensions", "disconnected"):
                with self.subTest(failure=failure), mock.patch("pendant.subprocess.run", side_effect=invalid):
                    with self.assertRaises(RuntimeError):
                        render(self.args)
                    self.assertFalse(self.args.output_dir.exists())
        self.assertEqual(list(self.work.iterdir()), [self.svg])

    def test_successful_export_preserves_inputs_and_editable_scene(self):
        before = self.svg.read_bytes()
        self.args.save_scene = True
        with mock.patch("pendant.shutil.which", return_value="/blender"), \
                mock.patch("pendant.subprocess.run", side_effect=self.fake_renderer):
            render(self.args)
        self.assertEqual(self.svg.read_bytes(), before)
        self.assertEqual({p.name for p in self.args.output_dir.iterdir()},
                         {"scene.json", "render.json", "render.log", "pendant.png", "pendant.blend"})
        self.assertEqual(set(self.work.iterdir()), {self.svg, self.args.output_dir})

    def test_cli_errors_are_explicit(self):
        with self.assertRaises(SystemExit) as error:
            main([str(self.svg), "--output-dir", str(self.work / "out"), "--wire-mm", "nan"])
        self.assertEqual(error.exception.code, 2)

    @unittest.skipUnless(os.environ.get("RIPPLE_TEST_BLENDER"), "Set RIPPLE_TEST_BLENDER for local Cycles integration")
    def test_real_blender_geometry_portable_scene_and_disconnected_body(self):
        from config import load_config
        from geometry import circular_svg

        settings = load_config(overrides={"algorithm": "energy", "height": 256, "max_width": 256, "stroke_width": 1})
        settings["render"]["points"] = 64
        outer = circular_svg(np.zeros(1, complex), settings["render"])
        self.svg.write_text(outer)
        self.args.blender = os.environ["RIPPLE_TEST_BLENDER"]
        self.args.resolution, self.args.samples, self.args.threads = 256, 1, 1
        self.args.save_scene = True
        render(self.args)
        report = json.loads((self.args.output_dir / "render.json").read_text())
        self.assertEqual(report["body_components"], 1)
        self.assertEqual(report["source_centerline_checks"], 128)
        result = subprocess.run(
            [self.args.blender, "--background", "--disable-autoexec",
             str(self.args.output_dir / "pendant.blend"), "--threads", "1", "--python-exit-code", "1",
             "--python-expr", "import bpy; assert bpy.context.scene.render.filepath == '//pendant.png'; "
             "assert bpy.context.scene.render.engine == 'CYCLES'; assert bpy.context.scene.render.threads == 1"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        svg, inner = ET.fromstring(outer), ET.fromstring(outer)[-1]
        scaled = parse_path(inner.get("d")).scaled(0.65, origin=128+128j)
        scaled[-1].end = scaled[0].start
        inner.set("d", scaled.d())
        svg.append(inner)
        self.svg.write_text(ET.tostring(svg, encoding="unicode"))
        self.args.output_dir = self.work / "disconnected"
        with self.assertRaisesRegex(RuntimeError, "Gold body has disconnected parts"):
            render(self.args)
        self.assertFalse(self.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
