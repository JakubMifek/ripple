import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
from svgpathtools import CubicBezier, Path as BezierPath, parse_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_print_templates import check_pack, controls
from print_templates import DEFAULT_SIZES, NS, generate, main
from utils import SVG_NS


CIRCLE = ("M 85 50 C 85 69.329966 69.329966 85 50 85 "
          "C 30.670034 85 15 69.329966 15 50 "
          "C 15 30.670034 30.670034 15 50 15 "
          "C 69.329966 15 85 30.670034 85 50 Z")


class PrintTemplateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.source = self.work/"overlay.svg"
        self.output = self.work/"pack"
        self.root = ET.Element(f"{{{SVG_NS}}}svg", {
            "viewBox": "0 0 100 100", "data-center": "50,50",
            "data-nominal-diameter": "70", "data-orientation": "recording"})
        ET.SubElement(self.root, f"{{{SVG_NS}}}path", {"d": CIRCLE, "data-role": "reference-circle"})
        self.curves = []
        for i in range(4):
            def point(z):
                return complex(50+(z.real-50)*(.7+.02*i)+i, 50+(z.imag-50)*(1-.04*i))
            curve = BezierPath(*[CubicBezier(*[point(z) for z in (s.start, s.control1, s.control2, s.end)])
                                for s in parse_path(CIRCLE)])
            self.curves.append(curve)
            group = ET.SubElement(self.root, f"{{{SVG_NS}}}g", {"data-word": f"Name {i+1}"})
            ET.SubElement(group, f"{{{SVG_NS}}}path", {"d": curve.d(), "stroke": "black"})
        self.write_source()

    def write_source(self):
        ET.ElementTree(self.root).write(self.source, encoding="utf-8")

    def test_default_pack_preserves_source_geometry_scale_and_registration(self):
        source_bytes = self.source.read_bytes()
        manifest = generate(self.source, self.output, svg_only=True)
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(check_pack(self.output, self.source), (5, 0))
        self.assertEqual(manifest["sizes_mm"], list(DEFAULT_SIZES))
        self.assertEqual([(p["kind"], p["wire_mm"]) for p in manifest["pages"]],
                         [("forming", 1), ("comparison", 1), ("forming", .9), ("comparison", .9), ("overlay", 1)])
        boxes = [curve.bbox() for curve in self.curves]
        span = max(max(b[1] for b in boxes)-min(b[0] for b in boxes),
                   max(b[3] for b in boxes)-min(b[2] for b in boxes))
        for info in manifest["pages"]:
            root = ET.parse(self.output/info["file"]).getroot()
            if info["kind"] == "forming":
                self.assertFalse(root.findall(".//*[@data-role='size-label']"))
                self.assertFalse([rect for rect in root.findall(".//s:rect[@fill='white']", NS)
                                  if rect.get("data-role") != "paper"])
                captions = [node.text for node in root.findall(".//s:text", NS)]
                self.assertEqual(captions.count("Inner 35 mm -> outer 80 mm; body sizes."), 4)
            origins = {}
            for group in root.findall(".//s:g[@data-role='drawing']", NS):
                maximum = float(group.get("data-max-mm"))
                wire = float(group.get("data-wire-mm"))
                origin = complex(float(group.get("data-x")), float(group.get("data-y")))
                scale = float(group.get("data-scale"))
                self.assertAlmostEqual(span/70*scale+wire, maximum, places=10)
                for use in group.findall("s:use[@data-role='wire']", NS):
                    i = int(use.get("data-word-index"))
                    definition = root.find(f"s:defs/s:path[@id='word-{i}']", NS)
                    actual = controls(parse_path(definition.get("d")))*scale+origin
                    expected = (controls(self.curves[i])-(50+50j))*(maximum-wire)/span+origin
                    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
                    self.assertAlmostEqual(float(use.get("stroke-width"))*scale, wire, places=12)
                    if info["kind"] == "forming":
                        origins.setdefault(i, set()).add(origin)
                    elif info["kind"] == "comparison":
                        origins.setdefault(maximum, set()).add(origin)
            if info["kind"] in ("forming", "comparison"):
                self.assertTrue(all(len(points) == 1 for points in origins.values()))
        with self.assertRaisesRegex(ValueError, "Both paper-size PDFs"):
            check_pack(self.output, self.source, require_pdf=True)

    def test_selected_sizes_and_one_wire_make_three_pages(self):
        manifest = generate(self.source, self.output, sizes=[65, 45], wires=[.9], svg_only=True)
        self.assertEqual(manifest["sizes_mm"], [45, 65])
        self.assertEqual(check_pack(self.output, self.source), (3, 0))

    def test_invalid_sizes_and_dense_contours_fail_without_output(self):
        for sizes, wires in [([35, 36], [1]), ([81], [1]), ([35.5], [1]),
                             ([35], [float("nan")]), ([35], [1, 1]), ([], [1])]:
            with self.subTest(sizes=sizes, wires=wires), self.assertRaises(ValueError):
                generate(self.source, self.output, sizes, wires, svg_only=True)
            self.assertFalse(self.output.exists())

    def test_unprintable_layout_is_rejected_not_shrunk(self):
        for node in self.root.findall("s:g/s:path", NS):
            node.set("d", CIRCLE)
        self.write_source()
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            generate(self.source, self.output, svg_only=True)
        self.assertFalse(self.output.exists())

    def test_requires_four_distinctly_named_loops(self):
        self.root[2].set("data-word", "Name 1")
        self.write_source()
        with self.assertRaisesRegex(ValueError, "distinct names"):
            generate(self.source, self.output, svg_only=True)
        self.root.remove(self.root[2])
        self.write_source()
        with self.assertRaisesRegex(ValueError, "exactly four"):
            generate(self.source, self.output, svg_only=True)

    def test_existing_directory_and_renderer_failures_leave_no_partial_pack(self):
        with mock.patch("print_templates.export_pdfs", side_effect=RuntimeError("renderer failed")):
            with self.assertRaisesRegex(RuntimeError, "renderer failed"):
                generate(self.source, self.output)
        self.assertEqual(list(self.work.iterdir()), [self.source])
        self.output.mkdir()
        sentinel = self.output/"keep.txt"
        sentinel.write_text("unchanged")
        with self.assertRaisesRegex(ValueError, "new directory"):
            generate(self.source, self.output, svg_only=True)
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_missing_pdf_dependencies_fail_explicitly_but_help_and_svg_work(self):
        with mock.patch.dict(sys.modules, {"cairosvg": None, "pymupdf": None}):
            with self.assertRaisesRegex(ValueError, "requirements-print.txt"):
                generate(self.source, self.output)
            self.assertFalse(self.output.exists())
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                main(["--help"])
            self.assertEqual(result.exception.code, 0)
            generate(self.source, self.output, sizes=[60], wires=[1], svg_only=True)

    def test_validator_detects_wrong_wire_source_and_inline_labels(self):
        manifest = generate(self.source, self.output, sizes=[35, 60], wires=[1], svg_only=True)
        filename = self.output/manifest["pages"][0]["file"]
        original = filename.read_bytes()
        for selector, attribute, value, message in [
            (".//s:use[@data-role='wire']", "stroke-width", "0.001", "thickness"),
            (".//s:g[@data-role='drawing']", "transform", "scale(2)", "transform"),
            (".//s:line[@data-role='calibration-50mm']", "y2", "18", "Calibration"),
        ]:
            root = ET.fromstring(original)
            root.find(selector, NS).set(attribute, value)
            ET.ElementTree(root).write(filename, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                check_pack(self.output, self.source)
        root = ET.fromstring(original)
        group = root.find(".//s:g[@data-role='drawing']", NS)
        x = float(group.get("data-x"))+1
        group.set("data-x", str(x))
        group.set("transform", f"translate({x},{group.get('data-y')}) scale({group.get('data-scale')})")
        ET.ElementTree(root).write(filename, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Nested sizes"):
            check_pack(self.output, self.source)
        root = ET.fromstring(original)
        label = ET.SubElement(root, f"{{{SVG_NS}}}g", {"data-role": "size-label"})
        ET.SubElement(label, f"{{{SVG_NS}}}rect", {"x": "50", "y": "50", "width": "4.4",
                                                 "height": "2.4", "fill": "white"})
        ET.SubElement(label, f"{{{SVG_NS}}}text").text = "60"
        ET.ElementTree(root).write(filename, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Inline size labels"):
            check_pack(self.output, self.source)
        label.attrib.pop("data-role")
        ET.ElementTree(root).write(filename, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "White masks"):
            check_pack(self.output, self.source)
        filename.write_bytes(original)
        self.source.write_bytes(self.source.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            check_pack(self.output, self.source)

    def test_actual_pdf_cubics_units_and_wrong_scale_detection(self):
        try:
            import cairosvg
            import pymupdf
        except (ImportError, OSError):
            self.skipTest("Optional PDF dependencies/native Cairo unavailable")
        generate(self.source, self.output, sizes=[35, 60, 80], wires=[1, .9])
        self.assertEqual(check_pack(self.output, self.source, require_pdf=True), (5, 2))
        pdf = self.output/"print-A4.pdf"
        original_bytes = pdf.read_bytes()
        with pymupdf.open(pdf) as masked:
            masked[0].draw_rect(pymupdf.Rect(120, 140, 135, 150), color=None, fill=(1, 1, 1))
            masked.saveIncr()
        with self.assertRaisesRegex(ValueError, "PDF white masks"):
            check_pack(self.output, self.source, require_pdf=True)
        pdf.write_bytes(original_bytes)
        with pymupdf.open(pdf) as original, pymupdf.open() as scaled:
            for i, original_page in enumerate(original):
                page = scaled.new_page(width=original_page.rect.width, height=original_page.rect.height)
                page.show_pdf_page(original_page.rect*.99, original, i)
            scaled.xref_set_key(scaled.pdf_catalog(), "ViewerPreferences", "<< /PrintScaling /None >>")
            scaled.save(self.work/"scaled.pdf")
        (self.work/"scaled.pdf").replace(pdf)
        with self.assertRaisesRegex(ValueError, "physical curve coordinates"):
            check_pack(self.output, self.source, require_pdf=True)


if __name__ == "__main__":
    unittest.main()
