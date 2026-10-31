"""Turn a Ripple artwork/overlay into a locally ray-traced gold pendant."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from svgpathtools import CubicBezier, parse_path
from utils import SVG_NS

DEFAULT_INPUT = (Path(__file__).resolve().parents[1] / "output_samples" /
                 "names-comparison/energy-timbre-pitch/overlay.svg")


def composition(svg, diameter_mm, wire_mm):
    """Preserve cubic centerlines, using one common SVG-to-metre conversion."""
    data = svg.read_bytes()
    root = ET.fromstring(data)
    try:
        center = [float(v) for v in root.attrib["data-center"].split(",")]
        nominal = float(root.attrib["data-nominal-diameter"])
        view = [float(v) for v in root.attrib["viewBox"].split()]
    except (KeyError, ValueError) as error:
        raise ValueError("Use a Ripple word SVG or overlay.svg, not a diagnostic or labeled preview.") from error
    if (len(center) != 2 or len(view) != 4 or not all(map(math.isfinite, [*center, nominal, *view]))
            or nominal <= 0 or min(view[2:]) <= 0):
        raise ValueError("SVG center, nominal diameter and viewBox must be finite and valid.")
    scale = diameter_mm / nominal / 1000
    words, anchors = [], []

    def point(z):
        return [(z.real - center[0]) * scale, (center[1] - z.imag) * scale, wire_mm / 2000]

    def visit(element, label=None):
        tag = element.tag.removeprefix(f"{{{SVG_NS}}}")
        if any(key in element.attrib for key in ("transform", "style", "display", "visibility")):
            raise ValueError("Transformed or CSS-styled SVGs are unsupported; use Ripple's unmodified artwork.")
        if tag not in ("svg", "g", "path", "title", "desc"):
            raise ValueError(f"Unsupported SVG element: {tag}")
        if tag == "svg" and element is not root:
            raise ValueError("Nested SVG viewports are unsupported; use a word SVG or overlay.svg.")
        label = element.get("data-word", label)
        if tag == "path" and element.get("data-role") != "reference-circle":
            try:
                path = parse_path(element.get("d", ""))
            except (ValueError, IndexError) as error:
                raise ValueError("Invalid SVG path data.") from error
            if (not path or not all(isinstance(segment, CubicBezier) for segment in path)
                    or not path.iscontinuous() or not path.isclosed()):
                raise ValueError("Each artwork path must be one closed, connected cubic Bezier loop.")
            cubics = [[point(getattr(seg, attr)) for attr in ("start", "control1", "control2", "end")]
                      for seg in path]
            if not np.isfinite(cubics).all() or path.length() <= 0:
                raise ValueError("Artwork paths must have finite, nonzero geometry.")
            words.append({"name": label or f"Word {len(words) + 1}", "cubics": cubics})
            # Attach at the highest crossing of the shared vertical centerline.
            for seg in path:
                polynomial = seg.poly()
                coefficients = np.real(polynomial.coeffs).copy()
                coefficients[-1] -= center[0]
                for t in np.roots(coefficients):
                    if abs(t.imag) < 1e-9 and 0 <= t.real <= 1:
                        anchors.append(point(seg.point(float(t.real)))[1])
        for child in element:
            visit(child, label)

    visit(root)
    if not words or not anchors or max(anchors) <= 0:
        raise ValueError("Artwork must contain loops crossing above the shared center for a bail attachment.")
    return {"source_name": svg.name, "source_sha256": hashlib.sha256(data).hexdigest(),
            "source_metadata": dict(root.attrib), "words": words, "anchor_y": max(anchors)}


def render(args):
    if not (math.isfinite(args.diameter_mm) and 10 <= args.diameter_mm <= 100):
        raise ValueError("Nominal diameter must be between 10 and 100 mm.")
    if not (math.isfinite(args.wire_mm) and 0.3 <= args.wire_mm <= args.diameter_mm / 15):
        raise ValueError("Wire diameter must be at least 0.3 mm and no more than 1/15 of the nominal diameter.")
    if not 256 <= args.resolution <= 8192 or not 1 <= args.samples <= 4096 or not 1 <= args.threads <= 64:
        raise ValueError("Use resolution 256-8192, samples 1-4096 and threads 1-64.")
    artwork = composition(args.input_svg, args.diameter_mm, args.wire_mm)
    output = args.output_dir.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Output must be a new directory inside an existing parent directory.")
    blender = shutil.which(args.blender)
    if blender is None:
        raise ValueError("Blender 4.5+ is required. Install it locally and set --blender to its executable.")
    job = {"version": 1, **artwork, "diameter_mm": args.diameter_mm, "wire_mm": args.wire_mm,
           "resolution": args.resolution, "samples": args.samples, "threads": args.threads,
           "save_scene": args.save_scene}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "scene.json").write_text(json.dumps(job, separators=(",", ":")) + "\n", encoding="utf-8")
        command = [blender, "--background", "--factory-startup", "--disable-autoexec",
                   "--threads", str(args.threads), "--python-exit-code", "1",
                   "--python", str(Path(__file__).with_name("pendant_scene.py")),
                   "--", str(stage / "scene.json")]
        print(f"Ray tracing {len(artwork['words'])} gold loops at {args.resolution}px with Cycles...", flush=True)
        with (stage / "render.log").open("w", encoding="utf-8") as log:
            try:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                tail = "\n".join((stage / "render.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
                raise RuntimeError(f"Blender rendering failed:\n{tail}") from error
        image = stage / "pendant.png"
        if not image.is_file():
            raise RuntimeError("Blender did not produce pendant.png.")
        with image.open("rb") as stream:
            header = stream.read(24)
        if (len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[12:16] != b"IHDR"
                or struct.unpack(">II", header[16:24]) != (args.resolution, args.resolution)):
            raise RuntimeError("Blender produced an invalid or incorrectly sized PNG.")
        report = json.loads((stage / "render.json").read_text(encoding="utf-8"))
        if report.get("body_components") != 1 or report.get("word_count") != len(artwork["words"]):
            raise RuntimeError("Rendered geometry did not preserve a connected pendant with all source words.")
        if args.save_scene and not (stage / "pendant.blend").is_file():
            raise RuntimeError("Blender did not save the requested editable scene.")
        if output.exists():
            raise ValueError("Output directory appeared during rendering; refusing to overwrite it.")
        stage.rename(output)
    print(f"Pendant: {output / 'pendant.png'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Photorealistic openwork gold pendant from Ripple SVG; requires local Blender 4.5+.")
    parser.add_argument("input_svg", type=Path, nargs="?", default=DEFAULT_INPUT,
                        help="Ripple artwork/overlay (default: four-name energy-timbre-pitch composition)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; parent must exist")
    parser.add_argument("--blender", default="blender", help="Blender executable name or path")
    parser.add_argument("--diameter-mm", type=float, default=32, help="Shared nominal diameter, not bounding-box fitting (default: 32)")
    parser.add_argument("--wire-mm", type=float, default=0.75, help="Round gold wire diameter (default: 0.75)")
    parser.add_argument("--resolution", type=int, default=1600, help="Square PNG size (default: 1600)")
    parser.add_argument("--samples", type=int, default=256, help="Maximum Cycles samples (default: 256)")
    parser.add_argument("--threads", type=int, default=4, help="CPU render threads (default: 4)")
    parser.add_argument("--save-scene", action="store_true", help="Also save an editable pendant.blend")
    args = parser.parse_args(argv)
    try:
        render(args)
    except (ValueError, OSError, RuntimeError, ET.ParseError) as error:
        parser.exit(2, f"Pendant error: {error}\n")


if __name__ == "__main__":
    main()
