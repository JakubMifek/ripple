"""Check physical SVG/PDF geometry against the original Ripple artwork."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from svgpathtools import parse_path
from print_templates import DEFAULT_INPUT, NS, PAPERS, load_artwork, scale_for, validate_options

MM_PER_PT = 25.4/72


def require(condition, message):
    if not condition:
        raise ValueError(message)


def controls(curve):
    return np.array([[segment.start, segment.control1, segment.control2, segment.end] for segment in curve])


def close(actual, expected, tolerance, message):
    require(np.shape(actual) == np.shape(expected)
            and bool(np.all(np.isfinite(actual)))
            and bool(np.all(np.abs(np.asarray(actual)-np.asarray(expected)) <= tolerance)), message)


def check_svg(root, words, bounds, manifest, page_info):
    require(root.get("width") == "210mm" and root.get("height") == "297mm"
            and root.get("viewBox") == "0 0 210 297", "SVG must use exact A4 millimeter coordinates.")
    require(root.get("data-source-sha256") == manifest["source_sha256"], "SVG source hash changed.")
    require(root.get("data-kind") == page_info["kind"], "Page kind changed.")
    require(not root.findall(".//*[@data-role='reference-circle']"), "Nominal reference must not become wire.")
    for i, (_, curve) in enumerate(words):
        definition = root.find(f"s:defs/s:path[@id='word-{i}']", NS)
        require(definition is not None, "Missing source curve definition.")
        close(controls(parse_path(definition.get("d"))), controls(curve), 1e-12, "Source curve geometry changed.")
    gauge = root.find(".//*[@data-role='calibration-50mm']")
    square = root.find(".//*[@data-role='calibration-10mm-square']")
    require(gauge is not None and square is not None, "Missing physical calibration gauges.")
    close([float(gauge.get("x2"))-float(gauge.get("x1")), float(gauge.get("y2"))-float(gauge.get("y1")),
           float(square.get("width")), float(square.get("height"))],
          [50, 0, 10, 10], 1e-10, "Calibration dimensions changed.")
    expected_pdf, signatures, centerlines = [], [], []
    forming_origins = {}
    for group in root.findall(".//s:g[@data-role='drawing']", NS):
        maximum, wire, scale, x, y = [float(group.get(key)) for key in
                                    ("data-max-mm", "data-wire-mm", "data-scale", "data-x", "data-y")]
        require(maximum in manifest["sizes_mm"] and wire == page_info["wire_mm"], "Unexpected body/wire size.")
        close(scale, scale_for(bounds, maximum, wire), 1e-12, "Nonuniform or incorrect physical scale.")
        require(group.get("transform") == f"translate({x},{y}) scale({scale})", "Unexpected drawing transform.")
        wire_uses = group.findall("s:use[@data-role='wire']", NS)
        if page_info["kind"] == "comparison":
            require(len(wire_uses) == 4, "Comparison words must share one origin/scale.")
        elif page_info["kind"] == "forming":
            require(len(wire_uses) == 1, "Each forming section must contain one name.")
            index = int(wire_uses[0].get("data-word-index"))
            forming_origins.setdefault(index, set()).add((x, y))
        for use in group.findall("s:use", NS):
            index = int(use.get("data-word-index"))
            require(0 <= index < 4 and use.get("href") == f"#word-{index}", "Wrong source curve reference.")
            role = use.get("data-role")
            require(role in ("wire", "centerline"), "Unexpected curve role.")
            thickness = wire if role == "wire" else .1
            close(float(use.get("stroke-width"))*scale, thickness, 1e-10, "Physical wire thickness changed.")
            expected_pdf.append((controls(words[index][1])*scale+complex(x, y), thickness))
            if role == "wire":
                signatures.append((maximum, index))
            else:
                centerlines.append((maximum, index))
    sizes = manifest["sizes_mm"]
    if page_info["kind"] in ("forming", "comparison"):
        require(sorted(signatures) == sorted((s, i) for s in sizes for i in range(4)),
                "Not every requested name/size appears exactly once.")
    else:
        chosen = min(sizes, key=lambda s: abs(s-60))
        require(signatures == [(chosen, i) for step in range(4) for i in range(step+1)],
                "Overlay alignment panels lost or reordered a word.")
    if page_info["kind"] == "forming":
        require(all(len(origins) == 1 for origins in forming_origins.values()),
                "Nested sizes must share a source-center origin.")
        require(sorted(centerlines) == sorted(signatures), "Every forming band needs its dashed centerline.")
        require(not root.findall(".//*[@data-role='size-label']"),
                "Inline size labels must not interrupt forming curves.")
        require(all(node.get("data-role") == "paper" for node in root.findall(".//*[@fill='white']")),
                "White masks must not interrupt forming curves.")
    else:
        require(not centerlines, "Unexpected forming centerlines.")
    return expected_pdf


def check_pdf(path, roots, expected_pages, paper):
    try:
        import pymupdf
    except ImportError as error:
        raise ValueError("PDF validation requires requirements-print.txt.") from error
    width, height = PAPERS[paper]
    with pymupdf.open(path) as document:
        require(document.page_count == len(roots), "PDF page count is incorrect.")
        require("/PrintScaling/None" in document.xref_get_key(document.pdf_catalog(), "ViewerPreferences")[1],
                "PDF does not request unscaled printing.")
        for number, (page, expected) in enumerate(zip(document, expected_pages), 1):
            close([page.rect.width*MM_PER_PT, page.rect.height*MM_PER_PT], [width, height], .001,
                  "PDF physical page size is incorrect.")
            require(not page.get_images(), "Templates must be vector paths, not raster images.")
            drawings = page.get_drawings()
            for drawing in drawings:
                rect = drawing["rect"]
                if drawing["fill"] == (1, 1, 1) and rect.width*MM_PER_PT > 200:
                    continue
                if roots[number-1].get("data-kind") == "forming":
                    require(drawing["fill"] != (1, 1, 1), "PDF white masks must not interrupt forming curves.")
                half = (drawing.get("width") or 0)*MM_PER_PT/2
                require(rect.x0*MM_PER_PT-half > 9 and rect.x1*MM_PER_PT+half < width-9
                        and rect.y0*MM_PER_PT-half > 9 and rect.y1*MM_PER_PT+half < height-9,
                        f"PDF page {number} has artwork outside the print margins.")
            segment_counts = {len(points) for points, _ in expected}
            curves = [d for d in drawings if len(d["items"]) in segment_counts
                      and all(item[0] == "c" for item in d["items"])]
            require(len(curves) == len(expected), f"PDF page {number} lost or added curves.")
            for drawing, (points, thickness) in zip(curves, expected):
                actual = np.array([[complex(p.x, p.y)*MM_PER_PT for p in item[1:]]
                                   for item in drawing["items"]])
                close(actual, points, .003, "PDF does not preserve the physical curve coordinates.")
                close(drawing["width"]*MM_PER_PT, thickness, .0001, "PDF wire thickness is incorrect.")
            lines = [item for d in drawings for item in d["items"] if item[0] == "l"]
            require(any(abs((item[2]-item[1]).x*MM_PER_PT-50) < .003
                        and abs(item[1].y*MM_PER_PT-17) < .003
                        and abs(item[2].y*MM_PER_PT-17) < .003 for item in lines), "PDF 50 mm gauge is incorrect.")
            squares = [item for d in drawings for item in d["items"] if item[0] == "re"]
            require(any(abs(item[1].width*MM_PER_PT-10) < .003
                        and abs(item[1].height*MM_PER_PT-10) < .003 for item in squares),
                    "PDF 10 mm calibration square is incorrect.")
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    x0, y0, x1, y1 = [v*MM_PER_PT for v in block["bbox"]]
                    require(9 < x0 < x1 < width-9 and 9 < y0 < y1 < height-9,
                            f"PDF page {number} has text outside the print margins.")
            require("ACTUAL SIZE / 100%" in page.get_text(), "PDF lacks unscaled-print instruction.")


def check_pack(directory, source, require_pdf=False):
    manifest = json.loads((directory/"dimensions.json").read_text(encoding="utf-8"))
    require(manifest["version"] == 1, "Unsupported print-pack version.")
    require(hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"], "Source SVG hash mismatch.")
    require(len(manifest["pages"]) == 2*len(manifest["wire_mm"])+1, "Wrong compact-pack page count.")
    sizes, wires = validate_options(manifest["sizes_mm"], manifest["wire_mm"])
    require(sizes == manifest["sizes_mm"], "Manifest sizes must be ascending.")
    require([(p["kind"], p["wire_mm"]) for p in manifest["pages"]]
            == [(kind, wire) for wire in wires for kind in ("forming", "comparison")]+[("overlay", wires[0])],
            "Incorrect compact-pack page order.")
    if require_pdf:
        require(set(manifest["pdf_files"]) == {"print-A4.pdf", "print-Letter.pdf"}, "Both paper-size PDFs are required.")
    words, bounds, _ = load_artwork(source)
    close(manifest["normalized_bounds"], bounds, 1e-12, "Manifest source bounds changed.")
    require(len(manifest["variants"]) == len(sizes)*len(wires), "Missing size measurements.")
    for measurement, (wire, maximum) in zip(manifest["variants"], ((w, s) for w in wires for s in sizes)):
        require(measurement["wire_mm"] == wire and measurement["maximum_body_mm"] == maximum,
                "Measurements refer to incorrect sizes.")
        scale = scale_for(bounds, maximum, wire)
        close([measurement["width_mm"], measurement["height_mm"], measurement["nominal_diameter_mm"]],
              [(bounds[1]-bounds[0])*scale+wire, (bounds[3]-bounds[2])*scale+wire, scale], 1e-10,
              "Incorrect outside body measurements.")
        for name, curve in words:
            close(measurement["projected_lengths_mm_no_allowance"][name], curve.length()*scale, 1e-8,
                  "Incorrect projected wire length.")
    roots, expected = [], []
    for info in manifest["pages"]:
        require(Path(info["file"]).name == info["file"], "Invalid page filename.")
        root = ET.parse(directory/info["file"]).getroot()
        roots.append(root)
        expected.append(check_svg(root, words, bounds, manifest, info))
    for file in manifest["pdf_files"]:
        require(file in ("print-A4.pdf", "print-Letter.pdf"), "Invalid PDF filename.")
        check_pdf(directory/file, roots, expected, file.removeprefix("print-").removesuffix(".pdf"))
    return len(roots), len(manifest["pdf_files"])


def render_previews(directory, output):
    import pymupdf
    require(not output.exists() and output.parent.is_dir(), "Preview output must be a new directory within an existing parent.")
    manifest = json.loads((directory/"dimensions.json").read_text(encoding="utf-8"))
    require(manifest["pdf_files"], "PDFs are required for rendered previews.")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for file in manifest["pdf_files"]:
            with pymupdf.open(directory/file) as document:
                for i, page in enumerate(document, 1):
                    page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(
                        stage/f"{Path(file).stem}-{i:02d}.png")
        require(not output.exists(), "Preview output appeared during rendering; refusing to overwrite it.")
        stage.rename(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a compact print pack against its source SVG and physical dimensions.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_INPUT, help="Original four-word overlay used for generation")
    parser.add_argument("--require-pdf", action="store_true", help="Require A4 and Letter PDFs, not SVG-only output")
    parser.add_argument("--preview-dir", type=Path, help="Render PDF pages as inspection PNGs in a new directory")
    args = parser.parse_args(argv)
    try:
        pages, pdfs = check_pack(args.directory, args.source, args.require_pdf)
        if args.preview_dir:
            render_previews(args.directory, args.preview_dir)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, ET.ParseError) as error:
        parser.exit(2, f"Print validation error: {error}\n")
    print(f"Verified {pages} SVG sheets and {pdfs} PDF packs against the source; physical scale and calibration agree.")


if __name__ == "__main__":
    main()
