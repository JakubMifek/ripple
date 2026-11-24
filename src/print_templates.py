"""Reusable, physical-size forming and comparison sheets for four Ripple loops."""

import argparse
import copy
from functools import lru_cache
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from svgpathtools import CubicBezier, Path as BezierPath
from pendant import composition
from utils import SVG_NS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output_samples/final-names-composition/energy-timbre-pitch/overlay.svg"
DEFAULT_SIZES = tuple(range(35, 81, 5))
PAPERS = {"A4": (210, 297), "Letter": (215.9, 279.4)}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#8E44AD")
NS = {"s": SVG_NS}


def load_artwork(source):
    # Reuse the strict Ripple artwork reader; normalize to one nominal-diameter unit.
    document = composition(source, 1000, 1)
    if len(document["words"]) != 4:
        raise ValueError("The four-section print layout requires exactly four artwork paths.")
    words = []
    for word in document["words"]:
        curve = BezierPath(*[CubicBezier(*[complex(p[0], -p[1]) for p in segment])
                            for segment in word["cubics"]])
        words.append((word["name"], curve))
    if len({name for name, _ in words}) != 4:
        raise ValueError("The four artwork paths need distinct names for labels and length measurements.")
    boxes = [curve.bbox() for _, curve in words]
    bounds = (min(b[0] for b in boxes), max(b[1] for b in boxes),
              min(b[2] for b in boxes), max(b[3] for b in boxes))
    return words, bounds, document


def validate_options(sizes, wires):
    if (not 1 <= len(sizes) <= 10 or len(set(sizes)) != len(sizes)
            or any(not math.isfinite(s) or not 20 <= s <= 80 or not float(s).is_integer() for s in sizes)
            or any(b-a < 5 for a, b in zip(sorted(sizes), sorted(sizes)[1:]))):
        raise ValueError("Choose 1-10 whole-mm body sizes between 20 and 80 mm, at least 5 mm apart.")
    if (not 1 <= len(wires) <= 2 or len(set(wires)) != len(wires)
            or any(not math.isfinite(w) or not 0.5 <= w <= 1.5 for w in wires)):
        raise ValueError("Choose one or two distinct round-wire diameters between 0.5 and 1.5 mm.")
    return sorted(sizes), list(wires)


def scale_for(bounds, maximum, wire):
    return (maximum-wire)/max(bounds[1]-bounds[0], bounds[3]-bounds[2])


def comparison_rows(sizes, wire, bounds):
    """Pack whole true-size silhouettes into at most three rows; never shrink to fit."""
    sizes = sorted(sizes, reverse=True)
    widths = [(bounds[1]-bounds[0])*scale_for(bounds, size, wire)+wire for size in sizes]
    candidates = {}
    for mask in range(1, 1 << len(sizes)):
        members = [i for i in range(len(sizes)) if mask & (1 << i)]
        if sum(widths[i] for i in members)+3*(len(members)-1) <= 190:
            candidates[mask] = members

    @lru_cache(None)
    def pack(mask, slots):
        if mask == 0:
            return 0, ()
        if slots == 0:
            return math.inf, ()
        best = math.inf, ()
        first = mask & -mask
        for row, members in candidates.items():
            if row & first and row & mask == row:
                rest, rows = pack(mask ^ row, slots-1)
                cost = max(sizes[i] for i in members)+6+rest
                if cost < best[0]:
                    best = cost, (row, *rows)
        return best

    height, rows = pack((1 << len(sizes))-1, 3)
    if height-6 > 230:
        raise ValueError("These sizes cannot fit one true-size comparison page; choose fewer/smaller sizes.")
    return [[sizes[i] for i in candidates[row]] for row in rows]


def element(parent, tag, **attrs):
    return ET.SubElement(parent, f"{{{SVG_NS}}}{tag}",
                         {key.replace("_", "-"): str(value) for key, value in attrs.items()})


def text(parent, x, y, value, size=3, bold=False, anchor="start"):
    node = element(parent, "text", x=x, y=y, font_size=size, fill="#222222",
                   font_weight="bold" if bold else "normal", text_anchor=anchor)
    node.text = value
    return node


def line(parent, x1, y1, x2, y2, width=.12, color="#777777", **attrs):
    return element(parent, "line", x1=x1, y1=y1, x2=x2, y2=y2,
                   stroke=color, stroke_width=width, **attrs)


def page(title, subtitle, words, source_hash, kind, wire):
    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "width": "210mm", "height": "297mm", "viewBox": "0 0 210 297",
        "font-family": "DejaVu Sans, Arial, sans-serif", "data-kind": kind,
        "data-source-sha256": source_hash, "data-wire-mm": str(wire)})
    element(root, "rect", x=0, y=0, width=210, height=297, fill="white", data_role="paper")
    definitions = element(root, "defs")
    for i, (_, curve) in enumerate(words):
        element(definitions, "path", id=f"word-{i}", d=curve.d()+" Z")
    element(root, "title").text = title
    text(root, 12, 16, title, size=4.2, bold=True)
    text(root, 12, 25, subtitle, size=2.55)
    element(root, "rect", x=109, y=10.5, width=10, height=10, fill="none",
            stroke="#111111", stroke_width=.12, data_role="calibration-10mm-square")
    text(root, 121, 14, "10 x 10", size=2.1)
    text(root, 121, 18, "mm", size=2.1)
    line(root, 145, 17, 195, 17, color="#111111", data_role="calibration-50mm")
    for x in range(145, 196, 10):
        line(root, x, 15.5, x, 18.5, color="#111111")
    text(root, 170, 13, "50 mm between tick centers", size=2.05, anchor="middle")
    return root


def draw(root, words, bounds, maximum, wire, origin, indices=range(4), mode="black"):
    scale = scale_for(bounds, maximum, wire)
    group = element(root, "g", transform=f"translate({origin.real},{origin.imag}) scale({scale})",
                    data_role="drawing", data_max_mm=maximum, data_wire_mm=wire,
                    data_scale=scale, data_x=origin.real, data_y=origin.imag)
    for i in indices:
        color = COLORS[i] if mode == "color" else "#b8b8b8" if mode == "forming" else "#111111"
        attributes = {"href": f"#word-{i}", "fill": "none", "stroke": color,
                      "stroke_width": wire/scale, "stroke_linecap": "round", "stroke_linejoin": "round",
                      "data_role": "wire", "data_word_index": i}
        element(group, "use", **attributes)
        if mode == "forming":
            attributes.update(stroke="#222222", stroke_width=.10/scale,
                              stroke_dasharray=f"{.7/scale} {.5/scale}", data_role="centerline")
            element(group, "use", **attributes)
        xmin, xmax, ymin, ymax = words[i][1].bbox()
        if (xmin*scale+origin.real-wire/2 < 10 or xmax*scale+origin.real+wire/2 > 200
                or ymin*scale+origin.imag-wire/2 < 30 or ymax*scale+origin.imag+wire/2 > 265):
            raise ValueError("Artwork exceeds the true-size print area; choose smaller sizes.")
    return group


def registration(root, origin, top=False):
    line(root, origin.real-2, origin.imag, origin.real+2, origin.imag, color="#999999")
    line(root, origin.real, origin.imag-2, origin.real, origin.imag+2, color="#999999")
    if top:
        text(root, origin.real, origin.imag-4, "TOP", size=2.1, anchor="middle")


def label_point(curve):
    points = [segment.point(t) for segment in curve for t in (0, .25, .5, .75)]
    return max(points, key=abs)


def forming_page(words, bounds, sizes, wire, source_hash):
    root = page(f"Nested forming guides / {wire:g} mm", "ACTUAL SIZE / 100%. Choose the SAME size for all four names. Sizes include wire; bail excluded.",
                words, source_hash, "forming", wire)
    for i, (name, curve) in enumerate(words):
        origin = complex((52, 151)[i % 2], (86, 190)[i // 2])
        left = (12, 111)[i % 2]
        line(root, left, origin.imag-44, left+5, origin.imag-44, width=.7, color=COLORS[i])
        text(root, left+8, origin.imag-43, name.removesuffix(" Final"), size=3.6, bold=True)
        for maximum in sizes:
            draw(root, words, bounds, maximum, wire, origin, [i], mode="forming")
        registration(root, origin, top=True)
        point = label_point(curve)
        angle = math.degrees(math.atan2(point.imag, point.real))+90
        angle = (angle+90) % 180-90
        for maximum in sizes:
            position = origin+point*scale_for(bounds, maximum, wire)
            label = element(root, "g", transform=f"translate({position.real},{position.imag}) rotate({angle})",
                            data_role="size-label", data_max_mm=maximum, data_word_index=i)
            element(label, "rect", x=-2.2, y=-1.2, width=4.4, height=2.4, fill="white")
            text(label, 0, .72, f"{maximum:g}", size=2.1, bold=True, anchor="middle")
        text(root, left, origin.imag+45, f"Inner {sizes[0]:g} mm -> outer {sizes[-1]:g} mm; labels are body sizes.",
             size=2.5)
    text(root, 12, 244, "Gray bands show wire diameter. Follow their dashed centers, not an edge. Numbers identify each contour.", size=2.65)
    text(root, 12, 250, "The little crosses mark the same source center. Keep the printed TOP direction; do not flip the shapes.", size=2.65)
    text(root, 12, 256, "Check the header gauges in BOTH axes. Short white label windows are annotations, not cuts in the wire.", size=2.65)
    text(root, 12, 266, "Cold forming only: do not solder on paper. Use the matching-wire comparison page for alignment.", size=2.65)
    return root


def comparison_page(words, bounds, sizes, wire, source_hash):
    root = page(f"Size comparison / {wire:g} mm wire",
                "ACTUAL SIZE / 100% - no Fit or Shrink. Maximum outside BODY dimensions; wire included, bail excluded.",
                words, source_hash, "comparison", wire)
    center = complex((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2)
    top = 35
    for row in comparison_rows(sizes, wire, bounds):
        widths = [(bounds[1]-bounds[0])*scale_for(bounds, s, wire)+wire for s in row]
        left = (210-sum(widths)-3*(len(row)-1))/2
        for maximum, width in zip(row, widths):
            x = left+width/2
            text(root, x, top-3, f"{maximum:g} mm", size=3, bold=True, anchor="middle")
            origin = complex(x, top+maximum/2)-center*scale_for(bounds, maximum, wire)
            draw(root, words, bounds, maximum, wire, origin)
            registration(root, origin)
            left += width+3
        top += max(row)+6
    text(root, 12, 268, "All shapes stay at 1:1; wire stays at the stated diameter. Crosses mark the shared source center.", size=2.45)
    return root


def overlay_page(words, bounds, sizes, wire, source_hash):
    maximum = min(sizes, key=lambda s: abs(s-60))
    root = page("How to align the four names",
                f"ACTUAL SIZE / 100%. This example is {maximum:g} mm body / {wire:g} mm wire ONLY. Other sizes: use their comparison.",
                words, source_hash, "overlay", wire)
    for i in range(4):
        origin = complex((52, 151)[i % 2], (79, 169)[i // 2])
        left = (12, 111)[i % 2]
        heading = text(root, left, origin.imag-45, f"{i+1}. Include {words[i][0].removesuffix(' Final')}",
                       size=3.1, bold=True)
        heading.set("fill", COLORS[i])
        draw(root, words, bounds, maximum, wire, origin, range(i+1), mode="color")
        registration(root, origin, top=True)
    notes = [
        "1. Choose one body size AND one wire diameter. Use that combination for all four loops.",
        "2. Keep the page orientation. Align the little center crosses, not the outside edges of the individual loops.",
        "3. Use the full-size comparison silhouette to check the assembled outline. Do not rotate or mirror a loop.",
        "4. Colors identify names. Panel numbering is NOT an over/under, stacking or soldering order.",
        "Round wires need a physical joint arrangement at crossings; software overlap is not a finished joint.",
        "Loop closures, attachment hardware and extra wire length must be planned separately. Do not solder on paper.",
    ]
    for i, note in enumerate(notes):
        text(root, 12, 222+i*6, note, size=2.65)
    return root


def make_pages(words, bounds, sizes, wires, source_hash):
    pages = []
    for wire in wires:
        pages.append(forming_page(words, bounds, sizes, wire, source_hash))
        pages.append(comparison_page(words, bounds, sizes, wire, source_hash))
    pages.append(overlay_page(words, bounds, sizes, wires[0], source_hash))
    return pages


def export_pdfs(pages, output):
    try:
        import cairosvg
        import pymupdf
    except (ImportError, OSError) as error:
        raise ValueError("PDF export needs requirements-print.txt and native Cairo; use --svg-only for SVGs.") from error
    for paper, (width, height) in PAPERS.items():
        with pymupdf.open() as combined:
            for original in pages:
                root = copy.deepcopy(original)
                root.set("width", f"{width}mm")
                root.set("height", f"{height}mm")
                root.set("viewBox", f"0 0 {width} {height}")
                background = root.find("s:rect[@data-role='paper']", NS)
                background.set("width", str(width))
                background.set("height", str(height))
                data = cairosvg.svg2pdf(bytestring=ET.tostring(root))
                with pymupdf.open(stream=data, filetype="pdf") as page_pdf:
                    combined.insert_pdf(page_pdf)
            combined.set_metadata({"title": f"Ripple compact forming pack - {paper}",
                                   "subject": "Physical-size wire templates; print at Actual size / 100%",
                                   "author": "Ripple"})
            combined.xref_set_key(combined.pdf_catalog(), "ViewerPreferences", "<< /PrintScaling /None >>")
            combined.save(output/f"print-{paper}.pdf", garbage=4, deflate=True)


def generate(source, output, sizes=DEFAULT_SIZES, wires=(1.0, .9), svg_only=False):
    sizes, wires = validate_options(sizes, wires)
    words, bounds, artwork = load_artwork(source)
    pages = make_pages(words, bounds, sizes, wires, artwork["source_sha256"])
    output = output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Output must be a new directory within an existing parent directory.")
    variants = []
    for wire in wires:
        for maximum in sizes:
            scale = scale_for(bounds, maximum, wire)
            variants.append({
                "maximum_body_mm": maximum, "wire_mm": wire,
                "width_mm": (bounds[1]-bounds[0])*scale+wire,
                "height_mm": (bounds[3]-bounds[2])*scale+wire,
                "nominal_diameter_mm": scale,
                "projected_lengths_mm_no_allowance": {name: curve.length()*scale for name, curve in words},
            })
    manifest = {"version": 1, "source_name": source.name, "source_sha256": artwork["source_sha256"],
                "source_metadata": artwork["source_metadata"], "sizes_mm": sizes, "wire_mm": wires,
                "size_definition": "Maximum outside BODY dimension including wire, excluding bail and solder",
                "coordinates": "Source-center-relative, divided by source nominal diameter; SVG Y direction preserved",
                "normalized_bounds": bounds, "variants": variants,
                "notice": "No joint, over-under, closure, bail or spare-wire allowance is designed here.",
                "pages": [], "pdf_files": [] if svg_only else [f"print-{paper}.pdf" for paper in PAPERS]}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for i, root in enumerate(pages, 1):
            filename = f"{i:02d}-{root.get('data-kind')}-{float(root.get('data-wire-mm')):g}mm.svg"
            (stage/filename).write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
            manifest["pages"].append({"file": filename, "kind": root.get("data-kind"),
                                      "wire_mm": float(root.get("data-wire-mm"))})
        if not svg_only:
            export_pdfs(pages, stage)
        (stage/"dimensions.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        if output.exists():
            raise ValueError("Output directory appeared during export; refusing to overwrite it.")
        stage.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compact full-size forming/print pack for four Ripple word loops.")
    parser.add_argument("input_svg", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; parent must exist")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES),
                        help="Whole-mm outside body dimensions, 20-80, at least 5 apart (default: 35 to 80 in 5 mm steps)")
    parser.add_argument("--wire-mm", nargs="+", type=float, default=[1.0, .9],
                        help="One/two wire diameters, in page order (default: 1.0 0.9)")
    parser.add_argument("--svg-only", action="store_true", help="Skip PDF export and its optional dependencies")
    args = parser.parse_args(argv)
    try:
        result = generate(args.input_svg, args.output_dir, args.sizes, args.wire_mm, args.svg_only)
    except (ValueError, OSError, RuntimeError, ET.ParseError) as error:
        parser.exit(2, f"Print template error: {error}\n")
    print(f"Print pack: {args.output_dir} ({len(result['pages'])} pages"
          f"{', SVG only' if args.svg_only else ', A4 and Letter PDFs'})")


if __name__ == "__main__":
    main()
