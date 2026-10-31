"""Compare algorithms and overlay multiple utterances on a shared nominal circle."""

import argparse
import copy
import hashlib
from pathlib import Path
import re
import wave
import xml.etree.ElementTree as ET
from config import ALGORITHMS, PITCH_ALGORITHMS, add_render_arguments, config_from_args
from main import render_file
from utils import SVG_NS, reference_first


def word_name(index, path):
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", path.stem).strip("-") or "word"
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]
    return f"{index:02d}-{slug}-{digest}"


def overlay(documents, labels, method):
    parsed = [ET.fromstring(document) for document in documents]
    root = ET.Element(parsed[0].tag, parsed[0].attrib)
    metadata = [{k: v for k, v in document.attrib.items() if k.startswith("data-pitch-")}
                for document in parsed]
    for key in metadata[0]:
        if any(values.get(key) != metadata[0][key] for values in metadata):
            root.attrib.pop(key)
    if len({values.get("data-pitch-source") for values in metadata}) > 1:
        root.set("data-pitch-source", "mixed")
    root.set("data-algorithm", method)
    for document, label, values in zip(parsed, labels, metadata):
        group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"data-word": label, **values})
        title = ET.SubElement(group, f"{{{SVG_NS}}}title")
        title.text = label
        for child in document:
            if child.get("data-role") == "reference-circle":
                root.append(child)
            else:
                group.append(child)
    root[:] = reference_first(root)
    return ET.tostring(root, encoding="unicode")


def comparison_preview(overlays, side):
    label_height = max(32, side * 0.06)
    width, height = side * len(overlays), side + label_height
    root = ET.Element(f"{{{SVG_NS}}}svg", width=str(width), height=str(height),
                      viewBox=f"0 0 {width} {height}")
    for i, (method, document) in enumerate(overlays.items()):
        overlay_root = ET.fromstring(document)
        group = ET.SubElement(root, f"{{{SVG_NS}}}g", transform=f"translate({i * side},0)")
        text = ET.SubElement(group, f"{{{SVG_NS}}}text", x=str(side / 2), y=str(label_height * 0.7),
                             attrib={"text-anchor": "middle", "font-size": str(label_height * 0.5)})
        text.text = method
        source = overlay_root.get("data-pitch-source")
        if source is not None:
            label = {"estimated": "estimated", "mixed": "mixed sources"}.get(source, "known")
            text.text += f" ({label})"
        art = ET.SubElement(group, f"{{{SVG_NS}}}g", transform=f"translate(0,{label_height})")
        art.attrib.update({k: v for k, v in overlay_root.attrib.items() if k.startswith("data-pitch-")})
        art.extend(list(overlay_root))
    return ET.tostring(root, encoding="unicode")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare algorithms; each panel overlays all words on the SAME diameter and center.")
    parser.add_argument("inputs", nargs="+", type=Path, help="16-bit PCM WAV words, in overlay order")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for all comparison outputs")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--pitch", nargs="+", type=Path, help="Optional known-ground-truth JSON per WAV; otherwise estimate pitch automatically")
    add_render_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.algorithm is not None:
            raise ValueError("Use --algorithms (plural) for comparisons.")
        if len(set(args.algorithms)) != len(args.algorithms):
            raise ValueError("Comparison algorithms must not be repeated.")
        if args.output_dir.exists():
            raise ValueError("Comparison output directory must not already exist.")
        needs_pitch = any(method in PITCH_ALGORITHMS for method in args.algorithms)
        if needs_pitch and args.pitch is not None and len(args.pitch) != len(args.inputs):
            raise ValueError("When --pitch is supplied, provide one known-source sidecar per input WAV.")
        if not needs_pitch and args.pitch is not None:
            raise ValueError("--pitch requires energy-pitch or energy-timbre-pitch in --algorithms.")
        base = config_from_args(args)
        labels = [p.stem for p in args.inputs]
        names = [word_name(i, p) for i, p in enumerate(args.inputs, 1)]
        outputs, overlays = {}, {}
        # Validate and render everything before creating any comparison files.
        for method in args.algorithms:
            settings = copy.deepcopy(base)
            settings["render"]["algorithm"] = method
            documents = []
            for i, (input_path, name) in enumerate(zip(args.inputs, names)):
                pitch_path = args.pitch[i] if args.pitch is not None and method in PITCH_ALGORITHMS else None
                diagnostic, circular = render_file(input_path, settings, pitch_path)
                outputs[Path(method) / "words" / f"{name}.svg"] = circular
                outputs[Path(method) / "diagnostics" / f"{name}.svg"] = diagnostic
                documents.append(circular)
            overlays[method] = overlay(documents, labels, method)
            outputs[Path(method) / "overlay.svg"] = overlays[method]
        side = min(base["render"]["max_width"], base["render"]["height"])
        outputs[Path("comparison.svg")] = comparison_preview(overlays, side)
        args.output_dir.mkdir(parents=True)
        for relative, content in outputs.items():
            destination = args.output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
    except (ValueError, OSError, wave.Error) as error:
        parser.exit(2, f"Comparison error: {error}\n")
    print(f"Comparison and shared-circle overlays: {args.output_dir}")
    if needs_pitch:
        print("Pitch: supplied known-source sidecars." if args.pitch else "Pitch: estimated automatically with Praat.")


if __name__ == "__main__":
    main()
