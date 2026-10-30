"""Single-recording Ripple CLI."""

import argparse
from pathlib import Path
import wave
from config import add_render_arguments, config_from_args
from pitch import load_pitch
from ripple import ripple


def render_file(input_wav, settings, pitch_path=None):
    method = settings["render"]["algorithm"]
    if method == "energy-pitch" and pitch_path is None:
        raise ValueError("energy-pitch requires --pitch with known ground truth; pitch is never estimated.")
    if method != "energy-pitch" and pitch_path is not None:
        raise ValueError("--pitch is only valid for energy-pitch.")
    with wave.open(str(input_wav), "rb") as wf:
        pitch = load_pitch(pitch_path, input_wav, wf.getframerate(), wf.getnframes()) if pitch_path else None
        return ripple(wf, settings=settings, pitch_data=pitch)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ripple: spoken-word keepsake curves (16-bit PCM WAV).")
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("output_svg", type=Path)
    add_render_arguments(parser)
    parser.add_argument("--pitch", type=Path, help="Versioned, WAV-bound known-ground-truth JSON (energy-pitch only)")
    parser.add_argument("--diagnostic", type=Path, help="Diagnostic SVG override; default: OUTPUT_STEM.diagnostic.svg")
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        settings = config_from_args(args)
        diagnostic = args.diagnostic or args.output_svg.with_suffix(".diagnostic.svg")
        protected = {args.input_wav.resolve(), Path(__file__).with_name("ripple.toml").resolve()}
        protected.update(p.resolve() for p in (args.config, args.pitch) if p is not None)
        if args.output_svg.resolve() == diagnostic.resolve() or any(
                p.resolve() in protected for p in (args.output_svg, diagnostic)):
            raise ValueError("Artwork and diagnostic outputs must be distinct and must not overwrite inputs/configuration.")
        processing, circular = render_file(args.input_wav, settings, args.pitch)
        # Parent directories remain explicit; do not silently create misspelled destinations.
        args.output_svg.write_text(circular, encoding="utf-8")
        diagnostic.write_text(processing, encoding="utf-8")
    except (ValueError, OSError, wave.Error) as error:
        parser.exit(2, f"Ripple error: {error}\n")
    print(f"Artwork: {args.output_svg}\nDiagnostic: {diagnostic}")


if __name__ == "__main__":
    main()
