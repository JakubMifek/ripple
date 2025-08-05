"""
main.py - Entry point for the Ripple sound wave project.
"""


import wave
import argparse
from enum import Enum
from ripple import ripple


class Mode(Enum):
    RIPPLE = 'Ripple'

def parse_args():
    parser = argparse.ArgumentParser(description="Ripple: Sound wave to SVG tool.")
    parser.add_argument('input_wav', type=str, help='Input WAV file name')
    parser.add_argument('output_svg', type=str, help='Output SVG file name')
    parser.add_argument('--mode', type=str, choices=[mode.value for mode in Mode], default=Mode.RIPPLE.value, help='Mode of operation (default: Ripple)')
    parser.add_argument('--smoothness', type=float, default=0.25, help='Waveform smoothness (0=raw, 1=flat line, 0.5=medium, default=0.25)')
    parser.add_argument('--poly-degree', type=int, default=16, help='Degree of polynomial for center curve (default: 16)')
    parser.add_argument('--max-width', type=int, default=2048, help='Maximum SVG width (default: 2048)')
    parser.add_argument('--height', type=int, default=256, help='SVG height (default: 256)')
    parser.add_argument('--fitting', type=str, choices=['fourier', 'poly'], default='fourier', help='Curve fitting method: fourier or poly (default: fourier)')
    return parser.parse_args()

def main():
    args = parse_args()
    mode = Mode(args.mode)
    print(f"Input WAV: {args.input_wav}")
    print(f"Output SVG: {args.output_svg}")
    print(f"Mode: {mode.value}")
    with wave.open(args.input_wav, 'rb') as wf:
        svg_content = ripple(
            wf,
            smoothness=args.smoothness,
            poly_degree=args.poly_degree,
            max_width=args.max_width,
            height=args.height,
            fitting=args.fitting
        )
        with open(args.output_svg, 'w') as svg_file:
            svg_file.write(svg_content)
        print(f"SVG written to {args.output_svg}")

if __name__ == "__main__":
    main()
