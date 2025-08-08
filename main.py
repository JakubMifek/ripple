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
    parser.add_argument('--smoothness', type=float, default=0.3333, help='Waveform smoothness (0=raw, 1=flat line, 0.5=medium, default=0.3)')
    parser.add_argument('--poly-degree', type=int, default=16, help='Degree of polynomial for center curve (default: 16)')
    parser.add_argument('--max-width', type=int, default=16384, help='Maximum SVG width (default: 16384)')
    parser.add_argument('--height', type=int, default=2048, help='SVG height (default: 2048)')
    parser.add_argument('--fitting', type=str, choices=['fourier', 'poly'], default='fourier', help='Curve fitting method: fourier or poly (default: fourier)')
    parser.add_argument('--color', type=str, default='orange', help='Color of the output curve (default: orange, accepts any SVG color)')
    parser.add_argument('--circle', action='store_true', help='If set, output the perfect circle as the SVG. Otherwise, output the waveform curve.')
    return parser.parse_args()

def main():
    args = parse_args()
    mode = Mode(args.mode)
    print(f"Input WAV: {args.input_wav}")
    print(f"Output SVG: {args.output_svg}")
    print(f"Mode: {mode.value} (Smoothness: {args.smoothness}, Poly Degree: {args.poly_degree}, Max Width: {args.max_width}, Height: {args.height}, Fitting: {args.fitting})")
    with wave.open(args.input_wav, 'rb') as wf:
        processing_svg, circular_svg = ripple(
            wf,
            smoothness=args.smoothness,
            poly_degree=args.poly_degree,
            max_width=args.max_width,
            height=args.height,
            fitting=args.fitting,
            color=args.color,
            circle=args.circle
        )
        
        with open(args.output_svg, 'w+') as svg_file:
            svg_file.write(circular_svg)
        print(f"Circle SVG written to {args.output_svg}")

        with open('tmp.svg', 'w+') as tmp_file:
            tmp_file.write(processing_svg)
        print("Processing SVG written to tmp.svg")

if __name__ == "__main__":
    main()
