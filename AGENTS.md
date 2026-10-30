# Agent instructions

## Project and code map

Ripple converts WAV audio into circular SVG artwork and a linear diagnostic SVG.
It is a collection of Python scripts, not an installed package or a web service.

- `main.py`: argparse CLI, WAV opening, and output writing.
- `ripple.py`: sample extraction, smoothing control, extrema and midpoint
  extraction, curve fitting dispatch, and circular geometry. `ripple()` returns
  `(processing_svg, circular_svg)` as strings.
- `utils.py`: channel averaging, cyclic median smoothing, polynomial and Fourier
  fitting, and SVG serialization.
- `merge.py`: a separate CLI that scales and overlays SVG paths onto one canvas;
  optionally renders a PNG with CairoSVG. It does not arrange a collage/grid.
- `names.sh`: runs four sample conversions concurrently, waits, then merges
  `output_names/`.
- `samples/`: committed WAV inputs. `output/`, `output_names/`,
  `output_samples/`, and root `tmp.svg` contain committed generated examples.

## Environment

Run commands from the repository root unless deliberately isolating output.
Use a virtual environment rather than installing into the system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Dependencies are NumPy, svgpathtools, and CairoSVG. CairoSVG is listed in
`requirements.txt`, but `merge.py` tolerates its absence and skips PNG export.
If PNG rendering fails, investigate the CairoSVG/native Cairo environment; do
not silently treat a missing PNG as a successful PNG check.

There is no configured test runner, formatter, linter, packaging metadata, or CI.
Do not invent existing commands such as `pytest` or introduce tooling unrelated
to the requested change.

## Behavior to preserve

- Audio processing supports 16-bit PCM only. Multichannel audio is averaged
  into mono before sample extraction; preserve the frame count and sample rate.
- Smoothing is cyclic, wrapping at the ends. `ripple.py` scales the user-facing
  smoothness before calling the shared smoother.
- Both fitting methods, `fourier` and `poly`, are supported. `--poly-degree`
  controls the polynomial degree or the Fourier term count, respectively.
- Circular output is always produced. `--circle` adds a gray reference circle;
  it does not replace the waveform or select linear output.
- The circular SVG is square, with side `min(svg_width, svg_height)`.
  The processing SVG includes the waveform, envelopes, and fitted curve.
- `main.py` writes the requested circular SVG and always overwrites `tmp.svg`
  in the current working directory. It does not create parent directories.
- `merge.py` reads every `.svg` in its input directory and scales path geometry
  to the requested dimensions, setting stroke width from `--thickness`.
  Keep merged output outside the input directory to avoid re-merging it.

Read implementation and CLI help before changing behavior. Some README defaults
are stale: `main.py` currently defaults to `--max-width 16384 --height 2048`;
the `ripple()` API has separate defaults. Update directly affected documentation
when changing arguments, defaults, or output contracts.

## Making changes

Keep changes focused and follow the existing function-based structure. Reuse
helpers in `utils.py` instead of duplicating smoothing, fitting, or SVG logic.
Preserve CLI names and return shapes unless the task explicitly changes them.
Report unsupported inputs and processing failures explicitly.

Do not modify sample WAVs or regenerate committed SVG/PNG examples unless the
task requires it. Do not run `names.sh` as a routine check: it overwrites tracked
examples and its parallel conversions race over the same root `tmp.svg`.

The current smoothing and coordinate-generation loops can be expensive on long
recordings. Prefer short inputs for checks; reducing `--max-width` alone does
not reduce the number of audio samples processed.

## Validation

For Python changes, start with syntax and CLI checks:

```bash
python3 -m py_compile main.py ripple.py utils.py merge.py
python3 main.py --help
python3 merge.py --help
```

For conversion or geometry changes, run a smoke check in an isolated temporary
working directory so the mandatory `tmp.svg` cannot overwrite the tracked one.
After activating the environment above, this example uses absolute script and
input paths while keeping all generated files outside the repository:

```bash
repo="$PWD"
python="$(command -v python3)"
work="$(mktemp -d)"
(
  cd "$work" || exit 1
  mkdir inputs
  "$python" "$repo/main.py" "$repo/samples/HelloWorld.wav" inputs/fourier.svg \
    --smoothness 0 --poly-degree 4 --max-width 512 --height 256 --circle &&
  "$python" "$repo/main.py" "$repo/samples/HelloWorld.wav" inputs/poly.svg \
    --smoothness 0 --poly-degree 4 --max-width 512 --height 256 --fitting poly &&
  "$python" "$repo/merge.py" inputs --width 256 --height 256 \
    --output merged.svg --png merged.png
)
```

Inspect the generated SVGs with `xml.etree.ElementTree`: require the SVG
namespace, expected dimensions, nonempty path data, and finite coordinates.
For this example, both circular SVGs and the merged SVG should be 256 by 256;
the reference-circle conversion should have two paths and the other one one.
When CairoSVG is available, also confirm the PNG exists and has the requested
dimensions. Remove only the temporary files/directory you created after checking.

Add focused regression tests for changed logic when appropriate, using small
synthetic 16-bit WAVs (including mono/stereo where relevant) and explicit SVG
assertions rather than only checking file existence. Exercise both fitting
methods for shared pipeline changes. Check relevant failure cases such as
unsupported sample widths and insufficient extrema for fitting; do not silently
change their behavior as part of an unrelated task.

Before committing, inspect `git diff` and `git status --short` for accidental
changes to generated examples. Documentation-only changes do not require audio
conversion or dependency installation.
