# Agent instructions

## Project map

Ripple is a Python 3.11+ script collection, not an installed package or service.
It makes artistic circular keepsakes from 16-bit PCM WAV recordings, not
guaranteed unique biometric signatures or physically tattoo-ready designs.

- `ripple.toml`, `config.py`: version-1 defaults, strict TOML overrides,
  validation and shared CLI options. Defaults -> explicit config -> CLI.
- `audio.py`: floating channel averaging, DC removal, conservative relative
  silence trimming, gain normalization; retains original sample offsets.
- `ripple.py`: orchestration, legacy signed-waveform baseline with cyclic median
  smoothing/extrema/recursive midpoints, diagnostic SVG. `ripple()` returns
  `(diagnostic_svg, circular_svg)` strings.
- `features.py`: millisecond RMS, relative spectral bands, supplied pitch.
- `pitch.py`: versioned, WAV-hash-bound known-ground-truth JSON; complete
  original-frame coverage. **No pitch estimation, including external or
  manually corrected speech estimates.** Read `docs/pitch-input.md`.
- `geometry.py`: bounded periodic Fourier profile, float radial geometry,
  analytic matching cubic tangents including the closed seam.
- `utils.py`: cyclic float median, baseline fits, XML SVG serialization.
- `main.py`: single-conversion CLI; output-specific diagnostics.
- `compare.py`: individual words, diagnostics, co-centered multiword overlays
  for each algorithm, labeled side-by-side algorithm preview.
- `generate_fixtures.py`: deterministic WAV/JSON from actual oscillator controls.
- `merge.py`: legacy path-only overlay/scaling, optional native-Cairo PNG.
- `names.sh`: first-three-method comparison of four example names into a new
  explicit output directory; no parallel/shared-diagnostic race.
- `tests/test_ripple.py`, `.github/workflows/tests.yml`: unittest and CI.

## Environment and validation

Use a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m py_compile main.py ripple.py utils.py merge.py config.py audio.py pitch.py features.py geometry.py compare.py generate_fixtures.py
python3 main.py --help
python3 compare.py --help
python3 merge.py --help
python3 -m unittest discover -s tests -v
```

No formatter/linter/packaging is configured. Do not invent existing tools.
Dependencies are NumPy, svgpathtools, CairoSVG. Missing CairoSVG **or native
Cairo** warns for implicit PNG export but fails for explicit `--png`.
`--no-png` is SVG-only. Never count absent PNG as a successful rendering check.
Tests exercise the host's available/unavailable branch and mock missing and
failed renderers; native PNG dimensions are checked when available.

## Contracts to preserve

- Shared preprocessing applies to **all four methods**, including baseline.
  Preserve internal pauses and sample-rate/original-frame alignment.
- Default algorithm is baseline. `--smoothness` is baseline-only; `--poly-degree`
  (alias `--detail`) controls fit detail. Polynomial fit is baseline-only and
  is projected to periodic geometry. New methods use Fourier profiles.
- `--baseline-gain` is a fixed display gain before the shared geometry limiter,
  not per-word normalization. It affects both baseline fits, not the energy
  methods; baseline diagnostics retain the unscaled extracted profile.
- Extraction is independent of export dimensions. Detail, deformation and
  roundness are separate. Do not stretch tiny residuals to full radial range.
- Every word is a **closed, connected smooth loop on the SAME nominal
  diameter and center**, independent of duration/sample count/method. Never
  auto-fit individual bounding boxes. Overlay overlap is intentional.
- Artwork side is `min(max_width, height)`; the light-gray reference circle is
  enabled by default (`--no-circle` hides it), never a substitute for the curve.
  Use floating geometry and matching seam tangents.
- `energy-pitch` always requires validated known-source input, even at zero
  pitch weight. No invented Hz, no silent energy fallback, no speech estimates.
  Absolute pitch must survive centering as a broad shape component; mean
  subtraction must not erase it. Unvoiced source modulation is zero.
- `main.py` writes `OUTPUT_STEM.diagnostic.svg` or `--diagnostic`; **never**
  writes shared root `tmp.svg`. Parent directories must exist.
- Comparisons require a new explicit output directory and render/validate all
  inputs before writing any files. Four-method comparison needs a valid
  sidecar per word; real speech explicitly selects the first three.
- `merge.py` reads all `.svg` in its input directory and scales from the
  canvas origin, not per-path bounds. Merge `METHOD/words/`, not diagnostics
  or labeled previews. `--output` is required; keep output outside input. General group transforms
  are not supported by its legacy path-only parser.

## Changes and checks

Read implementation and CLI help before changing behavior. Keep the
function-based structure and reuse shared helpers. Update related docs and
configuration for argument/default/output changes. Report processing errors
explicitly. Do not silently skip failed inputs or render success-shaped
partial pitch comparisons.

Do not modify committed WAVs or regenerate `output/`, `output_names/`,
`output_samples/`, root `tmp.svg` unless specifically required. Run new audio
checks in an isolated temporary directory, using absolute input/script paths.
Do not run `names.sh` as a routine check. Tests use temporary directories and
small analytic 16-bit mono/stereo fixtures; maintain exact feature assertions,
explicit numerical tolerances, finite/positive/closed/bounded geometry and
join/seam tangent checks. Exercise both baseline fits and all four methods.
Test size invariance, zero-feature ablations, pitch errors and trim alignment,
collision-safe names, shared-diameter overlays and PNG failure behavior.

Synthetic checks cannot establish beauty or real-world uniqueness. See
`docs/evaluation.md` for the human comparison protocol. Never invent true
pitch for natural samples. Check `git diff` and `git status --short` before
committing for accidental generated files.
