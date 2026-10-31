# Agent instructions

## Project map

Ripple is a Python 3.11+ script collection, not an installed package or service.
It makes artistic circular keepsakes from 16-bit PCM WAV recordings, not
guaranteed unique biometric signatures or physically tattoo-ready designs.
Python source lives in `src/`; configuration, tests and helper scripts remain
at the repository root. Execute CLIs as `python3 src/main.py`, not as a package.

- `ripple.toml`, `src/config.py`: version-1 defaults, strict TOML overrides,
  validation and shared CLI options. Defaults -> explicit config -> CLI.
- `src/audio.py`: floating channel averaging, DC removal, sustained-RMS trimming
  with quiet boundary extension, gain normalization; retains original offsets.
  `Audio.start_angle` uses original, unprocessed PCM plus rate/channels.
  `presets/legacy-trim.toml` reproduces the former sample-peak trimming.
- `src/ripple.py`: orchestration, legacy signed-waveform baseline with cyclic median
  smoothing/extrema/recursive midpoints, diagnostic SVG. `ripple()` returns
  `(diagnostic_svg, circular_svg)` strings.
- `src/features.py`: millisecond RMS, relative spectral bands, aligned pitch.
- `src/pitch.py`: automatic local Praat-Parselmouth pitch estimation, plus
  versioned WAV-hash-bound known-source JSON. Estimates are explicitly labeled,
  never declared as known ground truth. Read `docs/pitch-input.md`.
- `src/geometry.py`: bounded periodic Fourier profile, float radial geometry,
  analytic matching cubic tangents including the closed seam.
- `src/utils.py`: cyclic float median, baseline fits, XML SVG serialization.
- `src/main.py`: single-conversion CLI; output-specific diagnostics.
- `src/compare.py`: individual words, diagnostics, co-centered multiword overlays
  for each algorithm, labeled side-by-side algorithm preview.
- `src/generate_fixtures.py`: deterministic WAV/JSON from actual oscillator controls.
- `src/merge.py`: legacy path-only overlay/scaling, optional native-Cairo PNG.
- `src/pendant.py`, `src/pendant_scene.py`: optional local Blender/Cycles gold
  pendant renderer. The wrapper uses normal Python; the worker uses Blender's
  Python. Read `docs/pendant-guide.md`. Do not add Blender to core dependencies.
- `names.sh`: five-method comparison of four example names into a new
  explicit output directory; no parallel/shared-diagnostic race.
- `tests/test_ripple.py`, `.github/workflows/tests.yml`: unittest and CI.

## Environment and validation

Use a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m py_compile src/*.py
python3 src/main.py --help
python3 src/compare.py --help
python3 src/merge.py --help
python3 -m unittest discover -s tests -v
```

No formatter/linter/packaging is configured. Do not invent existing tools.
Dependencies are NumPy, svgpathtools, CairoSVG and praat-parselmouth (embedded
Praat; no desktop application required). Missing CairoSVG **or native
Cairo** warns for implicit PNG export but fails for explicit `--png`.
`--no-png` is SVG-only. Never count absent PNG as a successful rendering check.
Tests exercise the host's available/unavailable branch and mock missing and
failed renderers; native PNG dimensions are checked when available.

## Contracts to preserve

- Shared preprocessing applies to **all five methods**, including baseline.
  Preserve internal pauses and sample-rate/original-frame alignment.
  Default trimming uses 20 ms RMS / 5 ms hops, sustained -20 dB anchors,
  -30 dB edge extension, a 40 ms quiet-gap stop and 20 ms padding. Keep all
  internal regions between anchors; never use pitch voicing as a speech crop.
  This is not denoising or guaranteed speech detection. No sustained activity
  fails explicitly, never silently falls back to peak trimming. Preserve the
  legacy preset and original-peak gain normalization.
- Default algorithm is `energy-timbre-pitch`, with automatic local pitch
  estimation unless a known-source sidecar is supplied. Select baseline explicitly
  with `--algorithm baseline` or configuration. Comparisons still default to all
  five methods and validate settings against their requested algorithms.
  `--smoothness` is baseline-only; `--poly-degree`
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
- Default `render.orientation = "recording"` uses the versioned original-audio
  hash angle, identical across algorithms/export sizes and independent of paths
  or preprocessing settings. Bake the rigid rotation into points and tangents,
  never SVG group transforms. Do not rotate reference paths: keep them identical
  for deduplication. Preserve per-word orientation metadata through comparisons.
  `--orientation fixed` restores 0 degrees; use it for feature/gain/padding
  invariances so decorative hash rotation does not masquerade as a shape change.
- Artwork side is `min(max_width, height)`; the light-gray reference circle is
  enabled by default (`--no-circle` hides it), never a substitute for the curve.
  Mark it with `data-role="reference-circle"`: light `#c8c8c8`, 2 px, dotted.
  Share `REFERENCE_STROKE_WIDTH` from `src/config.py` across rendering and merging.
  Deduplicate identical references below all word curves in comparisons and
  legacy merges; artwork stroke settings must not change reference thickness.
  Use floating geometry and matching seam tangents.
- `energy-pitch` and `energy-timbre-pitch` estimate automatically without `--pitch`.
  Supplied known-source sidecars bypass estimation and retain strict validation;
  invalid sidecars never fall back to estimation. Missing/failed estimation or
  no voiced frames fail explicitly, even at zero pitch weight.
  Preserve original-frame alignment and zero modulation for unvoiced or
  unanalyzed frames. Keep estimated/known provenance in both SVGs, word groups
  and preview labels; estimates must never masquerade as known-source JSON.
  Absolute pitch must survive centering as a broad shape component; mean
  subtraction must not erase it. Unvoiced source modulation is zero.
  `pitch_octaves` scales average log-pitch; `pitch_inflection_octaves` separately
  scales deviations from its voiced mean. Both are fixed global scales, never
  per-word min-max normalization. `--pitch-sensitivity` overrides
  `features.pitch_weight` in both CLIs; it must not change estimation or Hz.
- Reuse `PITCH_ALGORITHMS` and `TIMBRE_ALGORITHMS` from `src/config.py` for
  feature selection, pitch routing and diagnostics. The combined method adds
  energy, timbre and pitch before the periodic fit using the existing weights;
  zero-feature ablations must reproduce the corresponding simpler method.
  Default timbre removes DC and the projection onto centered energy on the
  normalized-time grid; do not amplify the residual. Share `timbre_profile`
  between composition and the corrected diagnostic trace. Preserve exact old
  feature behavior through `presets/legacy-features.toml`.
- `src/main.py` writes `OUTPUT_STEM.diagnostic.svg` or `--diagnostic`; **never**
  writes shared root `tmp.svg`. Parent directories must exist.
- Comparisons require a new explicit output directory and render/validate all
  inputs before writing any files. Five-method comparison works from WAVs alone.
  An explicit `--pitch` override requires one valid known-source sidecar per word.
- `src/merge.py` reads all `.svg` in its input directory and scales from the
  canvas origin, not per-path bounds. Merge `METHOD/words/`, not diagnostics
  or labeled previews. `--output` is required; keep output outside input. General group transforms
  are not supported by its legacy path-only parser.
- Pendant rendering consumes word/overlay SVGs, never diagnostics or previews.
  Preserve source cubics and the shared nominal-diameter scale; omit marked
  reference circles. Reject unsupported SVG transforms/CSS rather than flattening
  them incorrectly. Keep real disconnected bodies and failed/missing Cycles
  output as errors; never substitute a flat image. Deliver a visual concept,
  not a claim of manufacturability. New output directories protect existing files.

## Changes and checks

Read implementation and CLI help before changing behavior. Keep the
function-based structure and reuse shared helpers. Update related docs and
configuration for argument/default/output changes. Report processing errors
explicitly. Do not silently skip failed inputs or render success-shaped
partial pitch comparisons.

Do not modify committed WAVs or regenerate `output/`, `output_names/`,
`output_samples/` unless specifically required. The obsolete root `tmp.svg`
has been removed; do not recreate it. Run new audio
checks in an isolated temporary directory, using absolute input/script paths.
Do not run `names.sh` as a routine check. Tests use temporary directories and
small analytic 16-bit mono/stereo fixtures; maintain exact feature assertions,
explicit numerical tolerances, finite/positive/closed/bounded geometry and
join/seam tangent checks. Exercise both baseline fits and all five methods.
Test size invariance, zero-feature ablations, pitch errors and trim alignment,
collision-safe names, shared-diameter overlays and PNG failure behavior.

Synthetic checks cannot establish beauty or real-world uniqueness. See
`docs/evaluation.md` for the human comparison protocol. Never invent true
pitch for natural samples. Check `git diff` and `git status --short` before
committing for accidental generated files.
