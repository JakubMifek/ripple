# Pitch: automatic estimates or known-source input

For microphone recording steps, start with the
[simple recording guide](recording-guide.md).

## Automatic pitch from a WAV

With the repository requirements installed, a **16-bit PCM WAV alone** is
enough. No Praat GUI, manual pitch labels or separate export is required:

```bash
mkdir -p artwork
python3 src/main.py recordings/my-name.wav artwork/my-name.svg --algorithm energy-pitch
python3 src/compare.py recordings/my-name.wav --output-dir artwork/my-name-comparison
```

Ripple calls [Praat-Parselmouth](https://parselmouth.readthedocs.io/en/stable/)
from Python, locally, using Praat's accurate autocorrelation analysis on the
shared mono, DC-removed, trimmed and gain-normalized audio. The first three
methods do not invoke the estimator. Passing `--pitch` bypasses estimation
and uses the strict known-source contract below instead.
Both `energy-pitch` and `energy-timbre-pitch` support this workflow. The latter
adds relative timbre using the existing `timbre_weight`; `pitch_weight` controls
pitch in both methods.

The `[pitch_estimation]` TOML defaults are:

```toml
[pitch_estimation]
floor_hz = 60.0
ceiling_hz = 600.0
hop_ms = 10.0
voicing_threshold = 0.45
```

An override file also needs `version = 1` before its tables. The ceiling must
exceed the floor and be below the WAV Nyquist frequency. The accurate analysis
window spans `6 / floor_hz` seconds, so the default needs at least 100 ms of
audio after trimming. The hop must span at least one audio sample.

Praat selects a frequency or rejects voicing at each analysis frame. Ripple
maps these estimates to original WAV frame indices using boundaries halfway
between analysis centers. The first/last analyzed intervals extend only half
a hop beyond their centers. Trimmed portions and unanalyzed boundary regions
have no pitch contribution, as do frames rejected as unvoiced. Frequencies are
not interpolated across gaps; this does not assert that every rejected or
unanalyzed portion is truly unvoiced.

If no voiced pitch is detected, analysis fails, the recording is too short,
or the dependency is unavailable, processing fails explicitly. Ripple does
not silently substitute energy-only output, even at zero pitch weight.
Comparisons validate all inputs before creating their output directory.

**Estimation is not ground truth.** Noise, breathy/creaky voices, multiple
speakers, and pitches outside the configured range can cause missed voicing
or octave errors. These estimates are suitable for an artistic comparison,
not proof of a speaker's identity or exact vocal frequency.

CLI output and the `energy-pitch (estimated)` and
`energy-timbre-pitch (estimated)` panel labels identify estimates.
Artwork and diagnostic SVGs carry `data-pitch-source="estimated"` and
`data-pitch-estimator` with Praat/Parselmouth versions; comparison word groups
retain their own provenance. Pitch modulation is blue in the `energy-pitch`
diagnostic and green in `energy-timbre-pitch`, where blue shows timbre.
These lines show modulation used by the geometry, not raw Hz. Keep the WAV and configuration
to reproduce a result. Automatic estimates are not written as known-source JSON.

## Optional known-source JSON (version 1)

`--pitch FILE.pitch.json` accepts **WAV + JSON**, not CSV or Praat exports.
JSON keeps provenance, audio binding and complete alignment together. This
format remains reserved for known-source frequencies, not estimated tracks.
Supplying an invalid or mismatched file is an error, never a request to
fall back to estimation.

### Accepted sources for known-source JSON

- `known-generating-controls`: actual oscillator fundamental-frequency
  commands or simulator/synthesizer source controls used to generate this
  precise recording. The commanded value must define the source's periodic
  fundamental, not a nominal musical note or intended speaking pitch.
- `direct-source-measurement`: exact source-domain frequency telemetry, such
  as a digital oscillator phase-increment/frequency register captured during
  generation, with exact sample alignment. This does **not** mean an acoustic
  measurement interpreted through a pitch estimator.

Real speech fundamental frequency cannot generally be recovered exactly from
arbitrary audio without estimation. Speech estimates, manually corrected
tracks, tuner readings, and intended musical notes must **not** be declared as
known-source sidecars. Use automatic estimation for microphone recordings.
A synthesizer's note controls count as known only if they actually define
the generating fundamental (account for detuning, bends and timing).

Source declarations and SHA-256 binding **cannot prove truthfulness**. Ripple
validates structure, declared category, coverage and audio binding, not whether
the user honestly supplied known ground truth. The user must verify provenance.

### JSON schema and alignment

Example for a hypothetical 8000 Hz, 4000-frame controlled WAV (replace the
placeholder with the actual SHA-256 of the entire WAV file):

```json
{
  "version": 1,
  "source": {
    "category": "known-generating-controls",
    "description": "Exact controls from my oscillator; 220 Hz, silence, then 330 Hz."
  },
  "wav_sha256": "REPLACE_WITH_64_CHARACTER_SHA256_OF_ENTIRE_WAV",
  "sample_rate": 8000,
  "intervals": [
    {"start": 0, "end": 1600, "hz": 220.0},
    {"start": 1600, "end": 2400, "hz": null},
    {"start": 2400, "end": 4000, "hz": 330.0}
  ]
}
```

All keys shown are mandatory and unknown keys are rejected. Version and sample
indices/rate are integers, not booleans. Intervals use **original WAV frame
indices** (a frame includes all channels), `[start, end)` with exclusive end.
They must be ordered, contiguous, nonempty, nonoverlapping and cover **every
frame from zero through the original frame count**, including surrounding
silence and internal pauses. Declare known unvoiced portions as `"hz": null`.
Missing/unknown data is an error; `null` must not be used to conceal unknown
voiced pitch. Voiced Hz must be finite, positive and strictly below Nyquist.
Changing frequencies can be supplied as consecutive known-control intervals,
including one interval per frame if needed; there is no pitch-track inference.

After trimming, Ripple retains the original start/end indices to align the
sidecar. Mono conversion does not change frame indices or sample rate.
Editing/re-exporting the WAV requires a newly verified sidecar and hash.
Use `sha256sum recording.wav` or Python's `hashlib.sha256` on the entire file.
No import from CSV or external estimated tracks is provided.

## Rendering, absolute pitch and unvoiced portions

Known or estimated frequency is converted to log2 relative to a globally configured
reference (`220 Hz`, two-octave scale by default). The mean voiced log-frequency
controls a **broad cosine component**, not a constant radius offset that would
disappear when centering a profile. Inflection around that mean adds local
detail. Thus absolute pitch is retained between words, while this utterance's
pitch changes remain encoded. Means are weighted by voiced sample duration.
The absolute and inflection weights and total pitch weight are explicit
artistic settings in `[features]`.

Unvoiced source samples contribute **zero pitch modulation**. No Hz is filled,
interpolated or invented across them. The modulation (not Hz) is resampled and
Fourier-smoothed to make a bounded, rounded, connected periodic curve. As with
any global Fourier representation, geometric effects can extend into adjacent
unvoiced angular regions; this is rendering, not evidence of voicing there.
The circular seam is also artistic, not a claim that the utterance is periodic.

Explicitly supplied missing/malformed/estimated-source/mismatched/incomplete
sidecars fail, even when pitch weight is zero. No energy-only fallback is
labeled as either pitch-based method.

## Generate a known-source demonstrator

```bash
python3 src/generate_fixtures.py /tmp/ripple-known-controls
mkdir -p artwork
python3 src/main.py /tmp/ripple-known-controls/interrupted.wav artwork/known.svg \
  --algorithm energy-pitch --pitch /tmp/ripple-known-controls/interrupted.pitch.json
```

The generator is deterministic and records the actual piecewise-constant
fundamental controls used for a phase-continuous sinusoid with a small second
harmonic and amplitude envelope. It creates constant, changing and interrupted
voicing. These are not real-speech fixtures. It refuses to overwrite a directory.
For comparisons, pass one sidecar per word in matching input order:

```bash
python3 src/compare.py /tmp/ripple-known-controls/constant.wav \
  /tmp/ripple-known-controls/changing.wav \
  --pitch /tmp/ripple-known-controls/constant.pitch.json \
  /tmp/ripple-known-controls/changing.pitch.json \
  --output-dir artwork/known-comparison
```
