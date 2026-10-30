# Known-ground-truth pitch input (version 1)

`energy-pitch` accepts **16-bit PCM WAV + JSON**, not CSV. JSON keeps provenance,
audio binding and complete alignment together. Ripple does **not** estimate,
detect, infer, correct or automatically fill pitch. This method is an
artistic encoding of supplied facts, not an acoustic pitch-analysis tool.

## What counts as a usable source?

- `known-generating-controls`: actual oscillator fundamental-frequency
  commands or simulator/synthesizer source controls used to generate this
  precise recording. The commanded value must define the source's periodic
  fundamental, not a nominal musical note or intended speaking pitch.
- `direct-source-measurement`: exact source-domain frequency telemetry, such
  as a digital oscillator phase-increment/frequency register captured during
  generation, with exact sample alignment. This does **not** mean an acoustic
  measurement interpreted through a pitch estimator.

Real speech fundamental frequency cannot generally be recovered exactly from
arbitrary audio without estimation. External speech estimators, manually
reviewed/corrected estimates, tuner readings, note annotations, score/MIDI
intent for a human performer, and inferred labels are **excluded**, even if
they look plausible. A synthesizer's note controls count only if they actually
define the generating fundamental (account for detuning, bends and timing).
Do not fabricate sidecars for the committed speech recordings.

Source declarations and SHA-256 binding **cannot prove truthfulness**. Ripple
validates structure, declared category, coverage and audio binding, not whether
the user honestly supplied known ground truth. The user must verify provenance.

## JSON schema and alignment

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
No conversion from CSV or estimated tracks is provided.

## Rendering, absolute pitch and unvoiced portions

Supplied frequency is converted to log2 relative to a globally configured
reference (`220 Hz`, two-octave scale by default). The mean known log-frequency
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

Missing/malformed/estimated-source/mismatched/incomplete sidecars fail, even
when pitch weight is zero. No energy-only fallback is labeled `energy-pitch`.

## Generate a valid demonstrator

```bash
python3 generate_fixtures.py /tmp/ripple-known-controls
mkdir -p artwork
python3 main.py /tmp/ripple-known-controls/interrupted.wav artwork/known.svg \
  --algorithm energy-pitch --pitch /tmp/ripple-known-controls/interrupted.pitch.json
```

The generator is deterministic and records the actual piecewise-constant
fundamental controls used for a phase-continuous sinusoid with a small second
harmonic and amplitude envelope. It creates constant, changing and interrupted
voicing. These are not real-speech fixtures. It refuses to overwrite a directory.
For comparisons, pass one sidecar per word in matching input order:

```bash
python3 compare.py /tmp/ripple-known-controls/constant.wav \
  /tmp/ripple-known-controls/changing.wav \
  --pitch /tmp/ripple-known-controls/constant.pitch.json \
  /tmp/ripple-known-controls/changing.pitch.json \
  --output-dir artwork/known-comparison
```
