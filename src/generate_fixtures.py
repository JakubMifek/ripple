"""Deterministic controlled synthesis, recording actual generating pitch controls."""

import argparse
import json
from pathlib import Path
import wave
import numpy as np
from pitch import wav_hash


def generate(directory, sample_rate=8000):
    directory = Path(directory)
    controls = {
        "constant": [(0.25, 220.0), (0.25, 220.0), (0.25, 220.0)],
        "changing": [(0.25, 165.0), (0.25, 220.0), (0.25, 330.0)],
        "interrupted": [(0.25, 220.0), (0.15, None), (0.25, 330.0)],
    }
    if directory.exists():
        raise ValueError("Fixture output directory must not already exist.")
    directory.mkdir(parents=True)
    for name, segments in controls.items():
        chunks, intervals, start, phase = [], [], 0, 0.0
        for duration, hz in segments:
            count = round(duration * sample_rate)
            t = np.arange(count) / sample_rate
            # The constant Hz controls below, not an acoustic estimate, define the sidecar.
            if hz is None:
                values = np.zeros(count)
            else:
                envelope = 0.55 * np.sin(np.pi * np.arange(count) / (count - 1)) ** 2
                values = envelope * (np.sin(phase + 2 * np.pi * hz * t)
                                     + 0.15 * np.sin(2 * phase + 4 * np.pi * hz * t))
                phase = (phase + 2 * np.pi * hz * count / sample_rate) % (2 * np.pi)
            chunks.append(values)
            intervals.append({"start": start, "end": start + count, "hz": hz})
            start += count
        wav_path = directory / f"{name}.wav"
        with wave.open(str(wav_path), "wb") as target:
            target.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
            target.writeframes(np.rint(np.concatenate(chunks) * 32767).astype("<i2").tobytes())
        data = {
            "version": 1,
            "source": {"category": "known-generating-controls",
                       "description": "generate_fixtures.py: phase-continuous fundamental plus 0.15 second harmonic; "
                                      "per-segment sine-squared amplitude envelope; Hz are exact oscillator controls. "
                                      "null segments are generated silence. PCM quantization is not pitch estimation."},
            "wav_sha256": wav_hash(wav_path), "sample_rate": sample_rate, "intervals": intervals}
        (directory / f"{name}.pitch.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="New fixture directory")
    args = parser.parse_args(argv)
    try:
        generate(args.output_dir)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Fixture error: {error}\n")
    print(f"Known generating controls and WAVs: {args.output_dir}")


if __name__ == "__main__":
    main()
