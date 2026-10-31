# Record your voice and understand pitch inputs

**Use Audacity to record your microphone.** It is free and available for
Windows, macOS, and Linux. Your recording can be used immediately with Ripple's
baseline, energy, and energy-timbre methods.

**A microphone recording cannot supply the exact pitch data required by the
fourth method under Ripple's current no-estimation rule.** A WAV contains the
sound, but not a verified timeline of its fundamental frequency. A better
microphone improves the recording; it does not create that timeline.
The WAV sample rate (such as 48000 Hz) is not your voice's pitch.

| What you want | What to use |
|---|---|
| A personal curve from someone saying a name | Microphone + Audacity; compare the first three methods |
| WAVs with valid known-pitch sidecars to try all four methods | Ripple's built-in controlled-sound generator |

## 1. Record a name with your microphone

Record on the computer connected to your microphone. No cloud upload is needed.

1. Install [Audacity](https://www.audacityteam.org/download/). Connect your USB
   microphone, or connect an XLR microphone through your audio interface.
2. In Audacity, choose **Audio Setup > Recording Device** and select your
   microphone or interface. Use **one recording channel (mono)**. If the device
   is missing, use **Rescan Audio Devices**. Menu wording varies by version.
3. Use a quiet room and keep a consistent microphone position, roughly
   15-20 cm away as a starting point. A pop filter helps. Disable automatic gain,
   noise suppression, and voice effects if your device offers them.
4. Make a short test recording and listen back. Adjust the microphone/interface
   gain so normal speech is around **-18 to -12 dB** on the meter, with room for
   louder syllables. If the meter turns red or the waveform clips, reduce gain
   and record again; lowering the volume afterward cannot undo clipping.
5. Record **3-5 natural takes** of the name. Leave about half a second of quiet
   before and after each take. Speak as you normally would; there is no need to
   hold a musical note. Keep the microphone position the same between takes.
6. Select one complete take, including a little surrounding quiet, and use
   **File > Export Audio**. Choose **selected audio/current selection**, not the
   entire session containing every take. Do not cut off consonants or remove
   pauses within the name.

Use these export settings:

| Setting | Value |
|---|---|
| Format | WAV / WAV (Microsoft) |
| Encoding | **Signed 16-bit PCM** |
| Channels | **Mono** |
| Sample rate | **48000 Hz** (44100 Hz is also supported) |
| Destination | A local folder, for example `recordings/` in the Ripple checkout |
| Filename | For example `my-name-take-01.wav` |

Do not export MP3, 24-bit PCM, or 32-bit float for Ripple. An Audacity `.aup3`
project is useful for keeping the original session, but is not the WAV export.
Skip normalization, compression, pitch correction, time stretching, and other
effects for your first comparison; Ripple already handles DC removal, gain
normalization, and conservative trimming. Keep an untouched original.
Keep recordings local unless you deliberately choose to share or commit them.

## 2. Generate a comparison from the recording

After the [repository setup](../README.md#setup), run this from the repository
root with the virtual environment active:

```bash
python3 src/compare.py recordings/my-name-take-01.wav \
  --algorithms baseline energy energy-timbre \
  --output-dir artwork/my-name-comparison
```

Open `artwork/my-name-comparison/comparison.svg` in a browser. It shows the
three curves with gray reference circles. The output directory must be new;
choose another name for another run.

To overlay several names, put their WAV paths after `src/compare.py`, before
`--algorithms`. The [combined-names example](../README.md#combined-names-example)
shows the full command. Every name will use the same nominal diameter and center.

**For your own voice recordings, you only need to prepare the WAV files.
Do not create pitch JSON by guessing or copying values from a tuner.**

## 3. Generate WAVs that really have known pitch

For a valid four-method demonstration, use the included generator rather than
recording your microphone:

```bash
python3 src/generate_fixtures.py artwork/known-pitch-audio
```

This creates three matched pairs:

| WAV | Matching pitch file | Sound |
|---|---|---|
| `constant.wav` | `constant.pitch.json` | Constant generating frequency |
| `changing.wav` | `changing.pitch.json` | Changing generating frequency |
| `interrupted.wav` | `interrupted.pitch.json` | Known voiced sections separated by silence |

These are synthetic sounds, **not recordings of a person's voice**. The
generator writes the exact frequency controls used to produce each WAV,
their sample timing, and the WAV's hash into its JSON. You do not need to
write JSON yourself. The output directory must not already exist.

Compare all four methods:

```bash
python3 src/compare.py artwork/known-pitch-audio/changing.wav \
  --pitch artwork/known-pitch-audio/changing.pitch.json \
  --output-dir artwork/known-pitch-comparison
```

Without `--algorithms`, comparison includes all four methods. Open
`artwork/known-pitch-comparison/comparison.svg` to see them. Keep each WAV and
its JSON together. Listening in Audacity is fine, but editing or re-exporting
the WAV invalidates the supplied file hash and may invalidate timing.

## Why not extract pitch from my recording?

Programs such as Praat and musical tuners **estimate** pitch. Manual correction,
singing an intended note, or knowing a melody does not make those values exact
ground truth. They are excluded by the current project contract.

Likewise, playing a known tone through a speaker and recording it with your
microphone is not a ready-made valid pair: the captured WAV has its own timing,
latency, and recording-clock alignment. Use direct digital generation with
matching source controls, not the sidecar for a different recording.

Using the fourth method on ordinary recorded speech would require changing
the no-estimation requirement. For the formats and accepted source information,
see the [technical pitch-input guide](pitch-input.md).

## Audacity references

- [Recording with a microphone](https://support.audacityteam.org/basics/recording-your-voice-and-microphone)
- [Exporting audio and selecting the export range](https://www.audacityteam.org/manual/getting-started/export-your-audio/)
- [WAV channels, sample rate, and signed 16-bit PCM](https://manual.audacityteam.org/man/wav_export_options.html)
