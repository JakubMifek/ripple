# Compare the art, not a biometric claim

Use `src/compare.py` with the **same configuration** for every word/algorithm.
The algorithm panels contain the actual overlapping composition; also inspect
the individual word curves and standalone `METHOD/overlay.svg`. Every word
uses the same canvas, center, nominal diameter, orientation and stroke.
Never independently fit each bounding box to the canvas: that destroys the
common-circle composition. Record the WAVs, configuration and CLI overrides
so comparisons can be repeated.

## Suggested listening and viewing study

1. With consent, record several repeated utterances of the same name by one
   speaker, the same name by several speakers, and different names by the
   same speaker. Keep microphone/distance and recording conditions consistent;
   include a separate varied-condition set. Preserve original WAVs.
2. Explicitly select `--algorithms baseline energy energy-timbre` for real
   speech unless genuine known-source pitch exists. Do not generate estimated
   speech sidecars to make a four-way panel. Use generated known controls for
   pitch experiments, clearly labeled as synthetic.
3. Shuffle/blind algorithm labels for human evaluation without changing
   geometry. Ask for roundness, visual appeal, seam quality, clutter/overlap,
   sense of personal connection, and preference at several viewing scales.
   Evaluate single curves **and overlapped words**, including words of very
   different durations. Audio listening and personal interpretation may be
   evaluated separately from blinded visual preference.
4. Compare repeated utterances against across-speaker/across-name differences.
   Record qualitative ratings and uncertainty; examine whether tiny recording
   changes dominate meaningful utterance differences. Do not interpret visual
   similarity as speaker identification or dissimilarity as uniqueness.
5. Before any tattoo, consult an artist about spacing, line weight, aging,
   placement, actual physical scale and overlap readability. SVG scalability
   alone does not establish safe or aesthetically durable tattoo dimensions.

The shared preprocessing reduces polarity/gain sensitivity for energy methods
and removes surrounding silence conservatively; it does not guarantee
invariance to pronunciation, noise, microphone coloration, clipping or room
acoustics. Baseline is deliberately signed-waveform based and is not expected
to resist polarity inversion. Synthetic tests validate numerical contracts,
not uniqueness, real-world identity, emotional value or beauty.

For feature ablations, use TOML overrides for `timbre_weight = 0` or
`pitch_weight = 0`: secondary-zero geometry must match energy. Vary absolute
known oscillator pitch separately from inflection controls and note that
changing a real sound's pitch can also change energy/band features. Do not
attribute every geometric change to just one feature without controlling
the others.
