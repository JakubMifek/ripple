# A gold pendant from the four-name composition

This is a **3D jewelry visualization**, not a recolored SVG or a photograph of
an existing object. The four word curves become openwork round gold wires.
The gray reference circle is excluded. The result is an artistic concept,
not a manufacturing-ready design or a certified gold alloy.

## Run it

Use the repository's Python environment and install
[Blender 4.5 LTS or newer](https://www.blender.org/download/lts/).
Blender is an optional, separate application; ordinary Ripple conversions do
not require it. No add-ons, HDRI downloads, paid assets or cloud service are used.
Blender's own Python builds and renders the scene, so do not install `bpy` in
Ripple's virtual environment.

From the repository root, with the virtual environment active:

```bash
mkdir -p artwork
python3 src/pendant.py --output-dir artwork/gold-pendant
```

The default source is
`output_samples/names-comparison/energy-timbre-pitch/overlay.svg`:
**Kuba, Eliska, Baltazar and Leontynka**, with their existing recording-derived
orientations. To use another Ripple artwork or multiword overlay:

```bash
python3 src/pendant.py path/to/overlay.svg \
  --output-dir artwork/another-pendant \
  --diameter-mm 32 --wire-mm 0.75 --resolution 1600 --samples 256 \
  --save-scene
```

If Blender is not on `PATH`, add `--blender /path/to/blender`. On macOS that is
typically `/Applications/Blender.app/Contents/MacOS/Blender`; on Windows,
pass the full path to `blender.exe`. A minimal Linux installation also needs
Blender's native libraries, including `libxkbcommon0`. Missing libraries or
failed Cycles renders are errors, never a fallback to a flat illustration.

The local renderer prepared in this checkout can be selected with
`--blender .cache/blender/run-blender`. This ignored local installation is not
part of a fresh clone.

The output directory must be new and its parent must exist. It contains:

- `pendant.png`: square, opaque, color-managed studio image.
- `scene.json`: input hash, exact source cubics, dimensions and render settings.
- `render.json`: renderer version, geometry checks and visualization caveats.
- `render.log`: local Blender output.
- `pendant.blend`: editable scene, only with `--save-scene`.

Saved Blender scenes use a relative image output path so they remain usable
after moving their output directory.

## What makes it look like jewelry

The script preserves the SVG's cubic centerlines and shared coordinate system,
including each word's orientation. It does not fit or rotate individual words.
The **32 mm nominal diameter** corresponds to the SVG's common reference
diameter, not the pendant's widest outside measurement. Deformation makes
the actual pendant wider; `render.json` records its finished mesh dimensions.
Wire diameter is independent of the SVG stroke width.

Round wire volumes are fused at their crossings and gently polished, rather
than left as intersecting surface tubes. A small attachment eye, connector ring
and alternating cable-chain links complete the composition. The renderer
rejects genuinely disconnected pendant bodies. Tiny isolated voxel shells
smaller than a real strand are explicitly logged and removed, and sampled
source centerlines are checked against the finished body.

Gold uses a metallic, energy-conserving shader with warm reflectance, subtle
roughness variation and microscopic surface relief. Three rectangular studio
lights create broad highlights and narrow edge reflections. A fine-grained
charcoal surface supplies contact shadows and dark reflected tones. Cycles
traces multiple glossy bounces; AgX manages bright specular highlights.
The slightly oblique product view keeps the full pendant sharp, like a
focus-stacked jewelry photograph, rather than hiding detail with heavy blur.

The default is **1600 x 1600, up to 256 samples, four CPU render threads**, with
adaptive sampling and denoising. For a quick preview use
`--resolution 768 --samples 64`. For a larger image try
`--resolution 2400 --samples 512`. Runtime depends on the machine.
The seed is fixed, but different Blender versions and hardware can change pixels.

Use a single word SVG or `METHOD/overlay.svg`, not diagnostics or the labeled
five-panel preview. Arbitrary SVG transforms, CSS, nested viewports and non-cubic
artwork are rejected instead of silently changing the composition.

Before making a real pendant, a jeweler must assess junction strength, chain
and bail clearance, casting tolerances, finishing, weight and wearability.
The visualization does not validate those properties.

## Renderer checks

Normal unittest discovery covers SVG conversion and mocked renderer failures
without installing Blender. To also exercise a real small Cycles render,
editable-scene relocation and disconnected-body rejection:

```bash
RIPPLE_TEST_BLENDER=/path/to/blender \
  python3 -m unittest discover -s tests -p test_pendant.py -v
```
