# Full-size silver-wire drafts: the final four names

**The current five-page pack covers 35-80 mm in 5 mm steps, for both
1.0 mm and 0.9 mm round wire.** Sizes mean maximum outside body dimension,
including wire but excluding the bail. These are forming and size-comparison
guides, not a guarantee of strength or an assembly-ready jewelry specification.

## Compact five-page pack

- [A4 print PDF](../output_samples/final-names-print-compact/print-A4.pdf)
- [US Letter print PDF](../output_samples/final-names-print-compact/print-Letter.pdf)
- [Editable SVG sheets and measurements](../output_samples/final-names-print-compact/)

The page order is:

1. **1.0 mm forming:** four named sections, each with all ten sizes nested.
2. **1.0 mm comparison:** all ten complete compositions at actual size.
3. **0.9 mm forming:** the same four-section nested layout.
4. **0.9 mm comparison:** all ten compositions at actual size.
5. **Overlay alignment:** colored cumulative panels at 60 mm / 1.0 mm,
   explaining how to register the four names.

Each nested contour is numbered with the **finished composition's size**,
not the individual loop's width or height. Choose the same number and wire
diameter for all four names. Gray bands show actual wire thickness; follow
their dashed centers. The small white number windows are labels, not gaps
to cut in the wire. Center crosses and `TOP` marks preserve registration.
The comparison pages pack sizes out of numerical order to save paper;
read each label.

The fifth page demonstrates alignment, **not an over/under or soldering
sequence**. For other sizes or 0.9 mm wire, use the matching comparison
silhouette rather than the 60 mm / 1.0 mm example.

All original cubic curves remain intact. One common scale is used for the
four names:

```text
scale = (maximum outside body size - wire diameter) / longest combined centerline span
printed point = (source point - shared source center) * scale + sheet location
```

The reference circle is omitted. Physical stroke width stays fixed while
the curves grow; the two wire versions use slightly different centerline
scales to achieve the same outside body size. Projected lengths for every
size/wire combination are in
[`dimensions.json`](../output_samples/final-names-print-compact/dimensions.json).
They exclude closure, crossing, trimming, spare-wire and bail allowances.
Do not treat them as ready-to-cut lengths.

## Generate and check another pack

These are repeatable scripts, not one-off generation commands. Activate the
project virtual environment and install the optional print dependencies:

```bash
python3 -m pip install -r requirements-print.txt
python3 src/print_templates.py --help
python3 src/check_print_templates.py --help
mkdir -p artwork
python3 src/print_templates.py \
  output_samples/final-names-composition/energy-timbre-pitch/overlay.svg \
  --output-dir artwork/print-pack
python3 src/check_print_templates.py artwork/print-pack \
  --source output_samples/final-names-composition/energy-timbre-pitch/overlay.svg \
  --require-pdf --preview-dir artwork/print-previews
```

Both output directories must be new, with existing parents. The input defaults
to the final four-name overlay above. Use an unmodified four-word Ripple
overlay with distinct names, not a labeled SVG, diagnostic or print sheet.
The strict artwork reader rejects unsupported transforms/CSS and retains
the original center/orientation; this does not use or require Blender.

The generator writes five A4 SVGs, A4 and Letter PDFs, and a measurement
manifest. It stages the entire pack before publishing the new directory;
missing/failed PDF export is an error, never a partial success.
PDF export requires **native Cairo** as well as Python CairoSVG and PyMuPDF.
On Debian/Ubuntu, the native library is provided by `libcairo2`.
`--svg-only` skips PDF export and its optional dependencies. CLI help also
works without them. Core audio dependencies remain unchanged.

To select fewer sizes or one wire diameter:

```bash
python3 src/print_templates.py --output-dir artwork/selected-print-pack \
  --sizes 45 50 55 60 65 70 --wire-mm 1.0
python3 src/check_print_templates.py artwork/selected-print-pack --require-pdf
```

One wire diameter produces three pages; two produce five, in the supplied
wire order. Body sizes must be whole millimeters between 20 and 80, at least
5 mm apart, with at most ten sizes. The compact layout rejects combinations
that cannot fit at 1:1 instead of quietly shrinking them; very broad or
off-center source curves may require fewer/smaller sizes.

The checker verifies source hashes, all source cubic controls, common scale,
wire diameters, size labels, measurements, page order, calibration gauges and
PDF vector coordinates/print margins. `--require-pdf` ensures SVG-only output
cannot count as a successful PDF check. `--preview-dir` renders both PDF packs
to PNGs for visual inspection; **print the PDFs, not these inspection images**.
Numerical checks cannot validate your printer, wire spring-back or joint design.

## Earlier 45-70 mm packs: 1.0 mm wire only (archive)

Start with the **one-page, actual-size comparison** showing all six options:

- [A4 comparison PDF](../output_samples/final-names-print-1mm-sizes/comparison-A4.pdf)
- [US Letter comparison PDF](../output_samples/final-names-print-1mm-sizes/comparison-Letter.pdf)
- [Editable comparison SVG](../output_samples/final-names-print-1mm-sizes/size-comparison.svg)

Once you select a size, the **six-page forming-guide pack** has one page per
size, in ascending order, with all four individual name curves and their
projected wire lengths:

- [A4 forming guides](../output_samples/final-names-print-1mm-sizes/forming-guides-A4.pdf)
- [US Letter forming guides](../output_samples/final-names-print-1mm-sizes/forming-guides-Letter.pdf)

**Every drawing in these new packs is 1:1, and every wire band is exactly
1.0 mm wide.** The curves enlarge together, but wire diameter does not increase.
The dimensions below include the wire and exclude the bail, solder fillets,
chain and other hardware. As the design is taller than wide, the stated
maximum dimension is its outside body height.

| Maximum body size | Outside width x height | Total projected wire, before any allowance | Individual SVG |
|---|---|---:|---|
| 45 mm | 41.94 x 45.00 mm | 472.4 mm | [45 mm guides](../output_samples/final-names-print-1mm-sizes/forming-45mm.svg) |
| 50 mm | 46.60 x 50.00 mm | 526.0 mm | [50 mm guides](../output_samples/final-names-print-1mm-sizes/forming-50mm.svg) |
| 55 mm | 51.25 x 55.00 mm | 579.7 mm | [55 mm guides](../output_samples/final-names-print-1mm-sizes/forming-55mm.svg) |
| 60 mm | 55.90 x 60.00 mm | 633.4 mm | [60 mm guides](../output_samples/final-names-print-1mm-sizes/forming-60mm.svg) |
| 65 mm | 60.56 x 65.00 mm | 687.1 mm | [65 mm guides](../output_samples/final-names-print-1mm-sizes/forming-65mm.svg) |
| 70 mm | 65.21 x 70.00 mm | 740.7 mm | [70 mm guides](../output_samples/final-names-print-1mm-sizes/forming-70mm.svg) |

These are nominal mathematical dimensions, not fabrication tolerances.
Lengths exclude all closure, over/under, trimming and bail allowances:
**do not reuse the earlier 120 mm-per-loop or 0.6 m stock suggestions for
these larger sizes**. Use the individual lengths on the selected sheet and
add spare appropriate to your joint construction and practice needs.

All original curves, recording orientations and their shared scale remain
intact. The formula is `(maximum body size - 1 mm) / longest combined
source-centerline span`. Full measurements and the source hash are in
[`dimensions.json`](../output_samples/final-names-print-1mm-sizes/dimensions.json).
Do not mix individual forming guides from different size pages.

Increasing size gives more open space relative to the same 1 mm wire, but
does not remove the 22 crossings. Longer unsupported spans can also be more
flexible; bigger is not automatically easier to build or stronger. Compare
the silhouettes on paper and prototype the selected size before committing
to silver. The soldered-joint limitations below still apply.

## Print settings for every pack

Use the matching paper size, **Actual size / 100%, one page per sheet**.
Disable Fit, Shrink and borderless enlargement. Measure the **50 mm line**
between end-tick centers and the **10 x 10 mm square** between line centers
in both axes before forming wire. These calibration marks are on every page.

## Earlier 40 mm drafts (archive)

- [A4 print pack](../output_samples/final-names-print/print-A4.pdf)
- [US Letter print pack](../output_samples/final-names-print/print-Letter.pdf)

Both PDFs contain the same geometry at the same physical size:

1. **0.9 mm wire**, 40 mm maximum outside body size, with four individual
   forming guides and a color reference.
2. **1.0 mm wire**, the same maximum size and guide layout.
3. **35 / 40 / 45 mm size comparison** at 1.0 mm wire, plus a side-by-side
   0.9 / 1.0 mm comparison at 40 mm.

The editable A4 sheets are also available as
[0.9 mm SVG](../output_samples/final-names-print/40mm-wire-0.9mm.svg),
[1.0 mm SVG](../output_samples/final-names-print/40mm-wire-1.0mm.svg), and
[size-comparison SVG](../output_samples/final-names-print/size-comparison.svg).
Prefer the PDFs for predictable printing. These sheets contain annotations;
they are **not inputs for the pendant renderer**.

Download the PDF, open it in a PDF reader, and print **Actual size / 100%,
one page per sheet** on the matching paper size. Disable Fit, Shrink, multiple
pages per sheet, and borderless enlargement. Do not print a browser screenshot
or the perspective pendant rendering.

Before using a sheet, measure its **50 mm calibration line between the centers
of the end ticks**, and the **10 x 10 mm square between the centers of its
opposite lines**. Check both axes. If these do not measure correctly, fix the
print settings before forming wire. The files encode physical dimensions, but
printer, paper and measurement tolerances remain. PDF viewer preferences also
request no scaling; the printer's settings still take precedence.

The upper-left assembly and all four individual forming guides are **1:1**.
The upper-right colored view is deliberately **2:1, for identification only**.
The entire comparison page is 1:1. Screen display size is not a ruler.

## What "40 mm" means

This design is taller than it is wide. Here, 40 mm means the **maximum of the
overall outside width and height, including the wire's diameter**, rather than
the SVG canvas or original gray nominal circle. It excludes the bail,
attachment rings, chain, solder fillets and any fabrication changes.

| Wire diameter | Outside body width | Outside body height | Shared nominal circle diameter |
|---|---:|---:|---:|
| 0.9 mm | 37.285 mm | 40.000 mm | 32.309 mm |
| 1.0 mm | 37.292 mm | 40.000 mm | 32.226 mm |

These are nominal mathematical dimensions, **not achievable manufacturing
tolerances**. In practice, think **37.3 x 40 mm**, plus the attachment hardware.
If you instead wanted 40 mm *wide*, the unchanged design would be about
42.9 mm high before adding the bail.

Every source cubic and its orientation is preserved. One uniform scale is
shared by all four names; there is no individual resizing, simplification,
rotation, or movement to create larger gaps. The two wire versions have
slightly different centerline scales so that their different wire radii still
give exactly the same 40 mm maximum outside size. Do not mix the individual
guides from the two versions.

The conversion is:

```text
scale = (40 mm - wire diameter) / longest combined source-centerline span
printed point = (source point - shared source center) * scale + sheet location
```

The original gray reference circle is omitted; it is not another wire.
The small crosses on individual guides mark the **same source center** for
alignment. Center the wire on the **dashed line**; the gray band represents its
full diameter, not an instruction to bend along one edge.
Exact scale factors, source hash and lengths are recorded in
[`dimensions.json`](../output_samples/final-names-print/dimensions.json).

## Will 1 mm at 4 cm be crowded or too open?

**Not generally crowded, but locally crowded.** At 40 mm / 1 mm, an approximately
23.6 mm diameter circle fits in the central opening around the original shared
center. That leaves substantial negative space. The sampled tightest centerline
bend radius is about 4.2 mm, so the main curves are broad bends rather than
tiny filigree. Neither measurement proves that the finished piece is easy to
make or robust.

The close-running top and bottom sections are the difficult areas. Some gaps
visually close at the requested wire thickness, and the original curves have
**22 distinct centerline crossings**. The silhouettes show that crowding
honestly rather than using hairline strokes that would exaggerate the gaps.
The colored enlargements help identify the strands, not promise separation.

- **35 mm:** smaller and more delicate in overall footprint, but less room
  for the same 1 mm wire and for working at adjacent joints.
- **40 mm:** my preferred starting compromise between manageable curves,
  visible detail and a modest openwork pendant.
- **45 mm:** more room and clearer separation in some near-parallel sections,
  but a noticeably larger piece. It does not remove any crossings.

**1.0 mm** is my first choice for a soldered prototype because it offers more
stiffness than 0.9 mm of the same alloy and temper. **0.9 mm** is a useful
lighter-looking alternative and is easier to bend. The visual difference is
subtle; stiffness changes more strongly with diameter. For an ideal round
wire in elastic bending, rigidity scales with diameter to the fourth power:
1.0 mm is about 52% stiffer than 0.9 mm, all else equal. That calculation is
not a strength rating for this finished pendant.

Use the millimeter diameter when buying wire rather than treating gauge
systems as interchangeable. These drafts assume **round** wire; flattened or
hammered wire has a different visible width. Sterling is the starting material
assumption, not softer fine silver. Soft/annealed stock is easier to shape;
do not assume that a half-hard purchase will retain its temper through
soldering. Discuss final stiffness and finishing with the maker.

## Soldered construction: what the drawing does not solve

The gold image fused overlapping volumes in software. **Two solid round
wires cannot occupy the same plane at a crossing.** A real build must resolve
that with a chosen over/under or other joint construction. A simple stacked
crossing is roughly two wire diameters thick; more layers can be thicker.
These sheets do not assign an over/under order, notch depths, solder locations,
closure positions, or a load-bearing bail.

The 22 crossings are **not instructions to make 22 solder joints**. Decide
which connections support the assembly, how each closed loop's ends join,
and how the bail transfers load before cutting silver to final length.
Near-parallel runs can turn into wider soldered bands and lose visible
separation. Do not automatically notch thin wire at every crossing or assume
the photorealistic image validated those joints.

Form and compare the four loops in inexpensive wire first. Align their
center marks and original orientation, then inspect the assembled prototype
at actual size. This is the best practical check of whether 40 mm feels right.
Copper is useful for a low-cost trial, although its temper and spring-back
will not necessarily match your silver.

| Name | 0.9 mm version: projected centerline length | 1.0 mm version: projected centerline length |
|---|---:|---:|
| Denisa | 103.6 mm | 103.3 mm |
| Eliska | 105.6 mm | 105.3 mm |
| Jiri | 103.8 mm | 103.6 mm |
| Jitka | 106.8 mm | 106.5 mm |
| **Total** | **419.7 mm** | **418.7 mm** |

These are planar curve lengths, **not final cut lengths**: they exclude every
joint, over/under adjustment, bail and trimming allowance. As a starting
allocation, about **120 mm per loop** gives room for fitting; **0.6 m of wire**
provides some additional spare for one trial, but is not a guaranteed material
list for an undecided joint design. Use more for practice or if your closure
method needs it.

Paper is only a cold-forming/checking guide. Move the work to an appropriate
jewelry soldering setup for joining; do not heat the printed sheet. A
silversmith should assess the joints, attachment, final hardness, sharp edges
and durability before treating it as finished wearable jewelry.

## Online guidance behind the recommendation

Sources checked on 24 November 2026:

- [Kernowcraft: compare jewelry made with different wire sizes](https://www.kernowcraft.com/blogs/how-to-guides/compare-jewellery-designs-using-different-sized-wire)
  shows a structured pendant using 0.7 mm wire and recommends 1 mm for
  pendants, including a geometric example. This supports 0.9-1.0 mm as a
  plausible starting range, not a certification of this four-loop design.
- [Cooksongold / Janet Dwyer: wire-wrapped pendant and earrings](https://www.cooksongold.com/blog/project/how-to-make-a-wire-wrapped-pendant-and-earrings/)
  uses 1 mm wire for a pendant frame. It is a different, wire-wrapped project,
  so it supports the material scale, not our solder-joint construction.
- [Kernowcraft: choosing jewelry wire](https://www.kernowcraft.com/blogs/how-to-guides/working-with-wire-top-tips-techniques-inspiration)
  recommends inexpensive copper for practice, distinguishes softer fine silver,
  and notes that stronger standalone structural frames often use thicker
  1.5-2 mm wire. Our openwork network needs well-designed connections; 1 mm
  is not automatically sufficient for every unsupported frame.
- [Adobe: printing PDF pages at Actual size](https://helpx.adobe.com/acrobat/desktop/print-documents/set-up-and-print-pdfs/page-size.html)
  distinguishes unscaled printing from Fit and custom scaling.

No source establishes an ideal diameter for this particular recording-derived
design. The 40 mm recommendation combines those examples with the measured
geometry and the printable comparisons, and still needs a physical prototype.
