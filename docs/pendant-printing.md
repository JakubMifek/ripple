# Full-size silver-wire drafts: the final four names

**Recommendation: start with a 40 mm maximum body dimension and 1.0 mm
round sterling-silver wire, first as an inexpensive-wire prototype.**
That is a reasonable starting size, not a guarantee of strength or an
assembly-ready jewelry specification.

## Download and print

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
