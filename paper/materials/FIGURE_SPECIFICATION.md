# Figure and data specification

The deliverable is an author materials package for a computer-vision paper. It
uses fixed-width vector figures, explicit paired controls, seed-level data and
standalone source records. It is not the final conference submission or a claim
of scientific acceptance readiness.

## Visual and statistical contract

- Export each figure as a vector PDF and SVG, with a 300 dpi PNG preview.
- Use a 7-inch two-column canvas with text at least 8 pt at that insertion size.
  Keep serif text consistent, explicit axis units, a visible zero for differences,
  and a colorblind-safe palette with markers or text as a second encoding.
- Display AP multiplied by 100; an absolute difference of 0.01 AP is one AP point.
  All CSVs retain unrounded AP in [0, 1] or its signed difference.
- For matched mainline results, show all three paired seed differences and their
  mean and sample SD (ddof=1). A seed SD is not a confidence interval. Do not
  imply independent events or perform a significance test from aggregate AP alone.
- Keep 2021 selection separate from 2022/2023 historical-test reporting. An
  all-year mean is an equal-weight average of setting/year/seed metrics, not a
  pooled-pixel AP. Keep single-run historical chains separate and without error bars.
- Show negative attribution cells and account for all requested comparison
  cells. Do not restrict a figure to the scenario or seed with the largest gain.
- The protocol illustration is schematic. It does not depict real fire maps,
  qualitative predictions, a new inference router or a deployed system.

## Planned use

Use the protocol and primary paired-effect figures in the main paper. Keep the
X17-versus-X14 attribution visible in the main experimental analysis. The
scenario decomposition and symbolic cost accounting can appear in the main
paper or supplement depending on space. The historical T1 attribution figure
and full method catalogue belong to supplementary material.

No event-level predictions were regenerated. Qualitative forecast overlays,
event bootstrap intervals, new missingness severities, router replay and matched
external-method results remain outside this materials build. They cannot be
manufactured from aggregate AP files.

## Venue guidance checked

The official [CVPR 2026 author guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)
require the conference template and an eight-page main paper limit including
figures/tables, with references allowed beyond it. They also describe supplementary
figures, tables and code and require anonymity for review. The
[ICCV 2025 author guidelines](https://iccv.thecvf.com/Conferences/2025/AuthorGuidelines)
are the companion venue reference consulted for this package. The eventual
submission must use the instructions for its actual conference edition.

The figure dimensions, palette and SD convention above are our design decisions,
not purported CVPR/ICCV mandates. Author provenance is deliberately preserved
here; an anonymous review copy must be prepared separately. The package is
self-contained and does not require access to its original cluster paths for
reading figures or regenerating them from the included CSVs.
