# SPOR — Scatterplot shape-Preserving Overlap Removal

**Grid-based overlap removal for dimensionality-reduction scatterplots with a fixed glyph size, explicit white-space control, and grid-aware adaptive sampling.**

SPOR takes a 2D projection produced by any dimensionality reduction (DR) technique (t-SNE, UMAP, PCA, LLE, …) and rearranges its points onto a uniform grid so that **no two glyphs overlap** — while treating the glyph size as a *hard constraint*. Instead of shrinking points or thumbnails when the projection is crowded (as prior grid methods do), SPOR keeps glyphs at the user-specified size and, when the visual space cannot accommodate all points, performs a **grid-aware adaptive sampling** that selects spatially representative instances. A **white-space ratio** parameter explicitly controls how much of the projection's original empty space (gaps and cluster separations) is preserved in the final layout.

> 📄 This repository contains the reference implementation and experimental scripts for the paper
> *"SPOR: Shape-preserving overlap removal for dimensionality reduction scatterplots by leveraging sampling and white-space control"* — currently under review.

---

## Key ideas

- **Readability first.** The glyph size `g` is never reduced. What you set is what is drawn — essential for image-centric analyses where thumbnails must remain legible.
- **Explicit white-space control.** The white-space ratio `w_s ∈ [0, 1]` decides how many of the grid's empty cells are preserved as "dummy" placeholders, keeping the gaps and boundaries that separate clusters visible in the overlap-free layout.
- **Adaptive sampling instead of shrinking.** When the number of points exceeds the remaining grid capacity, SPOR samples exactly the number of points that fit, spreading the budget evenly across occupied cells (uniform coverage) and picking, inside each cell, the point closest to the cell centroid (minimal displacement).
- **Optional class priorities.** Points of user-specified *priority labels* are retained before any sampling takes place — useful for task-driven emphasis on a class of interest.
- **Small, interpretable parameter set.** `(g, w_s, priorities)` is all an analyst needs to balance readability, coverage, and separation.
- **Structure preservation.** The final assignment reuses DGrid's recursive-bisection strategy, promoting small displacements, stable geometry, and preserved aspect ratio.

## How it works

SPOR proceeds in three phases (see Fig. 1 of the paper):

1. **Grid construction & dummy points (A, B, B2).** A uniform grid whose cell dimension equals the glyph size `g` is overlaid on the scatterplot's bounding box, fixing the capacity at `M = R × C` cells. Every empty cell becomes a *dummy-point candidate*; candidates are scored by local density (Gaussian-smoothed point counts) and by distance to the nearest data point (a bell-shaped profile that favors cells at cluster boundaries over deep voids or cells hugging cluster cores). The top `N_dum = ⌊w_s · U⌋` candidates are kept as dummies — they are the mechanism that preserves the original white space.
2. **Adaptive sampling of real points (C).** The dummy budget leaves a hard capacity of `M − N_dum` cells for real data. If the dataset exceeds it, SPOR samples exactly that many points with a fair-coverage policy: a base quota per occupied cell, residual budget going to denser cells first, and an in-cell selector (by default, the point closest to the cell centroid). Priority-labeled points, if any, are retained before sampling.
3. **Grid assignment (D, E).** Kept real points and dummies are jointly assigned to grid cells with DGrid's fast recursive bisection; dummies are then discarded. Because dummies occupy cells during assignment, real points cannot be pulled into gap regions — cluster separations survive in the final layout.

The overall complexity is `O(N (log N)²)`, or `O(N log N)` with axis pre-sorting.

## Installation

Clone the repository and install the dependencies



### ⚠️ Library-version notes

The rendering helper `scatterplot.py` relies on `matplotlib.cm.get_cmap`, which is **deprecated and scheduled for removal in Matplotlib 3.11** — so keep `matplotlib < 3.11`. In addition, `numba` and `numpy` must be a compatible pair (each numba release supports only a bounded range of numpy versions).

The pinned set in [`requirements.txt`](requirements.txt) is the environment used to develop the paper's experiments, re-verified end-to-end on **Python 3.11**:

| Package | Version |
| --- | --- |
| numpy | 2.2.5 |
| matplotlib | 3.10.1 |
| numba | 0.61.2 |
| scikit-learn | 1.6.1 |
| pandas *(figures script only)* | 2.2.3 |
| Pillow *(figures script only)* | 11.2.1 |



## Quick start

```python
import numpy as np
from SPOR import DGridAdaptive   # SPOR's implementation class
import scatterplot as sct

# y: (N, 2) coordinates from any DR technique (t-SNE, UMAP, ...)
y = np.load("my_projection.npy")
labels = np.load("my_labels.npy")          # optional, only needed for priorities

spor = DGridAdaptive(
    glyph_size=0.5,                        # grid-cell size == displayed glyph size (never shrunk)
    white_space_ratio=0.9,                 # keep 90% of the empty cells as white space
    sampling_strategy="gridfair",          # uniform-coverage sampling policy
)
coords = spor.fit_transform(y, labels=labels)   # (N_kept, 2) overlap-free coordinates

# Indices (into y) of the points that survived sampling, in output order:
kept = [p["id"] for p in spor.grid if not p["dummy"]]

# Render with fixed-size circles:
sct.circles(coords, glyph_width=0.5, glyph_height=0.5,
            label=labels[kept], cmap="Dark2")
sct.savefig("overlap_free.png", dpi=300)
```

> **Note.** When sampling occurs (`N` exceeds the remaining capacity), `fit_transform` returns coordinates only for the *kept* points. Always use `spor.grid` (as above) to map outputs back to the original rows — e.g., to fetch the matching thumbnails, labels, or metadata.

To guarantee a show-all, overlap-free layout without any sampling (possible whenever `N ≤ M`), use the automatic white-space policy:

```python
spor = DGridAdaptive(glyph_size=0.06, white_space_ratio="auto_overlap_free")
coords = spor.fit_transform(y)
```

## API

### `DGridAdaptive(...)`

| Parameter | Type / values | Default | Description |
| --- | --- | --- | --- |
| `glyph_size` | `float` | `1.0` | Grid-cell dimension in projection units; equals the displayed glyph size and is never reduced. Defines the visual capacity `M = R × C`. |
| `white_space_ratio` | `float` in `[0, 1]`, or `"auto_overlap_free"` | `1.0` | Fraction of the grid's empty cells preserved as white space (`N_dum = ⌊w_s · U⌋`). `1.0` keeps all original gaps; lower values release cells so more real points can be shown. `"auto_overlap_free"` computes the largest `w_s` that still places **all** `N` points without sampling (falls back to `0.0` when `N > M`). Values below the feasible minimum are auto-corrected with a warning. |
| `sampling_strategy` | `"gridfair"` \| `"gridfair_density"` | `"gridfair"` | Policy used when sampling is required. `gridfair` spreads the budget evenly across occupied cells (uniform spatial coverage — the policy evaluated in the paper); `gridfair_density` allocates the extra budget proportionally to each cell's point count. |
| `priority_labels` | `list[int]` or `None` | `None` | Labels whose points are retained before any sampling (requires `labels` in `fit_transform`). If priority points alone exceed capacity, sampling happens within the priority set only. |
| `type_search` | `2` \| `1` \| `0` | `2` | In-cell selector: `2` = points closest to the cell centroid (medoid-like, minimal displacement), `1` = first points found, `0` = random. |
| `random_state` | `int` or `None` | `None` | Accepted for API compatibility; the random in-cell selector (`type_search=0`) currently uses NumPy's global RNG. |
| `return_type` | `"coord"` \| `"index"` | `"coord"` | Reserved. `fit_transform` currently always returns coordinates; the kept indices are available through `spor.grid` (see below). |

### `fit_transform(y, labels=None) → np.ndarray`

Computes the overlap-free layout for `y` (an `(N, 2)` array of DR coordinates). Returns an `(N_kept, 2)` array with the cell-center coordinates of the retained points. `labels` (an `(N,)` array) is only required when using `priority_labels`.

### Attributes after fitting

- `spor.grid` — list of dicts for every assigned element (real and dummy) with keys `id`, `x`, `y`, `row`, `col`, `dummy` (real entries also carry `label`; dummy entries carry their scoring fields). Filter on `dummy == False` and read `id` to recover the original indices of the kept points (see Quick start).

### Rendering helpers (`scatterplot.py`)

`scatterplot.py` (adapted from F. V. Paulovich's [DGrid implementation](https://github.com/fpaulovich/dimensionality-reduction), MIT license) draws fixed-size glyphs at the returned coordinates: `circles`, `rectangles`, `starglyphs`, and `images` (for thumbnail mosaics), plus `title`, `savefig`, and `show`.

## Reproducing the paper figures

`paper_images.py` regenerates the layouts used in the paper's use case (points, raw thumbnail embeddings, DGrid baselines, and SPOR with/without priorities across `w_s ∈ {1.0, 0.9, 0.8}`). It expects:

- a pickle produced by the embedding-computation step with keys `X_embedded` (N×2), `y` (numeric labels), `class_map` (id → name) and `img_paths` (thumbnail paths relative to `data/`);
- the baseline `dgrid.py` from the [DGrid repository](https://github.com/fpaulovich/dimensionality-reduction) (folder `dgrid/`) on the Python path;
- the configuration constants at the top of the script (`PICKLE_NAME`, `GLYPH_SIZE`, `POINT_RADIUS`, `PRIORITY_NAMES`, …) adjusted to your data.

Outputs (PNG + PDF) are written to `outputs/grids/<embedding-name>/`.

## Evaluation summary

On 1,000 synthetic scatterplots (following the Sca²Gri evaluation protocol) and against DGrid and Sca²Gri, SPOR achieves the best overall Ranking Product across seven quality metrics — stress, trustworthiness, orthogonal ordering, aspect ratio, displacement, spread, and overlap — with the `w_s = 1.0` variant ranked first, while remaining competitive in runtime (and substantially faster than Sca²Gri in the `N < M` regime). The `auto_overlap_free` variant yields a median overlap of exactly 0. See Sec. 4 of the paper for the full study, including qualitative use cases on the Galaxy Zoo DECaLS image collection and the Brazilian Academic Genealogy dataset.

## Repository structure

```
├── SPOR.py            # SPOR implementation (class DGridAdaptive)
├── scatterplot.py     # Fixed-size glyph rendering helpers (circles, images, starglyphs, ...)
├── paper_images.py    # Script to regenerate the paper's use-case figures
├── requirements.txt   # Verified dependency versions
└── README.md
```



## License

`scatterplot.py` is adapted from code by Fernando V. Paulovich, released under the MIT license. See the file headers for details.
