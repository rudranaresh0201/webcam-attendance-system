"""Local Binary Patterns, written out in NumPy so every step is visible.

    circular LBP(P, R)  ->  uniform mapping (59 bins for P=8)  ->  spatial grid
    of cell histograms  ->  (weighted) chi-square distance

Also Local Ternary Patterns (Tan & Triggs 2010), which split each comparison
into three states with a tolerance t so that small noise around the centre
value no longer flips bits.

A descriptor is an array of shape (C, B): C cells (grid cells x scales x code
planes) each holding a normalised B-bin histogram. Keeping the cell axis
separate is what allows per-cell weights and the per-cell "where did the faces
differ" heatmap in the UI.
"""
from functools import lru_cache

import numpy as np


# --------------------------------------------------------------------------- codes
def _sample(img, dy, dx):
    """Bilinear sample of img shifted by (dy, dx) (sub-pixel), same shape as img.
    Border pixels are replicated."""
    h, w = img.shape
    y0, x0 = int(np.floor(dy)), int(np.floor(dx))
    fy, fx = dy - y0, dx - x0
    pad = int(np.ceil(max(abs(dy), abs(dx)))) + 1
    p = np.pad(img, pad, mode="edge")

    def at(oy, ox):
        return p[pad + oy:pad + oy + h, pad + ox:pad + ox + w]

    return ((1 - fy) * (1 - fx) * at(y0, x0) + (1 - fy) * fx * at(y0, x0 + 1)
            + fy * (1 - fx) * at(y0 + 1, x0) + fy * fx * at(y0 + 1, x0 + 1))


def _neighbours(gray, points, radius):
    """The P circular neighbours of every pixel, shape (P, H, W)."""
    g = gray.astype(np.float32)
    out = []
    for p in range(points):
        angle = 2 * np.pi * p / points
        dy, dx = -radius * np.sin(angle), radius * np.cos(angle)
        # snap tiny float error so axis-aligned neighbours are exact (no interpolation blur)
        dy, dx = round(dy, 6), round(dx, 6)
        out.append(_sample(g, dy, dx))
    return np.stack(out), g


def lbp_codes(gray, points=8, radius=1):
    """LBP code per pixel: bit p is 1 when neighbour p >= centre."""
    nb, c = _neighbours(gray, points, radius)
    weights = (1 << np.arange(points, dtype=np.int64))[:, None, None]
    return ((nb >= c - 1e-3).astype(np.int64) * weights).sum(0)


def ltp_codes(gray, points=8, radius=1, t=5.0):
    """Local Ternary Pattern split into its two binary halves (upper, lower):
    upper bit = neighbour >= centre + t, lower bit = neighbour <= centre - t."""
    nb, c = _neighbours(gray, points, radius)
    weights = (1 << np.arange(points, dtype=np.int64))[:, None, None]
    upper = ((nb >= c + t).astype(np.int64) * weights).sum(0)
    lower = ((nb <= c - t).astype(np.int64) * weights).sum(0)
    return upper, lower


@lru_cache(maxsize=None)
def uniform_table(points=8):
    """Maps every P-bit code to a bin. Codes with at most 2 circular 0/1
    transitions ("uniform": spots, edges, corners, line ends) each get their own
    bin; all others share the last bin. P=8 -> 58 uniform + 1 = 59 bins.
    Ojala et al. found uniform patterns cover ~90% of texture at (8,1)."""
    table = np.zeros(1 << points, dtype=np.int32)
    nxt = 0
    nonuniform = []
    for code in range(1 << points):
        bits = [(code >> i) & 1 for i in range(points)]
        transitions = sum(bits[i] != bits[(i + 1) % points] for i in range(points))
        if transitions <= 2:
            table[code] = nxt
            nxt += 1
        else:
            nonuniform.append(code)
    table[nonuniform] = nxt
    return table


def n_bins(points=8, uniform=True):
    return int(uniform_table(points).max()) + 1 if uniform else 1 << points


# --------------------------------------------------------------------------- histograms
@lru_cache(maxsize=32)
def _cell_index(h, w, gx, gy):
    ys = np.minimum(np.arange(h) * gy // h, gy - 1)
    xs = np.minimum(np.arange(w) * gx // w, gx - 1)
    return (ys[:, None] * gx + xs[None, :]).astype(np.int64)


def spatial_histogram(codes, bins, grid=(8, 8)):
    """(gx*gy, bins) array: a normalised histogram of codes in each grid cell."""
    gx, gy = grid
    cell = _cell_index(codes.shape[0], codes.shape[1], gx, gy)
    flat = np.bincount((cell * bins + codes).ravel(), minlength=gx * gy * bins)
    hist = flat.reshape(gx * gy, bins).astype(np.float32)
    hist /= hist.sum(1, keepdims=True) + 1e-9
    return hist


class LBPDescriptor:
    """Configurable multi-scale (uniform) LBP or LTP descriptor."""

    def __init__(self, scales=((8, 1),), grid=(8, 8), uniform=True, ternary=False, ltp_t=5.0):
        self.scales = tuple(tuple(s) for s in scales)
        self.grid = tuple(grid)
        self.uniform = uniform
        self.ternary = ternary
        self.ltp_t = ltp_t

    @property
    def signature(self):
        kind = "ltp" if self.ternary else ("ulbp" if self.uniform else "lbp")
        sc = "-".join(f"{p}.{r}" for p, r in self.scales)
        return f"{kind}_{sc}_g{self.grid[0]}x{self.grid[1]}"

    @property
    def n_cells(self):
        return len(self.scales) * self.grid[0] * self.grid[1] * (2 if self.ternary else 1)

    def _hist(self, codes, points):
        if self.uniform:
            codes = uniform_table(points)[codes]
        return spatial_histogram(codes, n_bins(points, self.uniform), self.grid)

    def __call__(self, chip):
        """chip: 2-D uint8 face -> (C, B) float32 descriptor."""
        blocks = []
        for points, radius in self.scales:
            if self.ternary:
                for codes in ltp_codes(chip, points, radius, self.ltp_t):
                    blocks.append(self._hist(codes, points))
            else:
                blocks.append(self._hist(lbp_codes(chip, points, radius), points))
        width = max(b.shape[1] for b in blocks)
        blocks = [np.pad(b, ((0, 0), (0, width - b.shape[1]))) for b in blocks]
        return np.concatenate(blocks).astype(np.float32)


# --------------------------------------------------------------------------- distances
def chi2_cells(query, gallery):
    """Per-cell chi-square distance from one (C, B) query to each of N (N, C, B)
    gallery descriptors -> (N, C). Each cell histogram sums to 1, so each cell
    distance lies in [0, 2]."""
    num = (gallery - query[None]) ** 2
    den = gallery + query[None] + 1e-9
    return (num / den).sum(-1)


def weighted_distance(cell_d, weights):
    """Weighted mean over cells: (..., C) -> (...). Weights need not sum to 1."""
    return cell_d @ (weights / weights.sum())


def learn_cell_weights(features, labels, pairs=4000, seed=0):
    """Data-driven version of Ahonen's hand-set region weights.

    For each cell, compare the chi-square distances of same-person pairs with
    those of different-person pairs and score the cell by the Fisher criterion
        J = (mu_between - mu_within)^2 / (var_between + var_within).
    Cells where identities separate well (eyes, brows, nose bridge) get large
    weights; cells dominated by hair, background or pose (corners, cheeks
    edges) get small ones. Weights are normalised to mean 1.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    by_label = {l: idx[labels == l] for l in np.unique(labels)}
    multi = [l for l, ids in by_label.items() if len(ids) >= 2]
    within, between = [], []
    for _ in range(pairs):
        l = multi[rng.integers(len(multi))]
        a, b = rng.choice(by_label[l], 2, replace=False)
        within.append(chi2_cells(features[a], features[b][None])[0])
        a, b = rng.choice(idx, 2, replace=False)
        while labels[a] == labels[b]:
            a, b = rng.choice(idx, 2, replace=False)
        between.append(chi2_cells(features[a], features[b][None])[0])
    w, b = np.array(within), np.array(between)
    j = np.maximum(b.mean(0) - w.mean(0), 0) ** 2 / (b.var(0) + w.var(0) + 1e-9)
    return (j / (j.mean() + 1e-12)).astype(np.float32)
