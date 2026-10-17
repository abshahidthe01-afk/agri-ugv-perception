"""
Soil with real detail under chosen plots: a textured sheet on the terrain, and its texture.

The terrain's own texture is a drone orthophoto at 6 cm per pixel. The dome cameras see the
ground at about 3 mm per pixel, so there it is a smooth brown blur, unlike real soil: a
crop/weed network trained on real field pictures takes most of it for plants. make_field_extras
therefore lays a thin sheet with a soil texture of about 1 mm per pixel on the terrain under
chosen plots (soil_sheet). The texture is made from real field pictures with soil masks by
image quilting (quilt, used by make_soil_texture).
"""

from agri_ugv_field.plants import plot_frame
import numpy as np


def soil_sheet(grid, plot, metres, margin=0.6, lift=0.004):
    """
    Return (vertices, uvs, faces) of a soil sheet under a plot and around it.

    The sheet covers the plot's rectangle grown by 'margin' on every side (as an axis-aligned
    box of whole terrain cells) and copies the terrain mesh there: the same grid points and
    the same two triangles per cell (agri_ugv_terrain.heightfield), 'lift' metres higher, so
    it lies parallel to the ground everywhere. The texture repeats every 'metres' in x and y.
    The faces point up (counter-clockwise seen from above).
    """
    xs, ys, heights = grid
    centre, along, across = plot_frame(plot)
    half_length, half_width = plot['length'] / 2 + margin, plot['width'] / 2 + margin
    corners = np.array([centre + a * half_length * along + b * half_width * across
                        for a in (-1, 1) for b in (-1, 1)])
    low, high = corners.min(axis=0), corners.max(axis=0)
    if high[0] < xs[0] or low[0] > xs[-1] or high[1] < ys[0] or low[1] > ys[-1]:
        raise ValueError(f'plot {plot["plot_id"]} lies outside the terrain')

    def cells(axis, a, b):
        first = int(np.clip(np.searchsorted(axis, a) - 1, 0, len(axis) - 2))
        last = int(np.clip(np.searchsorted(axis, b) + 1, first + 2, len(axis)))
        return slice(first, last)

    cols, rows = cells(xs, low[0], high[0]), cells(ys, low[1], high[1])
    xx, yy = np.meshgrid(xs[cols], ys[rows])
    vertices = np.column_stack([xx.ravel(), yy.ravel(), (heights[rows, cols] + lift).ravel()])
    index = np.arange(xx.size).reshape(xx.shape)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, 1:].ravel(), index[1:, :-1].ravel()
    faces = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return vertices, vertices[:, :2] / metres, faces


def soil_windows(labels, patch, margin=8, stride=16):
    """
    Return the top-left corners (row, col) of the patch x patch windows that are soil.

    labels: a picture's classes, 0 for soil. A window counts only if it and a band of
    'margin' pixels around it are all soil (no plant, weed or the edge of one).
    """
    other = np.pad((np.asarray(labels) != 0).astype(np.int64), ((1, 0), (1, 0)))
    integral = other.cumsum(axis=0).cumsum(axis=1)
    size = patch + 2 * margin
    rows, cols = np.asarray(labels).shape
    corners = []
    for r in range(0, rows - size + 1, stride):
        for c in range(0, cols - size + 1, stride):
            if (integral[r + size, c + size] - integral[r, c + size] -
                    integral[r + size, c] + integral[r, c]) == 0:
                corners.append((r + margin, c + margin))
    return corners


def even_colour(patches):
    """Give every patch (N, h, w, 3) the mean colour of all of them, keeping its detail."""
    patches = np.asarray(patches, dtype=np.float32)
    means = patches.mean(axis=(1, 2), keepdims=True)
    return patches - means + means.mean(axis=0, keepdims=True)


def seam(cost):
    """Return the column per row of the cheapest path from the top to the bottom of 'cost'."""
    rows, cols = cost.shape
    total = np.array(cost, dtype=float)
    for r in range(1, rows):
        left = np.r_[np.inf, total[r - 1, :-1]]
        right = np.r_[total[r - 1, 1:], np.inf]
        total[r] += np.minimum(np.minimum(left, total[r - 1]), right)
    path = np.empty(rows, dtype=int)
    path[-1] = int(np.argmin(total[-1]))
    for r in range(rows - 2, -1, -1):
        low = max(path[r + 1] - 1, 0)
        path[r] = low + int(np.argmin(total[r, low:path[r + 1] + 2]))
    return path


def quilt(patches, size, overlap, rng, tolerance=0.1):
    """
    Return a size x size x 3 texture laid from square patches (N, p, p, 3) that tiles.

    Image quilting (Efros & Freeman 2001) on a grid that wraps around: patches are laid row
    by row, 'overlap' pixels over their neighbours; each is chosen among those that differ
    least from what is laid already where they overlap (within 'tolerance' of the best), and
    the border between old and new follows the path of least difference through each
    overlap. The last patch of a row or column also overlaps the first one, so the texture's
    right edge continues at its left edge and its bottom edge at its top.
    """
    patches = np.asarray(patches, dtype=np.float32)
    patch = patches.shape[1]
    step = patch - overlap
    if overlap < 1 or step < 1 or size % step:
        raise ValueError(f'size {size} must be a multiple of patch - overlap = {step}')
    out = np.zeros((size, size, 3), np.float32)
    filled = np.zeros((size, size), bool)
    strips = {'left': (slice(None), slice(0, overlap)),
              'right': (slice(None), slice(step, patch)),
              'top': (slice(0, overlap), slice(None)),
              'bottom': (slice(step, patch), slice(None))}
    for i in range(size // step):
        for j in range(size // step):
            where = np.ix_((np.arange(patch) + i * step) % size,
                           (np.arange(patch) + j * step) % size)
            old, known = out[where], filled[where]
            if known.any():
                error = (((patches - old) ** 2).sum(axis=3) * known).sum(axis=(1, 2))
                new = patches[rng.choice(np.flatnonzero(error <= error.min() * (1 + tolerance)))]
            else:
                new = patches[rng.integers(len(patches))]
            take = np.ones((patch, patch), bool)
            cost = ((new - old) ** 2).sum(axis=2)
            for side, strip in strips.items():
                if not known[strip].all():
                    continue
                across = side in ('left', 'right')
                part = cost[strip] if across else cost[strip].T
                keep_old = np.zeros(part.shape, bool)
                for r, col in enumerate(seam(part)):
                    if side in ('left', 'top'):
                        keep_old[r, :col] = True         # the old patch lies before the seam
                    else:
                        keep_old[r, col + 1:] = True     # or after it
                take[strip] &= ~(keep_old if across else keep_old.T)
            out[where] = np.where(take[..., None], new, old)
            filled[where] = True
    return out
