"""The height of agri-ugv-sim's terrain mesh, for things that lie just above the ground."""

import numpy as np


def mesh_height(grid, x, y):
    """
    Return the height of the terrain mesh itself at points (x, y).

    grid: (x axis, y axis, heights[rows, cols]) as agri_ugv_field.ground.read_obj_grid reads
    it. The mesh (agri_ugv_terrain.heightfield) splits every cell into two flat triangles
    along its diagonal from the south-west to the north-east corner. Where the four corners
    of a cell do not lie in one plane, the bilinear agri_ugv_field.ground.ground_height
    differs from it inside the cell, so things that must lie a few millimetres above the
    visible ground use this. Points outside the grid get the height of the nearest edge.
    """
    xs, ys, heights = grid
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    fx = np.clip(np.interp(x, xs, np.arange(len(xs))), 0, len(xs) - 1)
    fy = np.clip(np.interp(y, ys, np.arange(len(ys))), 0, len(ys) - 1)
    j, i = np.minimum(fx.astype(int), len(xs) - 2), np.minimum(fy.astype(int), len(ys) - 2)
    tx, ty = fx - j, fy - i
    a, b = heights[i, j], heights[i, j + 1]
    c, d = heights[i + 1, j + 1], heights[i + 1, j]
    return np.where(tx >= ty, a + tx * (b - a) + ty * (c - b), a + ty * (d - a) + tx * (c - d))
