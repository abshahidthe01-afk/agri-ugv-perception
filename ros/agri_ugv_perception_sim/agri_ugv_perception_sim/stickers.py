"""
Weeds from real field pictures, laid flat on the soil of chosen plots as textured stickers.

The generated crops are 3D plants drawn by code. Weeds instead are cut out of real field
pictures (make_cutouts): each one is a small picture, transparent around the plant, at a
known size per pixel. make_field_extras packs them into one texture (atlas) and lays copies of
them in a plot at random places, turned and mirrored at random and not overlapping
(place). Each sticker is a small grid of flat faces that follows the terrain mesh a few
millimetres above it (sticker_mesh), above the soil sheet (soil.py) where there is one.
Being flat, they are seen by the cameras but not by the laser scanners.
"""

import math

from agri_ugv_field.plants import merge, plot_frame
import numpy as np

LIFT = 0.008            # [m] above the terrain mesh: 4 mm above the soil sheet (soil.py)


def atlas(pictures, width=2048, pad=4):
    """
    Pack RGBA pictures (H, W, 4) into one texture; return it and each picture's box.

    The pictures are laid in rows (tallest first), 'pad' pixels apart; the padding repeats
    each picture's edge colour (transparent), so that the pictures' colours do not bleed
    into each other where the texture is shown smaller. The texture is 'width' wide and as
    high as needed, rounded up to a power of two. A box is (column, row, width, height) of
    the picture in the texture.
    """
    pictures = [np.asarray(p, dtype=np.uint8) for p in pictures]
    if any(p.ndim != 3 or p.shape[2] != 4 for p in pictures):
        raise ValueError('the pictures must be RGBA (height, width, 4)')
    if any(p.shape[1] + 2 * pad > width for p in pictures):
        raise ValueError(f'a picture is wider than the texture ({width} px)')
    boxes, places = [None] * len(pictures), []
    x = y = row_height = 0
    for k in sorted(range(len(pictures)), key=lambda k: -pictures[k].shape[0]):
        h, w = pictures[k].shape[:2]
        if x + w + 2 * pad > width:
            x, y, row_height = 0, y + row_height, 0
        places.append((k, x, y))
        boxes[k] = (x + pad, y + pad, w, h)
        x, row_height = x + w + 2 * pad, max(row_height, h + 2 * pad)
    height = 2 ** int(math.ceil(math.log2(max(y + row_height, 1))))
    texture = np.zeros((height, width, 4), np.uint8)
    for k, x, y in places:
        p = pictures[k]
        h, w = p.shape[:2]
        texture[y:y + h + 2 * pad, x:x + w + 2 * pad, :3] = np.pad(
            p[:, :, :3], ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        texture[y + pad:y + pad + h, x + pad:x + pad + w, 3] = p[:, :, 3]
    return texture, boxes


def box_uvs(box, shape):
    """Return (u_left, u_right, v_bottom, v_top) of a box (atlas) in a texture of 'shape'."""
    column, row, w, h = box
    rows, cols = shape[:2]
    return column / cols, (column + w) / cols, 1.0 - (row + h) / rows, 1.0 - row / rows


def place(plot, sizes, count, rng, tries=50):
    """
    Return up to 'count' stickers in a plot: (x, y, yaw, mirrored, index) each.

    sizes: (width, height) in metres of each picture; index says which one. Every sticker
    lies wholly inside the plot's rectangle and clear of the others (their circles around
    the pictures do not overlap). A sticker that finds no free place in 'tries' draws is
    left out.
    """
    centre, along, across = plot_frame(plot)
    radii = 0.5 * np.hypot(*np.asarray(sizes, dtype=float).T)
    placed, xy, reach = [], np.zeros((0, 2)), np.zeros(0)
    for _ in range(count):
        index = int(rng.integers(len(sizes)))
        r = radii[index]
        half_length, half_width = plot['length'] / 2 - r, plot['width'] / 2 - r
        if half_length <= 0 or half_width <= 0:
            continue
        for _ in range(tries):
            a, b = rng.uniform(-half_length, half_length), rng.uniform(-half_width, half_width)
            point = centre + a * along + b * across
            if np.all(np.hypot(*(xy - point).T) >= reach + r):
                break
        else:
            continue
        yaw, mirrored = rng.uniform(0, 2 * math.pi), bool(rng.random() < 0.5)
        placed.append((float(point[0]), float(point[1]), yaw, mirrored, index))
        xy, reach = np.vstack([xy, point]), np.r_[reach, r]
    return placed


def sticker_mesh(placed, sizes, uvs, height, lift=LIFT, cells=4):
    """
    Return (vertices, uvs, faces) of stickers placed by place(), in world metres.

    sizes: (width, height) in metres per picture; uvs: its box_uvs in the texture;
    height(x, y): the terrain mesh's height (ground.mesh_height). Each sticker is a grid of
    cells x cells squares whose corners lie 'lift' above the terrain, so it follows the
    ground's bends closely. The picture's top edge points along yaw + 90 degrees; mirrored
    stickers show it flipped left to right. The faces point up.
    """
    s = np.linspace(0.0, 1.0, cells + 1)
    ss, tt = np.meshgrid(s, s)                        # rows of the grid: t, columns: s
    ss, tt = ss.ravel(), tt.ravel()
    index = np.arange((cells + 1) ** 2).reshape(cells + 1, cells + 1)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, 1:].ravel(), index[1:, :-1].ravel()
    one = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    parts = []
    for x, y, yaw, mirrored, k in placed:
        w, h = sizes[k]
        u_left, u_right, v_bottom, v_top = uvs[k]
        local_x, local_y = (ss - 0.5) * w, (tt - 0.5) * h
        cos, sin = math.cos(yaw), math.sin(yaw)
        px, py = x + cos * local_x - sin * local_y, y + sin * local_x + cos * local_y
        u = u_right - ss * (u_right - u_left) if mirrored else u_left + ss * (u_right - u_left)
        v = v_bottom + tt * (v_top - v_bottom)
        parts.append((np.column_stack([px, py, height(px, py) + lift]),
                      np.column_stack([u, v]), one))
    return merge(parts)
