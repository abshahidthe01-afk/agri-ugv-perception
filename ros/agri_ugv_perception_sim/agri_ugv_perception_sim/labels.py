"""
Soil, crop and weed labels for the robot's pictures, from the simulation's own geometry.

The field's plants and weeds are known exactly: agri_ugv_field grows the same crop plants
again from the same seed (grow), and the weed stickers are the meshes that make_field_extras
wrote. Their surfaces are sampled densely (leaf_points: only on the opaque parts of their
textures, as the cameras show them), moved into the robot's frame with its true pose, and
projected into each picture (agri_ugv_phenotyping.colour.project). A point labels its pixel
only where the picture's depth image agrees with the point's own depth, so hidden leaves and
weeds under leaves do not count; where points of both a crop and a weed fall on a pixel, the
one closest to the measured depth wins. Pixels that no point labels are soil, after a 3 x 3
closing has filled the gaps between the sample points inside each crop and weed. Pixels of
the robot itself (points more than ROBOT above its ground point) and pixels without depth
are left out of training and scoring (IGNORE).
"""

import numpy as np

SOIL, CROP, WEED, IGNORE = 0, 1, 2, 255
ROBOT = 0.45            # [m] points higher above base_footprint belong to the robot itself
TOLERANCE = 0.01        # [m] a true point is seen where the depth image agrees this well


def depth_points(depth, mount, k):
    """Return robot-frame points (H, W, 3) of a depth image in metres (NaN where none)."""
    h, w = depth.shape
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    v, u = np.mgrid[0:h, 0:w].astype(float)
    d = np.where(depth > 0, depth, np.nan)
    q = np.stack([d, (cx - u) * d / fx, (cy - v) * d / fy], axis=-1)   # optical link frame
    return q @ np.asarray(mount)[:3, :3].T + np.asarray(mount)[:3, 3]


def seen(points, depth, mount, k, tolerance=TOLERANCE):
    """
    Return (rows, columns, misfit) of the robot-frame points that a camera sees.

    A point is seen if it lies in front of the camera and inside the picture, and the depth
    image there agrees with its depth within 'tolerance'; misfit is that disagreement.
    """
    from agri_ugv_phenotyping.colour import project

    h, w = depth.shape
    if len(points) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    mount = np.asarray(mount)
    u, v, along = project(points, (mount[:3, :3], mount[:3, 3]), k)
    inside = (along > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    cols, rows, along = u[inside].astype(int), v[inside].astype(int), along[inside]
    misfit = np.abs(depth[rows, cols] - along)
    keep = (depth[rows, cols] > 0) & (misfit < tolerance)
    return rows[keep], cols[keep], misfit[keep]


def spread(mask, outside, combine):
    """Combine each pixel of a boolean mask with its 8 neighbours ('outside' beyond it)."""
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=outside)
    out = mask.copy()
    for dr in range(3):
        for dc in range(3):
            out = combine(out, padded[dr:dr + h, dc:dc + w])
    return out


def closing(mask):
    """
    Close a boolean mask with a 3 x 3 square (dilate, then erode); keep its own pixels.

    The mask is closed inside a border of one pixel, so that it neither grows nor shrinks
    along the picture's edge.
    """
    framed = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    closed = spread(spread(framed, False, np.logical_or), True, np.logical_and)
    return closed[1:-1, 1:-1] | mask


def label_picture(depth, mount, k, classes, tolerance=TOLERANCE, robot=ROBOT):
    """
    Return the labels (H, W, uint8) of one picture: SOIL, a class from 'classes', or IGNORE.

    classes maps a label (CROP, WEED) to the robot-frame points (N, 3) of its true surface;
    depth: the picture's depth image in metres (0 = nothing seen); mount: the camera's
    optical link in the robot frame (4 x 4); k: its calibration, row by row.
    """
    depth = np.asarray(depth, dtype=float)
    h, w = depth.shape
    best = np.full((h, w), np.inf)
    labels = np.full((h, w), SOIL, np.uint8)
    for label, points in classes.items():
        rows, cols, misfit = seen(points, depth, mount, k, tolerance)
        own = np.full((h, w), np.inf)
        np.minimum.at(own, (rows, cols), misfit)
        better = own < best
        labels[better], best[better] = label, own[better]
    for label in classes:                        # fill the gaps between sample points
        filled = closing(labels == label) & (labels == SOIL)
        labels[filled] = label
    height = depth_points(depth, mount, k)[:, :, 2]
    labels[~np.isfinite(height) | (height > robot)] = IGNORE
    return labels
