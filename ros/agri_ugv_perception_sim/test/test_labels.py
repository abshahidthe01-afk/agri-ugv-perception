"""Tests for the labels of a picture from true surface points and its depth image."""

from agri_ugv_perception_sim.labels import closing, CROP, depth_points, IGNORE, \
    label_picture, seen, SOIL, WEED
import numpy as np
import pytest

H, FX, CX, CY = 1.5, 500.0, 50.0, 40.0
K = [FX, 0.0, CX, 0.0, FX, CY, 0.0, 0.0, 1.0]


def looking_down(height=H):
    """Return the optical link (4 x 4) of a camera 'height' above the robot, looking down."""
    mount = np.eye(4)
    mount[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]     # view down, picture's top = +x
    mount[:3, 3] = [0.0, 0.0, height]
    return mount


def square(x0, x1, y0, y1, z, step=0.002):
    """Return points step apart on a horizontal square."""
    xx, yy = np.mgrid[x0:x1:step, y0:y1:step]
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])


def scene(shape=(80, 100)):
    """
    Return the depth image of a small scene under the camera.

    The ground at z 0, a leaf 0.1 m wide at z 0.2 over the robot, a weed on the ground
    beside it; the robot's own parts in the top 5 rows, and no depth in the next 3.
    """
    v, u = np.mgrid[0:shape[0], 0:shape[1]].astype(float)

    def hit(z):                                          # where each pixel's ray meets z
        d = H - z
        return d, (CY - v) * d / FX, (CX - u) * d / FX

    depth = np.full(shape, H)
    d, x, y = hit(0.005)
    weed = (x >= 0.08) & (x < 0.12) & (y >= 0.0) & (y < 0.04)
    depth[weed] = d
    d, x, y = hit(0.2)
    leaf = (np.abs(x) < 0.05) & (np.abs(y) < 0.05)
    depth[leaf] = d
    depth[:5] = 0.5                                      # 1 m above the ground: the robot
    depth[5:8] = 0.0                                     # nothing seen
    return depth


def test_depth_points_and_projection_agree():
    depth = scene()
    points = depth_points(depth, looking_down(), K)
    assert np.isnan(points[6, 10]).all()                 # no depth
    assert points[40, 50] == pytest.approx([0.0, 0.0, 0.2])     # the picture's centre: leaf
    rows, cols, misfit = seen(points[20:].reshape(-1, 3), depth, looking_down(), K)
    assert misfit.max() < 1e-9 and len(rows) == 60 * 100 - np.sum(depth[20:] == 0)


def test_seen_keeps_only_points_on_the_visible_surface():
    depth = np.full((80, 100), H)                        # bare ground
    points = np.array([[0.0, 0.0, 0.004],                # on the ground: seen
                       [0.0, 0.01, 0.1],                 # in the air above it: no
                       [5.0, 0.0, 0.0],                  # outside the picture: no
                       [0.0, 0.0, 2.0]])                 # behind the camera: no
    rows, cols, misfit = seen(points, depth, looking_down(), K)
    assert list(zip(rows, cols)) == [(40, 50)] and misfit[0] == pytest.approx(0.004)


def test_closing_fills_holes_and_keeps_the_edge_of_the_picture():
    mask = np.zeros((7, 7), bool)
    mask[1:5, 1:5] = True
    mask[2, 2] = False                                   # a hole
    mask[0, 6] = True                                    # a single pixel at the edge
    closed = closing(mask)
    assert closed[2, 2] and closed[0, 6]
    assert not closed[6, 0] and closed.sum() == 17


def test_labels_follow_the_visible_surfaces():
    depth = scene()
    leaf = square(-0.05, 0.05, -0.05, 0.05, 0.2)
    weeds = np.vstack([square(0.08, 0.12, 0.0, 0.04, 0.005),
                       square(-0.02, 0.02, -0.02, 0.02, 0.005)])  # the second under the leaf
    labels = label_picture(depth, looking_down(), K, {CROP: leaf, WEED: weeds})
    assert set(np.unique(labels)) == {SOIL, CROP, WEED, IGNORE}
    assert np.all(labels[:8] == IGNORE) and not np.any(labels[8:] == IGNORE)
    rows = slice(8, None)
    for label, d in ((CROP, H - 0.2), (WEED, H - 0.005)):
        truth = np.isclose(depth[rows], d)
        found = labels[rows] == label
        assert np.sum(found != truth) <= 0.1 * truth.sum()      # at most the outline differs
    assert labels[40, 50] == CROP                        # the weed under the leaf is hidden
