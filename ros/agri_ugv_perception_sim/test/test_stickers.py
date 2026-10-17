"""Tests for weed stickers: the texture of all pictures, their places and their meshes."""

import math

from agri_ugv_field.plants import plot_frame
from agri_ugv_perception_sim.stickers import atlas, box_uvs, place, sticker_mesh
import numpy as np
import pytest


def pictures(rng, shapes=((10, 6), (4, 12), (7, 7), (12, 3), (5, 20))):
    """Return RGBA pictures of random colours, opaque in their middle."""
    out = []
    for h, w in shapes:
        p = rng.integers(0, 256, (h, w, 4)).astype(np.uint8)
        p[:, :, 3] = 0
        p[1:-1, 1:-1, 3] = 255
        out.append(p)
    return out


def plot(length=4.0, width=2.0, heading_deg=30.0):
    return {'plot_id': 3, 'centre_x': 1.0, 'centre_y': -2.0, 'length': length, 'width': width,
            'heading_deg': heading_deg}


def test_atlas_holds_every_picture_apart_and_is_transparent_elsewhere():
    rng = np.random.default_rng(0)
    pics = pictures(rng)
    texture, boxes = atlas(pics, width=32, pad=2)
    assert texture.shape[1] == 32 and texture.shape[2] == 4
    assert texture.shape[0] in (8, 16, 32, 64)                       # a power of two
    used = np.zeros(texture.shape[:2], bool)
    for p, (x, y, w, h) in zip(pics, boxes):
        assert (w, h) == (p.shape[1], p.shape[0])
        assert np.array_equal(texture[y:y + h, x:x + w], p)
        grown = used[max(y - 2, 0):y + h + 2, max(x - 2, 0):x + w + 2]
        assert not grown.any()                                       # padding kept clear
        used[y:y + h, x:x + w] = True
    assert not texture[~used, 3].any()


def test_atlas_refuses_pictures_it_cannot_hold():
    with pytest.raises(ValueError, match='RGBA'):
        atlas([np.zeros((4, 4, 3), np.uint8)])
    with pytest.raises(ValueError, match='wider'):
        atlas([np.zeros((4, 40, 4), np.uint8)], width=32)


def test_box_uvs_count_v_upwards_from_the_bottom_of_the_texture():
    assert box_uvs((4, 2, 8, 4), (8, 16, 4)) == pytest.approx((0.25, 0.75, 0.25, 0.75))
    assert box_uvs((0, 0, 16, 8), (8, 16, 4)) == pytest.approx((0.0, 1.0, 0.0, 1.0))


def test_stickers_lie_inside_the_plot_and_clear_of_each_other():
    sizes = [(0.10, 0.06), (0.04, 0.04)]
    radii = 0.5 * np.hypot(*np.array(sizes).T)
    placed = place(plot(), sizes, 200, np.random.default_rng(1))
    assert len(placed) == 200
    assert placed == place(plot(), sizes, 200, np.random.default_rng(1))    # same seed
    centre, along, across = plot_frame(plot())
    xy = np.array([(x, y) for x, y, _, _, _ in placed])
    r = radii[[k for *_, k in placed]]
    a, b = (xy - centre) @ along, (xy - centre) @ across
    assert np.all(np.abs(a) + r <= 2.0 + 1e-9) and np.all(np.abs(b) + r <= 1.0 + 1e-9)
    gaps = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)) - (r[:, None] + r[None])
    assert np.all(gaps[~np.eye(len(xy), dtype=bool)] >= -1e-9)
    assert {k for *_, k in placed} == {0, 1}
    assert {m for _, _, _, m, _ in placed} == {True, False}
    assert all(0 <= yaw < 2 * math.pi for _, _, yaw, _, _ in placed)


def test_a_full_plot_takes_fewer_stickers_and_a_tiny_one_none():
    placed = place(plot(0.4, 0.4), [(0.1, 0.1)], 100, np.random.default_rng(2))
    assert 1 <= len(placed) < 100
    assert place(plot(0.05, 0.05), [(0.1, 0.1)], 10, np.random.default_rng(2)) == []


def test_sticker_mesh_lies_above_the_ground_and_shows_its_picture():
    def height(x, y):
        return 0.1 * x + 0.05 * y + 1.0

    sizes, uvs = [(0.2, 0.1)], [(0.25, 0.75, 0.25, 0.75)]
    placed = [(1.0, 2.0, math.pi / 2, False, 0), (0.0, 0.0, 0.0, True, 0)]
    vertices, uv, faces = sticker_mesh(placed, sizes, uvs, height, lift=0.008, cells=4)
    assert vertices.shape == (50, 3) and uv.shape == (50, 2) and faces.shape == (64, 3)
    assert vertices[:, 2] == pytest.approx(height(vertices[:, 0], vertices[:, 1]) + 0.008)
    v = vertices[faces]
    assert np.all(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 2] > 0)    # faces up
    first, second = slice(0, 25), slice(25, 50)
    # turned by 90 degrees: the picture's bottom-left corner (-0.1, -0.05) lands at +0.05, -0.1
    corner = np.flatnonzero(np.all(np.isclose(uv[first], [0.25, 0.25]), axis=1))
    assert vertices[first][corner[0], :2] == pytest.approx([1.05, 1.9])
    assert np.ptp(vertices[first, 0]) == pytest.approx(0.1)
    assert np.ptp(vertices[first, 1]) == pytest.approx(0.2)
    # mirrored: the same corner of the sticker shows the picture's right edge
    corner = np.flatnonzero(np.all(np.isclose(vertices[second, :2], [-0.1, -0.05]), axis=1))
    assert uv[second][corner[0]] == pytest.approx([0.75, 0.25])
    assert sticker_mesh([], sizes, uvs, height)[0].shape == (0, 3)
