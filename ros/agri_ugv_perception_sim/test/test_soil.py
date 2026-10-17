"""Tests for the soil sheet under plots and the quilted soil texture."""

from agri_ugv_perception_sim.make_soil_texture import main as make_soil_texture
from agri_ugv_perception_sim.soil import even_colour, quilt, seam, soil_sheet, soil_windows
from agri_ugv_terrain.heightfield import grid_axes
import numpy as np
import pytest


def plot(cx=0.0, cy=0.0, length=4.0, width=2.0, heading_deg=30.0):
    return {'plot_id': 7, 'centre_x': cx, 'centre_y': cy, 'length': length, 'width': width,
            'heading_deg': heading_deg}


def sloped_grid(rows=41, cols=61, spacing=0.45):
    """Return a terrain grid as read_obj_grid does, with a bumpy surface."""
    xs, ys = grid_axes(rows, cols, spacing)
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, 0.05 * xx + 0.02 * np.sin(3 * yy) + 0.1


def test_soil_sheet_copies_the_terrain_cells_lifted_and_covers_the_plot():
    grid = sloped_grid()
    xs, ys, heights = grid
    vertices, uvs, faces = soil_sheet(grid, plot(), metres=1.024, margin=0.6, lift=0.004)
    # every vertex is a terrain grid point, 4 mm above it
    j = np.searchsorted(xs, vertices[:, 0])
    i = np.searchsorted(ys, vertices[:, 1])
    assert np.allclose(xs[j], vertices[:, 0]) and np.allclose(ys[i], vertices[:, 1])
    assert np.allclose(vertices[:, 2], heights[i, j] + 0.004)
    # the texture repeats every 1.024 m in x and y
    assert np.allclose(uvs, vertices[:, :2] / 1.024)
    # faces point up, like the terrain's
    v = vertices[faces]
    normal_z = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 2]
    assert np.all(normal_z > 0)
    # the plot's corners grown by the margin lie inside the sheet
    h = np.radians(30.0)
    u, w = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    for a in (-1, 1):
        for b in (-1, 1):
            x, y = a * 2.6 * u + b * 1.6 * w
            assert vertices[:, 0].min() <= x <= vertices[:, 0].max()
            assert vertices[:, 1].min() <= y <= vertices[:, 1].max()


def test_a_plot_outside_the_terrain_is_refused():
    with pytest.raises(ValueError, match='outside'):
        soil_sheet(sloped_grid(), plot(cx=100.0), metres=1.0)


def test_soil_windows_avoid_plants_and_keep_a_margin():
    labels = np.zeros((64, 64), np.uint8)
    labels[40, 40] = 2                                   # one weed pixel
    corners = soil_windows(labels, patch=16, margin=4, stride=4)
    assert corners
    for r, c in corners:
        assert labels[r - 4:r + 20, c - 4:c + 20].max() == 0
    assert (4, 4) in corners                             # far from the weed


def test_seam_steps_at_most_one_column_per_row_and_finds_the_cheap_path():
    cost = np.ones((6, 5))
    cost[:, 3] = 0.0
    path = seam(cost)
    assert list(path) == [3] * 6
    path = seam(np.random.default_rng(0).random((30, 8)))
    assert np.all(np.abs(np.diff(path)) <= 1) and path.min() >= 0 and path.max() < 8


def test_even_colour_gives_all_patches_the_same_mean_and_keeps_their_detail():
    rng = np.random.default_rng(1)
    patches = rng.random((3, 8, 8, 3)) * 50 + np.array([0, 60, 120])[:, None, None, None]
    even = even_colour(patches)
    means = even.mean(axis=(1, 2))
    assert np.allclose(means, means[0], atol=1e-3)
    assert np.allclose(even - even.mean(axis=(1, 2), keepdims=True),
                       patches - patches.mean(axis=(1, 2), keepdims=True), atol=1e-3)


def test_quilt_fills_the_texture_from_the_patches_and_tiles_without_an_edge():
    stripes = np.zeros((16, 16, 3), np.float32)
    stripes[:, ::4] = 200.0                             # vertical stripes, period 4
    texture = quilt(np.array([stripes] * 3), size=48, overlap=4, rng=np.random.default_rng(2))
    assert texture.shape == (48, 48, 3)
    assert set(np.unique(texture)) <= {0.0, 200.0}       # only patch pixels, nothing blended
    # the stripe pattern continues across the wrap from the right edge to the left one
    columns = np.flatnonzero(texture[0, :, 0] == 200.0)
    assert np.all(np.diff(np.r_[columns, columns[0] + 48]) == 4)


def test_quilt_refuses_a_size_that_does_not_wrap():
    with pytest.raises(ValueError, match='multiple'):
        quilt(np.zeros((2, 16, 16, 3)), size=50, overlap=4, rng=np.random.default_rng(0))


def test_make_soil_texture_writes_a_texture_from_the_soil_of_the_pictures(tmp_path):
    image_module = pytest.importorskip('PIL.Image')
    rng = np.random.default_rng(3)
    (tmp_path / 'images').mkdir()
    (tmp_path / 'labels').mkdir()
    for k in range(2):
        picture = (rng.random((96, 96, 3)) * 80 + 60).astype(np.uint8)
        labels = np.zeros((96, 96), np.uint8)
        labels[:20, :20] = 1                             # a plant in one corner
        image_module.fromarray(picture).save(tmp_path / 'images' / f'05-15_{k}.png')
        image_module.fromarray(labels).save(tmp_path / 'labels' / f'05-15_{k}.png')
    out = tmp_path / 'soil.png'
    make_soil_texture(['--images', str(tmp_path / 'images'), '--semantics',
                       str(tmp_path / 'labels'), '--output', str(out), '--size', '56',
                       '--patch', '32', '--overlap', '4', '--match', '05-15'])
    texture = np.asarray(image_module.open(out))
    assert texture.shape == (56, 56, 3)
    with pytest.raises(SystemExit):
        make_soil_texture(['--images', str(tmp_path / 'images'), '--semantics',
                           str(tmp_path / 'labels'), '--output', str(out), '--match', '06-05'])
