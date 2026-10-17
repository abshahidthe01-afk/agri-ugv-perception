"""Tests for the true crop and weed surfaces around the robot."""

import math

from agri_ugv_field.ground import ground_height, read_obj_grid
from agri_ugv_field.layout import read_layout_csv
from agri_ugv_field.plants import grow
from agri_ugv_field.textures import leaf_outlines
from agri_ugv_perception_sim.make_field_extras import main as make_field_extras
from agri_ugv_perception_sim.scene import crop_meshes, plot_of, plots_near, points_near, \
    to_robot, weed_meshes, weed_plots
import numpy as np
import pytest
from test_make_field_extras import field, pictures


def layout(folder):
    """Write the test field; return (plots, ground, args)."""
    args = field(folder)
    plots = [p for p in read_layout_csv((folder / 'plots.csv').read_text())
             if p['type'] == 'plot']
    grid = read_obj_grid((folder / 'ground' / 'meshes' / 'ground.obj').read_text())
    return plots, (lambda x, y: ground_height(grid, x, y)), args


def test_plots_near_the_robot(tmp_path):
    plots, _, _ = layout(tmp_path)
    assert plots_near(plots, [(0.0, 0.0)], 1.5) == {101}
    assert plots_near(plots, [(-2.5, 0.0)], 1.5) == {100, 101}
    assert plots_near(plots, [(0.0, 20.0)], 1.5) == set()


def test_the_plot_that_holds_the_robot(tmp_path):
    plots, _, _ = layout(tmp_path)
    assert plot_of(plots, (0.0, 0.0)) == 101 and plot_of(plots, (-7.0, 0.0, 9.9)) == 100
    assert plot_of(plots, (-2.8, 0.0)) == 101                  # 0.2 m inside its edge ...
    assert plot_of(plots, (-2.8, 0.0), inset=0.4) is None     # ... not 0.4 m
    assert plot_of(plots, (0.0, 20.0)) is None


def test_crop_meshes_are_the_plants_that_make_plants_grows(tmp_path):
    pytest.importorskip('PIL')
    plots, ground, _ = layout(tmp_path)
    meshes = crop_meshes(plots, ground, {101})
    grown = [m for p, m, _ in grow(plots, ground, 1, leaf_outlines(1)) if p['plot_id'] == 101]
    assert len(meshes) == 1 and np.array_equal(meshes[0][0], grown[0][''][0])
    assert meshes[0][3].dtype == bool and meshes[0][3].any()


def test_weed_meshes_and_points_near_the_robot(tmp_path):
    image_module = pytest.importorskip('PIL.Image')
    _, _, args = layout(tmp_path)
    soil, weeds = pictures(tmp_path, image_module)
    make_field_extras(args + ['--weed-cutouts', weeds, '--weed-plots', '101'])
    meshes = weed_meshes(tmp_path / 'extras')
    assert len(meshes) == 1 and meshes[0][3].shape[1] == 2048
    points = points_near(meshes, (0.0, 0.0), 1.0, 0.003)
    assert len(points) > 0 and np.all(np.abs(points[:, :2]) <= 1.1)   # whole faces near
    assert np.allclose(points[:, 2], 0.008, atol=1e-3)
    assert points_near(meshes, (0.0, 30.0), 1.0, 0.003).shape == (0, 3)
    assert weed_meshes(tmp_path / 'nothing') == []
    assert weed_plots(tmp_path / 'extras') == {101} and weed_plots(tmp_path / 'nothing') == set()


def test_to_robot_turns_world_points_into_the_robot_frame():
    pose = (1.0, 2.0, 0.5, 0.0, 0.0, math.pi / 2)        # facing +y
    assert to_robot([[1.0, 3.0, 0.5], [0.0, 2.0, 1.5]], pose) == pytest.approx(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
