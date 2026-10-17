"""Tests for the height of the terrain mesh."""

from agri_ugv_field.ground import ground_height, read_obj_grid
from agri_ugv_perception_sim.terrain import mesh_height
from agri_ugv_terrain.heightfield import mesh_from_heights, obj_text, vertex_normals
import numpy as np
import pytest


def terrain_obj(heights, spacing):
    """Return OBJ text exactly as agri_ugv_terrain writes it."""
    vertices, faces = mesh_from_heights(heights, spacing)
    return obj_text(vertices, vertex_normals(heights, spacing), faces)


def test_mesh_height_lies_on_the_terrain_triangles():
    rng = np.random.default_rng(4)
    heights = np.round(rng.random((4, 5)) * 0.3, 4)          # as precise as the OBJ file
    vertices, faces = mesh_from_heights(heights, 0.5)
    grid = read_obj_grid(terrain_obj(heights, 0.5))
    # points inside every triangle of the mesh, as weighted means of its corners
    weights = rng.dirichlet(np.ones(3), size=len(faces))
    points = np.einsum('fk,fkd->fd', weights, vertices[faces])
    assert mesh_height(grid, points[:, 0], points[:, 1]) == pytest.approx(points[:, 2])


def test_mesh_height_differs_from_the_bilinear_height_in_a_twisted_cell():
    grid = read_obj_grid(terrain_obj(np.array([[0.0, 0.0], [0.0, 0.4]]), 1.0))
    assert mesh_height(grid, 0.0, 0.0) == pytest.approx(0.2)     # on the diagonal
    assert ground_height(grid, 0.0, 0.0) == pytest.approx(0.1)
    assert mesh_height(grid, 0.3, -0.3) == pytest.approx(0.08)   # 0.4 x 0.2 (south-east)
    assert mesh_height(grid, 5.0, -0.5) == pytest.approx(0.0)    # beyond the edge
