"""Tests for writing and reading OBJ meshes."""

from agri_ugv_perception_sim.meshes import obj_text, read_obj
import numpy as np
import pytest


def test_a_mesh_is_read_back_as_written():
    vertices = np.array([[0.0, 0.0, 0.0], [1.2345, 0.0, 0.5], [0.0, 2.0, 0.25]])
    uvs = np.array([[0.123456, 0.0], [1.0, 0.5], [0.33333, 0.99999]])
    faces = np.array([[0, 1, 2]])
    normals = np.array([[0.0, 0.0, 1.0]] * 3)
    back = read_obj(obj_text(vertices, uvs, faces, normals))
    assert back[0] == pytest.approx(np.round(vertices, 3))
    assert back[1] == pytest.approx(np.round(uvs, 5))
    assert back[2].tolist() == [[0, 1, 2]]


def test_faces_with_different_position_and_texture_indices_are_refused():
    with pytest.raises(ValueError, match='differ'):
        read_obj('v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/2 2/2 3/3\n')
