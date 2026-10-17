"""Tests for the parts of the picture recorder that need no simulation."""

import math

from agri_ugv_perception_sim.record_pictures import pose_at, sheet
import numpy as np
import pytest


def test_the_pose_between_two_samples_changes_evenly_and_turns_the_short_way():
    samples = [(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.pi - 0.1),
               (10.1, 1.0, 2.0, 0.0, 0.0, 0.0, -math.pi + 0.1)]
    pose = pose_at(samples, 10.05)
    assert pose[:3] == pytest.approx((0.5, 1.0, 0.0))
    assert abs(pose[5]) == pytest.approx(math.pi)        # through pi, not through 0
    assert pose_at(samples, 9.0) == pytest.approx(samples[0][1:])
    assert pose_at(samples, 11.0) == pytest.approx(samples[1][1:])


def test_sheet_lays_the_pictures_in_rows_at_half_size():
    pytest.importorskip('PIL')
    pictures = [np.full((40, 60, 3), k * 40, np.uint8) for k in range(6)]
    board = sheet(pictures, columns=4)
    assert board.shape == (40, 120, 3)
    assert board[5, 35, 0] == 40 and board[25, 35, 0] == 200 and board[25, 95, 0] == 0
