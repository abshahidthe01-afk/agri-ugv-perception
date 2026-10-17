"""Tests for the command-line tool that labels a recording into a dataset folder."""

import csv

from agri_ugv_perception_sim.labels import WEED
from agri_ugv_perception_sim.make_field_extras import main as make_field_extras
from agri_ugv_perception_sim.make_labels import main
from agri_ugv_perception_sim.meshes import read_obj
import numpy as np
import pytest
from test_labels import K, looking_down
from test_make_field_extras import field, pictures


def test_a_recording_becomes_pictures_and_labels(tmp_path):
    image_module = pytest.importorskip('PIL.Image')
    args = field(tmp_path)
    soil, weeds = pictures(tmp_path, image_module)
    make_field_extras(args[:-2] + ['--output', str(tmp_path / 'extras'), '--weed-cutouts',
                                   weeds, '--weed-plots', '101'])
    # two sets of one camera 1.5 m above flat ground: over the weed nearest to the middle of
    # plot 101 (at 0, 0), and far outside the plots
    vertices = read_obj((tmp_path / 'extras' / 'meshes' / 'weeds_101.obj').read_text())[0]
    centres = vertices.reshape(-1, 25, 3).mean(axis=1)
    x, y = centres[np.argmin(np.hypot(centres[:, 0], centres[:, 1])), :2]
    np.savez(tmp_path / 'rec.npz', t=np.array([3.0, 9.0]), cameras=np.array([16]),
             pictures=np.zeros((2, 1, 80, 100, 3), np.uint8),
             depth_mm=np.full((2, 1, 80, 100), 1500, np.uint16), k=np.array([K]),
             mounts=np.array([looking_down()]),
             truth=np.array([[3.0, x, y, 0.0, 0.0, 0.0, 1.0],
                             [9.0, 0.0, 20.0, 0.0, 0.0, 0.0, 1.0]]))
    labels_args = [str(tmp_path / 'rec.npz'), '--output', str(tmp_path / 'data'),
                   '--extras', str(tmp_path / 'extras')] + args[:4]
    main(labels_args)
    labels = np.asarray(image_module.open(tmp_path / 'data' / 'semantics' / 'rec_000_16.png'))
    assert labels.shape == (80, 100) and labels.dtype == np.uint8
    assert set(np.unique(labels)) <= {0, 1, 2, 255} and np.any(labels == WEED)
    assert (tmp_path / 'data' / 'images' / 'rec_000_16.png').is_file()
    assert (tmp_path / 'data' / 'check' / 'rec_000.png').is_file()
    rows = list(csv.DictReader(open(tmp_path / 'data' / 'pictures.csv')))
    assert [(r['name'], r['plot']) for r in rows] == [('rec_000_16.png', '101')]
    assert float(rows[0]['weed']) == pytest.approx(100 * np.mean(labels == WEED), abs=1e-3)
    main(labels_args + ['--all-sets'])                   # again: replaced, not added
    rows = list(csv.DictReader(open(tmp_path / 'data' / 'pictures.csv')))
    assert [(r['name'], r['plot']) for r in rows] == [('rec_000_16.png', '101'),
                                                      ('rec_001_16.png', '')]
