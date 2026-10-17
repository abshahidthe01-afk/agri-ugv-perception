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
    # one set of one camera 1.5 m above flat ground, over the first weed in plot 101
    first = read_obj((tmp_path / 'extras' / 'meshes' / 'weeds_101.obj').read_text())[0][:25]
    x, y = first[:, :2].mean(axis=0)
    np.savez(tmp_path / 'rec.npz', t=np.array([3.0]), cameras=np.array([16]),
             pictures=np.zeros((1, 1, 80, 100, 3), np.uint8),
             depth_mm=np.full((1, 1, 80, 100), 1500, np.uint16), k=np.array([K]),
             mounts=np.array([looking_down()]),
             truth=np.array([[3.0, x, y, 0.0, 0.0, 0.0, 1.0]]))
    labels_args = [str(tmp_path / 'rec.npz'), '--output', str(tmp_path / 'data'),
                   '--extras', str(tmp_path / 'extras')] + args[:4]
    main(labels_args)
    labels = np.asarray(image_module.open(tmp_path / 'data' / 'semantics' / 'rec_000_16.png'))
    assert labels.shape == (80, 100) and labels.dtype == np.uint8
    assert set(np.unique(labels)) <= {0, 1, 2, 255} and np.any(labels == WEED)
    assert (tmp_path / 'data' / 'images' / 'rec_000_16.png').is_file()
    assert (tmp_path / 'data' / 'check' / 'rec_000.png').is_file()
    main(labels_args)                                    # again: replaced, not added
    rows = list(csv.DictReader(open(tmp_path / 'data' / 'pictures.csv')))
    assert [r['name'] for r in rows] == ['rec_000_16.png']
    assert float(rows[0]['weed']) == pytest.approx(100 * np.mean(labels == WEED), abs=1e-3)
