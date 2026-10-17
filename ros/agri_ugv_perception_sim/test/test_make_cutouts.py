"""Tests for the command-line tool that cuts whole plants out of labelled pictures."""

from agri_ugv_perception_sim.make_cutouts import main, whole_plants
import numpy as np
import pytest


def labelled_picture():
    """Return (picture, semantics, instances) with plants of every kind, 64 x 64 pixels."""
    semantics = np.zeros((64, 64), np.uint16)
    instances = np.zeros((64, 64), np.uint16)

    def plant(rows, cols, label, plant_id):
        semantics[rows, cols], instances[rows, cols] = label, plant_id

    plant(slice(10, 20), slice(10, 25), 2, 5)         # a whole weed, 150 pixels
    plant(slice(30, 36), slice(30, 40), 2, 6)         # a weed partly marked as cut ...
    plant(slice(30, 32), slice(30, 40), 4, 6)         # ... by the picture's edge
    plant(slice(0, 6), slice(40, 46), 2, 7)           # a weed touching the edge
    plant(slice(50, 53), slice(50, 53), 2, 8)         # a small weed, 9 pixels
    plant(slice(40, 56), slice(5, 21), 1, 9)          # a whole crop
    picture = np.random.default_rng(0).integers(0, 256, (64, 64, 3)).astype(np.uint8)
    return picture, semantics, instances


def test_only_whole_plants_of_the_class_are_found():
    _, semantics, instances = labelled_picture()
    found = whole_plants(semantics, instances, 2, 4)
    assert [(plant, area) for plant, area, _ in found] == [(5, 150), (8, 9)]
    assert found[0][2] == (10, 10, 19, 24)
    assert [p for p, _, _ in whole_plants(semantics, instances, 2, 4, min_area=20)] == [5]
    assert [p for p, _, _ in whole_plants(semantics, instances, 1, 3)] == [9]
    assert whole_plants(np.zeros((8, 8)), np.zeros((8, 8)), 2, 4) == []


def test_cutouts_are_transparent_except_on_the_plant(tmp_path):
    image = pytest.importorskip('PIL.Image')
    picture, semantics, instances = labelled_picture()
    for folder, values in (('images', picture), ('semantics', semantics),
                           ('instances', instances)):
        (tmp_path / folder).mkdir()
        image.fromarray(values).save(tmp_path / folder / 'p0.png')
    args = ['--images', str(tmp_path / 'images'), '--semantics', str(tmp_path / 'semantics'),
            '--instances', str(tmp_path / 'instances'), '--min-area', '20']
    main(args + ['--output', str(tmp_path / 'weeds')])
    files = sorted((tmp_path / 'weeds').glob('*.png'))
    assert [f.name for f in files] == ['p0_5.png']
    rgba = np.asarray(image.open(files[0]))
    assert rgba.shape == (14, 19, 4)                   # the box with 2 pixels around it
    assert np.array_equal(rgba[:, :, :3], picture[8:22, 8:27])
    assert np.array_equal(rgba[:, :, 3] > 0, instances[8:22, 8:27] == 5)
    main(args + ['--output', str(tmp_path / 'crops'), '--plants', 'crop'])
    assert [f.name for f in (tmp_path / 'crops').glob('*.png')] == ['p0_9.png']
    with pytest.raises(SystemExit):
        main(args + ['--output', str(tmp_path / 'none'), '--min-area', '1000'])
