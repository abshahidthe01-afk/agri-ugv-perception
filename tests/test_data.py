"""Tests for reading PhenoBench and the training views of its pictures."""

import numpy as np
from PIL import Image
import pytest

from perception.data import enlarge, IGNORE, MEAN, names_in_plots, PhenoBench, random_crop, \
    random_view, STD, three_classes


def test_partial_plants_count_as_whole_ones():
    labels = np.array([[0, 1, 2], [3, 4, 0]], np.uint16)
    assert three_classes(labels).tolist() == [[0, 1, 2], [1, 2, 0]]
    with pytest.raises(ValueError):
        three_classes(np.array([[5]]))
    assert three_classes(np.array([[255, 4]], np.uint8)).tolist() == [[IGNORE, 2]]


def test_whole_pictures_are_read_with_their_labels(phenobench):
    data = PhenoBench(phenobench, 'val')
    picture, labels, name = data[0]
    assert len(data) == 2 and name == 'phenoBench_00000.png'
    assert picture.shape == (3, 64, 64) and labels.shape == (64, 64)
    raw = np.asarray(Image.open(phenobench / 'val' / 'images' / name).convert('RGB'))
    assert np.allclose(picture.numpy().transpose(1, 2, 0) * STD + MEAN, raw / 255.0, atol=1e-5)
    truth = three_classes(Image.open(phenobench / 'val' / 'semantics' / name))
    assert np.array_equal(labels.numpy(), truth)


def test_the_test_split_has_no_labels(phenobench):
    _, labels, _ = PhenoBench(phenobench, 'test')[1]
    assert (labels.numpy() == -1).all()


def test_views_and_crops_keep_pictures_and_labels_together():
    rng = np.random.default_rng(1)
    labels = np.kron(rng.integers(0, 3, (4, 4)), np.ones((32, 32), np.int64))  # 32 px blocks
    picture = np.repeat((labels * 100).astype(np.uint8)[..., None], 3, axis=2)
    for _ in range(20):
        p, lab = random_view(picture, labels, rng)
        p, lab = random_crop(p, lab, 48, rng)
        assert p.shape == (48, 48, 3) and lab.shape == (48, 48)
        # the picture's value tells the class: equal everywhere but at block borders
        assert (np.abs(p[..., 0].astype(int) - 100 * lab) < 2).mean() > 0.9


def test_training_squares_differ_by_epoch_and_repeat_for_the_same_epoch(phenobench):
    data = PhenoBench(phenobench, 'train', crop=32, seed=3)
    first = data[0][0].numpy()
    assert np.array_equal(first, data[0][0].numpy())
    data.epoch = 1
    assert not np.array_equal(first, data[0][0].numpy())
    assert data[0][0].shape == (3, 32, 32)


def test_enlarging_rounds_to_multiples_of_32_and_keeps_the_labels():
    labels = np.zeros((48, 64), np.int64)
    labels[:24, :32] = 1
    labels[-1] = IGNORE
    picture = np.repeat((labels * 100 % 256).astype(np.uint8)[..., None], 3, axis=2)
    big, big_labels = enlarge(picture, labels, 2.8)
    assert big.shape == (128, 192, 3) and big_labels.shape == (128, 192)   # 134 x 179 rounded
    assert set(np.unique(big_labels)) == {0, 1, IGNORE}
    assert big_labels[10, 10] == 1 and big_labels[100, 150] == 0 and big_labels[-1, 0] == IGNORE
    _, none = enlarge(picture, np.full((48, 64), -1), 2.0)
    assert none.shape == (96, 128) and (none == -1).all()
    same = enlarge(picture, labels, 1.0)
    assert same[0] is picture and same[1] is labels


def test_simulated_pictures_are_read_enlarged_and_by_plot(simulated):
    assert names_in_plots(simulated, [8]) == ['beet_002_16.png', 'beet_003_16.png']
    data = PhenoBench(simulated.parent, simulated.name, scale=2.0,
                      names=names_in_plots(simulated, ['7']))
    picture, labels, name = data[1]
    assert len(data) == 2 and name == 'beet_001_16.png'
    assert picture.shape == (3, 128, 192) and labels.shape == (128, 192)
    assert set(np.unique(labels.numpy())) == {0, 1, 2, IGNORE}
