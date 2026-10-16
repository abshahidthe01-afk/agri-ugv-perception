"""Tests for the IoU scores, also against the official PhenoBench evaluation."""

from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from perception.data import three_classes
from perception.metrics import confusion, iou, summary


def test_confusion_counts_true_classes_in_rows_and_skips_unlabelled_pixels():
    truth = np.array([0, 0, 1, 2, -1])
    prediction = np.array([0, 1, 1, 0, 2])
    assert confusion(truth, prediction).tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_iou_per_class_and_mean():
    matrix = np.array([[8, 2, 0], [1, 4, 0], [0, 0, 0]])
    values = iou(matrix)
    assert values[0] == pytest.approx(8 / 11) and values[1] == pytest.approx(4 / 7)
    assert np.isnan(values[2])                       # never there, never predicted
    scores = summary(matrix)
    assert scores['mIoU'] == pytest.approx(100 * (8 / 11 + 4 / 7) / 2)


def test_scores_match_the_official_evaluation(phenobench, tmp_path):
    official = pytest.importorskip('phenobench.evaluation.evaluate_semantics')
    rng = np.random.default_rng(5)
    predictions = tmp_path / 'predictions' / 'semantics'
    predictions.mkdir(parents=True)
    matrix = np.zeros((3, 3), np.int64)
    for path in sorted((phenobench / 'val' / 'semantics').glob('*.png')):
        truth = three_classes(Image.open(path))
        guess = np.where(rng.random(truth.shape) < 0.3, rng.integers(0, 3, truth.shape), truth)
        Image.fromarray(guess.astype(np.uint8)).save(predictions / path.name)
        matrix += confusion(truth, guess)
    args = vars(Namespace(phenobench_dir=Path(phenobench), prediction_dir=predictions.parent,
                          split='val'))
    result = official.evaluate_semantics(args)
    mine = summary(matrix)
    for name in ('soil', 'crop', 'weed', 'mIoU'):
        assert mine[name] == pytest.approx(result[name], abs=0.01)
