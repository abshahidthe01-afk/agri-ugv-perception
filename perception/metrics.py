"""Intersection over union per class, as the PhenoBench evaluation computes it."""

import numpy as np


def confusion(truth, prediction, classes=3):
    """Return the classes x classes count matrix: rows true class, columns predicted class."""
    truth = np.asarray(truth, np.int64).ravel()
    prediction = np.asarray(prediction, np.int64).ravel()
    keep = (truth >= 0) & (truth < classes)
    return np.bincount(classes * truth[keep] + prediction[keep],
                       minlength=classes * classes).reshape(classes, classes)


def iou(matrix):
    """IoU per class from a confusion matrix (a class never seen nor predicted: NaN)."""
    matrix = np.asarray(matrix, np.float64)
    hits = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - hits
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(union > 0, hits / union, np.nan)


def summary(matrix, names=('soil', 'crop', 'weed')):
    """IoU per class and their mean, in percent, as a dict."""
    values = 100 * iou(matrix)
    result = {name: float(v) for name, v in zip(names, values)}
    result['mIoU'] = float(np.nanmean(values))
    return result
