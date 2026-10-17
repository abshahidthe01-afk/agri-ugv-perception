"""Tests for the analysis of where a network's errors are."""

import numpy as np
import pytest

from perception.analyse import analyse, main, report

pytest.importorskip('scipy')


def scene():
    """Return (truth, predicted) of one picture: a plant with a gap, a weed, some soil."""
    truth = np.zeros((40, 40), np.int64)
    truth[5:20, 5:20] = 1                       # a crop plant ...
    truth[11:14, 11:14] = 0                     # ... with a small gap at its centre
    truth[25:35, 25:35] = 2                     # a weed of 100 pixels
    truth[30:32, 5:7] = 2                       # a weed of 4 pixels
    truth[0, :] = 255                           # left out
    predicted = np.where(truth == 255, 0, truth)
    predicted[10:15, 10:15] = 2                 # weed called around the gap ...
    predicted[11:14, 11:14] = 0
    predicted[30:32, 5:7] = 0                   # ... the small weed missed
    predicted[25:35, 25] = 0                    # and one edge column of the big one
    predicted[2, 30:33] = 2                     # a false weed on soil
    return truth, predicted


def test_the_errors_are_found_where_they_were_made():
    result = analyse([scene()], gap=20, near=1, sizes=(20,))
    matrix = result['confusion']
    assert matrix[1, 2] == 16 and matrix[0, 2] == 3 and matrix[2, 0] == 14
    # one step from the gap reaches the ring around it but not its 4 corners
    assert result['false_weed'] == {'soil': 3, 'crop': 16, 'crop near gaps': 12}
    assert result['missed_weed']['edge'] == (14, 40)        # 4 small + 10 of 36 outer
    assert result['missed_weed']['inside'] == (0, 64)
    small, big = result['patches']
    assert small[:4] == (0, 20, 1, 0.0) and big[:4] == (20, np.inf, 1, 1.0)
    assert big[4] == pytest.approx(0.9)
    assert len(report(result)) == 8                       # 4 table lines, 2 notes, 2 sizes


def test_the_tool_reads_a_simulated_split(simulated, tmp_path):
    from PIL import Image
    (tmp_path / 'pred' / 'semantics').mkdir(parents=True)
    for path in (simulated / 'semantics').glob('*.png'):
        truth = np.asarray(Image.open(path))
        Image.fromarray(np.where(truth == 255, 0, truth).astype(np.uint8)).save(
            tmp_path / 'pred' / 'semantics' / path.name)
    result = main(['--data', str(simulated.parent), '--split', simulated.name, '--predictions',
                   str(tmp_path / 'pred'), '--plots', '7'])
    matrix = result['confusion']
    assert np.all(matrix == np.diag(np.diag(matrix)))            # all right
    assert matrix.sum() == 2 * 60 * 96                            # 2 pictures, 4 rows left out
