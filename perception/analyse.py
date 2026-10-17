"""
Where a network's crop/weed errors are: its class maps compared with the labels of a split.

    python -m perception.analyse --data <folder> --split <name> --predictions <folder>
        [--plots 181,201]

--predictions is the folder that perception.predict (or scripts/eval_erfnet_baseline.py)
wrote, with semantics/<picture name>. Over all pictures it prints:
- the confusion matrix (rows: the true class, in % of its pixels), and the precision and
  recall of each class;
- where the false weed pixels lie: on true soil or on true crop, and how many of those on
  crop lie near small gaps inside a plant (holes of at most --gap pixels in the crop mask,
  grown by --near pixels; in the simulation these are the gaps between the leaf stalks at
  a beet's centre);
- how many true weed pixels are missed on a weed's outer ring of pixels and inside it;
- the weed patches (true weed pixels that touch) by size, with the share of them found
  (at least half of their pixels called weed) and their mean share of pixels called weed.
Pixels labelled IGNORE (the robot itself) are left out; partial plants count as whole ones.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from perception.data import CLASSES, names_in_plots, three_classes
from perception.metrics import confusion

SOIL, CROP, WEED = 0, 1, 2
SIZES = (20, 50, 100, 200)      # [pixels] limits between the size classes of weed patches


def analyse(pairs, gap=150, near=6, sizes=SIZES):
    """
    Return the error figures of (true labels, predicted classes) picture pairs, as a dict.

    confusion: 3 x 3 pixel counts (rows true, columns predicted); false_weed: counts of
    pixels called weed on true soil, on true crop, and on crop near small gaps inside it;
    missed_weed: (missed, all) true weed pixels on the outer ring of the weeds and inside;
    patches: (smallest, largest, count, share found, mean pixel recall) per size class.
    """
    from scipy import ndimage

    matrix = np.zeros((3, 3), np.int64)
    false_weed = {'soil': 0, 'crop': 0, 'crop near gaps': 0}
    missed = {'edge': [0, 0], 'inside': [0, 0]}
    patch_sizes, patch_recalls = [], []
    for truth, predicted in pairs:
        truth, predicted = np.asarray(truth), np.asarray(predicted)
        matrix += confusion(truth, predicted)
        called_weed = predicted == WEED
        crop, weed = truth == CROP, truth == WEED
        false_weed['soil'] += int(np.sum(called_weed & (truth == SOIL)))
        false_weed['crop'] += int(np.sum(called_weed & crop))
        holes = ndimage.binary_fill_holes(crop) & ~crop & (truth == SOIL)
        labels, count = ndimage.label(holes)
        if count:
            small = np.flatnonzero(ndimage.sum(holes, labels, range(1, count + 1)) <= gap) + 1
            gaps = ndimage.binary_dilation(np.isin(labels, small), iterations=near) & crop
            false_weed['crop near gaps'] += int(np.sum(called_weed & gaps))
        inside = ndimage.binary_erosion(weed)
        for key, part in (('edge', weed & ~inside), ('inside', inside)):
            missed[key][0] += int(np.sum(part & ~called_weed))
            missed[key][1] += int(np.sum(part))
        labels, count = ndimage.label(weed)
        if count:
            size = ndimage.sum(weed, labels, range(1, count + 1))
            patch_sizes += list(size)
            patch_recalls += list(ndimage.sum(called_weed, labels, range(1, count + 1)) / size)
    size, recall = np.array(patch_sizes), np.array(patch_recalls)
    limits = (0,) + tuple(sizes) + (np.inf,)
    patches = []
    for low, high in zip(limits[:-1], limits[1:]):
        chosen = (size >= low) & (size < high)
        patches.append((low, high, int(chosen.sum()),
                        float(np.mean(recall[chosen] >= 0.5)) if chosen.any() else float('nan'),
                        float(np.mean(recall[chosen])) if chosen.any() else float('nan')))
    return {'confusion': matrix, 'false_weed': false_weed,
            'missed_weed': {k: tuple(v) for k, v in missed.items()}, 'patches': patches}


def report(result):
    """Return the figures of analyse() as lines of text."""
    matrix = result['confusion']
    with np.errstate(invalid='ignore', divide='ignore'):
        rows = 100 * matrix / matrix.sum(axis=1, keepdims=True)
        recall = 100 * np.diag(matrix) / matrix.sum(axis=1)
        precision = 100 * np.diag(matrix) / matrix.sum(axis=0)
    lines = ['true \\ called  ' + ''.join(f'{c:>8}' for c in CLASSES) + '   recall precision']
    for k, name in enumerate(CLASSES):
        lines.append(f'{name:<14} ' + ''.join(f'{v:7.2f}%' for v in rows[k])
                     + f' {recall[k]:7.2f}% {precision[k]:7.2f}%')
    false = result['false_weed']
    total = false['soil'] + false['crop']
    lines.append(f'false weed pixels: {total} ({false["soil"]} on soil, {false["crop"]} on '
                 f'crop, of those {false["crop near gaps"]} near small gaps inside the crop)')
    edge, inside = result['missed_weed']['edge'], result['missed_weed']['inside']
    lines.append(f'missed weed pixels: {100 * edge[0] / max(edge[1], 1):.1f} % of those on the '
                 f'outer ring, {100 * inside[0] / max(inside[1], 1):.1f} % of those inside')
    for low, high, count, found, recall in result['patches']:
        span = f'{low:g}-{high:g}' if np.isfinite(high) else f'{low:g}+'
        lines.append(f'weed patches of {span:>8} pixels: {count:6d}, found {100 * found:5.1f} %,'
                     f' pixels called weed {100 * recall:5.1f} %')
    return lines


def main(argv=None):
    """Read the labels and class maps of a split, print where the errors are."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--data', required=True, type=Path,
                        help='PhenoBench folder, or the folder of a simulated dataset')
    parser.add_argument('--split', default='val', help='val or a dataset name')
    parser.add_argument('--predictions', required=True, type=Path,
                        help='folder with semantics/ (perception.predict)')
    parser.add_argument('--plots', help='simulated pictures: only those of these plots')
    parser.add_argument('--gap', type=int, default=150,
                        help='largest hole in the crop mask that counts as a gap (150 px)')
    parser.add_argument('--near', type=int, default=6, help='pixels around a gap (6)')
    args = parser.parse_args(argv)

    labels = args.data / args.split / 'semantics'
    plots = [v.strip() for v in (args.plots or '').split(',') if v.strip()]
    names = (names_in_plots(args.data / args.split, plots) if plots else
             sorted(p.name for p in labels.glob('*.png')))
    pairs = ((three_classes(Image.open(labels / n)),
              np.asarray(Image.open(args.predictions / 'semantics' / n))) for n in names)
    result = analyse(pairs, args.gap, args.near)
    print(f'{len(names)} pictures of {args.split}, class maps from {args.predictions}')
    print('\n'.join(report(result)))
    return result


if __name__ == '__main__':
    main()
