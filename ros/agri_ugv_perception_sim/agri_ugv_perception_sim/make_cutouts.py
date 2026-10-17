"""
Command-line tool: cut whole plants out of labelled field pictures, for weed stickers.

For make_field_extras --weed-cutouts (stickers.py). Made for the PhenoBench dataset (Weyler et al.
2024, www.phenobench.org): sugar beet field pictures at 1 mm per pixel with a class per pixel
(1 crop, 2 weed, 3 and 4 crop and weed cut by the picture's edge) and an ID per plant, e.g.
--images PhenoBench/train/images --semantics PhenoBench/train/semantics --instances
PhenoBench/train/plant_instances. Only whole plants are cut out: none of their pixels is
marked as cut by the edge, and they do not touch it. Each cut-out is a PNG picture of the
plant's box with a margin, transparent everywhere but on the plant. Use the training split
only, so that networks scored on the other splits never see these plants. PhenoBench is
licensed CC BY-SA 4.0, and so are cut-outs made from it: keep them outside this repository.
"""

import argparse
from pathlib import Path

import numpy as np

CLASSES = {'crop': (1, 3), 'weed': (2, 4)}      # PhenoBench: (whole label, cut by the edge)


def whole_plants(semantics, instances, label, partial, min_area=0):
    """
    Return (plant ID, area, box) of the whole plants of class 'label' in a picture.

    box = (first row, first column, last row, last column). A plant is whole if none of
    its pixels has the class 'partial' and it does not touch the picture's edge.
    """
    semantics, instances = np.asarray(semantics), np.asarray(instances)
    mine = (semantics == label) & (instances > 0)
    if not mine.any():
        return []
    cut = set(np.unique(instances[semantics == partial]).tolist())
    ids = instances[mine]
    rows, cols = np.nonzero(mine)
    order = np.argsort(ids, kind='stable')
    ids, rows, cols = ids[order], rows[order], cols[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    last_row, last_col = semantics.shape[0] - 1, semantics.shape[1] - 1
    found = []
    for start, end in zip(starts, ends):
        plant = int(ids[start])
        r, c = rows[start:end], cols[start:end]
        box = (int(r.min()), int(c.min()), int(r.max()), int(c.max()))
        if (plant in cut or end - start < min_area or box[0] == 0 or box[1] == 0
                or box[2] == last_row or box[3] == last_col):
            continue
        found.append((plant, int(end - start), box))
    return found


def cut_out(picture, semantics, instances, label, plant, box, margin=2):
    """Return the plant as an RGBA array of its box plus 'margin', transparent elsewhere."""
    h, w = np.asarray(semantics).shape
    r0, c0 = max(box[0] - margin, 0), max(box[1] - margin, 0)
    r1, c1 = min(box[2] + margin, h - 1) + 1, min(box[3] + margin, w - 1) + 1
    mask = (np.asarray(semantics)[r0:r1, c0:c1] == label) & \
        (np.asarray(instances)[r0:r1, c0:c1] == plant)
    rgba = np.zeros((r1 - r0, c1 - c0, 4), np.uint8)
    rgba[:, :, :3] = np.asarray(picture)[r0:r1, c0:c1, :3]
    rgba[:, :, 3] = 255 * mask
    return rgba


def main(argv=None):
    """List the whole plants, pick some at random, save each as a transparent PNG."""
    from PIL import Image   # Pillow (python3-pil)

    parser = argparse.ArgumentParser(description='Cut whole plants out of labelled field '
                                     'pictures, as transparent PNG pictures.')
    parser.add_argument('--images', required=True, help='folder of RGB pictures (PNG)')
    parser.add_argument('--semantics', required=True,
                        help='folder of class pictures with the same names')
    parser.add_argument('--instances', required=True,
                        help='folder of plant ID pictures with the same names')
    parser.add_argument('--output', required=True, help='folder to write the cut-outs to')
    parser.add_argument('--plants', default='weed', choices=sorted(CLASSES),
                        help='which plants to cut out (weed)')
    parser.add_argument('--count', type=int, default=300, help='how many (300)')
    parser.add_argument('--min-area', type=int, default=400,
                        help='leave out plants of fewer pixels (400: 4 cm2 at 1 mm per pixel)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args(argv)

    def read(folder, name):
        values = np.asarray(Image.open(Path(folder) / name))
        return values[..., 0] if values.ndim == 3 else values

    label, partial = CLASSES[args.plants]
    names = sorted(p.name for p in Path(args.images).glob('*.png'))
    found = [(name, *plant) for name in names
             for plant in whole_plants(read(args.semantics, name), read(args.instances, name),
                                       label, partial, args.min_area)]
    if not found:
        raise SystemExit(f'No whole {args.plants} plants of at least {args.min_area} pixels '
                         f'in {len(names)} pictures: check the folders')
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(len(found), size=min(args.count, len(found)), replace=False))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob('*.png'):                      # cut-outs of an earlier run
        old.unlink()
    by_picture = {}
    for k in chosen:
        by_picture.setdefault(found[k][0], []).append(found[k][1:])
    areas = []
    for name, plants in sorted(by_picture.items()):
        picture = np.asarray(Image.open(Path(args.images) / name).convert('RGB'))
        semantics, instances = read(args.semantics, name), read(args.instances, name)
        for plant, area, box in plants:
            rgba = cut_out(picture, semantics, instances, label, plant, box)
            Image.fromarray(rgba, 'RGBA').save(output / f'{Path(name).stem}_{plant}.png')
            areas.append(area)
    print(f'{output}: {len(chosen)} of {len(found)} whole {args.plants} plants of at least '
          f'{args.min_area} pixels in {len(names)} pictures; area of the chosen ones '
          f'(10th / 50th / 90th percentile): '
          f'{" / ".join(str(int(v)) for v in np.percentile(areas, [10, 50, 90]))} pixels')
