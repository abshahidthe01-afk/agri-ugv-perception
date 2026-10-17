"""
Write a network's class maps for a dataset split, as the official tools read them.

    python -m perception.predict --checkpoint runs/<name>/best.pt --data <PhenoBench folder>
        --split val --out predictions/<name>_val [--zip]

One PNG per picture, <out>/semantics/<picture name>, 0 soil 1 crop 2 weed, 8 bit. Score val
with the development kit: phenobench-eval --task semantics --phenobench_dir <PhenoBench
folder> --prediction_dir <out> --split val. For the test split, --zip also writes
<out>.zip with the folder semantics/ inside, the form the benchmark server takes (check it
with phenobench-validator first: every upload counts).

The same for simulated pictures (agri_ugv_perception_sim, make_labels): --data is the
folder that holds the dataset folder, --split its name, --scale enlarges each picture
before the network sees it (2.8: to PhenoBench's pixel size), and the class maps come out
at the picture's own size; --plots keeps only the pictures of these plots. Where the split
has labels, the class maps are also scored (IoU of soil, crop and weed, as metrics.py; pixels
labelled IGNORE do not count).
"""

import argparse
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import torch

from perception.data import names_in_plots, PhenoBench
from perception.metrics import confusion, summary
from perception.models import load


def write_zip(folder, path):
    """Zip folder/semantics as semantics/<file> (with the directory entry the server wants)."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('semantics/', '')
        for png in sorted((Path(folder) / 'semantics').glob('*.png')):
            archive.write(png, f'semantics/{png.name}')


def class_map(model, picture, scale=1.0):
    """
    Return the classes (H, W) of one picture tensor (1, 3, H, W), seen enlarged by scale.

    The enlarged size is rounded to whole multiples of 32 pixels, as the networks need.
    """
    size = picture.shape[-2:]
    if scale != 1:
        picture = torch.nn.functional.interpolate(
            picture, size=[32 * max(1, round(n * scale / 32)) for n in size], mode='bilinear',
            align_corners=False)
    logits = model(picture)
    if logits.shape[-2:] != size:
        logits = torch.nn.functional.interpolate(logits.float(), size=size, mode='bilinear',
                                                 align_corners=False)
    return logits.argmax(dim=1)[0]


def main(argv=None):
    """Write the class maps; return the scores where there are labels (else None)."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path,
                        help='PhenoBench folder, or the folder of a simulated dataset')
    parser.add_argument('--split', default='val', help='train, val, test or a dataset name')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--scale', type=float, default=1.0,
                        help='enlarge the pictures by this for the network (1.0)')
    parser.add_argument('--plots', help='simulated pictures: only those of these plots')
    parser.add_argument('--zip', action='store_true', help='also write <out>.zip')
    parser.add_argument('--limit', type=int, default=0, help='only N pictures (tests)')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    amp = device.type == 'cuda'
    model, checkpoint = load(args.checkpoint, device)
    plots = [v.strip() for v in (args.plots or '').split(',') if v.strip()]
    names = names_in_plots(args.data / args.split, plots) if plots else None
    data = PhenoBench(args.data, args.split, limit=args.limit, names=names)
    loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=args.workers)
    folder = args.out / 'semantics'
    folder.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((3, 3), np.int64)
    start = time.perf_counter()
    with torch.no_grad():
        for pictures, labels, names in loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                classes = class_map(model, pictures.to(device), args.scale)
            classes = classes.cpu().numpy().astype(np.uint8)
            Image.fromarray(classes).save(folder / names[0])
            matrix += confusion(labels[0].numpy(), classes)
    seconds = time.perf_counter() - start
    scores = summary(matrix) if data.has_labels else None
    print(f'{len(data)} class maps of the {args.split} split in {folder} '
          f'({1000 * seconds / len(data):.0f} ms per picture); checkpoint from epoch '
          f'{checkpoint["epoch"] + 1}, validation {checkpoint.get("metrics")}')
    if scores:
        print('Scores (IoU %): ' + ', '.join(f'{k} {v:.2f}' for k, v in scores.items()))
    if args.zip:
        write_zip(args.out, args.out.with_suffix('.zip'))
        print(f'Wrote {args.out.with_suffix(".zip")}')
    return scores


if __name__ == '__main__':
    main()
