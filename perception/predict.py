"""
Write a network's class maps for a PhenoBench split, as the official tools read them.

    python -m perception.predict --checkpoint runs/<name>/best.pt --data <PhenoBench folder>
        --split val --out predictions/<name>_val [--zip]

One PNG per picture, <out>/semantics/<picture name>, 0 soil 1 crop 2 weed, 8 bit. Score val
with the development kit: phenobench-eval --task semantics --phenobench_dir <PhenoBench
folder> --prediction_dir <out> --split val. For the test split, --zip also writes
<out>.zip with the folder semantics/ inside, the form the benchmark server takes (check it
with phenobench-validator first: every upload counts).
"""

import argparse
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import torch

from perception.data import PhenoBench
from perception.models import load


def write_zip(folder, path):
    """Zip folder/semantics as semantics/<file> (with the directory entry the server wants)."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('semantics/', '')
        for png in sorted((Path(folder) / 'semantics').glob('*.png')):
            archive.write(png, f'semantics/{png.name}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path, help='PhenoBench folder')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--zip', action='store_true', help='also write <out>.zip')
    parser.add_argument('--limit', type=int, default=0, help='only N pictures (tests)')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    amp = device.type == 'cuda'
    model, checkpoint = load(args.checkpoint, device)
    data = PhenoBench(args.data, args.split, limit=args.limit)
    loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=args.workers)
    folder = args.out / 'semantics'
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with torch.no_grad():
        for pictures, _, names in loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(pictures.to(device))
            classes = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(classes).save(folder / names[0])
    seconds = time.perf_counter() - start
    print(f'{len(data)} class maps of the {args.split} split in {folder} '
          f'({1000 * seconds / len(data):.0f} ms per picture); checkpoint from epoch '
          f'{checkpoint["epoch"] + 1}, validation {checkpoint.get("metrics")}')
    if args.zip:
        write_zip(args.out, args.out.with_suffix('.zip'))
        print(f'Wrote {args.out.with_suffix(".zip")}')


if __name__ == '__main__':
    main()
