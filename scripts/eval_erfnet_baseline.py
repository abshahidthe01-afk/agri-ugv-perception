"""
Run the published PhenoBench ERFNet baseline on a PhenoBench split, and score it.

The reference to compare our own networks with: the checkpoint of PRBonn/phenobench-baselines
(semantic_segmentation, semantic-seg-erfnet.ckpt), run in this repository's environment
instead of the baseline's own (Python 3.8, PyTorch 1.10, Lightning 1.6).

Uses the baseline's own ERFNet class (loaded from a clone of PRBonn/phenobench-baselines) and
the checkpoint's weights, with the baseline's own input steps: picture / 255, then each
colour channel standardised by the picture's own mean and standard deviation, full 1024 x 1024
picture, arg-max over the 3 classes (soil, crop, weed). It writes one label PNG per picture
(<out>/semantics/<name>.png, the layout phenobench-eval and the benchmark server expect) and
prints its own IoU per class as a cross-check of phenobench-eval (partial crops/weeds 3/4 count
as crop/weed, as in the official evaluation).

    python scripts/eval_erfnet_baseline.py --data <PhenoBench folder>
        --baselines <phenobench-baselines clone> --ckpt <semantic-seg-erfnet.ckpt>
        --out <prediction folder> [--split val] [--limit 20]
"""

import argparse
import importlib.util
from pathlib import Path
import pickle
import sys
import time
import types

import numpy as np
from PIL import Image
import torch

NAMES = ('soil', 'crop', 'weed')


def erfnet_class(baselines):
    """Return ERFNetModel from the baselines clone (needs only torch; wget only pretrains)."""
    path = Path(baselines) / 'semantic_segmentation' / 'modules' / 'erfnet' / 'erfnet_modified.py'
    if not path.is_file():
        sys.exit(f'Not found: {path} (is --baselines the phenobench-baselines folder?)')
    sys.modules.setdefault('wget', types.ModuleType('wget'))   # used only when pretrained=True
    spec = importlib.util.spec_from_file_location('erfnet_modified', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ERFNetModel


class _Placeholder(dict):
    """Stands in for a class the checkpoint names but this environment lacks (e.g. Lightning's)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _Placeholder


def read_checkpoint(ckpt):
    """torch.load the Lightning checkpoint; classes of packages not installed become placeholders
    (only the tensors in 'state_dict' are used)."""
    try:
        return torch.load(ckpt, map_location='cpu', weights_only=False)
    except (ModuleNotFoundError, AttributeError) as error:
        print(f'  note: {error}; reading the checkpoint without those classes')
        lenient = types.ModuleType('lenient_pickle')
        lenient.__dict__.update(pickle.__dict__)
        lenient.Unpickler = _LenientUnpickler
        return torch.load(ckpt, map_location='cpu', weights_only=False, pickle_module=lenient)


def load_network(baselines, ckpt, device):
    """Build ERFNet (3 classes) and load the checkpoint's 'network.*' weights, strictly."""
    network = erfnet_class(baselines)(num_classes=3, pretrained=False)
    saved = read_checkpoint(ckpt)                              # a Lightning checkpoint
    state = saved.get('state_dict', saved)
    weights = {k[len('network.'):]: v for k, v in state.items() if k.startswith('network.')}
    if not weights:
        sys.exit(f'No "network.*" weights in {ckpt}; keys start with: '
                 f'{sorted({k.split(".")[0] for k in state})}')
    network.load_state_dict(weights, strict=True)
    print(f'Loaded {len(weights)} tensors, '
          f'{sum(v.numel() for v in network.parameters()) / 1e6:.2f} M parameters; '
          f'Lightning {saved.get("pytorch-lightning_version", "?")}, '
          f'epoch {saved.get("epoch", "?")}')
    return network.to(device).eval()


def standardise(picture):
    """Picture (H x W x 3, uint8) -> 1 x 3 x H x W tensor, each channel to mean 0 and std 1."""
    x = torch.from_numpy(np.array(picture)).permute(2, 0, 1).float() / 255.0
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.flatten(1).std(dim=1).view(3, 1, 1)              # unbiased, like torch.std there
    return ((x - mean) / (std + 1e-17)).unsqueeze(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--data', required=True, type=Path,
                        help='PhenoBench folder (has train/, val/)')
    parser.add_argument('--baselines', required=True, type=Path, help='phenobench-baselines clone')
    parser.add_argument('--ckpt', required=True, type=Path, help='semantic-seg-erfnet.ckpt')
    parser.add_argument('--out', required=True, type=Path, help='prediction folder to write')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--limit', type=int, default=0, help='only the first N pictures (0: all)')
    parser.add_argument('--cpu', action='store_true', help='run on the CPU')
    args = parser.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'torch {torch.__version__}, device {device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))
    network = load_network(args.baselines, args.ckpt, device)

    images = args.data / args.split / 'images'
    labels = args.data / args.split / 'semantics'
    names = sorted(p.name for p in images.glob('*.png'))
    if not names:
        sys.exit(f'No PNG pictures in {images}')
    if args.limit:
        names = names[:args.limit]
    has_labels = labels.is_dir()
    out = args.out / 'semantics'
    out.mkdir(parents=True, exist_ok=True)

    confusion = np.zeros((3, 3), np.int64)                    # rows: truth, columns: prediction
    times = []
    for i, name in enumerate(names):
        picture = Image.open(images / name)
        if picture.mode != 'RGB':
            print(f'  note: {name} is {picture.mode}, converted to RGB')
            picture = picture.convert('RGB')
        x = standardise(np.asarray(picture)).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            prediction = network(x).argmax(dim=1)[0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        prediction = prediction.cpu().numpy().astype(np.uint8)
        Image.fromarray(prediction).save(out / name)
        if has_labels:
            truth = np.asarray(Image.open(labels / name)).astype(np.int64)
            if truth.ndim == 3:
                truth = truth[..., 0]
            truth = np.where(truth == 3, 1, np.where(truth == 4, 2, truth))
            keep = truth < 3
            confusion += np.bincount(3 * truth[keep] + prediction[keep],
                                     minlength=9).reshape(3, 3)
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(names):
            print(f'  {i + 1} / {len(names)} pictures')

    rate = np.median(times[1:] if len(times) > 1 else times)
    memory = (f', peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MB'
              if device.type == 'cuda' else '')
    print(f'{len(names)} predictions in {out}; {1000 * rate:.0f} ms per picture (median){memory}')
    if has_labels:
        iou = np.diag(confusion) / np.maximum(
            confusion.sum(0) + confusion.sum(1) - np.diag(confusion), 1)
        shares = confusion.sum(1) / confusion.sum()
        print('Own check (IoU %): ' + ', '.join(
            f'{n} {100 * v:.2f}' for n, v in zip(NAMES, iou)) + f'; mIoU {100 * iou.mean():.2f}')
        print('Pixel shares in the labels: ' + ', '.join(
            f'{n} {100 * s:.2f} %' for n, s in zip(NAMES, shares)))


if __name__ == '__main__':
    main()
