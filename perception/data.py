"""
PhenoBench pictures and labels for training and scoring crop/weed segmentation.

PhenoBench (Weyler et al., www.phenobench.org) has 1024 x 1024 drone pictures of sugar beet
fields at about 1 mm per pixel, with a class per pixel: 0 soil, 1 crop, 2 weed, 3 partial
crop and 4 partial weed (plants cut by the picture's edge). As in the official evaluation,
partial crop counts as crop and partial weed as weed, so the network learns 3 classes.

Training pictures are cut to random squares (with a random zoom), turned and mirrored in any
of the 8 ways (the camera looks straight down, so every way is a real view), and their
colours changed a little; scoring uses the whole pictures as they are. Pictures are scaled
to 0..1 and normalised with the ImageNet means and spreads the pretrained encoder expects:
one fixed normalisation for all pictures, so bare soil stays soil-coloured.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch

CLASSES = ('soil', 'crop', 'weed')
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def three_classes(labels):
    """Map PhenoBench labels to soil 0, crop 1, weed 2 (partial plants count as whole)."""
    labels = np.asarray(labels)
    if labels.ndim == 3:
        labels = labels[..., 0]
    out = labels.astype(np.int64)
    out[out == 3] = 1
    out[out == 4] = 2
    if out.max(initial=0) > 2 or out.min(initial=0) < 0:
        raise ValueError(f'unexpected label values {np.unique(labels)}')
    return out


def normalise(picture):
    """(H, W, 3) uint8 picture -> (3, H, W) float tensor, ImageNet normalisation."""
    x = (np.asarray(picture, np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def random_view(picture, labels, rng):
    """Turn by a multiple of 90 degrees and maybe mirror: one of the 8 top-down views."""
    k = int(rng.integers(4))
    picture, labels = np.rot90(picture, k), np.rot90(labels, k)
    if rng.random() < 0.5:
        picture, labels = picture[:, ::-1], labels[:, ::-1]
    return picture, labels


def random_crop(picture, labels, size, rng, zoom=(0.75, 1.25)):
    """Cut a size x size square after zooming by a random factor (labels: nearest pixel)."""
    factor = float(rng.uniform(*zoom))
    h, w = labels.shape
    side = min(int(round(size / factor)), h, w)          # square cut from the original ...
    r, c = int(rng.integers(h - side + 1)), int(rng.integers(w - side + 1))
    picture = Image.fromarray(np.ascontiguousarray(picture[r:r + side, c:c + side]))
    labels = Image.fromarray(np.ascontiguousarray(labels[r:r + side, c:c + side]).astype(np.uint8))
    return (np.asarray(picture.resize((size, size), Image.BILINEAR)),    # ... scaled to size
            np.asarray(labels.resize((size, size), Image.NEAREST)).astype(np.int64))


def random_colour(picture, rng, brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02):
    """Change brightness, contrast, saturation and hue a little (float 0..255 in and out)."""
    x = np.asarray(picture, np.float32) / 255.0
    x = x * (1 + rng.uniform(-brightness, brightness))
    mean = x.mean()
    x = (x - mean) * (1 + rng.uniform(-contrast, contrast)) + mean
    grey = x.mean(axis=2, keepdims=True)
    x = (x - grey) * (1 + rng.uniform(-saturation, saturation)) + grey
    shift = rng.uniform(-hue, hue)                       # a small turn of the colour wheel
    x = x + shift * (np.roll(x, 1, axis=2) - np.roll(x, -1, axis=2))
    return np.clip(x * 255.0, 0, 255)


class PhenoBench(torch.utils.data.Dataset):
    """
    One PhenoBench split (train, val or test) as (picture tensor, label tensor, file name).

    crop: None for whole pictures, else the side of the random training squares (the
    random views and colours are used only then). The test split has no labels: its label
    tensor is all -1.
    """

    def __init__(self, root, split, crop=None, seed=0, limit=0):
        self.root, self.split, self.crop = Path(root), split, crop
        self.images = self.root / split / 'images'
        self.labels = self.root / split / 'semantics'
        self.names = sorted(p.name for p in self.images.glob('*.png'))
        if limit:
            self.names = self.names[:limit]
        if not self.names:
            raise FileNotFoundError(f'no PNG pictures in {self.images}')
        self.has_labels = self.labels.is_dir()
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        picture = np.asarray(Image.open(self.images / name).convert('RGB'))
        if self.has_labels:
            labels = three_classes(Image.open(self.labels / name))
        else:
            labels = np.full(picture.shape[:2], -1, np.int64)
        if self.crop:
            # a different random draw for every picture and epoch, the same for a rerun
            rng = np.random.default_rng([self.seed, self.epoch, index])
            picture, labels = random_view(picture, labels, rng)
            picture, labels = random_crop(picture, labels, self.crop, rng)
            if rng.random() < 0.8:
                picture = random_colour(picture, rng)
        return normalise(picture), torch.from_numpy(np.ascontiguousarray(labels)), name
