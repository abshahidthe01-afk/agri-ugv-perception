"""A tiny PhenoBench-like folder for the tests: random pictures with known labels."""

import numpy as np
from PIL import Image
import pytest


def write_split(root, split, count, size, rng, labels=True):
    (root / split / 'images').mkdir(parents=True)
    if labels:
        (root / split / 'semantics').mkdir(parents=True)
    for k in range(count):
        name = f'phenoBench_{k:05d}.png'
        semantics = np.zeros((size, size), np.uint16)            # PhenoBench: 16 bit labels
        semantics[size // 8:size // 2, size // 8:size // 2] = 1          # a crop
        semantics[size // 2:size // 2 + size // 8, size // 2:] = 2       # a weed
        semantics[:size // 8, -size // 8:] = 3                          # partial crop
        semantics[-size // 8:, :size // 8] = 4                          # partial weed
        picture = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        picture[semantics % 2 == 1, 1] = 230                           # crops are greener
        Image.fromarray(picture).save(root / split / 'images' / name)
        if labels:
            Image.fromarray(semantics).save(root / split / 'semantics' / name)


@pytest.fixture
def phenobench(tmp_path):
    """Return the root of a PhenoBench-like folder: train 4, val 2, test 2 (no labels)."""
    rng = np.random.default_rng(0)
    write_split(tmp_path, 'train', 4, 64, rng)
    write_split(tmp_path, 'val', 2, 64, rng)
    write_split(tmp_path, 'test', 2, 64, rng, labels=False)
    return tmp_path
