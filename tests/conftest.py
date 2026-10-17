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


@pytest.fixture
def simulated(tmp_path):
    """
    Return a simulated dataset folder as make_labels writes it: 4 pictures of 64 x 96, two
    in plot 7 and two in plot 8, labels 0 / 1 / 2 and 255 (the robot) in the top rows.
    """
    rng = np.random.default_rng(1)
    folder = tmp_path / 'sim' / 'beet'
    (folder / 'images').mkdir(parents=True)
    (folder / 'semantics').mkdir()
    rows = ['name,recording,set,camera,t,x,y,yaw,plot,soil,crop,weed,ignore']
    for k in range(4):
        name = f'beet_{k:03d}_16.png'
        labels = np.zeros((64, 96), np.uint8)
        labels[20:40, 10:50] = 1
        labels[45:55, 60:70] = 2
        labels[:4] = 255
        picture = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        picture[labels == 1, 1] = 230
        Image.fromarray(picture).save(folder / 'images' / name)
        Image.fromarray(labels).save(folder / 'semantics' / name)
        rows.append(f'{name},beet,{k},16,{k}.0,0,0,0,{7 + k // 2},0,0,0,0')
    (folder / 'pictures.csv').write_text('\n'.join(rows) + '\n')
    return folder
