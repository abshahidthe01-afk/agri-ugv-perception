"""
Command-line tool: make a seamless soil texture from field pictures with soil masks.

For the soil sheet of make_field_extras (--soil-texture). Made for the PhenoBench dataset
(Weyler et al. 2024, www.phenobench.org): sugar beet field pictures at 1 mm per pixel with
labels per pixel (0 = soil), e.g. --images PhenoBench/train/images --semantics
PhenoBench/train/semantics --match 06-05 (the pictures of one day look alike). PhenoBench is
licensed CC BY-SA 4.0, and so is a texture made from it: keep it outside this repository
(make_field_extras copies it into the model it writes).
"""

import argparse
from pathlib import Path

from agri_ugv_perception_sim.soil import even_colour, quilt, soil_windows
import numpy as np


def soil_patches(images, labels, patch, match=''):
    """Return all pure-soil patch x patch windows of the matching pictures, and their count."""
    from PIL import Image   # Pillow (python3-pil)

    found, pictures = [], 0
    for path in sorted(Path(images).glob('*.png')):
        if match not in path.name:
            continue
        mask = np.asarray(Image.open(Path(labels) / path.name))
        mask = mask[..., 0] if mask.ndim == 3 else mask
        picture = np.asarray(Image.open(path).convert('RGB'))
        found += [picture[r:r + patch, c:c + patch] for r, c in soil_windows(mask, patch)]
        pictures += 1
    return found, pictures


def main(argv=None):
    """Find soil windows, quilt them into a texture, save it as PNG."""
    from PIL import Image   # Pillow (python3-pil)

    parser = argparse.ArgumentParser(description='Make a seamless soil texture by image '
                                     'quilting from field pictures with soil masks.')
    parser.add_argument('--images', required=True, help='folder of RGB pictures (PNG)')
    parser.add_argument('--semantics', required=True,
                        help='folder of label pictures with the same names, 0 = soil')
    parser.add_argument('--output', required=True, help='texture file to write (PNG)')
    parser.add_argument('--match', default='',
                        help='only pictures whose name contains this, e.g. a date (06-05)')
    parser.add_argument('--size', type=int, default=1024, help='texture size in pixels (1024)')
    parser.add_argument('--patch', type=int, default=160, help='patch size in pixels (160)')
    parser.add_argument('--overlap', type=int, default=32, help='patch overlap in pixels (32)')
    parser.add_argument('--choices', type=int, default=600,
                        help='patches drawn at random to choose from (600)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args(argv)

    found, pictures = soil_patches(args.images, args.semantics, args.patch, args.match)
    if not found:
        raise SystemExit(f'No {args.patch} px windows of pure soil in {pictures} pictures: '
                         'check the folders and --match')
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(found), size=min(args.choices, len(found)), replace=False)
    patches = even_colour([found[k] for k in pick])
    texture = np.clip(quilt(patches, args.size, args.overlap, rng), 0, 255).astype(np.uint8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(texture).save(args.output)
    print(f'{args.output}: {args.size} x {args.size} px from {len(patches)} of {len(found)} '
          f'soil windows in {pictures} pictures; mean colour (RGB) '
          f'{", ".join(str(int(v)) for v in texture.reshape(-1, 3).mean(axis=0))}')
