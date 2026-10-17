"""
Command-line tool: soil / crop / weed labels for recorded pictures, as a dataset folder.

    ros2 run agri_ugv_perception_sim make_labels <recording.npz> --output <folder>
        [--extras ~/agri_data/models/field_extras]

The recording (record_pictures) holds sets of pictures with depth, the cameras' places and
calibrations, and the robot's true pose. Only the sets taken in a plot with weeds (one of
the extras model) are used, with the robot at least --inset metres inside the plot, so that
the cameras see the soil sheet and the weeds and not the ground between plots (--all-sets
uses every set). Each picture gets its labels (labels.py) from the true crop plants and the
weed stickers around the robot (scene.py). The folder has the layout of a PhenoBench split,
so the same code reads both: images/<name>.png (the picture) and semantics/<name>.png (8
bit: 0 soil, 1 crop, 2 weed, 255 left out: the robot itself), named
<recording>_<set>_<camera>; pictures.csv (one row per picture: where it was taken, in
which plot, and its share of each label) and check/<recording>_<set>.png (pictures above
their labels).
Running it again for the same recording replaces its pictures. For each picture it prints
the shares and how often the labels agree with the robot's plant/soil colour threshold
(excess green above 0.1, agri_ugv_phenotyping.colour) on the labelled pixels.
"""

import argparse
import csv
from pathlib import Path

from agri_ugv_field.ground import ground_height, read_obj_grid
from agri_ugv_field.layout import read_layout_csv
from agri_ugv_perception_sim.labels import CROP, IGNORE, label_picture, SOIL, WEED
from agri_ugv_perception_sim.make_field_extras import installed
from agri_ugv_perception_sim.scene import crop_meshes, plot_of, plots_near, points_near, \
    to_robot, weed_meshes, weed_plots
import numpy as np

COLOURS = {SOIL: (150, 110, 70), CROP: (40, 200, 40), WEED: (230, 40, 40), IGNORE: (0, 0, 0)}
COLUMNS = ['name', 'recording', 'set', 'camera', 't', 'x', 'y', 'yaw', 'plot', 'soil', 'crop',
           'weed', 'ignore']


def coloured(labels):
    """Return the labels as an RGB picture (COLOURS)."""
    table = np.zeros((256, 3), np.uint8)
    for label, colour in COLOURS.items():
        table[label] = colour
    return table[labels]


def main(argv=None):
    """Label every picture of a recording and write them into the dataset folder."""
    from agri_ugv_phenotyping.colour import excess_green, GREEN
    from PIL import Image   # Pillow (python3-pil)

    parser = argparse.ArgumentParser(description='Label recorded pictures as soil, crop or '
                                     'weed from the geometry of the simulation.')
    parser.add_argument('recording', help='recording (.npz) made by record_pictures')
    parser.add_argument('--output', required=True, help='dataset folder to write into')
    parser.add_argument('--extras', default=str(Path.home() / 'agri_data' / 'models' /
                                                'field_extras'),
                        help='model folder of make_field_extras (its weeds)')
    parser.add_argument('--layout', help='plot layout CSV (default: the one of agri_ugv_field)')
    parser.add_argument('--terrain-model',
                        help='terrain model folder (default: must_c_field of agri_ugv_gazebo)')
    parser.add_argument('--seed', type=int, default=1, help='seed the plants were grown with (1)')
    parser.add_argument('--step', type=float, default=0.003,
                        help='metres between sample points on the plants (0.003)')
    parser.add_argument('--reach', type=float, default=1.5,
                        help='metres around the robot to sample (1.5)')
    parser.add_argument('--inset', type=float, default=0.4,
                        help='metres the robot is inside a plot with weeds at least (0.4)')
    parser.add_argument('--all-sets', action='store_true',
                        help='label every set, wherever the robot was')
    args = parser.parse_args(argv)

    layout = args.layout or installed('agri_ugv_field', 'data', 'must_c_field_plots.csv')
    terrain = args.terrain_model or installed('agri_ugv_gazebo', 'models', 'must_c_field')
    if layout is None or terrain is None:
        raise ValueError('without ROS, give --layout and --terrain-model')
    recording = np.load(args.recording)
    stem = Path(args.recording).stem
    pictures, depths = recording['pictures'], recording['depth_mm'] / 1000.0
    mounts, ks, cameras = recording['mounts'], recording['k'], recording['cameras']
    poses = recording['truth'][:, 1:7]
    if not np.all(np.isfinite(mounts)):
        raise ValueError(f'{args.recording}: the places of the cameras are missing')

    plots = [r for r in read_layout_csv(Path(layout).read_text()) if r['type'] == 'plot']
    terrain = Path(terrain)
    grid = read_obj_grid((terrain / 'meshes' / f'{terrain.name}.obj').read_text())

    def ground(x, y):
        return ground_height(grid, x, y)

    weedy = [p for p in plots if p['plot_id'] in weed_plots(args.extras)]
    where = [plot_of(weedy, pose, args.inset) for pose in poses]
    sets = [s for s in range(len(poses)) if args.all_sets or where[s] is not None]
    near = plots_near(plots, poses[sets], args.reach)
    crops = crop_meshes(plots, ground, near, args.seed)
    weeds = weed_meshes(args.extras)
    print(f'{args.recording}: {len(sets)} of {len(poses)} sets '
          f'({"all" if args.all_sets else f"in the {len(weedy)} plots with weeds"}) x '
          f'{len(cameras)} cameras; crops of {len(near)} plot(s), {len(weeds)} weed mesh(es) '
          f'from {args.extras}')

    output = Path(args.output)
    for folder in ('images', 'semantics', 'check'):
        (output / folder).mkdir(parents=True, exist_ok=True)
        for old in (output / folder).glob(f'{stem}_*.png'):      # an earlier run's pictures
            old.unlink()
    table = output / 'pictures.csv'
    rows = list(csv.DictReader(open(table))) if table.exists() else []
    rows = [r for r in rows if r['recording'] != stem]
    print('set camera | soil  crop  weed  left out % | colour threshold agrees %')
    for s in sets:
        pose = poses[s]
        classes = {CROP: to_robot(points_near(crops, pose, args.reach, args.step), pose),
                   WEED: to_robot(points_near(weeds, pose, args.reach, args.step), pose)}
        panels = []
        for c, camera in enumerate(cameras):
            labels = label_picture(depths[s, c], mounts[c], ks[c], classes)
            name = f'{stem}_{s:03d}_{camera}.png'
            Image.fromarray(pictures[s, c]).save(output / 'images' / name)
            Image.fromarray(labels).save(output / 'semantics' / name)
            panels.append((pictures[s, c], coloured(labels)))
            share = {label: 100 * float(np.mean(labels == label)) for label in COLOURS}
            used = labels != IGNORE
            plant = excess_green(pictures[s, c].reshape(-1, 3).astype(float)).reshape(
                labels.shape) > GREEN
            agree = 100 * float(np.mean((plant == (labels != SOIL))[used])) if used.any() \
                else float('nan')
            print(f'{s:3d} {camera:6d} | {share[SOIL]:5.1f} {share[CROP]:5.1f} '
                  f'{share[WEED]:5.2f} {share[IGNORE]:5.1f}      | {agree:5.1f}')
            rows.append({'name': name, 'recording': stem, 'set': s, 'camera': int(camera),
                         't': f'{recording["truth"][s, 0]:.2f}', 'x': f'{pose[0]:.3f}',
                         'y': f'{pose[1]:.3f}', 'yaw': f'{pose[5]:.4f}',
                         'plot': '' if where[s] is None else where[s],
                         **{key: f'{share[label]:.3f}' for key, label in
                            (('soil', SOIL), ('crop', CROP), ('weed', WEED),
                             ('ignore', IGNORE))}})
        top = np.concatenate([p for p, _ in panels], axis=1)
        bottom = np.concatenate([lab for _, lab in panels], axis=1)
        Image.fromarray(np.concatenate([top, bottom], axis=0)).save(
            output / 'check' / f'{stem}_{s:03d}.png')
    with open(table, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(sets) * len(cameras)} labelled pictures in {output} (images/, semantics/, '
          f'check/, pictures.csv)')
