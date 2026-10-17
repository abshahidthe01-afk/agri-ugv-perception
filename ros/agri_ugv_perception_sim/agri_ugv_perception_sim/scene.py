"""
The true surfaces of the crop plants and weeds around the robot, as sample points.

Crops: agri_ugv_field grows every plot's plants from one random generator, in the order of
the layout, so growing them again with the same layout, terrain and seed gives exactly the
meshes that Gazebo shows (as agri_ugv_phenotyping.truth does). Weeds: the sticker meshes
in the model folder that make_field_extras wrote. Each mesh comes with the opaque mask of
its texture, so that the sample points lie only where the cameras see a plant.
"""

from pathlib import Path

from agri_ugv_perception_sim.meshes import read_obj
import numpy as np


def plots_near(plots, poses, reach):
    """Return the IDs of the plots whose rectangle comes within reach + 0.5 m of a pose."""
    from agri_ugv_field.plants import plot_frame

    ids = set()
    for plot in plots:
        centre, along, across = plot_frame(plot)
        for pose in poses:
            d = np.asarray(pose[:2], dtype=float) - centre
            if max(abs(d @ along) - plot['length'] / 2,
                   abs(d @ across) - plot['width'] / 2) < reach + 0.5:
                ids.add(plot['plot_id'])
    return ids


def plot_of(plots, pose, inset=0.0):
    """Return the ID of the plot that holds pose (x, y, ...) at least 'inset' inside, or None."""
    from agri_ugv_field.plants import plot_frame

    for plot in plots:
        centre, along, across = plot_frame(plot)
        d = np.asarray(pose[:2], dtype=float) - centre
        if (abs(d @ along) <= plot['length'] / 2 - inset
                and abs(d @ across) <= plot['width'] / 2 - inset):
            return plot['plot_id']
    return None


def weed_plots(folder):
    """Return the IDs of the plots with weed stickers in a model folder (weeds_<id>.obj)."""
    return {int(f.stem.split('_')[1]) for f in (Path(folder) / 'meshes').glob('weeds_*.obj')}


def crop_meshes(plots, ground, ids, seed=1):
    """
    Return [(vertices, uvs, faces, opaque mask)] of the crop plants of the plots 'ids'.

    plots: all plots of the layout, in its order; ground(x, y): the terrain height as
    make_plants uses it; seed: make_plants's seed.
    """
    from agri_ugv_field.plants import CROPS, grow, PARTS
    from agri_ugv_field.textures import draw_texture, leaf_outlines, opaque

    meshes, masks, left = [], {}, set(ids)
    for plot, parts, _ in grow(plots, ground, seed, leaf_outlines(seed)):
        if not left:
            break                         # the generator grows the plots one after another
        if plot['plot_id'] not in left:
            continue
        left.discard(plot['plot_id'])
        for part, (vertices, uvs, faces) in parts.items():
            texture = PARTS[CROPS[plot['crop']]['shape']][part]
            if texture not in masks:
                masks[texture] = opaque(draw_texture(texture, seed))
            meshes.append((vertices, uvs, faces, masks[texture]))
    return meshes


def weed_meshes(folder):
    """Return [(vertices, uvs, faces, opaque mask)] of the weed stickers in a model folder."""
    from agri_ugv_field.textures import opaque
    from PIL import Image   # Pillow (python3-pil)

    folder = Path(folder)
    files = sorted((folder / 'meshes').glob('weeds_*.obj'))
    if not files:
        return []
    mask = opaque(Image.open(folder / 'textures' / 'weeds.png').convert('RGBA'))
    return [(*read_obj(f.read_text()), mask) for f in files]


def points_near(meshes, centre, reach, step):
    """
    Return world points (N, 3) about 'step' apart on the meshes' opaque parts near 'centre'.

    Only faces with a corner within 'reach' of centre in x and y are sampled.
    """
    from agri_ugv_phenotyping.truth import leaf_points

    found = []
    for vertices, uvs, faces, mask in meshes:
        corners = np.asarray(vertices)[faces][:, :, :2]
        near = np.any(np.all(np.abs(corners - np.asarray(centre)[:2]) <= reach, axis=2), axis=1)
        if near.any():
            found.append(leaf_points(vertices, uvs, faces[near], mask, step))
    return np.vstack(found) if found else np.zeros((0, 3))


def to_robot(points, pose):
    """Return world points in the robot frame of pose (x, y, z, roll, pitch, yaw)."""
    from agri_ugv_phenotyping.reconstruct import rotation_rpy

    turn = rotation_rpy(*pose[3:6])
    return (np.asarray(points, dtype=float) - np.asarray(pose[:3], dtype=float)) @ turn
