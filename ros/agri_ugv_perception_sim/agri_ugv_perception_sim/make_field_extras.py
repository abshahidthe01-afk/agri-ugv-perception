"""
Command-line tool: soil with real detail and weeds from real field pictures for chosen plots.

Writes a Gazebo model (default ~/agri_data/models/field_extras) that this package's
mission.launch.py adds to agri-ugv-sim's MuST-C field: a soil sheet under each plot of
--soil-plots (soil.py) and weed stickers in each plot of --weed-plots (stickers.py). The
field itself (terrain, plots, crop plants) stays exactly as agri-ugv-sim makes it.

    ros2 run agri_ugv_perception_sim make_field_extras --soil-texture <soil.png>
        --soil-plots 198 --weed-cutouts <folder of weed cut-outs> --weed-plots 198
"""

import argparse
from pathlib import Path

from agri_ugv_field.ground import read_obj_grid
from agri_ugv_field.layout import read_layout_csv
from agri_ugv_field.plants import vertex_normals
from agri_ugv_perception_sim.meshes import obj_text
from agri_ugv_perception_sim.soil import soil_sheet
from agri_ugv_perception_sim.stickers import atlas, box_uvs, place, sticker_mesh
from agri_ugv_perception_sim.terrain import mesh_height
from agri_ugv_terrain.heightfield import model_config
import numpy as np


def model_sdf(name, folder, meshes):
    """
    Return a static model with one visual per mesh and no collision.

    meshes maps a mesh name to its texture name; both files are named by their full path
    in 'folder', so the model can be added to a running simulation from its file. As
    agri_ugv_field's plants, every material shows its texture's picture and cuts away its
    transparent parts; the faces point up and are seen from above only.
    """
    folder = Path(folder).resolve()
    visuals = ''.join(f"""
      <visual name="{mesh}">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh><uri>{folder / 'meshes' / f'{mesh}.obj'}</uri></mesh>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <double_sided>false</double_sided>
          <pbr>
            <metal>
              <albedo_map>{folder / 'textures' / f'{texture}.png'}</albedo_map>
              <roughness>0.8</roughness>
              <metalness>0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>""" for mesh, texture in meshes.items())
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="extras">{visuals}
    </link>
  </model>
</sdf>
"""


def chosen_plots(text, plots, option):
    """Return the set of plot IDs in 'text' (comma-separated); refuse IDs not in plots."""
    ids = {int(v) for v in text.split(',') if v.strip()}
    missing = sorted(ids - {p['plot_id'] for p in plots})
    if missing:
        raise ValueError(f'{option}: no plots {missing} in the layout')
    return ids


def installed(package, *parts):
    """Return a path in an installed ROS package's share folder, or None without ROS."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return str(Path(get_package_share_directory(package), *parts))
    except (ImportError, LookupError):
        return None


def main(argv=None):
    """Check the options, then write the model folder: meshes, textures, model files."""
    from PIL import Image   # Pillow (python3-pil)

    parser = argparse.ArgumentParser(
        description='Make a Gazebo model of soil sheets and weed stickers for chosen plots '
        'of the agri-ugv-sim field.')
    parser.add_argument('--layout', help='plot layout CSV (default: the one of agri_ugv_field)')
    parser.add_argument('--terrain-model',
                        help='terrain model folder (default: must_c_field of agri_ugv_gazebo)')
    parser.add_argument('--output', default=str(Path.home() / 'agri_data' / 'models' /
                                                'field_extras'),
                        help='model folder to write; its name is the model name')
    parser.add_argument('--seed', type=int, default=1, help='random seed (same seed, same weeds)')
    parser.add_argument('--soil-texture',
                        help='soil picture (make_soil_texture) for a sheet under --soil-plots')
    parser.add_argument('--soil-plots', default='',
                        help='plot IDs, comma-separated, that get the soil sheet (e.g. 198)')
    parser.add_argument('--soil-mm', type=float, default=1.0,
                        help='millimetres of ground per pixel of the soil texture (1.0)')
    parser.add_argument('--weed-cutouts',
                        help='folder of weed pictures (make_cutouts) for --weed-plots')
    parser.add_argument('--weed-plots', default='',
                        help='plot IDs, comma-separated, that get weeds (e.g. 198)')
    parser.add_argument('--weeds-per-m2', type=float, default=20.0,
                        help='weeds per square metre of plot (20)')
    parser.add_argument('--weed-mm', type=float, default=1.0,
                        help='millimetres of ground per pixel of the weed pictures (1.0)')
    args = parser.parse_args(argv)

    layout = args.layout or installed('agri_ugv_field', 'data', 'must_c_field_plots.csv')
    terrain = args.terrain_model or installed('agri_ugv_gazebo', 'models', 'must_c_field')
    if layout is None or terrain is None:
        raise ValueError('without ROS, give --layout and --terrain-model')
    plots = [r for r in read_layout_csv(Path(layout).read_text()) if r['type'] == 'plot']
    soil_ids = chosen_plots(args.soil_plots, plots, '--soil-plots')
    weed_ids = chosen_plots(args.weed_plots, plots, '--weed-plots')
    if bool(soil_ids) != bool(args.soil_texture):
        raise ValueError('--soil-texture and --soil-plots go together')
    if bool(weed_ids) != bool(args.weed_cutouts):
        raise ValueError('--weed-cutouts and --weed-plots go together')
    if not soil_ids and not weed_ids:
        raise ValueError('nothing to make: give --soil-plots or --weed-plots (or both)')
    soil = Image.open(args.soil_texture).convert('RGB') if soil_ids else None
    if soil is not None and soil.width != soil.height:
        raise ValueError(f'{args.soil_texture}: the soil texture must be square')
    files = sorted(Path(args.weed_cutouts).glob('*.png')) if weed_ids else []
    if weed_ids and not files:
        raise ValueError(f'--weed-cutouts: no PNG pictures in {args.weed_cutouts}')

    terrain = Path(terrain)
    grid = read_obj_grid((terrain / 'meshes' / f'{terrain.name}.obj').read_text())
    model_dir = Path(args.output)
    for folder, kind in (('meshes', '*.obj'), ('textures', '*.png')):
        (model_dir / folder).mkdir(parents=True, exist_ok=True)
        for old in (model_dir / folder).glob(kind):           # files of an earlier run
            old.unlink()
    meshes, notes = {}, []

    def write(mesh, texture, vertices, uvs, faces):
        (model_dir / 'meshes' / f'{mesh}.obj').write_text(
            obj_text(vertices, uvs, faces, vertex_normals(vertices, faces)))
        meshes[mesh] = texture

    print(f'{model_dir}: extras for the plots of {Path(layout).name} on the {terrain.name} '
          f'terrain')
    if soil is not None:
        soil.save(model_dir / 'textures' / 'soil.png')
        metres = soil.width * args.soil_mm / 1000
        for plot in plots:
            if plot['plot_id'] in soil_ids:
                write(f'soil_{plot["plot_id"]}', 'soil', *soil_sheet(grid, plot, metres))
        ids = ', '.join(str(i) for i in sorted(soil_ids))
        print(f'  soil   sheet under plot(s) {ids}: {Path(args.soil_texture).name}, '
              f'{soil.width} x {soil.height} px repeating every {metres:.3f} m')
        notes.append(f'a soil sheet ({Path(args.soil_texture).name}) under plot(s) {ids}')
    if weed_ids:
        pictures = [np.asarray(Image.open(f).convert('RGBA')) for f in files]
        texture, boxes = atlas(pictures)
        Image.fromarray(texture, 'RGBA').save(model_dir / 'textures' / 'weeds.png')
        sizes = [(p.shape[1] * args.weed_mm / 1000, p.shape[0] * args.weed_mm / 1000)
                 for p in pictures]
        uvs = [box_uvs(box, texture.shape) for box in boxes]

        def height(x, y):
            return mesh_height(grid, x, y)

        laid = wanted = 0
        for plot in plots:
            if plot['plot_id'] in weed_ids:
                rng = np.random.default_rng([args.seed, plot['plot_id']])
                count = int(round(args.weeds_per_m2 * plot['length'] * plot['width']))
                placed = place(plot, sizes, count, rng)
                laid, wanted = laid + len(placed), wanted + count
                if placed:
                    write(f'weeds_{plot["plot_id"]}', 'weeds',
                          *sticker_mesh(placed, sizes, uvs, height))
        ids = ', '.join(str(i) for i in sorted(weed_ids))
        print(f'  weeds  {laid} stickers ({wanted} wanted) in plot(s) {ids}, from '
              f'{len(files)} pictures in {args.weed_cutouts}; texture {texture.shape[1]} x '
              f'{texture.shape[0]} px')
        notes.append(f'{laid} weed stickers ({Path(args.weed_cutouts).name}) in plot(s) {ids}')
    (model_dir / 'model.sdf').write_text(model_sdf(model_dir.name, model_dir, meshes))
    (model_dir / 'model.config').write_text(model_config(
        model_dir.name, f'extras for the plots of {Path(layout).name} on the {terrain.name} '
        f'terrain, made by agri_ugv_perception_sim (seed {args.seed}): ' + '; '.join(notes)))
