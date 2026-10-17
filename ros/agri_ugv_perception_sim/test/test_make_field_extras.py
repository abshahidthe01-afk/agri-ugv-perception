"""Tests for the command-line tool that writes the soil sheets and weed stickers model."""

from xml.etree import ElementTree

from agri_ugv_field.layout import field_layout, layout_csv, read_layout_csv
from agri_ugv_field.plants import plot_frame
from agri_ugv_perception_sim.make_field_extras import main
from agri_ugv_perception_sim.meshes import read_obj
from agri_ugv_terrain.make_terrain import write_model
import numpy as np
import pytest


def rectangle(cx, cy, length, width, heading_deg):
    """Return a closed ring of 5 points with the long side at heading_deg."""
    h = np.radians(heading_deg)
    u, v = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    corners = [np.array([cx, cy]) + a * length / 2 * u + b * width / 2 * v
               for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    return np.array(corners + corners[:1])


def field(folder):
    """Write a layout CSV with plots 100 and 101 and a flat terrain model; return args."""
    record = {'plot_ID': 0, 'crop': '', 'genotype': '', 'var': ''}
    shapes = [(rectangle(0.0, 0.0, 30.0, 12.0, -5.5), record)]
    shapes += [(rectangle(-7.0 + 7.0 * k, 0.0, 7.7, 6.0, 84.5),
                {**record, 'plot_ID': 100 + k, 'crop': crop})
               for k, crop in enumerate(['Sugar Beet', 'Potato'])]
    boundary, plots = field_layout(shapes, lambda points: points)
    (folder / 'plots.csv').write_text(layout_csv([boundary] + plots))
    write_model(folder, 'ground', np.zeros((41, 81)), 0.5, 'flat test ground')
    return ['--layout', str(folder / 'plots.csv'), '--terrain-model', str(folder / 'ground'),
            '--output', str(folder / 'extras')]


def pictures(folder, image_module):
    """Write a soil texture and 3 weed cut-outs; return (soil file, weed folder)."""
    image_module.new('RGB', (64, 64), (120, 90, 60)).save(folder / 'soil.png')
    (folder / 'weeds').mkdir()
    for k in range(3):
        rgba = np.zeros((40 + 10 * k, 30, 4), np.uint8)
        rgba[:, :, 1] = 150
        rgba[5:-5, 5:-5, 3] = 255
        image_module.fromarray(rgba, 'RGBA').save(folder / 'weeds' / f'weed_{k}.png')
    return str(folder / 'soil.png'), str(folder / 'weeds')


def test_the_model_has_a_soil_sheet_and_weeds_on_the_ground_of_the_chosen_plot(tmp_path):
    image_module = pytest.importorskip('PIL.Image')
    soil, weeds = pictures(tmp_path, image_module)
    args = field(tmp_path)
    main(args + ['--soil-texture', soil, '--soil-plots', '101', '--weed-cutouts', weeds,
                 '--weed-plots', '101', '--weeds-per-m2', '2'])
    model_dir = tmp_path / 'extras'
    model = ElementTree.parse(model_dir / 'model.sdf').getroot().find('model')
    assert model.get('name') == 'extras' and model.find('static').text == 'true'
    assert model.find('link/collision') is None
    visuals = {v.get('name'): v for v in model.findall('link/visual')}
    assert sorted(visuals) == ['soil_101', 'weeds_101']
    for name, texture in (('soil_101', 'soil'), ('weeds_101', 'weeds')):
        mesh = visuals[name].find('geometry/mesh/uri').text
        picture = visuals[name].find('material/pbr/metal/albedo_map').text
        assert mesh == str((model_dir / 'meshes' / f'{name}.obj').resolve())   # full paths
        assert picture == str((model_dir / 'textures' / f'{texture}.png').resolve())
    soil_points = read_obj((model_dir / 'meshes' / 'soil_101.obj').read_text())[0]
    weed_points, uvs, faces = read_obj((model_dir / 'meshes' / 'weeds_101.obj').read_text())
    assert np.allclose(soil_points[:, 2], 0.004)          # flat test ground, 4 mm above it
    assert np.allclose(weed_points[:, 2], 0.008)          # and the weeds 4 mm above that
    assert len(weed_points) == 25 * 92                    # 2 per m2 of 7.7 m x 6 m
    assert uvs.min() >= 0.0 and uvs.max() <= 1.0 and len(faces) == 32 * 92
    plot = next(p for p in read_layout_csv((tmp_path / 'plots.csv').read_text())
                if p['plot_id'] == 101)
    centre, along, across = plot_frame(plot)
    d = weed_points[:, :2] - centre
    assert np.all(np.abs(d @ along) <= plot['length'] / 2 + 1e-3)
    assert np.all(np.abs(d @ across) <= plot['width'] / 2 + 1e-3)
    assert '92 weed stickers' in (model_dir / 'model.config').read_text()
    # a run with weeds only leaves no soil behind
    main(args + ['--weed-cutouts', weeds, '--weed-plots', '100'])
    assert sorted(p.name for p in (model_dir / 'meshes').iterdir()) == ['weeds_100.obj']
    assert sorted(p.name for p in (model_dir / 'textures').iterdir()) == ['weeds.png']


def test_the_options_are_checked_before_anything_is_written(tmp_path):
    image_module = pytest.importorskip('PIL.Image')
    soil, weeds = pictures(tmp_path, image_module)
    args = field(tmp_path)
    (tmp_path / 'empty').mkdir()
    image_module.new('RGB', (64, 32)).save(tmp_path / 'oblong.png')
    for extra, message in (([], 'nothing to make'),
                           (['--soil-plots', '100'], 'together'),
                           (['--weed-cutouts', weeds], 'together'),
                           (['--weed-cutouts', weeds, '--weed-plots', '7'], 'no plots'),
                           (['--weed-cutouts', str(tmp_path / 'empty'), '--weed-plots', '100'],
                            'no PNG'),
                           (['--soil-texture', str(tmp_path / 'oblong.png'), '--soil-plots',
                             '100'], 'square')):
        with pytest.raises(ValueError, match=message):
            main(args + extra)
    assert not (tmp_path / 'extras').exists()
