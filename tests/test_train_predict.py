"""End-to-end: a tiny training run on the CPU, then class maps and a submission zip."""

import csv
import zipfile

import numpy as np
from PIL import Image
import pytest

pytest.importorskip('segmentation_models_pytorch')

from perception.models import load            # noqa: E402
from perception.predict import main as predict  # noqa: E402
from perception.train import main as train      # noqa: E402


def run(phenobench, out, epochs, *extra):
    train(['--data', str(phenobench), '--out', str(out), '--epochs', str(epochs),
           '--batch', '2', '--crop', '32', '--workers', '0', '--val-every', '1',
           '--no-pretrained', '--cpu', *extra])


def test_training_writes_a_log_and_checkpoints_and_can_continue(phenobench, tmp_path):
    out = tmp_path / 'run'
    run(phenobench, out, 2)
    rows = list(csv.DictReader(open(out / 'log.csv')))
    assert [r['epoch'] for r in rows] == ['0', '1']
    assert all(0 <= float(r['mIoU']) <= 100 for r in rows)
    model, checkpoint = load(out / 'best.pt')
    assert checkpoint['model'] == 'segformer_b0' and 'mIoU' in checkpoint['metrics']
    run(phenobench, out, 3, '--resume', str(out / 'last.pt'))
    rows = list(csv.DictReader(open(out / 'log.csv')))
    assert [r['epoch'] for r in rows] == ['0', '1', '2']


def test_class_maps_and_submission_zip_have_the_official_form(phenobench, tmp_path):
    out = tmp_path / 'run'
    run(phenobench, out, 1)
    predict(['--checkpoint', str(out / 'best.pt'), '--data', str(phenobench), '--split',
             'val', '--out', str(tmp_path / 'val'), '--workers', '0', '--cpu'])
    for name in ('phenoBench_00000.png', 'phenoBench_00001.png'):
        classes = np.asarray(Image.open(tmp_path / 'val' / 'semantics' / name))
        assert classes.dtype == np.uint8 and classes.shape == (64, 64)
        assert set(np.unique(classes)) <= {0, 1, 2}
    predict(['--checkpoint', str(out / 'best.pt'), '--data', str(phenobench), '--split',
             'test', '--out', str(tmp_path / 'test'), '--zip', '--workers', '0', '--cpu'])
    names = zipfile.ZipFile(tmp_path / 'test.zip').namelist()
    assert names == ['semantics/', 'semantics/phenoBench_00000.png',
                     'semantics/phenoBench_00001.png']


def test_training_on_simulated_pictures_mixed_with_a_few_real_ones(phenobench, simulated,
                                                                   tmp_path):
    out = tmp_path / 'mixed'
    run(phenobench, out, 2, '--real', '1', '--sim', str(simulated), '--sim-plots', '7',
        '--sim-scale', '1', '--epoch-size', '6', '--val-sim', str(simulated),
        '--val-sim-plots', '8')
    rows = list(csv.DictReader(open(out / 'log.csv')))
    assert len(rows) == 2 and all(0 <= float(r['sim_mIoU']) <= 100 for r in rows)
    run(phenobench, tmp_path / 'sim_only', 1, '--real', '0', '--sim', str(simulated),
        '--sim-scale', '1')
    with pytest.raises(ValueError, match='no training pictures'):
        run(phenobench, tmp_path / 'none', 1, '--real', '0')
    with pytest.raises(ValueError, match='only 4'):
        run(phenobench, tmp_path / 'too_many', 1, '--real', '5')


def test_simulated_pictures_are_scored_at_their_own_size(phenobench, simulated, tmp_path):
    run(phenobench, tmp_path / 'run', 1)
    scores = predict(['--checkpoint', str(tmp_path / 'run' / 'best.pt'), '--data',
                      str(simulated.parent), '--split', simulated.name, '--scale', '2',
                      '--plots', '8', '--out', str(tmp_path / 'sim'), '--workers', '0',
                      '--cpu'])
    written = sorted((tmp_path / 'sim' / 'semantics').glob('*.png'))
    assert [p.name for p in written] == ['beet_002_16.png', 'beet_003_16.png']
    classes = np.asarray(Image.open(written[0]))
    assert classes.shape == (64, 96) and set(np.unique(classes)) <= {0, 1, 2}
    assert set(scores) == {'soil', 'crop', 'weed', 'mIoU'}


def test_a_fixed_share_of_simulated_pictures(phenobench, simulated, tmp_path):
    run(phenobench, tmp_path / 'half', 1, '--real', '2', '--sim', str(simulated),
        '--sim-scale', '1', '--epoch-size', '8', '--sim-share', '0.5')
    assert len(list(csv.DictReader(open(tmp_path / 'half' / 'log.csv')))) == 1
    for extra in (['--real', '2', '--sim', str(simulated), '--sim-share', '0.5'],
                  ['--real', '0', '--sim', str(simulated), '--epoch-size', '8',
                   '--sim-share', '0.5'],
                  ['--sim', str(simulated), '--epoch-size', '8', '--sim-share', '1.5']):
        with pytest.raises(ValueError, match='sim-share'):
            run(phenobench, tmp_path / 'bad', 1, '--sim-scale', '1', *extra)
