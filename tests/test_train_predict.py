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
