# agri-ugv-perception

Crop/weed segmentation for a field robot: trained on real sugar beet fields
([PhenoBench](https://www.phenobench.org)), tested in simulation
([agri-ugv-sim](https://github.com/abshahidthe01-afk/agri-ugv-sim)), and used on the
simulated robot.

**Status: phase 1 in progress.** This README lists results only once they are measured.

## Plan

1. **Crop/weed segmentation on PhenoBench.** Train a network that labels every pixel of a
   field picture as soil, crop (sugar beet) or weed; score it with PhenoBench's official
   evaluation and compare it with the published ERFNet baseline.
2. **Simulation and real pictures.** Pictures and labels from the robot's cameras in
   agri-ugv-sim, and how well a network moves between simulated and real pictures.
3. **On the robot.** Replace the robot's hand-made plant/soil colour threshold with the
   network: crop/weed/soil-labelled 3D plot clouds, scored against the simulation's
   ground truth.
4. **Counting plants.** Follow plants from picture to picture with the robot's pose, so
   each plant is counted once.

## Setup

Python 3.12. Install PyTorch for your GPU first, then the rest. For example, Windows
PowerShell in this folder, with an NVIDIA GPU and CUDA 12.1:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pytest
```

The tests make a tiny PhenoBench-like folder and train for a few steps on the CPU; one of
them checks that our scores equal those of the official evaluation.

## Data

PhenoBench, labelled part (`PhenoBench-v110.zip`, 7.6 GB, from
[phenobench.org](https://www.phenobench.org/dataset.html)), unpacked so that the folder
holds `train/`, `val/` and `test/`. The data is licensed CC BY-SA 4.0 by its authors
(Weyler et al., see below) and is not part of this repository.

## Use

Train (writes `runs/<name>/log.csv`, `last.pt`, `best.pt`):

```powershell
python -m perception.train --data <PhenoBench folder> --out runs/segformer_b0
```

Class maps for the validation pictures, scored with the official evaluation:

```powershell
python -m perception.predict --checkpoint runs/segformer_b0/best.pt --data <PhenoBench folder> --split val --out predictions/segformer_b0_val
phenobench-eval --task semantics --phenobench_dir <PhenoBench folder> --prediction_dir predictions/segformer_b0_val --split val
```

The published baseline on the same pictures, for comparison (needs a clone of
[PRBonn/phenobench-baselines](https://github.com/PRBonn/phenobench-baselines) and its
semantic segmentation checkpoint `semantic-seg-erfnet.ckpt`):

```powershell
python scripts/eval_erfnet_baseline.py --data <PhenoBench folder> --baselines <phenobench-baselines folder> --ckpt <semantic-seg-erfnet.ckpt> --out predictions/erfnet_val
```

## Licence and credits

Code: MIT ([LICENSE](LICENSE)). Data, the baseline network and its weights belong to their
authors:

- J. Weyler et al., *PhenoBench: A Large Dataset and Benchmarks for Semantic Image
  Interpretation in the Agricultural Domain*, IEEE TPAMI 2024 (arXiv 2306.04557).
- E. Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with
  Transformers*, NeurIPS 2021; implementation from
  [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
- E. Romera et al., *ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic
  Segmentation*, IEEE T-ITS 2018 (the PhenoBench baseline).
