# efm3d

EFM3D benchmark and EVL (Egocentric Voxel Lifting) baseline model.
Paper: https://arxiv.org/abs/2406.10224

## What This Repo Does

- Defines the EFM3D benchmark: **3D object detection** and **surface reconstruction** on egocentric Aria data
- Implements **EVL**, a voxel-lifting baseline that takes multi-view RGBD frames as input
- Provides training, evaluation, and inference scripts

## Data

`data/` is a symlink → `/research/datasets/efm3d/`

Key splits:
- `ase_train/` — training sequences (ASE synthetic)
- `ase_eval/`, `ase_mesh/` — eval sequences + GT meshes
- `aeo/` — real-world object detection sequences
- `seq136_sample/` — small sample, good for smoke tests

## Key Commands

```bash
conda activate efm3d

# Inference
python infer.py

# Training (distributed)
bash sbatch_run.sh

# Evaluation
python eval.py
```

## Key Files

- `efm3d/` — main library (model, dataset, metrics)
- `train.py` / `eval.py` / `infer.py` — entry points
- `ckpt/` — local checkpoint dir (pretrained weights)
- `INSTALL.md` — installation instructions
- `benchmark.md` — benchmark details

## Dependencies

- `projectaria_tools` — for reading Aria VRS data
- `ATEK` — for data preprocessing pipeline
