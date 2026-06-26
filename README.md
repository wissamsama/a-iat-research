# A-IAT Research

Research codebase for FloodCastBench flood forecasting experiments, with a focus on latent temporal models and the official Mamba backend.

The repository also keeps older CIFAR/GTSRB adversarial-learning utilities, but the canonical current workflow is the WSL/Linux CUDA FloodCastBench setup.

## Canonical Workspace

Run heavy training from the Linux filesystem, not from `/mnt/c`.

```text
workspace:   /home/wissam/utem-workspace
repo:        /home/wissam/utem-workspace/code/a-iat-research
dataset:     /home/wissam/utem-workspace/data/FloodCastBench
experiments: /home/wissam/utem-workspace/experiments/FloodCastBench
checkpoints: /home/wissam/utem-workspace/checkpoints/FloodCastBench
logs:        /home/wissam/utem-workspace/logs/FloodCastBench
checksums:   /home/wissam/utem-workspace/checksums
```

Datasets, checkpoints, experiments, and logs are workspace artifacts. They are intentionally kept outside Git.

## Environment

Canonical environment:

```text
WSL2 Ubuntu-Research
conda env: floodcast-mamba
Python: 3.12
CUDA Toolkit: 13.0
PyTorch: 2.12.1+cu130
torchvision: 0.27.1+cu130
GPU: NVIDIA RTX 6000 Ada Generation
causal-conv1d: 1.6.2.post1
mamba-ssm: 2.3.2.post1
```

Activate:

```bash
cd /home/wissam/utem-workspace/code/a-iat-research
source /home/wissam/miniforge3/etc/profile.d/conda.sh
conda activate floodcast-mamba
```

Install guidance is in:

```text
docs/wsl_linux_setup.md
requirements-wsl-mamba.txt
```

`requirements.txt` is only a legacy pointer. Use `requirements-wsl-mamba.txt` for the active FloodCastBench/Mamba environment.

## Validation

Run tests:

```bash
pytest -q
```

Check CUDA/Mamba:

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba
import causal_conv1d
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print("mamba ok")
PY
```

Dry-run the current Mamba h72 configuration without launching training:

```bash
python tools/train_floodcastbench_forecasting.py \
  --config configs/floodcastbench_latent_temporal.yaml \
  --dataset-root /home/wissam/utem-workspace/data/FloodCastBench \
  --experiment-root /home/wissam/utem-workspace/experiments/FloodCastBench \
  --checkpoint-root /home/wissam/utem-workspace/checkpoints/FloodCastBench \
  --log-root /home/wissam/utem-workspace/logs/FloodCastBench \
  --temporal-module mamba \
  --horizon 72 \
  --batch-size 2 \
  --num-workers 2 \
  --epochs 100 \
  --device auto \
  --dry-run-config
```

## Training

Launch official Mamba at horizon 72:

```bash
python tools/train_floodcastbench_forecasting.py \
  --config configs/floodcastbench_latent_temporal.yaml \
  --dataset-root /home/wissam/utem-workspace/data/FloodCastBench \
  --experiment-root /home/wissam/utem-workspace/experiments/FloodCastBench \
  --checkpoint-root /home/wissam/utem-workspace/checkpoints/FloodCastBench \
  --log-root /home/wissam/utem-workspace/logs/FloodCastBench \
  --temporal-module mamba \
  --horizon 72 \
  --batch-size 2 \
  --num-workers 2 \
  --epochs 100 \
  --device auto
```

Outputs are written outside the repository:

```text
experiments/<run_name>/config.yaml
experiments/<run_name>/metrics.csv
experiments/<run_name>/summary.json
checkpoints/<run_name>/checkpoint_best.pth
checkpoints/<run_name>/checkpoint_last.pth
logs/<run_name>/
```

## Resume

Resume from the latest checkpoint:

```bash
python tools/train_floodcastbench_forecasting.py \
  --config configs/floodcastbench_latent_temporal.yaml \
  --dataset-root /home/wissam/utem-workspace/data/FloodCastBench \
  --experiment-root /home/wissam/utem-workspace/experiments/FloodCastBench \
  --checkpoint-root /home/wissam/utem-workspace/checkpoints/FloodCastBench \
  --log-root /home/wissam/utem-workspace/logs/FloodCastBench \
  --temporal-module mamba \
  --horizon 72 \
  --resume-from /home/wissam/utem-workspace/checkpoints/FloodCastBench/<run_name>/checkpoint_last.pth \
  --device auto
```

`checkpoint_last.pth` stores the latest epoch state. `checkpoint_best.pth` stores the best validation checkpoint.

## Artifact Policy

Keep in Git:

- source code;
- configs;
- tests;
- lightweight documentation;
- selected small reproducibility files, such as normalization JSONs.

Keep outside Git:

- raw datasets;
- generated experiments;
- checkpoints;
- logs;
- large generated figures;
- local shell logs;
- temporary caches.

The `.gitignore` is configured for this policy. Existing historical artifacts should be removed from Git only after they are verified in the workspace.
