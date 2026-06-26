# WSL/Linux FloodCastBench Mamba Setup

This project should run the heavy FloodCastBench/Mamba workflow from the Linux filesystem, not from `/mnt/c`.

Recommended layout:

```text
/home/wissam/projects/a-iat-research
/home/wissam/datasets/FloodCastBench
/home/wissam/experiments/FloodCastBench
/home/wissam/checkpoints/FloodCastBench
/home/wissam/logs/FloodCastBench
```

Avoid storing datasets, checkpoints, or training outputs under `/mnt/c`; filesystem overhead can slow training and data loading.

## Environment

Expected environment:

```text
WSL2 Ubuntu-Research
env name: floodcast-mamba
Python: 3.12
PyTorch: 2.12.1+cu130
CUDA Toolkit: 13.0
CUDA_HOME: /usr/local/cuda-13.0
GPU: NVIDIA RTX 6000 Ada Generation
causal-conv1d==1.6.2.post1
mamba-ssm==2.3.2.post1
```

Create and activate the environment:

```bash
python3.12 -m venv ~/.venvs/floodcast-mamba
source ~/.venvs/floodcast-mamba/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

Install the WSL/Mamba requirements from the repository:

```bash
cd /home/wissam/projects/a-iat-research
export CUDA_HOME=/usr/local/cuda-13.0
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-wsl-mamba.txt
```

## Checks

CUDA driver check:

```bash
nvidia-smi
```

PyTorch GPU check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print(torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else "no cuda")
PY
```

Mamba import check:

```bash
python - <<'PY'
from mamba_ssm import Mamba
import causal_conv1d
print("mamba ok")
print("causal_conv1d ok")
PY
```

Project smoke tests:

```bash
pytest -q
```

## Dataset Placement

Place the raw FloodCastBench dataset here:

```text
/home/wissam/datasets/FloodCastBench
```

The generic config keeps path roots as `null` for backward compatibility. On WSL, the training script defaults legacy repo-relative paths to:

```text
/home/wissam/datasets/FloodCastBench
/home/wissam/experiments/FloodCastBench
/home/wissam/checkpoints/FloodCastBench
/home/wissam/logs/FloodCastBench
```

You can still pass roots explicitly when you want a different location:

```bash
--dataset-root /home/wissam/datasets/FloodCastBench
--experiment-root /home/wissam/experiments/FloodCastBench
--checkpoint-root /home/wissam/checkpoints/FloodCastBench
--log-root /home/wissam/logs/FloodCastBench
```

## Example Training

Temporal convolution, horizon 20:

```bash
python tools/train_floodcastbench_forecasting.py \
  --config configs/floodcastbench_latent_temporal.yaml \
  --temporal-module temporal_conv \
  --horizon 20 \
  --dataset-root /home/wissam/datasets/FloodCastBench \
  --experiment-root /home/wissam/experiments/FloodCastBench \
  --checkpoint-root /home/wissam/checkpoints/FloodCastBench \
  --log-root /home/wissam/logs/FloodCastBench
```

Official Mamba, horizon 72:

```bash
python tools/train_floodcastbench_forecasting.py \
  --config configs/floodcastbench_latent_temporal.yaml \
  --temporal-module mamba \
  --horizon 72 \
  --dataset-root /home/wissam/datasets/FloodCastBench \
  --experiment-root /home/wissam/experiments/FloodCastBench \
  --checkpoint-root /home/wissam/checkpoints/FloodCastBench \
  --log-root /home/wissam/logs/FloodCastBench
```

## Resume Training

Resume from a previous checkpoint:

```bash
python tools/train_floodcastbench_forecasting.py \
  --config configs/floodcastbench_latent_temporal.yaml \
  --temporal-module mamba \
  --horizon 72 \
  --resume-from /home/wissam/checkpoints/FloodCastBench/<run_name>/checkpoint_last.pth \
  --dataset-root /home/wissam/datasets/FloodCastBench \
  --experiment-root /home/wissam/experiments/FloodCastBench \
  --checkpoint-root /home/wissam/checkpoints/FloodCastBench \
  --log-root /home/wissam/logs/FloodCastBench
```

`checkpoint_last.pth` is the latest epoch state. `checkpoint_best.pth` is the best validation checkpoint when a new validation best is observed.
