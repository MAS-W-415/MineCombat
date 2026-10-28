# Environment Setup

## Target hardware

The local development machine is:

- NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM
- 31 GB RAM, 24 CPU threads
- 264 GB free disk
- NVIDIA OpenGL 4.6 available on `DISPLAY=:0`

This is enough for environment bring-up, VPT 1x inference, and a small PPO
smoke test. Long PPO runs should use a 4090, A100, or a cloud instance.

## Prerequisites

- Python 3.10
- JDK 8 (MineStudio requires Java 8; Java 11 is not sufficient)
- A working OpenGL renderer:
  - a desktop session with NVIDIA OpenGL, or
  - VirtualGL, or
  - Xvfb with software rendering
- Git and Git LFS for model checkpoints

## Install system packages

Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk xvfb libgl1-mesa-glx
```

Select Java 8 if multiple JDKs are installed:

```bash
sudo update-alternatives --config java
java -version  # should report 1.8
```

## Create the Python environment

The repository uses a local virtual environment instead of conda:

```bash
cd MineCombat
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Or run:

```bash
bash scripts/setup.sh
```

## MineStudio engine

The first simulator run downloads the MineStudio engine from Hugging Face:

```bash
python -m minestudio.simulator.entry
```

If the command asks for confirmation, accept the download. The engine is stored
under the MineStudio cache directory.

## VPT 1x checkpoint

Download the foundation model once:

```bash
python - <<'PY'
from minestudio.models import VPTPolicy
policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_1x")
print(type(policy).__name__)
PY
```

The checkpoint is cached by Hugging Face. Do not commit model weights to git.

## Rendering

### Desktop session

If `DISPLAY` is set and `glxinfo` reports the NVIDIA renderer, no extra setup is
needed:

```bash
glxinfo -B | head
```

### Headless with Xvfb

```bash
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
```

Software rendering is slower; use it for correctness checks, not throughput
measurements.

### VirtualGL

For NVIDIA GPUs without a desktop session, install VirtualGL and run:

```bash
MINESTUDIO_GPU_RENDER=1 python -m minestudio.simulator.entry
```

## Single-GPU PPO notes

MineStudio's official online config assumes multiple GPUs:

- 2 rollout workers × 1 GPU
- 2 trainer workers × 1 GPU

`configs/ppo/zombie_vpt1x.yaml` uses:

- 1 rollout worker with `num_gpus_per_worker: 0.5`
- 1 trainer worker with `num_gpus_per_worker: 0.5`
- 8 environments
- batch size 1
- context length 64

If the process runs out of memory:

1. Reduce `worker_config.num_envs` from 8 to 4.
2. Reduce `context_length` from 64 to 32.
3. Set rollout `num_gpus_per_worker` to `0.25`.
4. Freeze the image encoder (requires a small code change in the policy).
5. Move the run to a 4090 or cloud GPU.

## Smoke tests

```bash
# Random actions
python scripts/smoke_test.py --config configs/env/zombie_stage1.yaml --steps 100

# VPT inference
python scripts/smoke_test.py --policy vpt --steps 600

# Baseline evaluation
python scripts/eval_baseline.py --policy random --episodes 5
```

Record the following numbers in the M0 issue:

- steps per second,
- peak VRAM,
- reset time,
- any rendering warnings.

## Troubleshooting

### `java.lang.UnsupportedClassVersionError`

MineStudio is running on Java 11+. Install JDK 8 and select it with
`update-alternatives`.

### `GLFW` or `OpenGL` errors

Check `DISPLAY`, `glxinfo`, and whether VirtualGL is required. Try a desktop
session before installing Xvfb.

### Simulator download prompt

The engine is downloaded on first use. Run the entry point interactively once
and accept the prompt.

### CUDA out of memory

Use the single-GPU profile and reduce the environment count, context length, or
batch size. The simulator and the trainer share the same 8 GB card.
