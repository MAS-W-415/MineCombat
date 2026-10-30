# 1M-Step PPO Runbook

This is the planned first real PPO run (#10). The local RTX 4060 / 31 GB machine
is not suitable: one environment runs at ~45 steps/s (about 6.2 hours for 1M),
and four local clients exhaust system RAM. Use a host with at least 64 GB RAM
and a 24 GB GPU (4090 or cloud).

## Step math

The base config (`configs/ppo/zombie_vpt1x.yaml`) uses:

- `batches_per_iteration: 200`
- `fragment_length: 256`
- 8 environments

One iteration is `200 × 256 = 51,200` environment steps. For 1M steps:

```text
20 iterations × 200 fragments × 256 steps = 1,024,000 steps
```

Do **not** use `--smoke` for the 1M run; it overrides the batch counts to 2.

## Preflight

```bash
export HF_ENDPOINT=https://hf-mirror.com
source .venv/bin/activate

# JDK 8 is required by the Minecraft engine.
export JAVA_HOME="$HOME/.local/opt/jdk8/usr/lib/jvm/java-8-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"

# Engine patches: Malmo launch race and left-click lock.
python scripts/patch_minestudio.py
python scripts/patch_minestudio_engine.py

# Check the resolved config without starting Ray.
PYTHONPATH=src python -m minecombat.train.ppo \
  --config configs/ppo/zombie_vpt1x.yaml \
  --num-iterations 20 \
  --num-envs 8 \
  --batch-size 4 \
  --dry-run
```

Confirm `num_iterations: 20`, `num_envs: 8`, `batches_per_iteration: 200`, and
`fragment_length: 256`.

## 8-environment smoke

Run one tiny iteration first. This validates that 8 Minecraft clients start and
that the host has enough RAM:

```bash
bash scripts/start_ray.sh
bash scripts/train_zombie.sh --smoke --num-envs 8 --batch-size 8 \
  --save-path experiments/ppo_8env_smoke
```

If this OOMs, reduce to 4 environments or move to a larger host. Do not start
the 1M run until the 8-env smoke completes.

## 1M run

W&B must be online for the run to be useful:

```bash
wandb login
unset WANDB_MODE
bash scripts/train_zombie.sh \
  --num-iterations 20 \
  --num-envs 8 \
  --batch-size 4 \
  --save-path experiments/ppo_1m \
  --name zombie_vpt1x_1m
```

Checkpoints are saved at iterations 10 and 20 (`save_interval: 10`). Videos are
recorded every 2 iterations (`record_video_interval: 2`).

## Evaluation

Evaluate each checkpoint on the same 50 seeds as the M2 baseline:

```bash
python scripts/eval_baseline.py \
  --policy vpt \
  --checkpoint experiments/ppo_1m/checkpoints/<timestamp>/10/model.ckpt \
  --episodes 50 --seed-start 0 --max-steps 600 --deterministic

python scripts/eval_baseline.py \
  --policy vpt \
  --checkpoint experiments/ppo_1m/checkpoints/<timestamp>/20/model.ckpt \
  --episodes 50 --seed-start 0 --max-steps 600 --deterministic
```

Record win rate, death rate, timeout rate, mean HP lost, and time-to-kill. The
gate needs at least 50% win rate on the 50 seeds (#11).

## Artifacts to link in #10

- W&B run URL
- `experiments/ppo_1m/checkpoints/<timestamp>/20/`
- Evaluation run directories
- At least 10 episode videos from the best checkpoint

## Cleanup

```bash
bash scripts/stop_ray.sh
pgrep -f '[m]cprec-6.13.jar' | xargs -r kill -9
nvidia-smi
```

Check for orphaned Java processes after every crashed run; they keep rendering
at 100% CPU and hold GPU memory.

## Known limitations

- The value-target shape warning is fixed (#20).
- The local machine supports at most two clients because of system RAM, not
  because of the display (#21).
- Dense damage is active after the `custom.damage_dealt` observation fix (#23).
  Re-run the 8-environment smoke after pulling that change so the reward
  matches the training config.
