# 1M Run Handoff — RTX 4090D (24 GB / 60 GB RAM / 24 h)

Status: ready after the 3×50 re-baseline. The source checkpoint and curriculum
are fixed; do not change them mid-run.

## Goal

Continue flat VPT + PPO from the best visible-combat checkpoint for 1,024,000
environment steps on the mixed distance + ±30° curriculum. This is the M3
visible-combat run, not the full 360° search run.

## Source performance (3×50 re-baseline)

- Combined 3–7 + ±30°: **16% mean**, range 10–22%, 61% deaths.
- Random 360°: **10% mean**, range 8–14%, 63% deaths.

The 1M run tests whether scale can close a gap that 100k did not. If the final
combined mean stays below ~30%, stop and change the curriculum or task
definition instead of starting another long run.

## Host

- GPU: NVIDIA RTX 4090 D, 24,564 MiB
- System RAM: ~60 GB
- Window: up to 24 hours
- JDK 8 required for the Minecraft engine
- W&B online for training curves

## Source

- Repo: `https://github.com/MAS-W-415/MineCombat`
- Base commit: `658a133` or later
- Environment config:
  `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`
- Warm-start checkpoint (271 MB, not in git):

```text
experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/model.ckpt
```

Transfer it with `scp`/`rsync`, or upload it to a private Hugging Face repo and
download it on the host. Preserve the filename `model.ckpt`.

SHA256:

```text
22b845c9ee010b4b7a4457d382ef0ab56094ca507483d6fb65a5cbc9c6e2e1e3
```

## Preflight

```bash
git clone https://github.com/MAS-W-415/MineCombat.git
cd MineCombat
git checkout 658a133

python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt   # if present

export JAVA_HOME="$HOME/.local/opt/jdk8/usr/lib/jvm/java-8-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com

python -m pytest -q
ruff check .
```

Restore the Minecraft engine if `/tmp/MineStudio` is missing:

```bash
python - <<'PY'
from minestudio.simulator.entry import check_engine
check_engine(skip_confirmation=True)
PY
python scripts/patch_minestudio.py
python scripts/patch_minestudio_engine.py
```

Start Ray:

```bash
bash scripts/start_ray.sh
```

## 8-environment smoke

Run one tiny iteration and record peak RAM/VRAM and wall time:

```bash
bash scripts/train_zombie.sh --smoke \
  --num-envs 8 --batch-size 8 \
  --save-path experiments/ppo_8env_smoke_4090d
```

Check:

- all 8 Minecraft clients start;
- peak system RAM stays below ~55 GB;
- peak VRAM stays below ~22 GB;
- `received_episode_statistics` and wall time give a throughput estimate.

If 8 environments fail, try 4, then 2. Do not start the 1M run until the smoke
completes.

## 1M run

```bash
bash scripts/train_zombie.sh \
  --env-config configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml \
  --init-checkpoint experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/model.ckpt \
  --num-iterations 20 \
  --num-envs 8 \
  --batch-size 4 \
  --save-interval 5 \
  --record-video-interval 5 \
  --save-path experiments/flat_1m_mixed_angle30_4090d \
  --name flat_1m_mixed_angle30_4090d
```

W&B must be online:

```bash
wandb login
unset WANDB_MODE
```

Target: 20 iterations × 51,200 steps = 1,024,000 environment steps. Check the
first iteration's `received_episode_statistics` value. If it reports 8× the
expected count, adjust `--num-iterations` to keep the total near 1.02M.

Checkpoints are expected at iterations 5, 10, 15, and 20 (250k, 500k, 750k,
1M).

## Evaluation

Run on the host after training. Use 50 fixed seeds for triage and 3×50 for the
final gate.

| Task | Config | Runs |
|---|---|---|
| easy front 3–4 | `configs/env/zombie_stage1_easy.yaml` | 1×50 |
| ±30° close | `configs/env/zombie_stage1_easy_angle30_mask.yaml` | 1×50 |
| combined 3–7 + ±30° | `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml` | 3×50 |
| random 360° | `configs/env/zombie_stage1.yaml` | 3×50 |

Example:

```bash
python scripts/eval.py \
  --policy vpt \
  --checkpoint experiments/flat_1m_mixed_angle30_4090d/checkpoints/<timestamp>/20/model.ckpt \
  --config configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml \
  --episodes 50 --seed-start 0 --max-steps 600 --deterministic
```

Repeat the combined and random evaluations three times with the same seeds and
report mean and range. Single 50-seed runs vary by up to ~14 points because
Minecraft mob AI is not reset between episodes. See
`docs/m3-evaluation-variance.md`.

Evaluate the 250k and 500k checkpoints with one 50-seed run each to see whether
the learning curve is still rising.

## Time budget

| Phase | Estimate |
|---|---|
| Setup + preflight | 45–60 min |
| 8-env smoke | 15–30 min |
| 1M training | 2–6 h (measure in smoke) |
| Checkpoint triage (250k/500k) | 20–30 min |
| Final 3×50 evaluation | 45–90 min |
| Buffer | rest of the 24 h |

## Fallbacks

- 8 envs OOM → 4 envs → 2 envs.
- 1 env only → 1M is ~6.2 h at 45 steps/s; still feasible, but run only the
  final 3×50 evaluation.
- Training loss diverges / too-high-KL warnings → keep the last finite
  checkpoint, stop, and report the curve; do not restart blindly.
- GPU or engine crash → check for orphaned Java processes and clean
  `/tmp/MineStudio/{MP_tmp,tmp}` before retrying.

## Artifacts to return

- W&B run URL and training curves.
- `experiments/flat_1m_mixed_angle30_4090d/checkpoints/<timestamp>/{5,10,15,20}/`
- Evaluation run directories with `metrics.json` and `episodes.csv`.
- At least 10 episode videos from the best checkpoint.
- A short notes file with the smoke throughput, peak RAM/VRAM, and any warnings.
