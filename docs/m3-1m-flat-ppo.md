# M3 — 1M Flat PPO on the Mixed + Angle Curriculum

Date: 2026-11-06  
Source checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Hardware: NVIDIA RTX 4090 D (24 GB), 60 GB RAM, Ubuntu 24.04  
Code: `b54c53d`  
Environment: Python 3.10.22, PyTorch 2.5.1+cu124, MineStudio 1.1.6, Ray 2.60.0, JDK 8u504

## Training

```text
environment steps: 1,024,000
PPO iterations: 20
wall time: 4,673.81 s (77.9 min)
peak GPU memory: 6,035 MiB
peak system RAM: 40 GB
NaN / Inf: 0
too-high-KL warnings: 2,150
```

The run used 8 environments, batch size 4, `save_interval=5`, and
`record_video_interval=5`. The training command was:

```bash
bash scripts/train_zombie.sh \
  --env-config configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml \
  --init-checkpoint experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/model.ckpt \
  --num-iterations 20 --num-envs 8 --batch-size 4 \
  --save-interval 5 --record-video-interval 5 \
  --save-path experiments/flat_1m_mixed_angle30_4090d \
  --name flat_1m_mixed_angle30_4090d
```

W&B was disabled for this run because the first startup hung on the online
HF/W&B checks. The continuation run uses `--wandb-mode offline` with
`HF_HUB_OFFLINE=1`.

### Checkpoint

```text
local: /home/u2024210916/MineCombat/experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/model.ckpt
share: /eaas/default/groups/mine_/home/share/minecombat_results/flat_1m_mixed_angle30_4090d/checkpoint_1m/model.ckpt
SHA256: 830f9c4038af5b6ddd7e5ef4fee37f80cfce47230b3727c4fc6b089faaebf8b4
```

The intermediate 5/10/15 checkpoints were deleted by the default
`keep_interval=40`. The continuation run uses `--keep-interval 5`.

## Evaluation

All evaluations use seeds 0–49, max 600 steps, deterministic policy.
Easy and ±30° were run once (1×50); combined and random were run three times
(3×50) because those are the gate metrics.

| Task | Win | Death | HP lost | Time-to-kill |
|---|---:|---:|---:|---:|
| easy front 3–4 | **0.72** | 0.28 | 12.00 | 190.9 |
| ±30° close | **0.52** | 0.46 | 12.50 | 196.5 |
| combined 3–7 + ±30° r1 | 0.32 | 0.66 | 15.00 | 228.2 |
| combined r2 | 0.38 | 0.58 | 14.50 | 223.3 |
| combined r3 | 0.52 | 0.46 | 14.20 | 230.9 |
| **combined mean** | **0.407** | **0.567** | **14.57** | **227.5** |
| random 360° r1 | 0.22 | 0.62 | 15.50 | 291.5 |
| random r2 | 0.26 | 0.66 | 16.50 | 289.2 |
| random r3 | 0.20 | 0.46 | 15.00 | 266.3 |
| **random mean** | **0.227** | **0.580** | **15.67** | **282.3** |

Combined range: **0.32–0.52**. Random range: **0.20–0.26**.

Compared with the 100k checkpoint:

| Task | 100k | 1M | Delta |
|---|---:|---:|---:|
| easy | 0.44 | 0.72 | +0.28 |
| ±30° | 0.34 | 0.52 | +0.18 |
| combined | ~0.18 | 0.407 | +0.23 |
| random | 0.04–0.10 | 0.227 | +0.13 to +0.19 |

## Video review

Ten evaluation episodes were recorded on the combined task: 4 wins, 6 deaths.

- **Win episodes**: the zombie is visible, the policy turns toward it, closes
  distance, swings the sword, and kills it. This is purposeful combat, not
  reward hacking.
- **Death episodes**: after the zombie leaves the center of the view or lands a
  hit, the camera often points at the sky or ground and the policy cannot
  re-acquire the target. Attack timing is not the main failure; **camera
  tracking / target re-acquisition** is.

Videos are in the shared directory:

```text
/eaas/default/groups/mine_/home/share/minecombat_results/flat_1m_mixed_angle30_4090d/eval/videos/
```

## Decision

Per the pre-registered rules, the combined task is in **case B**:

- mean 40.7%, range 32–52%;
- clearly above the 16% 100k baseline;
- not yet at the 50% case-A threshold;
- random is 22.7%, still far below 50%.

The immediate next step is a second 1M continuation from this checkpoint
(2.048M total), with `--keep-interval 5` to retain intermediate checkpoints and
`--wandb-mode offline` for curves. The target is to see whether the combined
mean crosses 50% and whether random continues to improve.

## Artifacts

- `experiments/eval_1m_4090d/` (logs, metrics, per-episode CSVs, video manifest)
- `experiments/eval_1m_4090d/20261106_093929_eval_vpt_ckpt/` (easy)
- `experiments/eval_1m_4090d/20261106_095237_eval_vpt_ckpt/` (±30°)
- `experiments/eval_1m_4090d/20261106_100658_eval_vpt_ckpt/` (combined r1)
- `experiments/eval_1m_4090d/20261106_104142_eval_vpt_ckpt/` (combined r2)
- `experiments/eval_1m_4090d/20261106_111247_eval_vpt_ckpt/` (combined r3)
- `experiments/eval_1m_4090d/20261106_102555_eval_vpt_ckpt/` (random r1)
- `experiments/eval_1m_4090d/20261106_105837_eval_vpt_ckpt/` (random r2)
- `experiments/eval_1m_4090d/20261106_113201_eval_vpt_ckpt/` (random r3)
